#!/usr/bin/env python
"""
ACT (Action Chunking with Transformers) -- unified single-file pipeline (split -> train/val -> test -> figures), fp32.
Same protocol as the SmolVLA / SmollerVLA reference notebooks (Train/Validation/Test Design Guide v1.0): episode-level 70/10/20
stratified split (same seed/manifest), validation every `val_every` steps on the validation episodes (stride-50 windows, episode
weighted, sharded by episode_id % world), a single best_valid checkpoint, test with the full metric table + bootstrap CIs, latency,
complexity, report and figures. Training is multi-GPU DDP (built for Kaggle 2xT4).

Model: STOCK lerobot ACT (ResNet18 ImageNet backbone, 4-layer transformer encoder, 1-layer decoder, CVAE with latent 32, L1 + kl_weight*KLD).
chunk_size = n_action_steps = 50 so the 50-step chunk protocol (validation windows, test windows, jerk, seam jump) is unchanged.
Everything else (optimizer AdamW lr 1e-5 / wd 1e-4 / grad-clip 10, no scheduler, no EMA) is lerobot's ACT default.

Run:  python act_train.py --stage all        # split -> train (auto torchrun on all GPUs) -> test -> plots
Single stages: --stage split | train | test | plots
"""
import argparse, contextlib, copy, csv, dataclasses, glob, hashlib, importlib, importlib.metadata, importlib.util, inspect, json, math
import os, platform, random, re, shutil, subprocess, sys, time, traceback, warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # Kaggle T4x2: no peer-to-peer
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

SCRIPT = Path(__file__).resolve()
OUT_ROOT = SCRIPT.parent if SCRIPT.parent.name == "small_test_vla" else SCRIPT.parent / "small_test_vla"
SPLIT_DIR = OUT_ROOT / "splits"
MANIFEST_PATH = SPLIT_DIR / "split_manifest.json"
SCHEMA = "1.0"

# ----------------------------------------------------------------------------------------------- config (unchanged protocol constants)
DEFAULT_DATASET = "yamaisanerd/Chunking_560_Skynet"
SEEDS = dict(split_seed=42, train_seed=42, val_seed=1234, test_seed=2024, bootstrap_seed=20240)
RATIOS = dict(train=0.70, val=0.10, test=0.20)
CHUNK = 50
BOOT_B = 10000
METRIC_KEYS = ["mae", "mse", "rmse", "nmae", "jerk_in_chunk", "accel_in_chunk", "gt_jerk_in_chunk", "smoothness_ratio", "jerk_norm", "accel_norm"]


# ===================================================================================================================
# generic helpers
# ===================================================================================================================
def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha_obj(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism(warn_log):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # guide 6: TF32 off for the final benchmark
    torch.backends.cudnn.allow_tf32 = False
    orig = warnings.showwarning

    def hook(message, category, filename, lineno, file=None, line=None):
        m = str(message)
        if "deterministic" in m.lower():
            warn_log.add(m[:200])
        orig(message, category, filename, lineno, file, line)

    warnings.showwarning = hook


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def run_cmd(cmd, cwd=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=120).stdout.strip()
    except Exception:
        return ""


def git_info():
    d = str(SCRIPT.parent)
    sha = run_cmd(["git", "rev-parse", "HEAD"], d)
    if not sha:
        return "nogit:" + file_sha256(SCRIPT)[:12], False
    return sha, bool(run_cmd(["git", "status", "--porcelain"], d))


def lerobot_commit():
    try:
        du = importlib.metadata.distribution("lerobot").read_text("direct_url.json")
        return json.loads(du).get("vcs_info", {}).get("commit_id", "unknown") if du else "unknown"
    except Exception:
        return "unknown"


def hf_sha(repo, repo_type="model"):
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        return (api.dataset_info(repo) if repo_type == "dataset" else api.model_info(repo)).sha
    except Exception:
        return None


def first_import(*pairs):
    for mod, name in pairs:
        try:
            return getattr(importlib.import_module(mod), name)
        except Exception:
            continue
    raise ImportError(f"cannot import any of {pairs}")


def call_filtered(fn, **kw):
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kw)
    return fn(**{k: v for k, v in kw.items() if k in params})


# ===================================================================================================================
# distributed helpers
# ===================================================================================================================
class _Dist:
    rank = 0
    world = 1
    local_rank = 0

    @property
    def is_main(self):
        return self.rank == 0


DIST = _Dist()


def dist_init():
    DIST.world = int(os.environ.get("WORLD_SIZE", "1"))
    DIST.rank = int(os.environ.get("RANK", "0"))
    DIST.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(DIST.local_rank)
    if DIST.world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=3))


def barrier():
    if DIST.world > 1:
        dist.barrier(device_ids=[DIST.local_rank])


def all_gather_obj(obj):
    if DIST.world == 1:
        return [obj]
    out = [None] * DIST.world
    dist.all_gather_object(out, obj)
    return out


def all_reduce_vec(vals, op="sum"):
    t = torch.tensor(vals, dtype=torch.float64, device=f"cuda:{DIST.local_rank}")
    if DIST.world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
    return t.tolist()


def log0(*a, **k):
    if DIST.is_main:
        print(*a, **k, flush=True)


@contextlib.contextmanager
def rank_zero_first():
    if DIST.world > 1 and not DIST.is_main:
        barrier()
    try:
        yield
    finally:
        if DIST.world > 1 and DIST.is_main:
            barrier()


# ===================================================================================================================
# 1. SPLIT (identical to the reference: same seed -> same episode split; an existing manifest is re-used and hash-verified)
# ===================================================================================================================
def round_half_up(x):
    return int(math.floor(x + 0.5))


def _quota(sizes, frac, total, caps):
    fr = Fraction(str(frac))
    base = {s: int(math.floor(fr * sizes[s])) for s in sizes}
    base = {s: min(base[s], caps[s]) for s in sizes}
    rem = total - sum(base.values())
    order = sorted(sizes, key=lambda s: (-(fr * sizes[s] - math.floor(fr * sizes[s])), s))
    guard = 0
    while rem > 0 and guard < 10000:
        for s in order:
            if rem == 0:
                break
            if base[s] < caps[s]:
                base[s] += 1
                rem -= 1
        guard += 1
    assert rem == 0, "cannot satisfy split quotas"
    return base


def make_split(ep_rows, seed, ratios=RATIOS, min_stratum=5):
    ids = sorted(int(r["episode_index"]) for r in ep_rows)
    N = len(ids)
    n_test = round_half_up(ratios["test"] * N)
    n_val = round_half_up(ratios["val"] * N)
    task_of = {int(r["episode_index"]): str(r["task"]) for r in ep_rows}
    groups = defaultdict(list)
    for e in ids:
        groups[task_of[e]].append(e)
    strata = defaultdict(list)
    for t, v in groups.items():
        strata[t if len(v) >= min_stratum else "POOL"].extend(v)
    strata = {s: sorted(v) for s, v in strata.items()}
    rng = np.random.default_rng(seed)
    for s in sorted(strata):
        rng.shuffle(strata[s])
    sizes = {s: len(v) for s, v in strata.items()}
    q_test = _quota(sizes, ratios["test"], n_test, sizes)
    left = {s: sizes[s] - q_test[s] for s in sizes}
    q_val = _quota(sizes, ratios["val"], n_val, left)
    test, val, train, counts = [], [], [], {}
    for s in sorted(strata):
        v = strata[s]
        t_ = v[: q_test[s]]
        va = v[q_test[s]: q_test[s] + q_val[s]]
        tr = v[q_test[s] + q_val[s]:]
        test += t_
        val += va
        train += tr
        counts[s] = dict(train=len(tr), val=len(va), test=len(t_))
    test, val, train = sorted(test), sorted(val), sorted(train)
    assert not (set(test) & set(val)) and not (set(test) & set(train)) and not (set(val) & set(train))
    assert sorted(test + val + train) == ids
    assert len(test) == n_test and len(val) == n_val and len(train) == N - n_test - n_val
    return train, val, test, counts


def split_hash_of(m):
    return sha_obj({k: m[k] for k in ("train_ids", "val_ids", "test_ids", "split_seed", "ratios", "dataset_revision")})


def load_hub_tables(repo, revision):
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=["meta/*", "meta/**", "data/*", "data/**"]))
    info = read_json(root / "meta" / "info.json")
    jl = root / "meta" / "episodes.jsonl"
    if jl.exists():  # lerobot v2.x
        ep_df = pd.DataFrame([json.loads(l) for l in open(jl) if l.strip()])
    else:  # lerobot v3
        files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
        assert files, "no episode table found under meta/"
        ep_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return root, info, ep_df


def episode_rows(ep_df):
    rows = []
    for _, r in ep_df.iterrows():
        t = r["tasks"] if "tasks" in ep_df.columns else r.get("task", "")
        if isinstance(t, (list, tuple, np.ndarray)):
            t = t[0] if len(t) else ""
        rows.append(dict(episode_index=int(r["episode_index"]), task=str(t), length=int(r["length"]) if "length" in ep_df.columns else -1))
    return rows


def episode_content_hashes(root):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    acc = defaultdict(lambda: hashlib.sha256())
    for f in files:
        df = pq.read_table(f, columns=["episode_index", "action", "observation.state"]).to_pandas()
        for e, g in df.groupby("episode_index"):
            a = np.round(np.stack(g["action"].to_numpy()) * 1e3).astype(np.int64)
            s = np.round(np.stack(g["observation.state"].to_numpy()) * 1e3).astype(np.int64)
            acc[int(e)].update(a.tobytes() + s.tobytes())
    return {e: h.hexdigest() for e, h in acc.items()}


def stage_split(args):
    if MANIFEST_PATH.exists():
        m = read_json(MANIFEST_PATH)
        assert split_hash_of(m) == m["split_hash"], "split manifest hash mismatch (file was modified)"
        print(f"[split] manifest exists, hash verified: {m['split_hash'][:12]} ({m['n_train']}/{m['n_val']}/{m['n_test']})")
        return
    rev = hf_sha(args.dataset_repo, "dataset")
    root, info, ep_df = load_hub_tables(args.dataset_repo, rev)
    rows = episode_rows(ep_df)
    N = len(rows)
    warns = []
    if int(info.get("total_episodes", N)) != N:
        warns.append(f"meta/info.json says {info.get('total_episodes')} episodes but episode table has {N}; episode table used")
    print(f"[split] dataset {args.dataset_repo} rev={rev} episodes={N} frames(info)={info.get('total_frames')}")
    train, val, test, counts = make_split(rows, SEEDS["split_seed"])
    hashes = episode_content_hashes(root)  # G4
    which = {**{e: "train" for e in train}, **{e: "val" for e in val}, **{e: "test" for e in test}}
    seen = {}
    for e, h in sorted(hashes.items()):
        if h in seen and which[seen[h]] != which[e]:
            sys.exit(f"[G4] episodes {seen[h]} ({which[seen[h]]}) and {e} ({which[e]}) have identical content -> leakage, abort")
        seen.setdefault(h, e)
    cand = [c for c in ep_df.columns if re.search(r"session|operator|source|recording|collector|scene|setup", c, re.I)]  # G5
    if cand:
        for c in cand:
            sets = {n: set(ep_df[ep_df.episode_index.isin(ids)][c].astype(str)) for n, ids in (("train", train), ("val", val), ("test", test))}
            print(f"[G5] field '{c}': cross-split overlap train&test={len(sets['train'] & sets['test'])} train&val={len(sets['train'] & sets['val'])}")
            warns.append(f"G5 field {c}: overlap train&test={len(sets['train'] & sets['test'])}")
    else:
        print("[G5] no session/operator/source field in episode table")
    m = dict(schema_version=SCHEMA, dataset_repo_id=args.dataset_repo, dataset_revision=rev or "main(unpinned)",
             split_seed=SEEDS["split_seed"], ratios=RATIOS, n_total=N, n_train=len(train), n_val=len(val), n_test=len(test),
             train_ids=train, val_ids=val, test_ids=test, strata_counts=counts, excluded_episodes=[], warnings=warns)
    m["split_hash"] = split_hash_of(m)
    write_json(MANIFEST_PATH, m)
    print(f"[split] wrote {MANIFEST_PATH}  {len(train)}/{len(val)}/{len(test)}  hash={m['split_hash'][:12]}")


# ===================================================================================================================
# 2. MODEL: stock lerobot ACT (nothing is patched). cfg built by lerobot's own CLI parser, every default stays ACT's default,
#    except chunk_size = n_action_steps = 50 (the protocol's chunk length).
# ===================================================================================================================
ACT_FILES = ("configuration_act.py", "modeling_act.py", "processor_act.py")


def ensure_patch():
    """Keeps the installed lerobot STOCK (no patching at all for ACT). Returns the sha256 of the ACT source files for the manifest."""
    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("lerobot is not installed.")
    d = Path(list(spec.submodule_search_locations)[0]) / "policies" / "act"
    return {n: file_sha256(d / n) for n in ACT_FILES if (d / n).exists()}


def run_id_of(args):
    return args.run_id or f"act_c{CHUNK}_s{args.train_seed}"


def cfg_argv(args, run_dir):
    return [
        "--policy.type=act", "--policy.push_to_hub=false", f"--policy.device={args.device}",
        f"--policy.chunk_size={CHUNK}", f"--policy.n_action_steps={CHUNK}",
        f"--dataset.repo_id={args.dataset_repo}", f"--dataset.revision={args.dataset_revision}", f"--dataset.video_backend={args.video_backend}",
        f"--batch_size={args.batch_size}", f"--num_workers={args.num_workers}", f"--steps={args.steps}", "--log_freq=50",
        "--save_checkpoint=false", "--tf32=false", f"--seed={args.train_seed}", "--job_name=act",
        f"--output_dir={run_dir / '_lerobot_unused'}",
    ]


def parse_cfg(argv):
    TrainPipelineConfig = first_import(("lerobot.configs.train", "TrainPipelineConfig"))
    import lerobot.policies.act.configuration_act  # noqa: F401
    lr_parser = importlib.import_module("lerobot.configs.parser")

    @lr_parser.wrap()
    def _inner(cfg: TrainPipelineConfig):
        return cfg

    old = sys.argv
    sys.argv = [old[0]] + list(argv)
    try:
        cfg = _inner()
    finally:
        sys.argv = old
    cfg.validate()
    assert getattr(cfg, "ema", None) is None, "EMA must be disabled"
    assert not cfg.policy.pretrained_path, "R8: no pretrained policy checkpoint may be loaded"
    assert cfg.policy.chunk_size == CHUNK and cfg.policy.n_action_steps == CHUNK
    return cfg


def make_ds(cfg, episodes):
    make_dataset = first_import(("lerobot.datasets.factory", "make_dataset"), ("lerobot.datasets", "make_dataset"))
    c = copy.deepcopy(cfg)
    c.dataset.episodes = [int(e) for e in episodes]
    return make_dataset(c)


def make_pol_and_proc(cfg, ds):
    make_policy = first_import(("lerobot.policies.factory", "make_policy"))
    make_pp = first_import(("lerobot.policies.factory", "make_pre_post_processors"))
    policy = call_filtered(make_policy, cfg=cfg.policy, ds_meta=ds.meta, rename_map=None)
    policy.float()  # fp32 everywhere
    pre, post = call_filtered(make_pp, policy_cfg=cfg.policy, pretrained_path=None, dataset_stats=ds.meta.stats,
                              preprocessor_overrides={"device_processor": {"device": str(cfg.policy.device)}})
    return policy, pre, post


def _wif(worker_id):
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


_COLLATE_WARNED = set()


def safe_collate(batch):
    from torch.utils.data import default_collate

    batch = [{k: v for k, v in b.items() if not k.startswith("language_")} for b in batch]
    try:
        return default_collate(batch)
    except Exception:
        out = {}
        for k in batch[0]:
            vals = [b[k] for b in batch]
            try:
                out[k] = default_collate(vals)
            except Exception:
                if k not in _COLLATE_WARNED:
                    _COLLATE_WARNED.add(k)
                    print(f"[collate] dropping ragged key '{k}'", flush=True)
        return out


def make_loader(ds, bs, shuffle, seed, nw, indices=None, drop_last=False, sampler=None):
    dset = torch.utils.data.Subset(ds, list(indices)) if indices is not None else ds
    g = torch.Generator()
    g.manual_seed(seed)
    own_shuffle = shuffle and sampler is None
    kw = dict(batch_size=bs, shuffle=own_shuffle, num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
              generator=g if own_shuffle else None, worker_init_fn=_wif, sampler=sampler, collate_fn=safe_collate)
    if nw > 0:
        kw.update(persistent_workers=shuffle, prefetch_factor=2, multiprocessing_context="spawn")
    return torch.utils.data.DataLoader(dset, **kw)


# ---- validation speed-up (same as the SmolVLA notebook): validation windows are decoded from video ONCE and kept in RAM as uint8
# (lossless: decoded frames are k/255). Training and the test stage never use it.
def cache_val_batches(dl, tag="val"):
    out, nbytes, t0 = [], 0, time.time()
    for bi, batch in enumerate(dl):
        b = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.is_floating_point() and v.ndim == 4 and k.startswith("observation.images"):
                b[k] = (v * 255).round().clamp(0, 255).to(torch.uint8)
            else:
                b[k] = v
            if torch.is_tensor(b[k]):
                nbytes += b[k].numel() * b[k].element_size()
        out.append(b)
        if DIST.is_main and (bi % 5 == 0 or bi == len(dl) - 1):
            print(f"[{tag}] decoding validation windows: batch {bi + 1}/{len(dl)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[{tag}] rank {DIST.rank}: cached {len(out)} batches, {nbytes / 2 ** 30:.2f} GB in RAM, {time.time() - t0:.0f}s", flush=True)
    return out


def replay_batches(cached):
    for b in cached:
        yield {k: (v.float() / 255 if (torch.is_tensor(v) and v.dtype == torch.uint8 and v.ndim == 4 and k.startswith("observation.images")) else v)
               for k, v in b.items()}


def col_np(ds, name):
    try:
        return ds.hf_dataset.data.column(name).to_numpy()
    except Exception:
        return np.asarray([int(x) for x in ds.hf_dataset[name]])


def episode_index_map(ds):
    ep, fr = col_np(ds, "episode_index"), col_np(ds, "frame_index")
    out = defaultdict(dict)
    for i, (e, f) in enumerate(zip(ep, fr)):
        out[int(e)][int(f)] = i
    return out


def act_chunk(policy, batch):
    """Deterministic ACT inference on a preprocessed batch (eval mode: latent z = 0). Same call as ACTPolicy.predict_action_chunk
    without the action queue. Returns NORMALISED actions [B, CHUNK, action_dim]."""
    from lerobot.utils.constants import OBS_IMAGES

    b = dict(batch)
    b[OBS_IMAGES] = [b[k] for k in policy.config.image_features]
    return policy.model(b)[0]


def pad_key(batch):
    for k in ("action_is_pad", "actions_is_pad"):
        if k in batch:
            return batch[k].bool()
    return None


def forward_loss(policy, batch):
    out = policy.forward(batch)
    return out[0] if isinstance(out, (tuple, list)) else out


def _group_of(n):
    if ".backbone." in n:
        return "ResNet18 backbone (ImageNet-pretrained)"
    if "vae_encoder" in n:
        return "CVAE encoder (train-time only)"
    return "transformer encoder/decoder + projections"


def param_report(policy):
    groups = defaultdict(lambda: [0, 0])
    for n, p in policy.named_parameters():
        g = _group_of(n)
        groups[g][0] += p.numel()
        if p.requires_grad:
            groups[g][1] += p.numel()
    tot = sum(v[0] for v in groups.values())
    trn = sum(v[1] for v in groups.values())
    lines = [f"  {g:42s} total {v[0]:>13,d}   trainable {v[1]:>13,d}" for g, v in groups.items()]
    lines.append(f"  {'TOTAL':42s} total {tot:>13,d}   trainable {trn:>13,d}  ({100 * trn / max(tot, 1):.2f}% trainable)")
    print("[params]\n" + "\n".join(lines), flush=True)
    return {g: dict(total=v[0], trainable=v[1]) for g, v in groups.items()}, tot, trn


def n_params(policy):
    tot = sum(p.numel() for p in policy.parameters())
    trn = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    act = tot - sum(p.numel() for n, p in policy.named_parameters() if "vae_encoder" in n)  # CVAE encoder is unused at inference
    return tot, trn, act


def t_hash(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def init_audit(policy, run_dir):
    """R9 init record: backbone = ImageNet-pretrained (torchvision), everything else = scratch. All parameters are trainable in stock ACT."""
    rows, warns = [], []
    for n, p in policy.named_parameters():
        g = _group_of(n)
        src = "pretrained_imagenet" if g.startswith("ResNet18") else "scratch_trained"
        if not p.requires_grad:
            warns.append(f"{n}: parameter is frozen (stock ACT trains all parameters)")
        rows.append(dict(name=n, group=g, init_source=src, requires_grad=p.requires_grad, numel=p.numel(), sha256=t_hash(p)))
    write_json(run_dir / "init_audit.json", dict(summary=dict(params_by_source={s: sum(r["numel"] for r in rows if r["init_source"] == s) for s in {r["init_source"] for r in rows}}),
                                                 warnings=warns, rows=rows))
    return warns


# ===================================================================================================================
# 5. validation. ACT inference is deterministic (latent z = 0, no sampled noise / time), so the "fixed noise" of the flow-matching
#    protocol has no counterpart: val_loss = masked L1 (normalised actions, real dims, valid chunk steps) per stride-50 window,
#    episode-weighted, one pass that also yields the real-inference predictions for the test-style metrics.
# ===================================================================================================================
@torch.no_grad()
def eval_pass(policy, pre, dl, W, real, amean, astd, do_sample):
    policy.eval()
    lw = np.zeros(W)
    preds = np.zeros((W, CHUNK, real), np.float32) if do_sample else None
    gts = np.zeros((W, CHUNK, real), np.float32) if do_sample else None
    valids = np.zeros((W, CHUNK), bool) if do_sample else None
    a = 0
    for batch in dl:
        b = pre(batch)
        actions = b["action"]
        pad = pad_key(b)
        B = actions.shape[0]
        valid_b = (~pad) if pad is not None else torch.ones(B, actions.shape[1], dtype=torch.bool, device=actions.device)
        valid = valid_b.float()
        out = act_chunk(policy, b)
        m = (out - actions).abs()[..., :real].mean(-1)
        lw[a:a + B] = ((m * valid).sum(1) / valid.sum(1).clamp(min=1)).cpu().numpy()
        if do_sample:
            preds[a:a + B] = out[..., :real].float().cpu().numpy() * astd + amean
            gts[a:a + B] = actions[..., :real].float().cpu().numpy() * astd + amean
            valids[a:a + B] = valid_b.cpu().numpy()
        a += B
    assert a == W, f"validation loader yielded {a} windows, expected {W}"
    return lw, preds, gts, valids


def run_val(policy, pre, dl, wins, real, amean, astd, std_train, do_sample, gather):
    """Guide 2.4: each rank scores its own episodes (id % world == rank); results are merged in episode-id order so the value
    does not depend on world_size. Returns (val_loss, sd, per-episode loss, test-style summary, per-episode metrics, pack)."""
    lw, preds, gts, valids = eval_pass(policy, pre, dl, len(wins), real, amean, astd, do_sample)
    loss_ep = defaultdict(list)
    for (e, _), l in zip(wins, lw):
        loss_ep[e].append(l)
    loss_local = {e: float(np.mean(v)) for e, v in loss_ep.items()}
    met_local, pack_local = {}, None
    if do_sample:
        loc = defaultdict(list)
        for i, (e, _) in enumerate(wins):
            loc[e].append(window_metrics(preds[i], gts[i], valids[i], std_train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            met_local = {e: {k: float(np.nanmean([m[k] for m in ms])) for k in METRIC_KEYS} for e, ms in loc.items()}
        if gather:
            pack_local = dict(pred=preds, gt=gts, valid=valids, episode_id=np.array([w[0] for w in wins]), start=np.array([w[1] for w in wins]))
    le, per_ep, packs = {}, {}, []
    for d, m, p in all_gather_obj((loss_local, met_local, pack_local)):
        le.update(d)
        per_ep.update(m)
        if p is not None:
            packs.append(p)
    le, per_ep = dict(sorted(le.items())), dict(sorted(per_ep.items()))
    vals = np.array(list(le.values()))
    vl, vsd = float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    summary = None
    if do_sample:
        summary = {}
        for k in METRIC_KEYS:
            v = np.array([per_ep[e][k] for e in per_ep], dtype=float)
            f = v[np.isfinite(v)]
            summary[k] = dict(mean=float(f.mean()) if len(f) else float("nan"), sd=float(f.std(ddof=1)) if len(f) > 1 else 0.0,
                              median=float(np.median(f)) if len(f) else float("nan"))
    pack = None
    if packs:
        pack = {k: np.concatenate([p[k] for p in packs]) for k in packs[0]}
        o = np.lexsort((pack["start"], pack["episode_id"]))
        pack = {k: v[o] for k, v in pack.items()}
    return vl, vsd, le, summary, per_ep, pack


# ===================================================================================================================
# 6. checkpoint policy (single best_valid; the full ACT state_dict is stored: every parameter is trained, FrozenBN buffers included)
# ===================================================================================================================
def save_best(policy, run_dir, meta, cfg):
    ck = run_dir / "checkpoints"
    tmp, final, old = ck / "best_valid.tmp", ck / "best_valid", ck / "best_valid.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    from safetensors.torch import save_file

    sd = {k: v.detach().cpu().clone().contiguous() for k, v in policy.state_dict().items()}
    save_file(sd, str(tmp / "model.safetensors"))
    write_json(tmp / "policy_config.json", dataclasses.asdict(cfg.policy))
    meta = dict(meta, sha256=file_sha256(tmp / "model.safetensors"), n_tensors=len(sd))
    write_json(tmp / "checkpoint_meta.json", meta)
    for f in tmp.iterdir():
        fd = os.open(f, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    if final.exists():
        if old.exists():
            shutil.rmtree(old)
        os.rename(final, old)
    os.rename(tmp, final)
    if old.exists():
        shutil.rmtree(old)
    return meta["sha256"]


# ===================================================================================================================
# 7. TRAIN stage (loop structure unchanged: DDP, DistributedSampler, clip, step, validate every val_every steps, single best ckpt)
# ===================================================================================================================
METRIC_COLS = ["event", "step", "samples_seen", "wall_time_s", "loss_total", "loss_action", "lr", "grad_norm", "clipped", "nan_inf_count",
               "step_s", "data_s", "samples_per_s", "peak_vram_gb", "val_loss", "val_loss_sd_over_episodes", "n_val_episodes", "is_best", "best_val_loss"]
TESTLOG_COLS = (["step", "samples_seen", "wall_time_s", "val_loss", "is_best", "n_val_episodes"]
                + [f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "sd", "median")])


def env_info():
    return dict(python=platform.python_version(), torch=torch.__version__, torchvision=importlib.metadata.version("torchvision"),
                cuda=torch.version.cuda, driver=run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split("\n")[0],
                lerobot_commit=lerobot_commit(), lerobot_version=importlib.metadata.version("lerobot"))


def stage_train(args):
    dist_init()
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    if DIST.world > torch.cuda.device_count():
        sys.exit(f"world_size={DIST.world} but only {torch.cuda.device_count()} CUDA devices are visible")
    patch_hashes = ensure_patch()
    barrier()
    device = torch.device(f"cuda:{DIST.local_rank}")
    args.device = "cuda"
    warn_log = set()
    set_determinism(warn_log)
    seed_all(args.train_seed)
    log0(f"[train] world_size={DIST.world}  per-GPU batch={args.batch_size}  effective batch={args.batch_size * DIST.world}")
    try:
        import av.option  # noqa: F401
    except Exception:
        sys.exit("pyav problem: install `pip install -U 'av>=15,<16'` (lerobot needs av.option)")

    m = read_json(MANIFEST_PATH)  # only train/val ids survive in this process (G1, G2)
    assert split_hash_of(m) == m["split_hash"], "split_hash mismatch"
    train_ids, val_ids = m["train_ids"], m["val_ids"]
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(m["test_ids"])) and not (set(val_ids) & set(m["test_ids"]))
    split_hash = m["split_hash"]
    args.dataset_repo, args.dataset_revision = m["dataset_repo_id"], (m["dataset_revision"] if "unpinned" not in m["dataset_revision"] else "main")
    del m

    rid = run_id_of(args)
    run_dir = OUT_ROOT / "runs" / rid
    if DIST.is_main:
        if run_dir.exists():
            try:
                old_status = read_json(run_dir / "manifest.json").get("status", "unknown")
            except Exception:
                old_status = "unknown"
            if old_status in ("complete", "degenerate"):
                sys.exit(f"run {rid} already finished (status={old_status}); overwriting is an error (use --run_id or delete the folder)")
            print(f"[train] removing unfinished earlier run {rid} (status={old_status})", flush=True)
            shutil.rmtree(run_dir)
        for sub in ("checkpoints", "env", "cache", "test", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        (run_dir / "env" / "pip_freeze.txt").write_text(run_cmd([sys.executable, "-m", "pip", "freeze"]))
    barrier()

    argv = cfg_argv(args, run_dir)
    cfg = parse_cfg(argv)
    pc = cfg.policy
    assert pc.chunk_size == CHUNK and int(pc.n_action_steps) == CHUNK
    resolved = json.loads(canon(dataclasses.asdict(cfg)))
    for k in ("output_dir", "job_name"):
        resolved.pop(k, None)
    resolved["dataset"].pop("root", None)
    resolved["dataset"].pop("episodes", None)
    sch = cfg.scheduler
    config_doc = dict(
        policy=resolved["policy"], backbone=dict(id=pc.vision_backbone, weights=pc.pretrained_backbone_weights, dtype="float32"),
        data=dict(repo_id=args.dataset_repo, revision=args.dataset_revision, split_manifest=str(MANIFEST_PATH.relative_to(OUT_ROOT)), split_hash=split_hash,
                  video_backend=args.video_backend),
        optim=dict(lr=cfg.optimizer.lr, betas=list(cfg.optimizer.betas), eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay,
                   grad_clip_norm=cfg.optimizer.grad_clip_norm, scheduler=None if sch is None else sch.__class__.__name__,
                   warmup_steps=getattr(sch, "num_warmup_steps", None), decay_steps=getattr(sch, "num_decay_steps", None), decay_lr=getattr(sch, "decay_lr", None)),
        train=dict(steps=args.steps, batch_size=args.batch_size, world_size=DIST.world, effective_batch_size=args.batch_size * DIST.world,
                   num_workers=args.num_workers, val_every=args.val_every, save_freq=0, ema=None, grad_accum=1, precision="fp32",
                   parallelism="DDP" if DIST.world > 1 else "single"),
        eval=dict(val=dict(metric="masked L1 (normalised actions, real dims), deterministic z=0", stride=CHUNK, dims="real"),
                  test=dict(windows_per_episode=9, draws=1, test_seed=SEEDS["test_seed"], bootstrap_B=BOOT_B)),
        seeds=SEEDS | dict(train_seed=args.train_seed),
        determinism=dict(deterministic_algorithms="warn_only", cudnn_deterministic=True, tf32=False, cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"]),
        cli_argv=argv,
    )
    config_hash = sha_obj({k: v for k, v in config_doc.items() if k != "cli_argv"})
    if DIST.is_main:
        try:
            import yaml

            (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(canon(config_doc)), sort_keys=False))
        except Exception:
            (run_dir / "config.yaml").write_text(json.dumps(config_doc, indent=2, default=str))

    gsha, gdirty = git_info()
    manifest = dict(
        schema_version=SCHEMA, run_id=rid, model_key=f"ACT_C{CHUNK}", status="running", valid=False,
        model=dict(arch="ACT (stock lerobot)", chunk_size=CHUNK, use_vae=pc.use_vae, kl_weight=pc.kl_weight, backbone_id=pc.vision_backbone, gated=False),
        revisions=dict(backbone_weights=pc.pretrained_backbone_weights, dataset=args.dataset_revision),
        seeds=config_doc["seeds"], hashes=dict(config_hash=config_hash, split_hash=split_hash, best_ckpt_sha256=None),
        code=dict(git_sha=gsha, git_dirty=gdirty, script_sha256=file_sha256(SCRIPT), lerobot_files_sha256=patch_hashes, **env_info()),
        hardware=dict(gpu_names=[torch.cuda.get_device_name(i) for i in range(DIST.world)], world_size=DIST.world, mixed_precision="no (fp32)", tf32=False),
        training=dict(steps=args.steps, batch_size_per_device=args.batch_size, effective_batch_size=args.batch_size * DIST.world),
        timestamps=dict(started_utc=now_utc()), warnings=[],
    )
    if DIST.is_main:
        write_json(run_dir / "manifest.json", manifest)
    cfg.policy.device = str(device)

    with rank_zero_first():
        train_ds = make_ds(cfg, train_ids)
        val_ds = make_ds(cfg, val_ids)
    assert sorted({int(e) for e in col_np(train_ds, "episode_index")}) == sorted(train_ids), "G2: train dataset episodes != manifest"
    with rank_zero_first():
        policy, pre, post = make_pol_and_proc(cfg, train_ds)
    policy.to(device)
    real_dims = int(policy.config.action_feature.shape[0])
    assert real_dims == 6, f"expected 6 real action dims, got {real_dims}"
    audit_warn = init_audit(policy, run_dir) if DIST.is_main else []
    manifest["warnings"] += audit_warn
    tot, trn, act = n_params(policy)
    if DIST.is_main:
        by_group, _, _ = param_report(policy)
        manifest["model"]["params_by_group"] = by_group
    manifest["model"].update(params_total=tot, params_trainable=trn, params_active=act)

    std_obj = None
    if DIST.is_main:
        acts = np.asarray(train_ds.hf_dataset.data.column("action").to_pylist(), dtype=np.float64)[:, :real_dims]
        write_json(run_dir / "train_action_std.json", dict(std=acts.std(0).tolist(), mean=acts.mean(0).tolist(), n_frames=len(acts)))
        std_obj = acts.std(0).tolist()
        del acts
    std_train = np.asarray(all_gather_obj(std_obj)[0], dtype=np.float64)
    amean = np.asarray(train_ds.meta.stats["action"]["mean"], dtype=np.float64)[:real_dims]
    astd = np.asarray(train_ds.meta.stats["action"]["std"], dtype=np.float64)[:real_dims]
    val_metrics_on = not args.no_val_metrics

    make_opt = first_import(("lerobot.optim.factory", "make_optimizer_and_scheduler"))
    optimizer, scheduler = make_opt(cfg, policy)  # ACT preset: AdamW, no scheduler -> scheduler is None
    trainable = [p for p in policy.parameters() if p.requires_grad]
    clip = float(cfg.optimizer.grad_clip_norm)

    # ---- validation windows of this rank (episode_id % world == rank)
    t_c = time.time()
    shard_ids = [e for e in val_ids if e % DIST.world == DIST.rank]
    idx_map = episode_index_map(val_ds)
    wins, idxs = [], []
    for e in sorted(shard_ids):
        fmap = idx_map[e]
        for s in range(0, len(fmap), CHUNK):
            wins.append((e, s))
            idxs.append(fmap[s])
    n_val_windows = sum(all_gather_obj(len(wins)))
    val_batches = cache_val_batches(make_loader(val_ds, args.val_batch, False, SEEDS["val_seed"], args.num_workers, indices=idxs))
    val_dl = lambda: replay_batches(val_batches)
    cache_build_s = time.time() - t_c
    n_val_eps = len(val_ids)

    hist = wr = tlog = twr = None
    if DIST.is_main:
        hist = open(run_dir / "metrics_history.csv", "w", newline="")
        wr = csv.DictWriter(hist, fieldnames=METRIC_COLS, restval="")
        wr.writeheader()
        if val_metrics_on:
            tlog = open(run_dir / "test_log.csv", "w", newline="")
            twr = csv.DictWriter(tlog, fieldnames=TESTLOG_COLS, restval="")
            twr.writeheader()
    sampler = torch.utils.data.DistributedSampler(train_ds, num_replicas=DIST.world, rank=DIST.rank, shuffle=True, seed=args.train_seed, drop_last=True)
    dl = make_loader(train_ds, args.batch_size, True, args.train_seed, args.num_workers, drop_last=True, sampler=sampler)
    epoch = 0
    sampler.set_epoch(epoch)
    it = iter(dl)
    ddp = policy
    if DIST.world > 1:  # find_unused_parameters=True as in the SmolVLA pipeline (lerobot's own DDP setting)
        ddp = nn.parallel.DistributedDataParallel(policy, device_ids=[DIST.local_rank], output_device=DIST.local_rank, find_unused_parameters=True)
    seed_all(args.train_seed + DIST.rank)
    eff_bs = args.batch_size * DIST.world
    best, first_val, best_step, status = float("inf"), None, -1, "running"
    wall_best = trainonly_best = val_best = None
    val_time = 0.0
    win = dict(loss=0.0, step=0.0, data=0.0, gn=0.0, clip=0.0, nan=0, n=0)
    t0 = time.time()
    log0(f"[train] starting: {args.steps} steps, logging every {args.log_every}, validation every {args.val_every}"
         f"{' (+ test-style metrics -> test_log.csv)' if val_metrics_on else ''}")
    ddp.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(1, args.steps + 1):
            ts = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                it = iter(dl)
                batch = next(it)
            data_s = time.perf_counter() - ts
            batch = pre(batch)
            out = ddp(batch)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            bad = float(not bool(torch.isfinite(loss)))
            if DIST.world > 1:
                bad = all_reduce_vec([bad], "max")[0]
            if bad:
                win["nan"] += 1
                raise FloatingPointError(f"non-finite training loss at step {step}")
            loss.backward()
            win["n"] += 1
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, clip))
            optimizer.step()
            if scheduler is not None:  # ACT has no scheduler preset
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            win["loss"] += float(loss.detach())
            win["gn"] += gn
            win["clip"] += float(gn > clip)
            win["data"] += data_s
            win["step"] += time.perf_counter() - ts
            if step % args.log_every == 0 or step == 1 or step == args.steps:
                n = win["n"]
                d = n * DIST.world
                avg_loss, avg_gn, avg_clip, avg_data, avg_step = [x / d for x in all_reduce_vec([win["loss"], win["gn"], win["clip"], win["data"], win["step"]])]
                if DIST.is_main:
                    wr.writerow(dict(event="train", step=step, samples_seen=step * eff_bs, wall_time_s=round(time.time() - t0, 2),
                                     loss_total=avg_loss, loss_action=avg_loss, lr=optimizer.param_groups[0]["lr"], grad_norm=avg_gn,
                                     clipped=avg_clip, nan_inf_count=win["nan"], step_s=avg_step, data_s=avg_data,
                                     samples_per_s=eff_bs * n / max(win["step"], 1e-9), peak_vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
                    hist.flush()
                    print(f"[train] step {step}/{args.steps} loss={avg_loss:.4f} gn={avg_gn:.2f} lr={optimizer.param_groups[0]['lr']:.2e} {avg_step:.2f}s/step  ETA {(time.time() - t0 - val_time) / step * (args.steps - step) / 60:.0f} min", flush=True)
                win = dict(loss=0.0, step=0.0, data=0.0, gn=0.0, clip=0.0, nan=0, n=0)
            if step % args.val_every == 0 or step == args.steps:
                tv = time.time()
                log0(f"[val] step {step}: evaluating {n_val_eps} validation episodes ...")
                # loss + real-inference metrics in ONE pass; raw predictions are always gathered (small)
                vl, vsd, le, vm, vm_ep, vm_pack = run_val(policy, pre, val_dl(), wins, real_dims, amean, astd, std_train,
                                                          do_sample=val_metrics_on, gather=val_metrics_on)
                finite = math.isfinite(vl)
                if first_val is None and finite:
                    first_val = vl
                is_best = finite and vl < best and not (abs(vl - best) <= 1e-9)
                ddp.train()
                val_time += time.time() - tv
                if is_best:
                    best, best_step = vl, step
                    wall_best = time.time() - t0
                    val_best = val_time
                    trainonly_best = wall_best - val_best
                    if DIST.is_main:
                        sha = save_best(policy, run_dir, dict(step=step, val_loss=vl, split_hash=split_hash, config_hash=config_hash, git_sha=gsha,
                                                               backbone_weights=pc.pretrained_backbone_weights, wall_time_s=wall_best), cfg)
                        manifest["hashes"]["best_ckpt_sha256"] = sha
                        with open(run_dir / "checkpoints" / "best_history.csv", "a", newline="") as f:
                            w = csv.writer(f)
                            if f.tell() == 0:
                                w.writerow(["step", "val_loss", "wall_time_s"])
                            w.writerow([step, vl, round(wall_best, 2)])
                        with open(run_dir / "val_per_episode_best.csv", "w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["episode_id", "val_loss"])
                            w.writerows(le.items())
                        if vm_pack is not None:
                            np.savez_compressed(run_dir / "val_predictions_best.npz", step=step, **vm_pack)
                            with open(run_dir / "val_per_episode_metrics_best.csv", "w", newline="") as f:
                                w = csv.writer(f)
                                w.writerow(["episode_id"] + METRIC_KEYS)
                                for e, d_ in vm_ep.items():
                                    w.writerow([e] + [d_[k] for k in METRIC_KEYS])
                if DIST.is_main:
                    wr.writerow(dict(event="val", step=step, samples_seen=step * eff_bs, wall_time_s=round(time.time() - t0, 2), val_loss=vl,
                                     val_loss_sd_over_episodes=vsd, n_val_episodes=n_val_eps, is_best=bool(is_best), best_val_loss=best,
                                     peak_vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
                    hist.flush()
                    print(f"[val] step {step} val_loss={vl:.5f} (best {best:.5f} @ {best_step}){' *' if is_best else ''}  [{time.time() - tv:.0f}s]", flush=True)
                    if vm is not None:
                        row = dict(step=step, samples_seen=step * eff_bs, wall_time_s=round(time.time() - t0, 2), val_loss=vl, is_best=bool(is_best), n_val_episodes=n_val_eps)
                        for k in METRIC_KEYS:
                            for s_ in ("mean", "sd", "median"):
                                row[f"{k}_{s_}"] = vm[k][s_]
                        twr.writerow(row)
                        tlog.flush()
                        print(f"[test_log] step {step} (validation episodes, real inference): MAE={vm['mae']['mean']:.4f} nMAE={vm['nmae']['mean']:.4f} "
                              f"MSE={vm['mse']['mean']:.4f} RMSE={vm['rmse']['mean']:.4f} jerk_norm={vm['jerk_norm']['mean']:.4f} "
                              f"smooth={vm['smoothness_ratio']['mean']:.3f}", flush=True)
        total_wall = time.time() - t0
        if first_val is None or best_step < 0:
            status = "failed"
        elif best > 0.8 * first_val:
            status = "degenerate"
        else:
            status = "complete"
    except Exception as e:
        status = "failed"
        manifest["warnings"].append("".join(traceback.format_exception_only(type(e), e)).strip())
        traceback.print_exc()
        total_wall = time.time() - t0
    if hist:
        hist.close()
    if tlog:
        tlog.close()

    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    manifest.update(status=status, valid=status == "complete")
    manifest["training"].update(
        best_step=best_step, best_val_loss=None if best_step < 0 else best, first_val_loss=first_val,
        wall_time_total_s=wall_best, wall_time_train_only_s=trainonly_best, wall_time_val_s=val_best,
        gpu_hours=None if wall_best is None else wall_best / 3600.0,
        wall_time_full_run_s=total_wall, wall_time_full_run_val_s=val_time, val_setup_s=cache_build_s,
        steps_per_s=(args.steps / max(total_wall - val_time, 1e-9)), samples_per_s=(args.steps * eff_bs / max(total_wall - val_time, 1e-9)),
        peak_train_vram_gb=peak, n_val_windows=n_val_windows, world_size=DIST.world,
        val_note="ACT validation is deterministic masked L1 (no sampled noise / time); training loss is L1 + kl_weight*KLD",
        test_log_csv=val_metrics_on,
    )
    manifest["timestamps"]["finished_utc"] = now_utc()
    manifest["warnings"] += sorted(warn_log)
    if DIST.is_main:
        write_json(run_dir / "manifest.json", manifest)
        print(f"[train] status={status} best_val={best:.5f}@{best_step} time_to_best={wall_best}")
        try:
            print(build_text_report(run_dir), flush=True)
            stage_plots(args)
        except Exception:
            print("[train] report / figure generation failed (re-run with --stage plots):")
            traceback.print_exc()
    if status == "failed":
        sys.exit(2)
    if DIST.world > 1:
        barrier()
        dist.destroy_process_group()


# ===================================================================================================================
# 8. TEST stage
# ===================================================================================================================
def window_metrics(pred, gt, valid, std):
    n = int(valid.sum())
    p, g = pred[valid], gt[valid]
    err = p - g
    mae_d = np.abs(err).mean(0)
    out = dict(mae=float(mae_d.mean()), mse=float((err ** 2).mean()), nmae=float((mae_d / std).mean()))
    out["rmse"] = math.sqrt(out["mse"])
    for name, order in (("jerk", 3), ("accel", 2)):
        if n >= order + 1:
            dp, dg = np.abs(np.diff(p, n=order, axis=0)), np.abs(np.diff(g, n=order, axis=0))
            out[f"{name}_in_chunk"], out[f"gt_{name}_in_chunk"] = float(dp.mean()), float(dg.mean())
            out[f"{name}_norm"] = float((dp.mean(0) / std).mean())
        else:
            out[f"{name}_in_chunk"] = out[f"gt_{name}_in_chunk"] = out[f"{name}_norm"] = float("nan")
    gj = out["gt_jerk_in_chunk"]
    out["smoothness_ratio"] = out["jerk_in_chunk"] / gj if gj and gj > 1e-12 else float("nan")
    return out


def test_starts(L, n=9):
    feas = max(L - CHUNK, 0)
    return [int(x) for x in np.unique(np.round(np.linspace(0, feas, n)).astype(int))]


def bootstrap_ci(x, B, seed, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def summarize(x, seed):
    x = np.asarray(x, dtype=float)
    f = x[np.isfinite(x)]
    lo, hi = bootstrap_ci(f, BOOT_B, seed)
    return dict(mean=float(f.mean()), sd=float(f.std(ddof=1)) if len(f) > 1 else 0.0, median=float(np.median(f)), p90=float(np.percentile(f, 90)), ci_lo=lo, ci_hi=hi)


def seam_jump(chunk_a, chunk_b):
    return float(np.linalg.norm(chunk_a[-1] - chunk_b[0]))


def stage_test(args):
    patch_hashes = ensure_patch()
    device = torch.device(args.device)
    warn_log = set()
    set_determinism(warn_log)
    rid = run_id_of(args)
    run_dir = OUT_ROOT / "runs" / rid
    tdir = run_dir / "test"
    man = read_json(run_dir / "manifest.json")
    ck = run_dir / "checkpoints"
    errs = []
    if man["status"] not in ("complete", "degenerate"):
        errs.append(f"P1 status={man['status']}")
    if file_sha256(ck / "best_valid" / "model.safetensors") != read_json(ck / "best_valid" / "checkpoint_meta.json")["sha256"]:
        errs.append("P2 checkpoint sha256 mismatch")
    sm = read_json(MANIFEST_PATH)
    if split_hash_of(sm) != sm["split_hash"] or sm["split_hash"] != man["hashes"]["split_hash"]:
        errs.append("P3 split_hash mismatch")
    if (tdir / "test.lock").exists():
        errs.append("P5 test.lock exists (test already run or crashed; remove it deliberately to re-run)")
    if len(sm["test_ids"]) != sm["n_test"]:
        errs.append("P6 n_test mismatch")
    extra = [p.name for p in ck.iterdir() if p.name not in ("best_valid", "best_history.csv")]
    if extra:
        errs.append(f"P7 unexpected files in checkpoints/: {extra}")
    cfg_doc = None
    if not errs:
        txt = (run_dir / "config.yaml").read_text()
        try:
            cfg_doc = json.loads(txt)
        except json.JSONDecodeError:
            import yaml

            cfg_doc = yaml.safe_load(txt)
        args.dataset_repo, args.dataset_revision = cfg_doc["data"]["repo_id"], cfg_doc["data"]["revision"]
    if errs:
        sys.exit("[test] preconditions failed:\n  " + "\n  ".join(errs))
    test_ids = sm["test_ids"]
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "test.lock").write_text("running " + now_utc())
    seed_all(SEEDS["test_seed"])

    cfg = parse_cfg(cfg_doc["cli_argv"])
    resolved = json.loads(canon(dataclasses.asdict(cfg)))
    resolved.pop("output_dir", None)
    resolved.pop("job_name", None)
    resolved["dataset"].pop("root", None)
    resolved["dataset"].pop("episodes", None)
    if resolved["policy"] != cfg_doc["policy"]:
        sys.exit("P4 config_hash mismatch: policy config differs from the training run (different lerobot version?)")
    ds = make_ds(cfg, test_ids)
    policy, pre, post = make_pol_and_proc(cfg, ds)
    policy.to(device)
    from safetensors.torch import load_file

    sd = load_file(str(ck / "best_valid" / "model.safetensors"))
    res = policy.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, f"S2 strict load failed: missing={res.missing_keys[:5]} unexpected={res.unexpected_keys[:5]}"
    policy.eval()
    real = int(policy.config.action_feature.shape[0])
    amean = np.asarray(ds.meta.stats["action"]["mean"], dtype=np.float64)[:real]
    astd = np.asarray(ds.meta.stats["action"]["std"], dtype=np.float64)[:real]
    assert str(cfg.policy.normalization_mapping["ACTION"]).endswith("MEAN_STD")
    tr = read_json(run_dir / "train_action_std.json")
    std_train = np.asarray(tr["std"])

    idx_map = episode_index_map(ds)
    rows = []
    for e in sorted(test_ids):
        fmap = idx_map[e]
        L = len(fmap)
        for wi, s in enumerate(test_starts(L)):
            rows.append(dict(ep=e, start=s, widx=str(wi), kind="main", ds_idx=fmap[s]))
        if L >= 2 * CHUNK:
            rg = np.random.default_rng(stable_seed(SEEDS["test_seed"], e, "seam"))
            s0 = int(rg.integers(0, L - 2 * CHUNK + 1))
            rows.append(dict(ep=e, start=s0, widx="seam_a", kind="seam", ds_idx=fmap[s0]))
            rows.append(dict(ep=e, start=s0 + CHUNK, widx="seam_b", kind="seam", ds_idx=fmap[s0 + CHUNK]))
    dl = make_loader(ds, args.test_batch, False, SEEDS["test_seed"], args.num_workers, indices=[r["ds_idx"] for r in rows])
    preds, gts, valids = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if bi % 10 == 0 or bi == len(dl) - 1:
                print(f"[test] inference batch {bi + 1}/{len(dl)}", flush=True)
            gt = batch["action"].clone().float().numpy()[..., :real]
            pad = batch["action_is_pad"].numpy() if "action_is_pad" in batch else np.zeros(gt.shape[:2], bool)
            b = pre(batch)
            out = act_chunk(policy, b)[..., :real].float().cpu().numpy() * astd + amean
            for j in range(gt.shape[0]):
                preds.append(out[j])
                gts.append(gt[j])
                valids.append(~pad[j])
    preds, gts, valids = np.stack(preds), np.stack(gts), np.stack(valids)
    print(f"[test] inferred {len(rows)} windows ({sum(r['kind'] == 'main' for r in rows)} main + seam pairs)")

    wrows, per_ep = [], defaultdict(list)
    keys = METRIC_KEYS
    for i, r in enumerate(rows):
        if r["kind"] != "main":
            continue
        m = window_metrics(preds[i], gts[i], valids[i], std_train)
        wrows.append(dict(episode_id=r["ep"], window_idx=r["widx"], frame_index=r["start"], draw=0, **{k: m[k] for k in keys}))
        per_ep[r["ep"]].append(m)
    task_of = {}
    try:
        _, _, ep_df = load_hub_tables(args.dataset_repo, None)
        task_of = {x["episode_index"]: x["task"] for x in episode_rows(ep_df)}
    except Exception:
        pass
    ep_out = []
    for e in sorted(per_ep):
        ms = per_ep[e]
        ep_out.append(dict(run_id=rid, model_key=man["model_key"], episode_id=e, task=task_of.get(e, ""), condition="", n_windows=len(ms),
                           **{k: float(np.nanmean([m[k] for m in ms])) for k in keys}, checkpoint_step=man["training"]["best_step"]))
    with open(tdir / "test_per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_out[0].keys()))
        w.writeheader()
        w.writerows(ep_out)
    with open(tdir / "test_per_window.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0].keys()))
        w.writeheader()
        w.writerows(wrows)
    mi = [i for i, r in enumerate(rows) if r["kind"] == "main"]
    np.savez_compressed(tdir / "predictions.npz", pred=preds[mi].astype(np.float32), gt=gts[mi].astype(np.float32), valid=valids[mi],
                        episode_id=np.array([rows[i]["ep"] for i in mi]), start=np.array([rows[i]["start"] for i in mi]))
    seams = []
    for i, r in enumerate(rows):
        if r["widx"] == "seam_a":
            seams.append(dict(episode_id=r["ep"], start_a=r["start"], l2_jump=seam_jump(preds[i], preds[i + 1])))
    with open(tdir / "bjump_seams.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "start_a", "l2_jump"])
        w.writeheader()
        w.writerows(seams)
    jumps = np.array([s["l2_jump"] for s in seams])
    wl = []
    if len(jumps) < 100:
        wl.append(f"bjump indicative only: n_seams={len(jumps)} < 100")

    # ---- latency (batch 1, fp32; act_chunk on prepared device tensors -> actions on CPU)
    g = np.random.default_rng(SEEDS["test_seed"])
    lat_main = [r for r in rows if r["kind"] == "main"]
    pick = [lat_main[i]["ds_idx"] for i in g.choice(len(lat_main), size=min(100, len(lat_main)), replace=False)]
    prepared = []
    for batch in make_loader(ds, 1, False, 0, 0, indices=pick):
        prepared.append(pre(batch))
    torch.cuda.reset_peak_memory_stats()
    lat = []

    def one(p):
        torch.cuda.synchronize()
        t = time.perf_counter()
        act_chunk(policy, p)[..., :real].float().cpu().numpy()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    with torch.no_grad():
        for k in range(10):
            one(prepared[k % len(prepared)])
        for k in range(300):
            lat.append(one(prepared[k % len(prepared)]))
    peak_infer = torch.cuda.max_memory_allocated() / 2 ** 30
    lat = np.array(lat)
    with open(tdir / "latency_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "latency_ms"])
        w.writerows((i, a) for i, a in enumerate(lat))
    latency = dict(n_runs=300, warmup=10, p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)), p99_ms=float(np.percentile(lat, 99)),
                   mean_ms=float(lat.mean()), std_ms=float(lat.std(ddof=1)), device=torch.cuda.get_device_name(0), dtype="float32")

    # ---- complexity
    from torch.utils.flop_counter import FlopCounterMode

    def fl(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops()

    one_batch = None
    for batch in make_loader(ds, 1, False, 0, 0, indices=pick[:1]):
        one_batch = pre(batch)
    policy.eval()
    with torch.no_grad():
        f_inf = fl(lambda: act_chunk(policy, one_batch))
    policy.train()  # training forward (CVAE encoder active) for the forward / train-step FLOPs
    with torch.no_grad():
        f_fwd = fl(lambda: forward_loss(policy, one_batch))
    f_trn = fl(lambda: forward_loss(policy, one_batch).backward())
    policy.zero_grad(set_to_none=True)
    policy.eval()
    tot, trn, act = n_params(policy)
    param_report(policy)
    complexity = dict(gflops_infer_chunk=f_inf / 1e9, gmacs_infer_chunk=f_inf / 2e9, gflops_forward=f_fwd / 1e9, gflops_train_step=f_trn / 1e9,
                      params_total=tot, params_trainable=trn, params_active=act, peak_infer_vram_gb=peak_infer, counter="torch.utils.flop_counter.FlopCounterMode",
                      counter_note="matmul/bmm/conv/sdpa only (1 MAC = 2 FLOPs); elementwise ops and softmax are not counted")

    metrics = {k: summarize([r[k] for r in ep_out], SEEDS["bootstrap_seed"]) for k in
               ["mae", "nmae", "mse", "rmse", "jerk_norm", "gt_jerk_in_chunk", "accel_norm", "smoothness_ratio"]}
    metrics["gt_jerk"] = metrics.pop("gt_jerk_in_chunk")
    metrics["jerk_in_chunk"] = summarize([r["jerk_in_chunk"] for r in ep_out], SEEDS["bootstrap_seed"])
    if metrics["mse"]["sd"] > 0.7 * metrics["mse"]["mean"]:
        wl.append("MSE SD is close to its mean: a few episodes dominate (see median/p90)")
    report = dict(
        schema_version=SCHEMA, run_id=rid, model_key=man["model_key"], valid=man["valid"],
        checkpoint=dict(step=man["training"]["best_step"], val_loss=man["training"]["best_val_loss"], sha256=man["hashes"]["best_ckpt_sha256"]),
        n_test_episodes=len(ep_out), n_test_windows=len(wrows), metrics=metrics,
        bjump=dict(mean=float(jumps.mean()) if len(jumps) else None, sd=float(jumps.std(ddof=1)) if len(jumps) > 1 else None, n_seams=len(jumps),
                   mode="seeded_one_seam_pair_per_episode"),
        latency=latency, complexity=complexity,
        training=dict(wall_time_total_s=man["training"]["wall_time_total_s"], gpu_hours=man["training"]["gpu_hours"], steps_per_s=man["training"]["steps_per_s"]),
        protocol=dict(windows_per_episode=9, draws_per_window=1, bootstrap_B=BOOT_B, test_seed=SEEDS["test_seed"], train_action_std=tr["std"],
                      nmae="mean_d(MAE_d / std_d(train))", jerk_norm="mean_d(mean|3rd diff|_d / std_d(train))", dtype=dict(model="fp32"),
                      chunk_size=CHUNK, inference="deterministic (z=0)"),
        warnings=wl + man.get("warnings", []) + sorted(warn_log),
    )
    write_json(tdir / "test_report.json", report)
    (tdir / "test.lock").write_text("done " + now_utc())
    print(f"[test] MAE={metrics['mae']['mean']:.4f} [{metrics['mae']['ci_lo']:.4f},{metrics['mae']['ci_hi']:.4f}]  nMAE={metrics['nmae']['mean']:.4f}  "
          f"p95={latency['p95_ms']:.1f}ms  GFLOPs/chunk={complexity['gflops_infer_chunk']:.1f}")
    try:
        print(build_text_report(run_dir), flush=True)
    except Exception:
        print("[test] text report failed (re-run with --stage plots):")
        traceback.print_exc()


# ===================================================================================================================
# 9. TEXT REPORT
# ===================================================================================================================
def _f(x, fmt=".4f"):
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "n/a"
        return format(x, fmt)
    except Exception:
        return str(x)


def build_text_report(run_dir):
    import pandas as pd

    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 200)
    run_dir = Path(run_dir)
    L = []

    def hdr(s):
        L.append("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)

    man = read_json(run_dir / "manifest.json")
    t, mo = man.get("training", {}), man.get("model", {})
    hdr(f"RUN {man.get('run_id')}   status={man.get('status')}   valid={man.get('valid')}")
    L.append(f"model_key        : {man.get('model_key')}   (ACT, chunk_size={mo.get('chunk_size')}, use_vae={mo.get('use_vae')})")
    L.append(f"params           : total {mo.get('params_total', 0):,}   trainable {mo.get('params_trainable', 0):,}   active {mo.get('params_active', 0):,}")
    for g, v in (mo.get("params_by_group") or {}).items():
        L.append(f"   {g:42s} total {v['total']:>13,d}   trainable {v['trainable']:>13,d}")
    L.append(f"steps / batch    : {t.get('steps')} steps, batch/GPU {t.get('batch_size_per_device')}, effective {t.get('effective_batch_size')}, GPUs {man.get('hardware', {}).get('world_size')}")
    L.append(f"best step        : {t.get('best_step')}   best val loss {_f(t.get('best_val_loss'), '.5f')}   first val loss {_f(t.get('first_val_loss'), '.5f')}")
    if t.get("first_val_loss") and t.get("best_val_loss"):
        L.append(f"val improvement  : {100 * (1 - t['best_val_loss'] / t['first_val_loss']):.1f}%  (degenerate if < 20%)")
    L.append(f"time to best     : {(t.get('wall_time_total_s') or 0) / 60:.1f} min   full run {(t.get('wall_time_full_run_s') or 0) / 60:.1f} min   "
             f"{t.get('samples_per_s', 0) or 0:.1f} samples/s   peak train VRAM {t.get('peak_train_vram_gb', 0) or 0:.1f} GB")
    if man.get("warnings"):
        L.append("warnings         :")
        L += [f"   - {w}" for w in man["warnings"]]

    hp = run_dir / "metrics_history.csv"
    if hp.exists():
        hist = pd.read_csv(hp)
        tr, va = hist[hist.event == "train"], hist[hist.event == "val"]
        hdr("TRAINING HISTORY")
        L.append(f"{len(tr)} train rows, {len(va)} validation rows")
        if len(tr):
            last = tr.iloc[-1]
            L.append(f"last train row   : step {int(last.step)}  loss {_f(last.loss_total)}  grad_norm {_f(last.grad_norm, '.3f')}  lr {last.lr:.3e}")
        if len(va):
            show = pd.concat([va[va.is_best.astype(str) == "True"], va.tail(10)]).drop_duplicates(subset="step").sort_values("step")
            L.append("Validation rows (every new best + last 10):")
            L.append(show[["step", "val_loss", "val_loss_sd_over_episodes", "n_val_episodes", "is_best", "best_val_loss", "wall_time_s"]].round(5).to_string(index=False))

    tp = run_dir / "test_log.csv"
    if tp.exists():
        tl = pd.read_csv(tp)
        if len(tl):
            hdr("TEST-STYLE METRICS DURING TRAINING (validation episodes, real 10-step sampler)  [test_log.csv]")
            cols = ["step", "val_loss", "is_best", "mae_mean", "nmae_mean", "mse_mean", "rmse_mean", "jerk_norm_mean", "accel_norm_mean", "smoothness_ratio_mean"]
            show = pd.concat([tl[tl.is_best.astype(str) == "True"], tl.tail(10)]).drop_duplicates(subset="step").sort_values("step")
            L.append("Rows (every new best + last 10):")
            L.append(show[cols].round(5).to_string(index=False))
            b = tl[tl.is_best.astype(str) == "True"]
            if len(b):
                r = b.iloc[-1]
                L.append(f"\nAt the best checkpoint (step {int(r.step)}): " + "  ".join(f"{k}={r[f'{k}_mean']:.4f}±{r[f'{k}_sd']:.4f}" for k in ["mae", "nmae", "mse", "rmse", "jerk_norm"]))

    tdir = run_dir / "test"
    if (tdir / "test_report.json").exists():
        rep = read_json(tdir / "test_report.json")
        hdr(f"TEST REPORT   (valid={rep['valid']}, {rep['n_test_episodes']} episodes, {rep['n_test_windows']} windows, checkpoint step {rep['checkpoint']['step']})")
        m = pd.DataFrame(rep["metrics"]).T[["mean", "sd", "median", "p90", "ci_lo", "ci_hi"]]
        m.round(6).to_csv(tdir / "test_metrics_summary.csv")
        L.append("Metrics over test episodes (raw action units unless 'n'-prefixed; ci = 95% bootstrap over episodes):")
        L.append(m.round(4).to_string())
        ms = rep["metrics"]["mse"]
        L.append(f"\nMSE  mean {ms['mean']:.4f}   sd {ms['sd']:.4f}   median {ms['median']:.4f}   p90 {ms['p90']:.4f}   95% CI [{ms['ci_lo']:.4f}, {ms['ci_hi']:.4f}]")
        L.append("\nBoundary jump (bjump): " + str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep["bjump"].items()}))
        L.append("\nLatency per action chunk (batch 1):")
        L.append(pd.Series(rep["latency"]).to_string())
        L.append("\nComplexity:")
        L.append(pd.Series(rep["complexity"]).to_string())
        L.append("\nProtocol: " + str({k: v for k, v in rep["protocol"].items() if k != "train_action_std"}))
        L.append("train_action_std per dim: " + str(np.round(rep["protocol"]["train_action_std"], 4)))
        if rep.get("warnings"):
            L.append("\nwarnings:")
            L += [f"   - {w}" for w in rep["warnings"]]
        ep = pd.read_csv(tdir / "test_per_episode.csv")
        hdr("PER-EPISODE TEST RESULTS")
        L.append("By task:")
        bt = ep.groupby("task")[["mae", "nmae", "mse", "jerk_norm"]].agg(["mean", "count"]).round(4)
        bt.to_csv(tdir / "test_by_task.csv")
        L.append(bt.to_string())
        cols = ["episode_id", "task", "n_windows", "mae", "nmae", "mse", "jerk_norm"]
        worst = ep.sort_values("mae", ascending=False).head(10)[cols].round(4)
        worst.to_csv(tdir / "test_worst10.csv", index=False)
        L.append("\nWorst 10 episodes by MAE:")
        L.append(worst.to_string(index=False))
        L.append("\nBest 5 episodes by MAE:")
        L.append(ep.sort_values("mae").head(5)[cols].round(4).to_string(index=False))
    else:
        L.append("\n(no test report yet: run --stage test)")

    text = "\n".join(L)
    (run_dir / "test_log.txt").write_text(text)
    return text


# ===================================================================================================================
# 10. FIGURES (layout identical to the reference notebooks)
# ===================================================================================================================
def _pred_figures(P, tag, save, plt, start_no):
    pred, gt, valid = P["pred"], P["gt"], P["valid"]
    D = pred.shape[-1]
    err = np.abs(pred - gt) * valid[..., None]
    nv = np.maximum(valid.sum(0), 1)[:, None]
    n = start_no
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].plot(err.sum((0, 2)) / (nv[:, 0] * D))
    ax[0].set(title=f"{tag}: MAE vs position in the chunk", xlabel="chunk step", ylabel="MAE")
    ax[1].bar(range(D), (err.sum(0) / nv).mean(0))
    ax[1].set(title=f"{tag}: MAE per action dimension", xlabel="dim")
    w_mae = err.sum((1, 2)) / np.maximum(valid.sum(1) * D, 1)
    ax[2].hist(w_mae, bins=40)
    ax[2].set(title=f"{tag}: per-window MAE", xlabel="MAE", ylabel="windows")
    save(fig, f"{n:02d}_{tag}_error_vs_horizon_dim_window.png")
    n += 1
    order = np.argsort(w_mae)
    for name, i in (("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])):
        fig, ax = plt.subplots(1, D, figsize=(3 * D, 2.8))
        ax = np.atleast_1d(ax)
        for d in range(D):
            ax[d].plot(gt[i, :, d], label="ground truth")
            ax[d].plot(pred[i, :, d], "--", label="prediction")
            ax[d].set_title(f"dim {d}")
        ax[0].legend(fontsize=7)
        fig.suptitle(f"{tag} {name.upper()} window: episode {int(P['episode_id'][i])}, start frame {int(P['start'][i])}, MAE {w_mae[i]:.3f}", y=1.04)
        save(fig, f"{n:02d}_{tag}_{name}_window.png")
        n += 1
    return n


def stage_plots(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rid = run_id_of(args)
    run_dir = OUT_ROOT / "runs" / rid
    fig_dir = OUT_ROOT / "figures" / rid
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
    try:
        build_text_report(run_dir)
    except Exception:
        traceback.print_exc()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, bbox_inches="tight")
        plt.close(fig)

    n = 1
    hist = pd.read_csv(run_dir / "metrics_history.csv")
    tr, va = hist[hist.event == "train"], hist[hist.event == "val"]
    roll = max(len(tr) // 20, 1)
    smooth = tr.loss_total.rolling(roll, min_periods=1).mean()
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    ax = ax.ravel()
    ax[0].plot(tr.step, smooth, label="loss")
    ax[0].set(title="Training loss (L1 + kl_weight*KLD)", xlabel="step", yscale="log")
    ax[0].legend()
    if len(va):
        ax[1].plot(va.step, va.val_loss, "-", label="val loss")
        b = va[va.is_best.astype(str) == "True"]
        ax[1].scatter(b.step, b.val_loss, c="red", zorder=3, label="new best (checkpoint written)")
        ax[1].set(title="Validation loss (masked L1, deterministic)", xlabel="step")
        ax[1].legend()
        ax[2].plot(tr.step, smooth, label="train")
        ax[2].plot(va.step, va.val_loss, "-", label="val")
        ax[2].set(title="Train vs validation", xlabel="step")
        ax[2].legend()
    ax[3].plot(tr.step, tr.grad_norm)
    ax[3].set(title="Gradient norm (pre-clip)", xlabel="step")
    ax[4].plot(tr.step, tr.lr)
    ax[4].set(title="Learning rate", xlabel="step")
    ax[5].plot(tr.step, tr.step_s, label="step time")
    ax[5].plot(tr.step, tr.data_s, label="data wait")
    ax[5].set(title="Seconds per step", xlabel="step")
    ax[5].legend()
    save(fig, f"{n:02d}_training_overview.png")
    n += 1

    if len(va) and "n_val_episodes" in va:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(va.step, va.val_loss, yerr=va.val_loss_sd_over_episodes / np.sqrt(va.n_val_episodes), marker="o", ms=3, capsize=2, label="val loss (±SE over episodes)")
        b = va[va.is_best.astype(str) == "True"]
        if len(b):
            ax.scatter(b.step, b.val_loss, c="red", zorder=3, label="new best")
        ax.set(title="Validation loss with standard error", xlabel="step", ylabel="masked L1 (normalised)")
        ax.legend()
        save(fig, f"{n:02d}_validation_loss_se.png")
        n += 1
    vp = run_dir / "val_per_episode_best.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(v.val_loss, bins=20)
        ax.set(title="Per-episode validation loss at best checkpoint", xlabel="val loss", ylabel="episodes")
        save(fig, f"{n:02d}_val_per_episode.png")
        n += 1

    tp = run_dir / "test_log.csv"
    if tp.exists():
        tl = pd.read_csv(tp)
        if len(tl):
            fig, ax = plt.subplots(2, 3, figsize=(17, 8))
            ax = ax.ravel()
            bst = tl[tl.is_best.astype(str) == "True"]
            for a, k, ttl in zip(ax, ["mae", "nmae", "mse", "rmse", "jerk_norm", "smoothness_ratio"],
                                 ["MAE", "nMAE", "MSE", "RMSE", "jerk_norm", "smoothness ratio (pred jerk / GT jerk)"]):
                a.plot(tl.step, tl[f"{k}_mean"], label="mean over val episodes")
                a.fill_between(tl.step, tl[f"{k}_mean"] - tl[f"{k}_sd"], tl[f"{k}_mean"] + tl[f"{k}_sd"], alpha=0.15, label="±1 SD")
                a.plot(tl.step, tl[f"{k}_median"], ":", label="median")
                if len(bst):
                    a.scatter(bst.step, bst[f"{k}_mean"], c="red", s=14, zorder=3, label="new best ckpt")
                if k == "smoothness_ratio":
                    a.axhline(1.0, c="k", ls="--", lw=1)
                a.set(title=f"{ttl} during training (validation episodes)", xlabel="step")
            ax[0].legend(fontsize=7)
            save(fig, f"{n:02d}_test_log_curves.png")
            n += 1

    vpred = run_dir / "val_predictions_best.npz"
    if vpred.exists():
        n = _pred_figures(np.load(vpred), "val", save, plt, n)

    tdir = run_dir / "test"
    if not (tdir / "test_report.json").exists():
        print(f"[plots] no test report yet: training / validation figures only -> {fig_dir}")
        return
    rep = read_json(tdir / "test_report.json")
    ep = pd.read_csv(tdir / "test_per_episode.csv")
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, k in zip(ax, ["mae", "nmae", "mse", "jerk_norm"]):
        a.hist(ep[k].dropna(), bins=25)
        mm = rep["metrics"][k]
        a.axvline(mm["mean"], c="r", label=f"mean {mm['mean']:.3g} [{mm['ci_lo']:.3g}, {mm['ci_hi']:.3g}]")
        a.axvline(mm["median"], c="k", ls="--", label=f"median {mm['median']:.3g}")
        a.set(title=f"Per-episode {k}", ylabel="episodes")
        a.legend(fontsize=7)
    save(fig, f"{n:02d}_test_per_episode_distributions.png")
    n += 1

    n = _pred_figures(np.load(tdir / "predictions.npz"), "test", save, plt, n)

    sj = pd.read_csv(tdir / "bjump_seams.csv")
    lat = pd.read_csv(tdir / "latency_samples.csv")
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].hist(lat.latency_ms, bins=40)
    for k, c in (("p50_ms", "g"), ("p95_ms", "orange"), ("p99_ms", "r")):
        ax[0].axvline(rep["latency"][k], c=c, label=f"{k[:3]} {rep['latency'][k]:.0f} ms")
    ax[0].set(title="Latency per chunk (bs 1)", xlabel="ms")
    ax[0].legend()
    ax[1].hist(sj.l2_jump, bins=25)
    ax[1].set(title=f"Chunk-boundary jump (n={len(sj)})", xlabel="L2 jump (raw units)")
    ax[2].scatter(ep.gt_jerk_in_chunk, ep.jerk_in_chunk, s=12)
    lim = [0, float(np.nanmax([ep.gt_jerk_in_chunk.max(), ep.jerk_in_chunk.max()]))]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set(title="Jerk: prediction vs ground truth", xlabel="GT jerk", ylabel="predicted jerk")
    save(fig, f"{n:02d}_latency_boundary_jerk.png")
    n += 1

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, k in zip(ax, ["mae", "nmae", "mse", "jerk_norm"]):
        mm = rep["metrics"][k]
        a.bar([0], [mm["mean"]], yerr=[[mm["mean"] - mm["ci_lo"]], [mm["ci_hi"] - mm["mean"]]], capsize=6)
        a.set(title=f"{k} mean + 95% bootstrap CI", xticks=[])
    save(fig, f"{n:02d}_test_metric_summary_ci.png")
    print(f"[plots] figures written to {fig_dir}")


# ===================================================================================================================
# 11. entry point
# ===================================================================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=["all", "split", "train", "test", "plots"])
    p.add_argument("--dataset_repo", default=DEFAULT_DATASET)
    p.add_argument("--dataset_revision", default="main")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16, help="per GPU; effective batch = batch_size x num_gpus")
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--val_batch", type=int, default=8)
    p.add_argument("--test_batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2, help="dataloader workers PER GPU process")
    p.add_argument("--video_backend", default="pyav")
    p.add_argument("--train_seed", type=int, default=SEEDS["train_seed"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--run_id", default=None)
    p.add_argument("--num_gpus", type=int, default=0, help="GPUs for the train stage (0 = all visible)")
    p.add_argument("--no_val_metrics", action="store_true", help="do not compute the test-style metrics (test_log.csv) at every validation")
    return p.parse_args()


def forward_args(a):
    out = []
    for k, v in vars(a).items():
        if k == "stage" or v is None:
            continue
        out += [f"--{k}", str(v)] if not isinstance(v, bool) else ([f"--{k}"] if v else [])
    return out


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
        ngpu = args.num_gpus or torch.cuda.device_count() or 1
        for st in ("split", "train", "test", "plots"):
            print(f"\n===== stage: {st} =====" + (f"  (DDP, {ngpu} GPUs)" if st == "train" and ngpu > 1 else ""), flush=True)
            if st == "train" and ngpu > 1:
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ngpu}", str(SCRIPT), "--stage", st]
            else:
                cmd = [sys.executable, str(SCRIPT), "--stage", st]
            rc = subprocess.call(cmd + forward_args(args), env=env)
            if rc != 0:
                sys.exit(f"stage '{st}' failed with exit code {rc}")
        return
    {"split": stage_split, "train": stage_train, "test": stage_test, "plots": stage_plots}[args.stage](args)


if __name__ == "__main__":
    main()
