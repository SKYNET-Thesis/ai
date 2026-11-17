#!/usr/bin/env python
"""
SmollerVLA — unified single-file pipeline (split -> train/val -> test -> figures), single GPU, fp32.

Architecture (from smoller_vla.ipynb): frozen LFM2.5-VL-450M (first 8 LM layers) -> AdaptiveConditionalGatedKV
(trainable) -> per-layer prefix K/V -> SmolVLA action expert with G1 gated attention (16 gates), flow-matching loss.
Protocol (from SmollerVLA Train/Validation/Test Design Guide v1.0): episode-level 70/10/20 stratified split,
fixed-noise validation loss, a single best_valid checkpoint, test with the full metric table, bootstrap CIs.

Run (everything, default):      python smollervla_train.py
Single stages (each is its own process): --stage split | train | test | plots
Defaults: dataset yamaisanerd/Chunking_560_Skynet, 10000 steps, batch 8, 1 GPU, fp32, TF32 off.
All outputs go next to this file when it sits in a folder called `small_test_vla` (else ./small_test_vla).

Environment (one-time):
    pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git"
    pip install -U "av>=15,<16" "transformers>=5.5,<6" matplotlib
    pip install causal-conv1d      # optional, speeds up the LFM conv layers (latency report records if it is absent)
NOTE: gate_patch (embedded below) edits three files of the INSTALLED lerobot (.orig backups are kept). It aborts if the
string anchors do not match your lerobot version; the pip freeze / patched-file hashes are stored in every run.
"""
import argparse
import copy
import csv
import dataclasses
import glob
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT = Path(__file__).resolve()
OUT_ROOT = SCRIPT.parent if SCRIPT.parent.name == "small_test_vla" else SCRIPT.parent / "small_test_vla"
SPLIT_DIR = OUT_ROOT / "splits"
MANIFEST_PATH = SPLIT_DIR / "split_manifest.json"
SCHEMA = "1.0"

DEFAULT_DATASET = "yamaisanerd/Chunking_560_Skynet"
SEEDS = dict(split_seed=42, train_seed=42, val_seed=1234, test_seed=2024, bootstrap_seed=20240)
RATIOS = dict(train=0.70, val=0.10, test=0.20)
CHUNK = 50
VAL_K = 10
BOOT_B = 10000
BACKBONE_ID = "LiquidAI/LFM2.5-VL-450M"
BB = dict(id=BACKBONE_ID, keep_layers=8, vis_tokens=48, img_size=256, dtype="float32")


# =====================================================================================================================
# generic helpers
# =====================================================================================================================
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
    torch.backends.cuda.matmul.allow_tf32 = False  # guide §6: TF32 off for the final benchmark
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
    dirty = bool(run_cmd(["git", "status", "--porcelain"], d))
    return sha, dirty


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
        info = api.dataset_info(repo) if repo_type == "dataset" else api.model_info(repo)
        return info.sha
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


# =====================================================================================================================
# 1. SPLIT (pure numpy / pandas; no lerobot import)
# =====================================================================================================================
def round_half_up(x):
    return int(math.floor(x + 0.5))


def _quota(sizes, frac, total, caps):
    """floor(frac*size) per stratum, remainder to the largest fractional parts (ties: sorted stratum key)."""
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
    """ep_rows: list of dict(episode_index, task). Returns train/val/test id lists + strata counts."""
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
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=["meta/*", "meta/**", "data/*", "data/**"]))
    info = read_json(root / "meta" / "info.json")
    jl = root / "meta" / "episodes.jsonl"
    if jl.exists():  # lerobot v2.x
        rows = [json.loads(l) for l in open(jl) if l.strip()]
        ep_df = pd.DataFrame(rows)
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
    # G4: duplicated episode content across splits
    hashes = episode_content_hashes(root)
    which = {**{e: "train" for e in train}, **{e: "val" for e in val}, **{e: "test" for e in test}}
    seen = {}
    for e, h in sorted(hashes.items()):
        if h in seen and which[seen[h]] != which[e]:
            sys.exit(f"[G4] episodes {seen[h]} ({which[seen[h]]}) and {e} ({which[e]}) have identical content -> leakage, abort")
        seen.setdefault(h, e)
    # G5: session/operator/source ids
    cand = [c for c in ep_df.columns if re.search(r"session|operator|source|recording|collector|scene|setup", c, re.I)]
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


# =====================================================================================================================
# 2. gate patch (embedded, identical to the notebook's gate_patch.py)
# =====================================================================================================================
def _sub(text, old, new, name):
    n = text.count(old)
    if n != 1:
        sys.exit(f"[patch failed] anchor '{name}' found {n}x (expected 1). Your lerobot version differs from the one this patch targets.")
    return text.replace(old, new)


def _patch_config(t):
    return _sub(
        t,
        '    compile_mode: str = "max-autotune"  # Torch compile mode\n',
        '    compile_mode: str = "max-autotune"  # Torch compile mode\n'
        "\n"
        "    # Gated Attention (sigmoid gate after SDPA) in every layer of the action expert\n"
        "    use_gated_attn: bool = False\n"
        "    gated_attn_headwise: bool = False  # False: elementwise (n x heads x head_dim), True: one scalar per head\n"
        "    gated_attn_bias_init: float = 4.0  # sigmoid(4) = 0.982\n",
        "config.fields",
    )


GATE_CLASS = '''class AttnOutputGate(nn.Module):
    """Gated Attention, position G1 (Qiu et al. 2025): Y' = Y * sigmoid(X W + b).
    Y = SDPA output [B, L, H*D] (before o_proj), X = normed hidden state of the query tokens [B, L, E].
    Head-specific. Elementwise: one score per channel (H*D). Headwise: one score per head (H)."""

    def __init__(self, hidden_size, num_heads, head_dim, headwise=False, bias_init=4.0):
        super().__init__()
        self.headwise = headwise
        self.head_dim = head_dim
        self.proj = nn.Linear(hidden_size, num_heads if headwise else num_heads * head_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, bias_init)

    def forward(self, x, att_out):
        g = torch.sigmoid(self.proj(x.to(self.proj.weight.dtype)))
        if self.headwise:
            g = g.repeat_interleave(self.head_dim, dim=-1)
        return att_out * g.to(att_out.dtype)


class SmolVLMWithExpertModel(nn.Module):'''

GATE_BUILD = '''        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        # Gated attention: one gate per action-expert layer (self- and cross-attention layers)
        self.use_gated_attn = use_gated_attn
        self.attn_gates = nn.ModuleDict()
        if use_gated_attn:
            assert self.num_expert_layers == self.num_vlm_layers, (
                "Gated attention assumes num_expert_layers == num_vlm_layers (default config)"
            )
            for _i in range(self.num_expert_layers):
                self.attn_gates[str(_i)] = AttnOutputGate(
                    lm_expert_config.hidden_size,
                    lm_expert_config.num_attention_heads,
                    lm_expert_config.head_dim,
                    gated_attn_headwise,
                    gated_attn_bias_init,
                )
'''


def _patch_vlm(t):
    t = _sub(t, "class SmolVLMWithExpertModel(nn.Module):", GATE_CLASS, "class")
    t = _sub(t, 'torch_dtype="bfloat16",', 'torch_dtype="float32",', "fp32 loader")
    t = _sub(
        t,
        '        device: str = "auto",\n    ):\n        super().__init__()\n        require_package("transformers", extra="smolvla")',
        '        device: str = "auto",\n'
        "        use_gated_attn: bool = False,\n"
        "        gated_attn_headwise: bool = False,\n"
        "        gated_attn_bias_init: float = 4.0,\n"
        '    ):\n        super().__init__()\n        require_package("transformers", extra="smolvla")',
        "init signature",
    )
    t = _sub(t, "        # Remove unused embed_tokens\n        self.lm_expert.embed_tokens = None\n", GATE_BUILD, "gate build")
    t = _sub(
        t,
        "        query_states = []\n        key_states = []\n        value_states = []\n",
        "        gate_x = None\n        query_states = []\n        key_states = []\n        value_states = []\n",
        "gate_x init",
    )
    t = _sub(
        t,
        "            hidden_states = layer.input_layernorm(hidden_states)\n",
        "            hidden_states = layer.input_layernorm(hidden_states)\n"
        "            if i == 1 and self.use_gated_attn:  # expert (action) tokens only\n"
        "                gate_x = hidden_states\n",
        "gate_x capture",
    )
    t = _sub(
        t,
        "        att_output = attention_interface(\n"
        "            attention_mask_, batch_size, head_dim, query_states, key_states, value_states\n"
        "        )\n"
        "        return [att_output], past_key_values\n",
        "        att_output = attention_interface(\n"
        "            attention_mask_, batch_size, head_dim, query_states, key_states, value_states\n"
        "        )\n"
        "        if gate_x is not None:  # gate only the expert tokens (the last Ls query positions)\n"
        "            _ls = gate_x.shape[1]\n"
        "            _gated = self.attn_gates[str(layer_idx)](gate_x, att_output[:, -_ls:])\n"
        "            att_output = torch.cat([att_output[:, :-_ls], _gated], dim=1)\n"
        "        return [att_output], past_key_values\n",
        "self-attn gate apply",
    )
    t = _sub(
        t,
        "                expert_value_states,\n            )\n            att_outputs.append(att_output)\n",
        "                expert_value_states,\n            )\n"
        "            if self.use_gated_attn:\n"
        "                att_output = self.attn_gates[str(layer_idx)](expert_hidden_states, att_output)\n"
        "            att_outputs.append(att_output)\n",
        "cross-attn gate apply",
    )
    return t


def _patch_modeling(t):
    t = _sub(
        t,
        '            device=self.config.device if self.config.device is not None else "auto",\n        )\n',
        '            device=self.config.device if self.config.device is not None else "auto",\n'
        "            use_gated_attn=self.config.use_gated_attn,\n"
        "            gated_attn_headwise=self.config.gated_attn_headwise,\n"
        "            gated_attn_bias_init=self.config.gated_attn_bias_init,\n"
        "        )\n",
        "pass flags",
    )
    t = _sub(
        t,
        "        self.prefix_length = self.config.prefix_length\n        self.rtc_processor = rtc_processor\n",
        "        self.prefix_length = self.config.prefix_length\n        self.rtc_processor = rtc_processor\n"
        "        self.float()  # fp32 everywhere\n",
        "fp32 cast",
    )
    return t


PATCHES = {"configuration_smolvla.py": _patch_config, "smolvlm_with_expert.py": _patch_vlm, "modeling_smolvla.py": _patch_modeling}


def ensure_patch():
    """Must run BEFORE any lerobot.policies.smolvla import in this process."""
    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("lerobot is not installed.")
    d = Path(list(spec.submodule_search_locations)[0]) / "policies" / "smolvla"
    if "use_gated_attn" in (d / "configuration_smolvla.py").read_text():
        print("[gate_patch] already patched")
    else:
        new = {n: fn((d / n).read_text()) for n, fn in PATCHES.items()}  # all-or-nothing
        for n, text in new.items():
            shutil.copy(d / n, d / (n + ".orig"))
            (d / n).write_text(text)
            print("[gate_patch] patched", n)
    return {n: file_sha256(d / n) for n in PATCHES}


# =====================================================================================================================
# 3. LFM2.5-VL backbone adapter (from the notebook: AdaptiveConditionalGatedKV + class-level patch of VLAFlowMatching)
# =====================================================================================================================
PREPROC = []  # seconds spent in the CPU LFM image pre-processing, one entry per image


class AdaptiveConditionalGatedKV(nn.Module):
    def __init__(self, input_dim, state_dim, kv_dim, kv_heads, head_dim, num_layers):
        super().__init__()
        self.input_dim, self.kv_dim = input_dim, kv_dim
        self.kv_heads, self.head_dim, self.num_layers = kv_heads, head_dim, num_layers
        self.mapper = nn.Sequential(nn.Linear(input_dim, input_dim), nn.SiLU(), nn.LayerNorm(input_dim))
        self.state_bridge = nn.Linear(state_dim, input_dim)
        self.kv_projection = nn.Linear(input_dim, 2 * kv_dim)
        self.gate_network = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, num_layers))
        self.layer_bias = nn.Parameter(torch.zeros(num_layers, 2, kv_dim))

    def forward(self, vl_features, state_features, mask=None):
        x = self.mapper(vl_features.float())
        state_token = self.state_bridge(state_features.float()).unsqueeze(1)
        x = torch.cat([x, state_token], dim=1)
        B, P, _ = x.shape
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            m = torch.cat([mask.to(x.dtype), torch.ones(B, 1, dtype=x.dtype, device=x.device)], dim=1).unsqueeze(-1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        gates = torch.sigmoid(self.gate_network(pooled))
        base_k, base_v = self.kv_projection(x).chunk(2, dim=-1)
        ks, vs = [], []
        for layer_id in range(self.num_layers):
            gate = gates[:, layer_id].reshape(B, 1, 1)
            k = (base_k + self.layer_bias[layer_id, 0]) * gate
            v = (base_v + self.layer_bias[layer_id, 1]) * gate
            ks.append(k.reshape(B, P, self.kv_heads, self.head_dim).transpose(1, 2).contiguous())
            vs.append(v.reshape(B, P, self.kv_heads, self.head_dim).transpose(1, 2).contiguous())
        return ks, vs


class LFMAdapter(nn.Module):
    """Frozen LFM2.5-VL (first keep_layers LM layers, fp32) + trainable AdaptiveConditionalGatedKV."""

    def __init__(self, model):
        super().__init__()
        from transformers import AutoProcessor, AutoTokenizer, Lfm2VlForConditionalGeneration

        ve = model.vlm_with_expert
        vcfg = ve.vlm.config.text_config  # SmolVLA KV geometry that the expert cross-attention expects
        kv_heads, head_dim = int(vcfg.num_key_value_heads), int(vcfg.head_dim)
        self.num_layers = len(ve.lm_expert.layers)
        self.cache_format = "dynamic"
        self.bb_dtype = torch.float32
        keep = BB["keep_layers"]
        backbone = Lfm2VlForConditionalGeneration.from_pretrained(BB["id"], dtype=self.bb_dtype, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(BB["id"])
        language = backbone.model.language_model
        n_full = len(language.layers)
        assert n_full >= keep, f"{BB['id']} has only {n_full} LM layers"
        layer_types = list(backbone.config.text_config.layer_types)
        language.layers = nn.ModuleList(list(language.layers[:keep]))  # FIRST pretrained layers
        for cfg in (language.config, backbone.config.text_config):
            cfg.num_hidden_layers = keep
            cfg.layer_types = layer_types[:keep]
        backbone.lm_head = nn.Identity()
        backbone.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone
        self.lfm_dim = int(backbone.config.text_config.hidden_size)
        self.backbone_revision = getattr(backbone.config, "_commit_hash", None)
        self.conditioner = AdaptiveConditionalGatedKV(
            input_dim=self.lfm_dim, state_dim=model.state_proj.out_features, kv_dim=kv_heads * head_dim,
            kv_heads=kv_heads, head_dim=head_dim, num_layers=self.num_layers,
        ).float()
        tok = getattr(getattr(ve, "processor", None), "tokenizer", None)
        self.task_tokenizer = tok or AutoTokenizer.from_pretrained(model.config.vlm_model_name)
        self._text_ids = {}
        self.cam_order, self.cam_labels = None, None
        print(f"[LFMAdapter] {BB['id']}: {keep}/{n_full} LM layers, hidden={self.lfm_dim}, layer_types={layer_types[:keep]}, dtype={self.bb_dtype}", flush=True)

    def set_cameras(self, keys):
        """Fix the camera order to head, wrist, side (the notebook's camera1..3) whatever the dataset key order is."""
        rank = lambda k: 0 if "head" in k else 1 if "wrist" in k else 2 if "side" in k else 3
        self.cam_order = sorted(range(len(keys)), key=lambda i: (rank(keys[i]), i))
        roles = ["Head camera: ", "\nWrist camera: ", "\nSide camera: "]
        known = all(rank(keys[i]) < 3 for i in self.cam_order) and len({rank(k) for k in keys}) == len(keys)
        self.cam_labels = {}
        for j, i in enumerate(self.cam_order):
            self.cam_labels[i] = roles[j] if (known and j < 3) else ("" if j == 0 else "\n") + f"Camera {j + 1}: "

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):  # frozen backbone is not checkpointed
        return self.conditioner.state_dict(destination=destination, prefix=prefix + "conditioner.", keep_vars=keep_vars)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def _embed_text(self, text, language):
        if text not in self._text_ids:
            self._text_ids[text] = self.processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return language.embed_tokens(self._text_ids[text].to(self.device))

    def _embed_image(self, x):  # x: [3,H,W] in [-1,1]
        t0 = time.perf_counter()
        x = ((x.detach().float() + 1.0) / 2.0).clamp(0, 1)
        S = BB["img_size"]
        if x.shape[-1] != S or x.shape[-2] != S:
            x = F.interpolate(x[None], size=(S, S), mode="bilinear", antialias=True, align_corners=False)[0]
        from PIL import Image

        pil = Image.fromarray((x.permute(1, 2, 0) * 255).round().byte().cpu().numpy())
        inputs = self.processor.image_processor(images=pil, return_tensors="pt")
        PREPROC.append(time.perf_counter() - t0)
        required = ("pixel_values", "spatial_shapes", "pixel_attention_mask")
        assert all(k in inputs for k in required), f"LFM image preprocessing mismatch: {list(inputs.keys())}"
        prepared = {k: inputs[k].to(self.device) for k in required}
        prepared["pixel_values"] = prepared["pixel_values"].to(self.bb_dtype)
        out = self.backbone.get_image_features(**prepared, return_dict=True)
        visual = out.pooler_output
        if isinstance(visual, (list, tuple)):
            visual = torch.cat(list(visual), dim=0)
        if visual.ndim == 3:
            visual = visual.reshape(-1, visual.shape[-1])
        assert visual.ndim == 2 and visual.shape[-1] == self.lfm_dim
        visual = F.adaptive_avg_pool1d(visual.float().transpose(0, 1).unsqueeze(0), BB["vis_tokens"]).transpose(1, 2)
        return visual.to(self.bb_dtype)

    def _task_text(self, tokens, mask):
        return self.task_tokenizer.decode(tokens[mask.bool()].tolist(), skip_special_tokens=True).strip()

    @torch.no_grad()
    def features(self, images, img_masks, lang_tokens, lang_masks):
        language = self.backbone.model.language_model
        B = images[0].shape[0]
        order = self.cam_order if self.cam_order is not None else list(range(len(images)))
        seqs = []
        for b in range(B):
            parts = []
            for c in order:
                if not bool(img_masks[c][b]):
                    continue
                label = self.cam_labels[c] if self.cam_labels else f"\nCamera {c + 1}: "
                parts.append(self._embed_text(label, language))
                parts.append(self._embed_image(images[c][b]))
            parts.append(self._embed_text("\n" + self._task_text(lang_tokens[b], lang_masks[b]), language))
            seqs.append(torch.cat([p.to(self.bb_dtype) for p in parts], dim=1)[0])
        L, D = max(s.shape[0] for s in seqs), seqs[0].shape[-1]
        x = torch.zeros(B, L, D, dtype=self.bb_dtype, device=self.device)
        m = torch.zeros(B, L, dtype=torch.bool, device=self.device)
        for b, s in enumerate(seqs):
            x[b, : s.shape[0]] = s
            m[b, : s.shape[0]] = True
        out = language(inputs_embeds=x, attention_mask=m.long(), use_cache=False, return_dict=True)
        hidden = out.last_hidden_state.float()
        assert torch.isfinite(hidden).all(), "non-finite LFM features"
        return hidden, m

    def make_cache_from_features(self, model, feats, mask, state):
        state_features = model.state_proj(state.float())
        ks, vs = self.conditioner(feats, state_features.to(feats.device), mask.to(feats.device))
        B = ks[0].shape[0]
        prefix_mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=mask.device)], dim=1)
        if self.cache_format == "dict":
            cache = {i: {"key_states": k, "value_states": v} for i, (k, v) in enumerate(zip(ks, vs))}
        else:
            from transformers import DynamicCache

            cache = DynamicCache()
            for i, (k, v) in enumerate(zip(ks, vs)):
                cache.update(k, v, i)
        return cache, prefix_mask

    def make_cache(self, model, images, img_masks, lang_tokens, lang_masks, state):
        feats, mask = self.features(images, img_masks, lang_tokens, lang_masks)
        return self.make_cache_from_features(model, feats, mask, state)


_ORIG = {}


def _flow_cls():
    import lerobot.policies.smolvla.modeling_smolvla as mod

    cls = getattr(mod, "VLAFlowMatching", None)
    if cls is None:
        for name, obj in vars(mod).items():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and "FlowMatching" in name:
                cls = obj
    assert cls is not None, "cannot find SmolVLA's flow-matching class"
    return cls


@torch.no_grad()
def _detect_cache_format(model):
    try:
        ve = model.vlm_with_expert
        dev, dt = next(ve.vlm.parameters()).device, next(ve.vlm.parameters()).dtype
        hid = int(ve.vlm.config.text_config.hidden_size)
        pe = torch.zeros(1, 4, hid, device=dev, dtype=dt)
        _, pkv = ve.forward(attention_mask=torch.ones(1, 4, 4, dtype=torch.bool, device=dev), position_ids=torch.arange(4, device=dev)[None],
                            past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
        return "dict" if isinstance(pkv, dict) else "dynamic"
    except Exception as e:
        print("[LFMAdapter] cache-format probe failed, using DynamicCache:", repr(e), flush=True)
        return "dynamic"


def _strip_smolvlm(model):
    """Remove SmolVLM2 (vision encoder, connector, text layers, embeddings, lm_head). Only config + the (unused) final norm stay."""
    ve = model.vlm_with_expert
    before = sum(p.numel() for p in ve.vlm.parameters())
    inner = getattr(ve.vlm, "model", None)
    if inner is not None:
        for name in ("vision_model", "connector"):
            if hasattr(inner, name):
                setattr(inner, name, nn.Identity())
        text = inner.text_model
        text.layers = nn.ModuleList([nn.Identity() for _ in range(len(text.layers))])
        text.embed_tokens = nn.Identity()
    if hasattr(ve.vlm, "lm_head"):
        ve.vlm.lm_head = nn.Identity()
    return before, sum(p.numel() for p in ve.vlm.parameters())


def _lfm_forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, **kw):
    if noise is None:
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)
    t = time[:, None, None]
    x_t = t * noise + (1 - t) * actions
    u_t = noise - actions
    cache, prefix_mask = self.lfm_adapter.make_cache(self, images, img_masks, lang_tokens, lang_masks, state)
    v_t = self.denoise_step(prefix_pad_masks=prefix_mask, past_key_values=cache, x_t=x_t, timestep=time)
    return F.mse_loss(u_t, v_t, reduction="none")


def _lfm_sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
    bsize, device = state.shape[0], state.device
    if noise is None:
        noise = torch.randn(bsize, self.config.chunk_size, self.config.max_action_dim, device=device, dtype=torch.float32)
    cache, prefix_mask = self.lfm_adapter.make_cache(self, images, img_masks, lang_tokens, lang_masks, state)
    n = int(self.config.num_steps)
    x_t = noise
    for i in range(n):
        tt = torch.full((bsize,), 1.0 - i / n, dtype=torch.float32, device=device)
        v_t = self.denoise_step(prefix_pad_masks=prefix_mask, past_key_values=cache, x_t=x_t, timestep=tt)
        x_t = x_t - v_t / n
    return x_t


def install_adapter():
    """Class-level patch of SmolVLA's flow-matching model. NO shape-tolerant state-dict loader (guide R8 / S2)."""
    cls = _flow_cls()
    if cls in _ORIG:
        return cls
    _ORIG[cls] = (cls.__init__, cls.forward, cls.sample_actions)
    orig_init = cls.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        adapter = LFMAdapter(self)
        adapter.cache_format = _detect_cache_format(self)
        self.lfm_adapter = adapter
        before, after = _strip_smolvlm(self)
        print(f"[LFMAdapter] cache={adapter.cache_format}; SmolVLM2 removed: {before:,} -> {after:,} params left (final-norm stub only)", flush=True)

    cls.__init__ = __init__
    cls.forward = _lfm_forward
    cls.sample_actions = _lfm_sample_actions
    return cls


# =====================================================================================================================
# 4. lerobot plumbing (cfg built through lerobot's own CLI parser so every default stays the framework default)
# =====================================================================================================================
def run_id_of(args):
    return args.run_id or f"lfm{args.lfm_layers}_w{int(round(args.expert_width * 100)):03d}_s{args.train_seed}"


def cfg_argv(args, run_dir):
    return [
        "--policy.type=smolvla", "--policy.push_to_hub=false", f"--policy.device={args.device}",
        "--policy.load_vlm_weights=false",  # R8: no pretrained smolvla_base; SmolVLM2 is removed anyway
        "--policy.use_gated_attn=true", "--policy.gated_attn_headwise=false", "--policy.gated_attn_bias_init=4.0",
        f"--policy.expert_width_multiplier={args.expert_width}", f"--policy.scheduler_decay_steps={args.steps}",
        f"--dataset.repo_id={args.dataset_repo}", f"--dataset.revision={args.dataset_revision}", f"--dataset.video_backend={args.video_backend}",
        f"--batch_size={args.batch_size}", f"--num_workers={args.num_workers}", f"--steps={args.steps}", "--log_freq=50",
        "--save_checkpoint=false", "--tf32=false", f"--seed={args.train_seed}", "--job_name=smollervla",
        f"--output_dir={run_dir / '_lerobot_unused'}",  # must not exist (lerobot validate), never created
    ]


def parse_cfg(argv):
    TrainPipelineConfig = first_import(("lerobot.configs.train", "TrainPipelineConfig"))
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  (registers the policy type)
    lr_parser = importlib.import_module("lerobot.configs.parser")

    @lr_parser.wrap()
    def _inner(cfg: TrainPipelineConfig):  # annotation must be a real class (no `from __future__ import annotations`)
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
    return cfg


def make_ds(cfg, episodes):
    make_dataset = first_import(("lerobot.datasets.factory", "make_dataset"), ("lerobot.datasets", "make_dataset"))
    c = copy.deepcopy(cfg)
    c.dataset.episodes = [int(e) for e in episodes]
    return make_dataset(c)


def make_pol_and_proc(cfg, ds):
    make_policy = first_import(("lerobot.policies.factory", "make_policy"))
    make_pp = first_import(("lerobot.policies.factory", "make_pre_post_processors"))
    install_adapter()
    policy = call_filtered(make_policy, cfg=cfg.policy, ds_meta=ds.meta, rename_map=None)
    policy.model.lfm_adapter.set_cameras(list(policy.config.image_features.keys()))
    stats = ds.meta.stats
    pre, post = call_filtered(make_pp, policy_cfg=cfg.policy, pretrained_path=None, dataset_stats=stats,
                              preprocessor_overrides={"device_processor": {"device": str(cfg.policy.device)}})
    return policy, pre, post


def _wif(worker_id):
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, bs, shuffle, seed, nw, indices=None, drop_last=False):
    dset = torch.utils.data.Subset(ds, list(indices)) if indices is not None else ds
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
              generator=g if shuffle else None, worker_init_fn=_wif)
    if nw > 0:
        kw.update(persistent_workers=shuffle, prefetch_factor=2, multiprocessing_context="spawn")
    return torch.utils.data.DataLoader(dset, **kw)


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


def unpack(policy, batch):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state, actions


def pad_key(batch):
    for k in ("action_is_pad", "actions_is_pad"):
        if k in batch:
            return batch[k].bool()
    return None


def forward_loss(policy, batch):
    out = policy.forward(batch)
    return out[0] if isinstance(out, (tuple, list)) else out


def n_params(policy):
    tot = sum(p.numel() for p in policy.parameters())
    trn = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    act = tot - sum(p.numel() for n, p in policy.named_parameters() if ".vlm." in n)
    return tot, trn, act


# =====================================================================================================================
# 5. R9 init audit
# =====================================================================================================================
def t_hash(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def init_audit(policy, run_dir, check_base):
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download, snapshot_download

    rows, warns = [], []
    bb_ref = {}
    try:  # pretrained-frozen reference: the Hugging Face LFM tensors
        d = snapshot_download(BB["id"], allow_patterns=["*.safetensors"])
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    bb_ref[k] = (f, k)
    except Exception as e:
        warns.append(f"could not read LFM reference weights: {e!r}")
    base = {}
    if check_base:
        try:
            p = hf_hub_download("lerobot/smolvla_base", "model.safetensors")
            with safe_open(p, framework="pt") as sf:
                for k in sf.keys():
                    base[k] = p
        except Exception as e:
            warns.append(f"could not fetch lerobot/smolvla_base for the scratch-hash check: {e!r}")
    pre = "model.lfm_adapter.backbone."
    n_ver = n_unver = 0
    for n, p in policy.named_parameters():
        if n.startswith(pre):
            src, grp = "pretrained_frozen", "lfm_backbone"
            assert not p.requires_grad, f"{n}: pretrained tensor must be frozen"
            h = t_hash(p)
            ref = bb_ref.get(n[len(pre):]) or bb_ref.get(n[len("model.lfm_adapter."):])
            if ref:
                with safe_open(ref[0], framework="pt") as sf:
                    ok = t_hash(sf.get_tensor(ref[1])) == h
                assert ok, f"{n}: differs from the Hugging Face LFM tensor"
                n_ver += 1
            else:
                n_unver += 1
        elif ".vlm." in n:
            src, grp, h = "unused_stub", "smolvlm2_final_norm_stub", t_hash(p)
            assert not p.requires_grad
        else:
            src, grp = "scratch_trained", "conditioner" if "lfm_adapter.conditioner" in n else "gates" if "attn_gates" in n else "expert_projections"
            assert p.requires_grad, f"{n}: scratch tensor must be trainable"
            h = t_hash(p)
            if n in base:
                with safe_open(base[n], framework="pt") as sf:
                    t = sf.get_tensor(n)
                if tuple(t.shape) == tuple(p.shape):
                    assert t_hash(t) != h, f"{n}: identical to the smolvla_base tensor -> pretrained weights leaked in"
        rows.append(dict(name=n, group=grp, init_source=src, requires_grad=p.requires_grad, numel=p.numel(), sha256=h))
    if n_unver:
        warns.append(f"{n_unver} LFM tensors could not be matched to Hugging Face key names; {n_ver} were verified equal")
    write_json(run_dir / "init_audit.json", dict(summary=dict(verified_pretrained=n_ver, unverified_pretrained=n_unver,
               params_by_source={s: sum(r["numel"] for r in rows if r["init_source"] == s) for s in {r["init_source"] for r in rows}}),
               warnings=warns, rows=rows))
    return warns


# =====================================================================================================================
# 6. validation (fixed noise / fixed timesteps, cached frozen-backbone features)
# =====================================================================================================================
def val_windows(ep_len):
    return list(range(0, ep_len, CHUNK))


def build_val_cache(policy, pre, ds, val_ids, cache_path, key, args, device):
    if cache_path.exists():
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        if c.get("key") == key:
            print(f"[val] feature cache hit ({len(c['windows'])} windows)")
            return c
        print("[val] feature cache key mismatch -> rebuilding")
    idx_map = episode_index_map(ds)
    wins, idxs = [], []
    for e in sorted(val_ids):
        fmap = idx_map[e]
        L = len(fmap)
        for s in val_windows(L):
            wins.append((e, s))
            idxs.append(fmap[s])
    print(f"[val] building feature cache: {len(wins)} windows over {len(val_ids)} episodes")
    dl = make_loader(ds, args.val_batch, False, SEEDS["val_seed"], args.num_workers, indices=idxs)
    model, ad = policy.model, policy.model.lfm_adapter
    policy.eval()
    out = dict(key=key, windows=wins, feats=[], masks=[], state=[], actions=[], pad=[])
    with torch.no_grad():
        for batch in dl:
            b = pre(batch)
            images, img_masks, lang, lm, state, actions = unpack(policy, b)
            feats, mask = ad.features(images, img_masks, lang, lm)
            pad = pad_key(b)
            for i in range(state.shape[0]):
                n = int(mask[i].sum())
                out["feats"].append(feats[i, :n].cpu())
                out["masks"].append(torch.ones(n, dtype=torch.bool))
                out["state"].append(state[i].cpu())
                out["actions"].append(actions[i].cpu())
                out["pad"].append(pad[i].cpu() if pad is not None else torch.zeros(actions.shape[1], dtype=torch.bool))
    assert len(out["feats"]) == len(wins)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    return out


def make_val_noise(wins, max_dim):
    noise = torch.empty(len(wins), VAL_K, CHUNK, max_dim)
    for i, (e, s) in enumerate(wins):
        for k in range(VAL_K):
            g = torch.Generator()
            g.manual_seed(stable_seed(SEEDS["val_seed"], e, s, k))
            noise[i, k] = torch.randn(CHUNK, max_dim, generator=g)
    return noise


@torch.no_grad()
def eval_val(policy, cache, noise, real_dims, device, bs=32):
    """Flow-matching velocity MSE, fixed noise, t_k=(k+0.5)/K (k=0..K-1), valid steps x real dims; episode-weighted."""
    model, ad = policy.model, policy.model.lfm_adapter
    policy.eval()
    W = len(cache["windows"])
    lw = np.zeros(W)
    for a in range(0, W, bs):
        sl = list(range(a, min(W, a + bs)))
        B = len(sl)
        L = max(cache["feats"][i].shape[0] for i in sl)
        D = cache["feats"][sl[0]].shape[-1]
        feats = torch.zeros(B, L, D)
        mask = torch.zeros(B, L, dtype=torch.bool)
        for j, i in enumerate(sl):
            n = cache["feats"][i].shape[0]
            feats[j, :n] = cache["feats"][i]
            mask[j, :n] = True
        feats, mask = feats.to(device), mask.to(device)
        state = torch.stack([cache["state"][i] for i in sl]).to(device)
        actions = torch.stack([cache["actions"][i] for i in sl]).to(device)
        valid = (~torch.stack([cache["pad"][i] for i in sl])).to(device).float()  # [B,chunk]
        kv, pmask = ad.make_cache_from_features(model, feats, mask, state)
        acc = torch.zeros(B, device=device)
        for k in range(VAL_K):
            t = (k + 0.5) / VAL_K
            time_ = torch.full((B,), t, device=device)
            nz = noise[sl, k].to(device)
            x_t = t * nz + (1 - t) * actions
            v = model.denoise_step(prefix_pad_masks=pmask, past_key_values=kv, x_t=x_t, timestep=time_)
            err = ((nz - actions) - v)[..., :real_dims] ** 2
            acc += (err.mean(-1) * valid).sum(1) / valid.sum(1).clamp(min=1)
        lw[a:a + B] = (acc / VAL_K).cpu().numpy()
    per_ep = defaultdict(list)
    for (e, _), l in zip(cache["windows"], lw):
        per_ep[e].append(l)
    le = {e: float(np.mean(v)) for e, v in sorted(per_ep.items())}
    vals = np.array(list(le.values()))
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, le


# =====================================================================================================================
# 7. checkpoint policy (single best_valid)
# =====================================================================================================================
def save_best(policy, run_dir, meta, cfg):
    ck = run_dir / "checkpoints"
    tmp, final, old = ck / "best_valid.tmp", ck / "best_valid", ck / "best_valid.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    from safetensors.torch import save_file

    sd = {k: v.detach().cpu().clone().contiguous() for k, v in policy.state_dict().items() if not k.startswith("model.lfm_adapter.backbone.")}
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


# =====================================================================================================================
# 8. TRAIN stage
# =====================================================================================================================
METRIC_COLS = ["event", "step", "samples_seen", "wall_time_s", "loss_total", "loss_action", "lr", "grad_norm", "clipped", "nan_inf_count",
               "step_s", "data_s", "samples_per_s", "peak_vram_gb", "val_loss", "val_loss_sd_over_episodes", "n_val_episodes", "is_best", "best_val_loss"]


def env_info():
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=importlib.metadata.version("transformers"),
                cuda=torch.version.cuda, driver=run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split("\n")[0],
                lerobot_commit=lerobot_commit(), lerobot_version=importlib.metadata.version("lerobot"),
                causal_conv1d=importlib.util.find_spec("causal_conv1d") is not None)


def stage_train(args):
    patch_hashes = ensure_patch()
    configure_backbone(args)
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available(), "a CUDA GPU is required"
    warn_log = set()
    set_determinism(warn_log)
    seed_all(args.train_seed)
    try:
        import av.option  # noqa: F401
    except Exception:
        sys.exit("pyav problem: install `pip install -U 'av>=15,<16'` (lerobot needs av.option)")
    if tuple(int(x) for x in importlib.metadata.version("transformers").split(".")[:2]) < (5, 1):
        sys.exit("transformers>=5.1 is required for LFM2.5-VL (pip install -U 'transformers>=5.5,<6')")

    # ---- split manifest: only train/val ids survive in this process (G1, G2)
    m = read_json(MANIFEST_PATH)
    assert split_hash_of(m) == m["split_hash"], "split_hash mismatch"
    train_ids, val_ids = m["train_ids"], m["val_ids"]
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(m["test_ids"])) and not (set(val_ids) & set(m["test_ids"]))
    split_hash, n_test_declared = m["split_hash"], m["n_test"]
    args.dataset_repo, args.dataset_revision = m["dataset_repo_id"], (m["dataset_revision"] if "unpinned" not in m["dataset_revision"] else "main")
    del m  # test_ids never reach the training process beyond this point

    rid = run_id_of(args)
    run_dir = OUT_ROOT / "runs" / rid
    if (run_dir / "manifest.json").exists():
        sys.exit(f"run {rid} already exists; overwriting is an error (use --run_id)")
    for sub in ("checkpoints", "env", "cache", "test", "figures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "env" / "pip_freeze.txt").write_text(run_cmd([sys.executable, "-m", "pip", "freeze"]))

    argv = cfg_argv(args, run_dir)
    cfg = parse_cfg(argv)
    gate_ok = cfg.policy.use_gated_attn and not cfg.policy.gated_attn_headwise
    assert gate_ok and float(cfg.policy.expert_width_multiplier) == args.expert_width and cfg.policy.chunk_size == CHUNK
    assert int(cfg.policy.num_steps) == 10
    resolved = json.loads(canon(dataclasses.asdict(cfg)))
    for k in ("output_dir", "job_name"):
        resolved.pop(k, None)
    resolved["dataset"].pop("root", None)
    resolved["dataset"].pop("episodes", None)
    config_doc = dict(
        policy=resolved["policy"], backbone=dict(BB, vis_tokens_per_cam=BB["vis_tokens"]),
        data=dict(repo_id=args.dataset_repo, revision=args.dataset_revision, split_manifest=str(MANIFEST_PATH.relative_to(OUT_ROOT)), split_hash=split_hash,
                  video_backend=args.video_backend),
        optim=dict(lr=cfg.optimizer.lr, betas=list(cfg.optimizer.betas), eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay,
                   grad_clip_norm=cfg.optimizer.grad_clip_norm, scheduler=cfg.scheduler.__class__.__name__,
                   warmup_steps=cfg.scheduler.num_warmup_steps, decay_steps=cfg.scheduler.num_decay_steps, decay_lr=cfg.scheduler.decay_lr),
        train=dict(steps=args.steps, batch_size=args.batch_size, num_workers=args.num_workers, val_every=args.val_every, save_freq=0, ema=None,
                   grad_accum=1, precision="fp32"),
        eval=dict(val=dict(K=VAL_K, stride=CHUNK, noise_seed=SEEDS["val_seed"], dims="real", t="(k+0.5)/K, k=0..K-1"),
                  test=dict(windows_per_episode=9, draws=1, num_steps=10, test_seed=SEEDS["test_seed"], bootstrap_B=BOOT_B)),
        seeds=SEEDS | dict(train_seed=args.train_seed),
        determinism=dict(deterministic_algorithms="warn_only", cudnn_deterministic=True, tf32=False, cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"]),
        cli_argv=argv,
    )
    config_hash = sha_obj({k: v for k, v in config_doc.items() if k != "cli_argv"})
    try:
        import yaml

        (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(canon(config_doc)), sort_keys=False))
    except Exception:
        (run_dir / "config.yaml").write_text(json.dumps(config_doc, indent=2, default=str))  # JSON is valid YAML

    gsha, gdirty = git_info()
    manifest = dict(
        schema_version=SCHEMA, run_id=rid, model_key=f"L{args.lfm_layers}_W{str(args.expert_width).replace('.', '')}", status="running", valid=False,
        model=dict(lfm_layers=args.lfm_layers, expert_width_multiplier=args.expert_width, gated=True, gate_headwise=False, gate_bias_init=4.0, backbone_id=BB["id"]),
        revisions=dict(lfm=None, smolvlm2_config_tokenizer=hf_sha(cfg.policy.vlm_model_name), dataset=args.dataset_revision, smolvla_base_used_only_for_audit=hf_sha("lerobot/smolvla_base")),
        seeds=config_doc["seeds"], hashes=dict(config_hash=config_hash, split_hash=split_hash, best_ckpt_sha256=None),
        code=dict(git_sha=gsha, git_dirty=gdirty, script_sha256=file_sha256(SCRIPT), patched_lerobot_files_sha256=patch_hashes, **env_info()),
        hardware=dict(gpu_names=[torch.cuda.get_device_name(0)], world_size=1, mixed_precision="no (fp32)", tf32=False),
        training=dict(steps=args.steps, batch_size_per_device=args.batch_size, effective_batch_size=args.batch_size),
        timestamps=dict(started_utc=now_utc()), warnings=[],
    )
    write_json(run_dir / "manifest.json", manifest)

    # ---- data / policy / optimiser
    train_ds = make_ds(cfg, train_ids)
    val_ds = make_ds(cfg, val_ids)
    assert sorted({int(e) for e in col_np(train_ds, "episode_index")}) == sorted(train_ids), "G2: train dataset episodes != manifest"
    policy, pre, post = make_pol_and_proc(cfg, train_ds)
    policy.to(device)
    manifest["revisions"]["lfm"] = policy.model.lfm_adapter.backbone_revision or hf_sha(BB["id"])
    real_dims = int(policy.config.action_feature.shape[0])
    assert real_dims == 6, f"expected 6 real action dims, got {real_dims}"
    audit_warn = init_audit(policy, run_dir, check_base=not args.skip_base_hash_check)
    manifest["warnings"] += audit_warn
    tot, trn, act = n_params(policy)
    manifest["model"].update(params_total=tot, params_trainable=trn, params_active=act)

    acts = np.asarray(train_ds.hf_dataset.data.column("action").to_pylist(), dtype=np.float64)[:, :real_dims]
    write_json(run_dir / "train_action_std.json", dict(std=acts.std(0).tolist(), mean=acts.mean(0).tolist(), n_frames=len(acts)))
    del acts

    make_opt = first_import(("lerobot.optim.factory", "make_optimizer_and_scheduler"))
    optimizer, scheduler = make_opt(cfg, policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    clip = float(cfg.optimizer.grad_clip_norm)

    # ---- validation cache
    t_c = time.time()
    key = sha_obj(dict(split_hash=split_hash, lfm=manifest["revisions"]["lfm"], img=dict(size=BB["img_size"], vis=BB["vis_tokens"], labels=policy.model.lfm_adapter.cam_labels),
                       keep=BB["keep_layers"], dtype=BB["dtype"], chunk=CHUNK, K=VAL_K))
    cache = build_val_cache(policy, pre, val_ds, val_ids, run_dir / "cache" / "val_features.pt", key, args, device)
    noise = make_val_noise(cache["windows"], int(policy.config.max_action_dim))
    cache_build_s = time.time() - t_c
    n_val_eps = len(val_ids)

    # ---- loop
    hist_path = run_dir / "metrics_history.csv"
    hist = open(hist_path, "w", newline="")
    wr = csv.DictWriter(hist, fieldnames=METRIC_COLS, restval="")
    wr.writeheader()
    dl = make_loader(train_ds, args.batch_size, True, args.train_seed, args.num_workers, drop_last=True)
    it = iter(dl)
    best, first_val, best_step, status = float("inf"), None, -1, "running"
    wall_best = trainonly_best = val_best = None
    val_time = 0.0
    win = dict(loss=0.0, step=0.0, data=0.0, gn=0.0, clip=0.0, nan=0)
    t0 = time.time()
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(1, args.steps + 1):
            ts = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            data_s = time.perf_counter() - ts
            batch = pre(batch)
            loss = forward_loss(policy, batch)
            if not torch.isfinite(loss):
                win["nan"] += 1
                raise FloatingPointError(f"non-finite training loss at step {step}")
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            win["loss"] += float(loss)
            win["gn"] += gn
            win["clip"] += float(gn > clip)
            win["data"] += data_s
            win["step"] += time.perf_counter() - ts
            if step % 50 == 0:
                n = 50
                wr.writerow(dict(event="train", step=step, samples_seen=step * args.batch_size, wall_time_s=round(time.time() - t0, 2),
                                 loss_total=win["loss"] / n, loss_action=win["loss"] / n, lr=optimizer.param_groups[0]["lr"], grad_norm=win["gn"] / n,
                                 clipped=win["clip"] / n, nan_inf_count=win["nan"], step_s=win["step"] / n, data_s=win["data"] / n,
                                 samples_per_s=args.batch_size * n / max(win["step"], 1e-9), peak_vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
                hist.flush()
                print(f"[train] step {step}/{args.steps} loss={win['loss'] / n:.4f} gn={win['gn'] / n:.2f} lr={optimizer.param_groups[0]['lr']:.2e} {win['step'] / n:.2f}s/step", flush=True)
                win = dict(loss=0.0, step=0.0, data=0.0, gn=0.0, clip=0.0, nan=0)
            if step % args.val_every == 0 or step == args.steps:
                tv = time.time()
                vl, vsd, le = eval_val(policy, cache, noise, real_dims, device, args.val_batch)
                policy.train()
                val_time += time.time() - tv
                finite = math.isfinite(vl)
                if first_val is None and finite:
                    first_val = vl
                is_best = finite and vl < best and not (abs(vl - best) <= 1e-9)
                if is_best:
                    best, best_step = vl, step
                    wall_best = time.time() - t0
                    val_best = val_time
                    trainonly_best = wall_best - val_best
                    sha = save_best(policy, run_dir, dict(step=step, val_loss=vl, split_hash=split_hash, config_hash=config_hash, git_sha=gsha,
                                                           lfm_revision=manifest["revisions"]["lfm"], smolvlm2_revision=manifest["revisions"]["smolvlm2_config_tokenizer"],
                                                           backbone_id=BB["id"], wall_time_s=wall_best), cfg)
                    manifest["hashes"]["best_ckpt_sha256"] = sha
                    with open(run_dir / "checkpoints" / "best_history.csv", "a", newline="") as f:  # metadata only
                        w = csv.writer(f)
                        if f.tell() == 0:
                            w.writerow(["step", "val_loss", "wall_time_s"])
                        w.writerow([step, vl, round(wall_best, 2)])
                    with open(run_dir / "val_per_episode_best.csv", "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["episode_id", "val_loss"])
                        w.writerows(le.items())
                wr.writerow(dict(event="val", step=step, samples_seen=step * args.batch_size, wall_time_s=round(time.time() - t0, 2), val_loss=vl,
                                 val_loss_sd_over_episodes=vsd, n_val_episodes=n_val_eps, is_best=bool(is_best), best_val_loss=best,
                                 peak_vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
                hist.flush()
                print(f"[val] step {step} val_loss={vl:.5f} (best {best:.5f} @ {best_step}){' *' if is_best else ''}", flush=True)
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
    hist.close()

    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    manifest.update(status=status, valid=status == "complete")
    manifest["training"].update(
        best_step=best_step, best_val_loss=None if best_step < 0 else best, first_val_loss=first_val,
        wall_time_total_s=wall_best,  # = time until the best (lowest val-loss) checkpoint was reached (user definition)
        wall_time_train_only_s=trainonly_best, wall_time_val_s=val_best,
        gpu_hours=None if wall_best is None else wall_best / 3600.0,  # world_size = 1
        wall_time_full_run_s=total_wall, wall_time_full_run_val_s=val_time, val_cache_build_s=cache_build_s,
        steps_per_s=(args.steps / max(total_wall - val_time, 1e-9)), samples_per_s=(args.steps * args.batch_size / max(total_wall - val_time, 1e-9)),
        peak_train_vram_gb=peak, n_val_windows=len(cache["windows"]),
        val_vs_train_t_distribution_note="validation uses a uniform t grid; training samples t from SmolVLA's Beta-shaped sample_time",
    )
    manifest["timestamps"]["finished_utc"] = now_utc()
    manifest["warnings"] += sorted(warn_log)
    write_json(run_dir / "manifest.json", manifest)
    print(f"[train] status={status} best_val={best:.5f}@{best_step} time_to_best={wall_best}")
    if status == "failed":
        sys.exit(2)


def configure_backbone(args):
    BB.update(keep_layers=args.lfm_layers, vis_tokens=args.lfm_vis_tokens, img_size=args.img_size)


# =====================================================================================================================
# 9. TEST stage  (pure-numpy metric helpers first: unit-testable)
# =====================================================================================================================
def window_metrics(pred, gt, valid, std):
    """pred, gt: [T,D] raw units; valid: [T] bool; std: [D] train-split per-dim action std (metric denominator only)."""
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
    configure_backbone(args)
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
    (tdir).mkdir(parents=True, exist_ok=True)
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
    bad = [k for k in res.missing_keys if not k.startswith("model.lfm_adapter.backbone.")]
    assert not bad and not res.unexpected_keys, f"S2 strict load failed: missing={bad[:5]} unexpected={res.unexpected_keys[:5]}"
    policy.eval()
    assert getattr(cfg, "ema", None) is None
    model = policy.model
    real = int(policy.config.action_feature.shape[0])
    amean = np.asarray(ds.meta.stats["action"]["mean"], dtype=np.float64)[:real]
    astd = np.asarray(ds.meta.stats["action"]["std"], dtype=np.float64)[:real]
    assert str(cfg.policy.normalization_mapping["ACTION"]).endswith("MEAN_STD")
    tr = read_json(run_dir / "train_action_std.json")
    std_train = np.asarray(tr["std"])

    # ---- windows (9 evenly spaced per episode + one seam pair per episode for bjump)
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
        for batch in dl:
            gt = batch["action"].clone().float().numpy()[..., :real]
            pad = batch["action_is_pad"].numpy() if "action_is_pad" in batch else np.zeros(gt.shape[:2], bool)
            b = pre(batch)
            i0 = len(preds)
            nz = torch.stack([torch.randn(CHUNK, int(policy.config.max_action_dim), generator=torch.Generator().manual_seed(
                stable_seed(SEEDS["test_seed"], rows[i0 + j]["ep"], rows[i0 + j]["widx"], 0))) for j in range(gt.shape[0])]).to(device)
            images, img_masks, lang, lm, state, _ = unpack(policy, b)
            out = model.sample_actions(images, img_masks, lang, lm, state, noise=nz)[..., :real].float().cpu().numpy() * astd + amean
            for j in range(gt.shape[0]):
                preds.append(out[j])
                gts.append(gt[j])
                valids.append(~pad[j])
    preds, gts, valids = np.stack(preds), np.stack(gts), np.stack(valids)
    print(f"[test] inferred {len(rows)} windows ({sum(r['kind'] == 'main' for r in rows)} main + seam pairs)")

    # ---- per-window / per-episode metrics
    wrows, per_ep = [], defaultdict(list)
    keys = ["mae", "mse", "rmse", "nmae", "jerk_in_chunk", "accel_in_chunk", "gt_jerk_in_chunk", "smoothness_ratio", "jerk_norm", "accel_norm"]
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
    # bjump
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

    # ---- S9 latency (batch 1, fp32, device tensors -> unnormalised actions on CPU)
    g = np.random.default_rng(SEEDS["test_seed"])
    lat_main = [r for r in rows if r["kind"] == "main"]
    pick = [lat_main[i]["ds_idx"] for i in g.choice(len(lat_main), size=min(100, len(lat_main)), replace=False)]
    prepared = []
    for batch in make_loader(ds, 1, False, 0, 0, indices=pick):
        prepared.append(unpack(policy, pre(batch))[:5])
    torch.cuda.reset_peak_memory_stats()
    fixed_noise = torch.randn(1, CHUNK, int(policy.config.max_action_dim), generator=torch.Generator().manual_seed(SEEDS["test_seed"])).to(device)
    lat, pre_ms = [], []

    def one(p):
        PREPROC.clear()
        torch.cuda.synchronize()
        t = time.perf_counter()
        o = model.sample_actions(*p, noise=fixed_noise)[..., :real].float().cpu().numpy() * astd + amean
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000, sum(PREPROC) * 1000

    with torch.no_grad():
        for k in range(10):
            one(prepared[k % len(prepared)])
        for k in range(300):
            a, b = one(prepared[k % len(prepared)])
            lat.append(a)
            pre_ms.append(b)
    peak_infer = torch.cuda.max_memory_allocated() / 2 ** 30
    lat = np.array(lat)
    with open(tdir / "latency_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "latency_ms", "preprocess_ms"])
        w.writerows((i, a, b) for i, (a, b) in enumerate(zip(lat, pre_ms)))
    latency = dict(n_runs=300, warmup=10, p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)), p99_ms=float(np.percentile(lat, 99)),
                   mean_ms=float(lat.mean()), std_ms=float(lat.std(ddof=1)), preprocess_mean_ms=float(np.mean(pre_ms)), device=torch.cuda.get_device_name(0),
                   backbone_dtype=BB["dtype"], causal_conv1d_installed=importlib.util.find_spec("causal_conv1d") is not None)
    if not latency["causal_conv1d_installed"]:
        wl.append("causal_conv1d not installed: LFM conv layers use the slow reference path (latency is pessimistic)")

    # ---- S10 complexity
    from torch.utils.flop_counter import FlopCounterMode

    def fl(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops()

    one_batch = None
    for batch in make_loader(ds, 1, False, 0, 0, indices=pick[:1]):
        one_batch = pre(batch)
    inp = unpack(policy, one_batch)
    with torch.no_grad():
        f_inf = fl(lambda: model.sample_actions(*inp[:5]))
        f_fwd = fl(lambda: forward_loss(policy, one_batch))
    policy.train()
    f_trn = fl(lambda: forward_loss(policy, one_batch).backward())
    policy.zero_grad(set_to_none=True)
    policy.eval()
    tot, trn, act = n_params(policy)
    complexity = dict(gflops_infer_chunk=f_inf / 1e9, gmacs_infer_chunk=f_inf / 2e9, gflops_forward=f_fwd / 1e9, gflops_train_step=f_trn / 1e9,
                      params_total=tot, params_trainable=trn, params_active=act, peak_infer_vram_gb=peak_infer, counter="torch.utils.flop_counter.FlopCounterMode",
                      counter_note="matmul/bmm/conv/sdpa only (1 MAC = 2 FLOPs); elementwise ops, softmax/sigmoid and the CPU LFM image processor are not counted")

    # ---- S12 statistics
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
        protocol=dict(windows_per_episode=9, draws_per_window=1, num_steps=10, bootstrap_B=BOOT_B, test_seed=SEEDS["test_seed"], train_action_std=tr["std"],
                      nmae="mean_d(MAE_d / std_d(train))", jerk_norm="mean_d(mean|3rd diff|_d / std_d(train))", dtype=dict(lfm_backbone="fp32", expert_gates="fp32")),
        warnings=wl + man.get("warnings", []) + sorted(warn_log),
    )
    write_json(tdir / "test_report.json", report)
    (tdir / "test.lock").write_text("done " + now_utc())
    print(f"[test] MAE={metrics['mae']['mean']:.4f} [{metrics['mae']['ci_lo']:.4f},{metrics['mae']['ci_hi']:.4f}]  nMAE={metrics['nmae']['mean']:.4f}  "
          f"p95={latency['p95_ms']:.1f}ms  GFLOPs/chunk={complexity['gflops_infer_chunk']:.1f}")


# =====================================================================================================================
# 10. FIGURES (rebuilt purely from the saved CSV / JSON / NPZ files)
# =====================================================================================================================
def stage_plots(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rid = run_id_of(args)
    run_dir = OUT_ROOT / "runs" / rid
    fig_dir = OUT_ROOT / "figures" / rid
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
    hist = pd.read_csv(run_dir / "metrics_history.csv")
    tr, va = hist[hist.event == "train"], hist[hist.event == "val"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name)
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(tr.step, tr.loss_total)
    ax[0].set(title="Training loss (mean of 50 steps)", xlabel="step", ylabel="flow-matching MSE")
    ax[0].set_yscale("log")
    ax[1].plot(tr.step, tr.grad_norm)
    ax[1].set(title="Gradient norm (pre-clip)", xlabel="step")
    ax[2].plot(tr.step, tr.lr)
    ax[2].set(title="Learning rate", xlabel="step")
    save(fig, "01_training_curves.png")

    if len(va):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(va.step, va.val_loss, yerr=va.val_loss_sd_over_episodes / np.sqrt(va.n_val_episodes), marker="o", capsize=2, label="val loss (±SE over episodes)")
        b = va[va.is_best.astype(str) == "True"]
        if len(b):
            ax.scatter(b.step, b.val_loss, c="red", zorder=3, label="new best (checkpoint written)")
        ax.set(title="Validation loss (fixed noise, fixed t)", xlabel="step", ylabel="velocity MSE")
        ax.legend()
        save(fig, "02_validation_loss.png")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tr.step, tr.step_s, label="step time")
        ax.plot(tr.step, tr.data_s, label="data wait")
        ax.set(title="Throughput", xlabel="step", ylabel="s / step")
        ax.legend()
        save(fig, "03_step_time.png")
        vp = run_dir / "val_per_episode_best.csv"
        if vp.exists():
            v = pd.read_csv(vp)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(v.val_loss, bins=20)
            ax.set(title="Per-episode validation loss at best checkpoint", xlabel="val loss", ylabel="episodes")
            save(fig, "04_val_per_episode.png")

    tdir = run_dir / "test"
    if not (tdir / "test_report.json").exists():
        print("[plots] no test report yet: training figures only")
        return
    rep = read_json(tdir / "test_report.json")
    ep = pd.read_csv(tdir / "test_per_episode.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, k, t in zip(ax, ["mae", "nmae", "mse"], ["MAE (raw units)", "nMAE", "MSE (raw units)"]):
        a.hist(ep[k], bins=25)
        m = rep["metrics"][k]
        a.axvline(m["mean"], c="r", label=f"mean {m['mean']:.3g} [{m['ci_lo']:.3g}, {m['ci_hi']:.3g}]")
        a.axvline(m["median"], c="k", ls="--", label=f"median {m['median']:.3g}")
        a.set(title=f"Per-episode {t}", xlabel=t, ylabel="episodes")
        a.legend(fontsize=7)
    save(fig, "05_test_error_distributions.png")

    P = np.load(tdir / "predictions.npz")
    pred, gt, valid = P["pred"], P["gt"], P["valid"]
    err = np.abs(pred - gt) * valid[..., None]
    n_valid_t = valid.sum(0)[:, None]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(err.sum(0).sum(1) / np.maximum(n_valid_t[:, 0] * pred.shape[-1], 1))
    ax[0].set(title="MAE vs. position in the action chunk", xlabel="chunk step", ylabel="MAE")
    pd_mae = err.sum(0) / np.maximum(n_valid_t, 1)
    ax[1].bar(range(pred.shape[-1]), pd_mae.mean(0))
    ax[1].set(title="MAE per action dimension", xlabel="dim", ylabel="MAE")
    save(fig, "06_error_vs_horizon_and_dim.png")

    w_mae = (np.abs(pred - gt) * valid[..., None]).sum((1, 2)) / np.maximum(valid.sum(1) * pred.shape[-1], 1)
    order = np.argsort(w_mae)
    picks = [("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])]
    fig, ax = plt.subplots(3, pred.shape[-1], figsize=(2.6 * pred.shape[-1], 7), sharex=True)
    for r, (name, i) in enumerate(picks):
        for d in range(pred.shape[-1]):
            ax[r, d].plot(gt[i, :, d], label="ground truth")
            ax[r, d].plot(pred[i, :, d], "--", label="prediction")
            if r == 0:
                ax[r, d].set_title(f"dim {d}")
            if d == 0:
                ax[r, d].set_ylabel(f"{name}\nep {int(P['episode_id'][i])}")
    ax[0, 0].legend(fontsize=7)
    save(fig, "07_predicted_vs_ground_truth_chunks.png")

    sj = pd.read_csv(tdir / "bjump_seams.csv")
    lat = pd.read_csv(tdir / "latency_samples.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(lat.latency_ms, bins=40)
    for p, c in (("p50_ms", "g"), ("p95_ms", "orange"), ("p99_ms", "r")):
        ax[0].axvline(rep["latency"][p], c=c, label=f"{p[:3]} {rep['latency'][p]:.0f} ms")
    ax[0].set(title="Latency per action chunk (bs 1)", xlabel="ms")
    ax[0].legend()
    ax[1].hist(sj.l2_jump, bins=25)
    ax[1].set(title=f"Chunk-boundary jump (n={len(sj)})", xlabel="L2 jump (raw units)")
    ax[2].scatter(ep.gt_jerk_in_chunk, ep.jerk_in_chunk, s=12)
    lim = [0, float(np.nanmax([ep.gt_jerk_in_chunk.max(), ep.jerk_in_chunk.max()]))]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set(title="Jerk: prediction vs ground truth", xlabel="GT jerk", ylabel="predicted jerk")
    save(fig, "08_latency_boundary_jerk.png")

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, k in zip(ax, ["mae", "nmae", "mse", "jerk_norm"]):
        m = rep["metrics"][k]
        a.bar([0], [m["mean"]], yerr=[[m["mean"] - m["ci_lo"]], [m["ci_hi"] - m["mean"]]], capsize=6)
        a.set(title=f"{k} mean + 95% bootstrap CI", xticks=[])
    save(fig, "09_metric_summary_ci.png")
    print(f"[plots] figures written to {fig_dir}")


# =====================================================================================================================
# 11. entry point
# =====================================================================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=["all", "split", "train", "test", "plots"])
    p.add_argument("--dataset_repo", default=DEFAULT_DATASET)
    p.add_argument("--dataset_revision", default="main")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lfm_layers", type=int, default=8)
    p.add_argument("--expert_width", type=float, default=0.5)
    p.add_argument("--lfm_vis_tokens", type=int, default=48)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_batch", type=int, default=32)
    p.add_argument("--test_batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--video_backend", default="pyav")
    p.add_argument("--train_seed", type=int, default=SEEDS["train_seed"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--run_id", default=None)
    p.add_argument("--skip_base_hash_check", action="store_true", help="skip downloading lerobot/smolvla_base (~0.9 GB) for the R9 scratch-hash check")
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
        for st in ("split", "train", "test", "plots"):
            print(f"\n===== stage: {st} =====", flush=True)
            rc = subprocess.call([sys.executable, str(SCRIPT), "--stage", st] + forward_args(args), env=env)
            if rc != 0:
                sys.exit(f"stage '{st}' failed with exit code {rc}")
        return
    {"split": stage_split, "train": stage_train, "test": stage_test, "plots": stage_plots}[args.stage](args)


if __name__ == "__main__":
    main()
