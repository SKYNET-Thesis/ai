# SmollerVLA Train / Validation / Test Design Guide

**Version:** 1.0 · **Target reader:** coding assistant (PyTorch) · **Scope:** SmollerVLA = frozen pretrained LFM2.5-VL-450M (first 8 layers) + AdaptiveConditionalGatedKV conditioner + SmolVLA action expert with G1 gated attention, trained with the lerobot flow-matching pipeline. Everything except the pretrained VLM is trained from scratch, and the result is compared with the original SmolVLA under the same rule (sections 10 and 11).

**Terms.** RUN = one (lfm\_layers L, expert\_width W) configuration. WINDOW = (episode\_id, start\_frame) producing one action chunk of 50 steps. REAL DIMS = the first 6 of the 32 padded action dims. Every number below marked (obs.) was observed in the supplied CSVs or notebook; everything else is a design decision.

## 0. Hard Requirements

| ID | Requirement |
| --- | --- |
| R1 | Split at EPISODE level, 70/10/20, one shared manifest for every RUN. Test set is identical across RUNs (paired tests depend on it). |
| R2 | Checkpoint selection uses validation loss only. Test data is never read before the final test step. |
| R3 | Exactly one checkpoint is ever written, only when validation loss reaches a new best, and it is the only checkpoint loaded for test: best\_valid. No last-step, EMA, averaging or ensembling. |
| R4 | The test script emits every metric in the metric table (ranks 1–7 and 6a) into test\_report.json and test\_per\_episode.csv. |
| R5 | No custom normalization step is part of this protocol. Every arm uses the framework's default pipeline unchanged. Known side effect: lerobot's default statistics are dataset-wide, so they include val and test episodes. This affects all arms equally. |
| R6 | Every artifact carries split\_hash, config\_hash and git\_sha. A mismatch aborts the run. |
| R7 | A RUN whose validation loss fails the degeneracy gate (section 3.6) is flagged invalid and excluded from rankings. |

R8 and R9 (fair initialization and the step-0 init audit) are defined in section 10 and apply to every arm. Section 11 checks this guide against the SmollerVLA code.

## 1. Data Split Protocol

### 1.1 Parameters

| Field | Value |
| --- | --- |
| unit | episode\_index (never frames or windows) |
| ratios | train 0.70, val 0.10, test 0.20 |
| split\_seed | 42 (obs.: split\_seed=42 in all CSVs; keep it so the test set stays comparable with the existing benchmark) |
| stratify\_by | task string; strata with fewer than 5 episodes are pooled into one stratum named POOL |
| dataset | yamaisanerd/Chunking\_560\_Skynet (560 episodes, 656,950 frames, 3 cameras at 640x480 AV1), pinned to a commit SHA (never main) |

### 1.2 Algorithm

```
ids = sorted(all episode_index); N = len(ids)
n_test = round_half_up(0.2*N); n_val = round_half_up(0.1*N); n_train = N - n_test - n_val
strata = group ids by task; merge groups with size < 5 into POOL
rng = numpy.random.default_rng(split_seed)
for s in sorted(strata): rng.shuffle(strata[s])
quota_test[s]: floor(0.2*|s|), then hand out the remainder to the largest fractional parts
    (ties broken by sorted stratum key) until sum == n_test
quota_val[s]: same procedure with 0.1, applied to the episodes left after the test draw
take the first quota_test[s] shuffled ids as test, the next quota_val[s] as val, the rest as train
assert disjoint and union == ids; sort each list ascending
```

Expected sizes: N=560 gives exactly 392 / 56 / 112. The earlier n=111 test set came from a split of unknown composition, so the new manifest supersedes it (section 7, Q1). Build the id list from the episode table, because meta/info.json is stale: it still says 571 episodes and 666,669 frames.

### 1.3 Split manifest: splits/split\_manifest.json

| Field | Type | Description |
| --- | --- | --- |
| schema\_version | str | 1.0 |
| dataset\_repo\_id, dataset\_revision | str | repo id and commit SHA |
| split\_seed, ratios | int, object | inputs of 1.1 |
| n\_total, n\_train, n\_val, n\_test | int | counts |
| train\_ids, val\_ids, test\_ids | list\[int\] | sorted episode ids |
| strata\_counts | object | task -> counts per split |
| split\_hash | str | sha256 of canonical JSON (sorted keys, no whitespace) of {train\_ids, val\_ids, test\_ids, split\_seed, ratios, dataset\_revision} |
| excluded\_episodes | list\[object\] | each {episode\_id, reason}; empty by default |

The manifest is generated once. If the file exists, load it and recompute the hash. Never regenerate silently.

### 1.4 Leakage guards

| ID | Guard |
| --- | --- |
| G1 | Assert pairwise disjointness of the three id lists at every process start. |
| G2 | Restrict the dataset per split with the lerobot dataset episodes argument. The training process receives train\_ids and val\_ids only. It never receives test\_ids. |
| G3 | Not used. There is no custom normalization step (see R5). |
| G4 | Per-episode content hash (sha256 of quantized actions and states). Identical hashes in different splits abort the run. |
| G5 | If episode metadata has a session, operator or source id, log the cross-split overlap. See Q3. |
| G6 | Validation and test windows, noise and timesteps come from their own seeded generators, independent of the training RNG. |
| G7 | Test data is touched only by evaluate\_test.py after training has ended (section 4). |

## 2. Validation Loss Definition

### 2.1 Decision

The selection criterion is the **flow-matching velocity MSE on the validation split with fixed noise and fixed timesteps**. It is the training objective with its randomness removed. Rationale:

- It is deterministic and low-variance, so checkpoint ranking is stable.
- It needs no 10-step integration.
- In the existing logs (obs.) val\_fixed\_noise\_loss falls together with val\_mae for the healthy runs (L8 W0.5: 0.397 -> 0.227 while val MAE fell 13.7 -> 9.3).
- Action MAE belongs to the test protocol, where it is the rank-1 metric.

### 2.2 Formula

For window w in episode e, timestep t\_k and noise eps\_wk:

- x\_t = t\_k \* eps\_wk + (1 - t\_k) \* a (matches the implemented convention, noise at t=1)
- u = eps\_wk - a
- v = policy.denoise\_step(x\_t, t\_k, prefix of w)

```
l_w   = mean over k in 1..K, valid chunk steps s, real dims d of (u - v)^2
l_e   = mean over windows w of episode e of l_w
val_loss = mean over val episodes e of l_e        # episode-weighted
```

### 2.3 Settings

| Setting | Value |
| --- | --- |
| space | the model's own training space (framework default pipeline, unchanged) |
| dims | 6 real dims only (never the 26 padded dims) |
| chunk steps | mask padded steps with actions\_is\_pad |
| timesteps | K=10, t\_k=(k+0.5)/10 |
| noise | torch.Generator seeded by hash(val\_seed=1234, episode\_id, start\_frame, k) |
| windows | all start frames 0, 50, 100, ... (stride = chunk\_size) in every val episode; no subsampling (the prior fast\_16ep\_1window\_1draw mode is retired) |
| mode | model.eval(), torch.no\_grad(), no augmentation, fp32 expert and gates |
| frozen-backbone cache | the LFM backbone is frozen and validation is deterministic, so precompute its hidden states once into cache/val\_features.pt. Key the cache on split\_hash + LFM revision + image preprocessing config (IMG\_SIZE, VIS\_TOKENS\_PER\_CAM=48, camera label strings) + keep\_layers + backbone dtype. For the SmolVLA baseline, cache the SmolVLM2 prefix key/value tensors instead, with the same keying. Every evaluation then runs only conditioner + expert. Rebuild automatically on key mismatch. |

Backbone features are cached, but state\_proj and the conditioner stay live because they are trained.

### 2.4 Distributed aggregation

Shard val episodes by episode\_id modulo world\_size. Each rank returns {episode\_id: l\_e}. Rank 0 gathers them, sorts by id and averages. The result must not depend on world\_size.

### 2.5 Logging

| Event | Cadence | Fields |
| --- | --- | --- |
| train step | every 50 steps | loss\_total, loss\_action, lr, grad\_norm, clipped, nan\_inf\_count, step\_s, data\_s, samples\_per\_s, peak\_vram\_gb |
| validation | every 250 steps and at the final step | val\_loss, val\_loss\_sd\_over\_episodes, n\_val\_episodes, is\_best, best\_val\_loss |

Write each validation as one row of metrics\_history.csv. Per-episode val losses at the best step go to val\_per\_episode\_best.csv.

## 3. Checkpoint Policy

**Rule: a checkpoint is written only when validation loss reaches a new best. That single best\_valid checkpoint is the only file ever loaded for testing. Nothing else is saved.**

### 3.1 Trigger

Save when val\_loss is finite and `val_loss < best_val_loss` (strict, min\_delta = 0). Evaluate at every validation event.

### 3.2 Tie-breaking

If `abs(val_loss - best_val_loss) <= 1e-9`, keep the existing checkpoint (the earlier step wins).

### 3.3 Write procedure (atomic)

1. Write checkpoints/best\_valid.tmp/ with all files, fsync, then compute the sha256 of model.safetensors.
2. Rename it over checkpoints/best\_valid/ (remove the old directory first).
3. Append {step, val\_loss, wall\_time\_s} to checkpoints/best\_history.csv. This is metadata only, never a loadable checkpoint.

### 3.4 Files kept on disk

| Path | Content |
| --- | --- |
| checkpoints/best\_valid/model.safetensors | every trainable tensor (action expert, gates, projections, conditioner). The frozen pretrained VLM (LFM2.5-VL for SmollerVLA, SmolVLM2 for the SmolVLA baseline) is not saved. It is re-fetched from the pinned Hugging Face revision recorded in checkpoint\_meta.json. SmollerVLA also needs the SmolVLM2 tokenizer revision, because the task text is decoded with it and re-tokenized by LFM. |
| checkpoints/best\_valid/policy\_config.json | resolved policy config |
| checkpoints/best\_valid/checkpoint\_meta.json | step, val\_loss, sha256, split\_hash, config\_hash, git\_sha, backbone revisions |

### 3.5 Forbidden

- Disable lerobot periodic saving (save\_freq=2500 in the notebook). It would create step checkpoints that are not best\_valid.
- Disable EMA (assert ema is None).
- No last-step evaluation, checkpoint averaging, SWA, ensembling or best-of-seeds selection.
- No use of any test metric for any decision.
- No other checkpoint of any kind: no periodic, last-step, optimizer or resume state. A crashed run restarts from step 0. best\_history.csv is metadata only and is never loadable.

Training runs for a fixed step budget (4000 in the notebook) with the cosine schedule fixed up front. There is no early stopping, because it would change the LR schedule.

### 3.6 Run status and degeneracy gate

| Status | Condition |
| --- | --- |
| failed | non-finite training loss, or no finite validation loss ever recorded. Abort and run no test. |
| degenerate | `best_val_loss > 0.8 * first_val_loss` (less than 20% improvement). The test may run, but the report sets valid=false. The observed L8 W0.75 run improved only about 9% (2.26 -> 2.06) and would be flagged. Healthy runs improved about 43%. |
| complete | everything else |

## 4. Final Test Protocol

Entry point: `python evaluate_test.py --run_dir <RUN>`. It is a separate process from training.

### 4.1 Preconditions

P1: manifest.json status is complete or degenerate. P2: sha256 of best\_valid/model.safetensors equals checkpoint\_meta.json. P3: split\_hash in the run equals the shared split\_manifest.json. P4: config\_hash matches. P5: test.lock does not exist. P6: n\_test equals the manifest count (112 expected for N=560). P7: checkpoints/ contains only best\_valid/ and best\_history.csv, and anything else aborts the test.

### 4.2 Sequence

| Step | Action |
| --- | --- |
| S1 | Run P1–P6. Create test.lock. |
| S2 | Call lfm\_backbone\_adapter.install() (SmollerVLA arm only), then load best\_valid only. Eval mode. Assert no EMA. Load strictly: every missing key must start with model.lfm\_adapter.backbone., and there must be no unexpected keys and no shape-skipped tensors. The notebook's shape-tolerant loader was a warm-start workaround and must be off. |
| S3 | Load test episodes from test\_ids. |
| S4 | Windows: 9 evenly spaced valid start frames per episode, from numpy.linspace over the feasible starts with rounding (matches the 999 windows per model, obs.). draws\_per\_window = 1 (see Q5). 10 Euler steps. Noise from a generator seeded by hash(test\_seed=2024, episode\_id, window\_idx, draw). |
| S5 | Per window, compare the unnormalized predicted chunk with the ground-truth chunk over valid steps and 6 real dims. mae = mean abs error. mse = mean squared error in raw units. nmae = mean over dims of (MAE\_d / std\_d), where std\_d is the per-dim action std of the train split, computed once and stored in test\_report.json. It is a metric denominator only, never a model input. |
| S6 | Smoothness: jerk\_in\_chunk = mean abs third finite difference along the chunk time axis of the predicted chunk. Also compute gt\_jerk\_in\_chunk from ground truth with the same operator, and the per-window ratio. Report predicted vs ground-truth jerk. |
| S7 | bjump: for consecutive windows (i, i+1) of the same episode spaced exactly chunk\_size apart, l2\_jump = L2 norm of (chunk\_i\[last\] - chunk\_{i+1}\[first\]) in raw units over 6 dims. Record n\_seams and bjump\_mode (existing runs used fast\_fixed with 32 seams). Report as indicative whenever n\_seams < 100. |
| S8 | Aggregate windows to episodes (mean over draws, then windows). Write test\_per\_episode.csv. Also write predictions.npz (predicted and ground-truth chunks for every test window, raw units, float32) so figures and statistics can be rebuilt without the model. |
| S9 | Latency pass on the same checkpoint: batch size 1, 10 warmup runs, N=300 timed runs (the prior 100 gives a very noisy p99) over 100 seeded test windows. Timer spans device tensors to unnormalized actions on CPU, with cuda.synchronize before each clock read. Record p50, p95, p99, mean and std of latency\_ms per action chunk. Report p95 and p99 as the headline, never the mean. In SmollerVLA the LFM image processor runs on CPU inside the model call, once per image, so preprocessing is part of the measured latency. Time it separately with a hook around that call. Also record the backbone dtype and whether causal\_conv1d is installed; without it the LFM conv layers use a slow reference path. |
| S10 | Complexity: torch.utils.flop\_counter.FlopCounterMode, batch size 1, one full inference chunk. Emit gflops, gmacs (= gflops / 2), params\_total, params\_trainable and params\_active. Reset peak memory stats before S9 and record peak\_infer\_vram\_gb as max\_memory\_allocated. |
| S11 | Within-run statistics (S12) and write test\_report.json. Mark test.lock as done. |
| S12 | See 4.3. |

### 4.3 Statistics settings

| Item | Setting |
| --- | --- |
| bootstrap | percentile method, resample episodes with replacement, B=10000, bootstrap\_seed=20240, 95% CI |
| CI targets | mean of mae, nmae, mse, jerk\_norm per run; mean paired difference per comparison |
| paired test | scipy.stats.wilcoxon on per-episode differences, two-sided, zero\_method=wilcox, normal approximation (n = number of test episodes, 112 for N=560) |
| supplementary | paired t-test, win rate, rank-biserial effect size |
| multiplicity | Holm correction across the family {mae, nmae, mse, jerk\_norm} per comparison; alpha=0.05 |
| pairing rule | compare\_runs.py refuses to pair two runs unless their test\_ids and split\_hash are identical |
| reporting | MSE: mean and SD across episodes. SD near the mean (obs. 120 vs 176) means a few episodes dominate. Also report the median and the 90th percentile of per-episode MSE. |

### 4.4 Metric table to emitted field map

| Rank | Metric | Where it is written |
| --- | --- | --- |
| 1 | MAE / nMAE | test\_per\_episode.csv (mae, nmae); test\_report.json metrics.mae, metrics.nmae with mean, ci\_lo, ci\_hi |
| 2 | Latency p95/p99 | test\_report.json latency.p95\_ms, p99\_ms |
| 3 | jerk\_norm | test\_per\_episode.csv jerk\_norm, gt\_jerk\_in\_chunk; report metrics.jerk\_norm |
| 4 | bjump | test\_report.json bjump.mean, n\_seams, mode |
| 5 | MSE\_test@best\_valid | test\_per\_episode.csv mse; report metrics.mse mean, sd, median, p90 |
| 6 | GFLOPs/GMACs and params | test\_report.json complexity.\* |
| 6a | wall\_time\_total | manifest.json training.wall\_time\_total\_s (copied into the report) |
| 7 | Peak inference VRAM | test\_report.json complexity.peak\_infer\_vram\_gb |

## 5. Save Format and Artifact Schema

### 5.1 Directory tree

```
splits/split_manifest.json                  # shared by all runs
runs/{run_id}/
  config.yaml
  manifest.json
  env/pip_freeze.txt
  metrics_history.csv
  val_per_episode_best.csv
  cache/val_features.pt                     # excluded from archives
  checkpoints/
    best_valid/{model.safetensors, policy_config.json, checkpoint_meta.json}
    best_history.csv                        # metadata only, never loadable
  test/
    test.lock
    test_report.json
    test_per_episode.csv
    test_per_window.csv
    predictions.npz                         # predicted + ground-truth chunks, all test windows, raw units
    bjump_seams.csv
    latency_samples.csv
results/{experiment_id}/                    # section 12
comparisons/{runA}__vs__{runB}.json         # from compare_runs.py
```

run\_id format: `lfm{L}_w{W*100}_s{train_seed}`, for example lfm8\_w050\_s42. Run ids must be unique. Overwriting is an error.

### 5.2 manifest.json

| Field | Type | Description |
| --- | --- | --- |
| schema\_version | str | 1.0 |
| run\_id, model\_key | str | model\_key like L8\_W05 |
| status | str | complete, degenerate or failed |
| valid | bool | false when degenerate or failed |
| model | object | lfm\_layers, expert\_width\_multiplier, gated, gate\_headwise, gate\_bias\_init, backbone\_id |
| revisions | object | HF commit SHAs for LFM2.5-VL, lerobot/smolvla\_base, SmolVLM2 (config and tokenizer), dataset |
| seeds | object | split\_seed, train\_seed, val\_seed, test\_seed, bootstrap\_seed |
| hashes | object | config\_hash, split\_hash, best\_ckpt\_sha256 |
| code | object | git\_sha, git\_dirty, lerobot\_commit, python, torch, transformers, cuda, driver |
| hardware | object | gpu\_names, world\_size, mixed\_precision, tf32 |
| training | object | steps, batch\_size\_per\_device, effective\_batch\_size, best\_step, best\_val\_loss, first\_val\_loss, wall\_time\_total\_s, wall\_time\_train\_only\_s, wall\_time\_val\_s, gpu\_hours (= wall\_time\_total\_s \* world\_size / 3600), steps\_per\_s, samples\_per\_s, peak\_train\_vram\_gb |
| timestamps | object | started\_utc, finished\_utc |

wall\_time\_total\_s must accumulate across crash resumes. Persist it in \_resume state.

### 5.3 config.yaml

The fully resolved configuration after CLI merge and defaults, not the CLI strings. Required top-level keys: policy (every field including use\_gated\_attn, gated\_attn\_headwise, gated\_attn\_bias\_init, expert\_width\_multiplier, chunk\_size, num\_steps), backbone (id, keep\_layers, vis\_tokens\_per\_cam, img\_size), data (repo\_id, revision, rename\_map, split\_manifest\_path), optim (lr, betas, eps, weight\_decay, grad\_clip\_norm, scheduler, warmup\_steps, decay\_steps, decay\_lr), train (steps, batch\_size, num\_workers, val\_every, save\_freq=0, ema=null), eval (val and test settings from sections 2 and 4), seeds, determinism.

### 5.4 metrics\_history.csv

| Column | Type | Note |
| --- | --- | --- |
| event | str | train or val |
| step | int |  |
| samples\_seen | int |  |
| wall\_time\_s | float | cumulative |
| loss\_total, loss\_action | float | train rows |
| lr, grad\_norm | float | train rows |
| clipped, nan\_inf\_count | float | train rows |
| step\_s, data\_s, samples\_per\_s | float | train rows |
| peak\_vram\_gb | float |  |
| val\_loss | float | val rows (selection criterion) |
| val\_loss\_sd\_over\_episodes | float | val rows |
| n\_val\_episodes | int | val rows |
| is\_best | bool | val rows |
| best\_val\_loss | float | val rows |

### 5.5 test\_report.json

| Field | Type | Description |
| --- | --- | --- |
| schema\_version, run\_id, model\_key | str |  |
| valid | bool | copied from manifest |
| checkpoint | object | step, val\_loss, sha256 |
| n\_test\_episodes, n\_test\_windows | int | expect 112 episodes (N=560) |
| metrics | object | for each of mae, nmae, mse, rmse, jerk\_norm, gt\_jerk, accel\_norm, smoothness\_ratio: mean, sd, median, p90, ci\_lo, ci\_hi (rmse and accel\_norm are informational) |
| bjump | object | mean, sd, n\_seams, mode |
| latency | object | n\_runs, warmup, p50\_ms, p95\_ms, p99\_ms, mean\_ms, std\_ms, preprocess\_mean\_ms, device |
| complexity | object | gflops\_infer\_chunk, gmacs\_infer\_chunk, gflops\_forward, gflops\_train\_step, params\_total, params\_trainable, params\_active, peak\_infer\_vram\_gb, counter, counter\_note |
| training | object | wall\_time\_total\_s, gpu\_hours, steps\_per\_s |
| protocol | object | windows\_per\_episode, draws\_per\_window, num\_steps, bootstrap\_B, test\_seed |
| warnings | list\[str\] |  |

### 5.6 test\_per\_episode.csv

One row per test episode, sorted by episode\_id. Columns: run\_id, model\_key, episode\_id, task, condition, n\_windows, mae, mse, rmse, nmae, jerk\_in\_chunk, accel\_in\_chunk, gt\_jerk\_in\_chunk, smoothness\_ratio, jerk\_norm, accel\_norm, checkpoint\_step. Every run writes the identical episode set. test\_per\_window.csv adds window\_idx, frame\_index and draw.

## 6. Reproducibility Checklist

| Item | Requirement |
| --- | --- |
| Seeds | Set split\_seed, train\_seed, val\_seed, test\_seed and bootstrap\_seed explicitly and store them in the manifest. Seed random, numpy, torch and cuda. Set PYTHONHASHSEED. Use worker\_init\_fn and a seeded DataLoader generator. See Q2 for the train\_seed discrepancy. |
| Determinism | torch.use\_deterministic\_algorithms(True), CUBLAS\_WORKSPACE\_CONFIG=:4096:8, cudnn.deterministic=True, cudnn.benchmark=False. Record the TF32 setting (the notebook used tf32=True; set it to False for the final benchmark). Record the dtype of each component: the notebook log shows the LFM backbone in bfloat16 on a T4, where bf16 is emulated and slow, while gate\_patch.py forces SmolVLM2 and the expert to fp32. If an op has no deterministic implementation, fall back to warn\_only=True and list the op in manifest warnings. |
| Determinism scope | Two-GPU DDP reductions and the bf16 LFM backbone are not bit-exact across hardware. Promise identical results only on identical hardware and software, and a tolerance of 1e-4 on val\_loss elsewhere. |
| config\_hash | sha256 of canonical JSON (sorted keys) of the resolved config, excluding output\_dir, timestamps and paths. |
| Code | git\_sha is required and the working tree must be clean unless allow\_dirty is set, in which case record git\_dirty=true. Pin the lerobot commit. The notebook installs from the repository head, which is not reproducible, and gate\_patch.py aborts if its string anchors do not match. The architecture is defined by runtime patches, so also record the sha256 of gate\_patch.py, lfm\_backbone\_adapter.py, lfm\_train\_entry.py and the three patched lerobot files. Pin transformers, because LFM2.5-VL needs 5.1 or newer. |
| Model and data revisions | Pin the Hugging Face revision SHA for LFM2.5-VL-450M, lerobot/smolvla\_base, SmolVLM2 (config and tokenizer) and the dataset. |
| Hashes | split\_hash, val-feature-cache key, best\_ckpt\_sha256. All are checked at load time. |
| Environment | Save pip freeze to env/pip\_freeze.txt and record GPU names and driver. |

## 7. Open Questions / Assumptions

| ID | Issue | Working assumption |
| --- | --- | --- |
| Q1 | Episode count. Resolved for the new dataset: 560 episodes (392 / 56 / 112). The old n=111 and excluded episode 555 came from an earlier split. | Use 392 / 56 / 112. Re-evaluate every model on this manifest, because the earlier benchmark numbers are not comparable. Check that meta/info.json (571) disagrees with the episode table (560) before building ids. |
| Q2 | train\_seed is 42 in the CSVs but the notebook passes --seed=1000. | Use 42 and record it. Confirm what the benchmark runs used. |
| Q3 | Does the combined dataset have session, operator or source ids? | Run G5. If cross-split overlap exists, decide whether to group-split on that field. This would change the existing test set. |
| Q4 | Existing validation used 16 episodes, 1 window, 1 draw (obs.). | Replaced by full validation. Old val numbers are not comparable. |
| Q5 | test\_window has 999 rows per model, but the draw column reaches 2. | Assume 9 windows and 1 draw. Confirm the intended number of draws. |
| Q6 | Definitions of nMAE and jerk normalization are not in the extract. | nMAE = per-dim MAE / train action std, averaged. jerk is the third finite difference of the predicted chunk. Confirm against the old scripts. |
| Q7 | Validation loss versus training loss. | They are the same objective on different data and randomness: val uses fixed noise and timesteps, no augmentation and eval mode. |
| Q8 | L8 W0.75 appears invalid (MAE 35, flat validation loss, rows copied from L8 W0.5). | Retrain it from scratch under this protocol. Do not reuse its old rows. |
| Q9 | Width W0.75 versus W0.5 and L8 versus L9 are confounded with trainable parameter count (114M vs 59M). | Report them separately. Any claim about width needs a matched-capacity control. |

## 8. Pipeline Test Plan

These tests verify the training and evaluation code before any real run. Run them in CI with a tiny synthetic dataset (about 40 episodes, 3 tasks).

| Area | Type | Cases | Target |
| --- | --- | --- | --- |
| Split | unit | disjoint ids; union equals all; exact sizes (560 -> 392/56/112); same seed gives the same hash; different seed gives a different hash; strata with fewer than 5 episodes are pooled; a repeated episode content hash across splits fails | 100% of split.py |
| Leakage | integration | train process never receives test\_ids; the dataset episodes argument matches the manifest | all guards G1–G7 |
| Validation loss | unit | hand-computed loss on a toy model; padded steps and dims 7–32 do not change the value; result is identical with world\_size 1 and 2; fixed noise makes two evaluations bit-equal | all branches |
| Feature cache | unit | cache hit gives the same loss as a cold run; changed split\_hash invalidates it | key logic |
| Checkpoint policy | unit | saves only on strict improvement; tie keeps the earlier step; NaN val\_loss is skipped; kill during write leaves the previous best intact; \_resume/ is removed at finish; EMA and periodic save raise at config validation | all branches |
| Degeneracy gate | unit | a flat loss curve (L8 W0.75 style) yields degenerate and valid=false | gate logic |
| Test protocol | integration | refuses a missing hash, a wrong split\_hash, an existing test.lock, and a non-best checkpoint; emits every field in 5.5 and 5.6; sorted episode ids | all preconditions |
| Metrics | unit | MAE, nMAE, MSE, jerk and bjump match hand computation on synthetic chunks; a constant chunk has zero jerk; matching seam values give bjump 0 | exact |
| Statistics | unit | bootstrap CI contains the true mean on simulated data; Wilcoxon matches scipy on a fixture; Holm adjustment correct; unpaired test\_ids are refused | exact |
| Latency | smoke | warmup excluded; p99 is at least p95; synchronization is called | basic |
| Reproducibility | end-to-end | two 20-step runs with the same config give the same val\_loss on the same hardware; changing any config field changes config\_hash | 1 run pair |
| Gate and adapter safety | regression | keep the existing tests: gate=1 reproduces the baseline, uncached equals cached path, finite gradients on all 16 gates and the conditioner | existing suite |

## 9. Decision Records

| ADR | Decision | Alternatives considered | Consequence |
| --- | --- | --- | --- |
| ADR-1 | Select checkpoints on fixed-noise flow-matching velocity MSE over the full val split. | A: action MAE from sampling (costly, noisy). B: random-noise flow loss (noisy ranking). C: both with a weighted score (arbitrary weights). | Cheap, deterministic, monotone in the observed logs. It is not the test metric, so a small selection mismatch is possible. Revisit if best\_valid and test MAE disagree across seeds. |
| ADR-2 | Keep a single best\_valid checkpoint. | A: also keep the last step (violates R3). B: EMA or averaging (forbidden by constraint). | Simplest audit trail. Training budget is fixed, so no early stopping. |
| ADR-3 | Stratified episode-level split with one shared manifest. | A: random episode split (task imbalance). B: frame-level split (leaks). C: leave-session-out (needs metadata, see Q3). | Paired tests across models become valid. The test set cannot be changed after the first benchmark without invalidating old results. |
| ADR-4 | Precompute and cache frozen-backbone features for validation. | A: run the backbone at every evaluation (about 5–10 minutes per evaluation on 2×T4). | Validation every 250 steps is cheap. Adds a cache-invalidation requirement. |

## 10. Fair-Comparison Protocol: SmollerVLA vs Original SmolVLA

The comparison isolates the architecture. In every arm the pretrained VLM is frozen, and every other parameter is trained from scratch on the same data, budget and evaluation.

### 10.1 Arms

| Arm | Frozen pretrained VLM | Trained from scratch | Gates | Expert width | run\_id prefix |
| --- | --- | --- | --- | --- | --- |
| A. SmolVLA (original) | SmolVLM2-500M-Video-Instruct, 16 layers, loaded with load\_vlm\_weights=true | action expert, state/action/time projections | no | 0.75 (the original width) | smolvla\_w075 |
| B. SmollerVLA | LFM2.5-VL-450M, first 8 layers | action expert, projections, 16 G1 gates, AdaptiveConditionalGatedKV | yes | 0.5 and 0.75 | lfm8\_w050, lfm8\_w075 |
| C. SmolVLA + gates (recommended control) | as A | as A plus 16 G1 gates | yes | 0.75 | smolvla\_gate\_w075 |

Arm C separates the effect of gating from the effect of the backbone swap. Add L9 arms by the same rule if needed.

### 10.2 Hard rules

| ID | Rule |
| --- | --- |
| R8 | No arm may load lerobot/smolvla\_base. The notebook's `--policy.path=lerobot/smolvla_base` loads the pretrained expert and projections. Use `--policy.type=smolvla` so that only the VLM comes from a checkpoint. Disable the shape-tolerant state-dict loader for training. |
| R9 | Step-0 init audit, written to init\_audit.json. For every parameter record: group, init source (pretrained\_frozen or scratch\_trained), requires\_grad, numel. Assert that every pretrained\_frozen tensor has requires\_grad=False and a sha256 equal to the Hugging Face VLM tensor. Assert that every scratch\_trained tensor has requires\_grad=True and a hash different from the smolvla\_base tensor of the same name and shape. Abort on any other state. |

The old notebook was a warm start. At W0.5 it skipped 136 shape-mismatched tensors, so those started fresh. At W0.75 the shapes match the checkpoint, so the pretrained expert was loaded. Any earlier W0.5 vs W0.75 comparison is therefore confounded by initialization.

### 10.3 Controlled variables

| Item | Rule |
| --- | --- |
| Data | Same split manifest and same dataset revision. |
| Optimization | Identical steps, per-device and effective batch size, optimizer, learning-rate schedule, gradient clipping and seeds. |
| Training budget | Fix it from an arm A pilot, using validation loss only. Run A until validation loss improves by less than 1% over 2000 steps, then give every arm that step count. The notebook's 4000 steps x 128 = 512k samples is about 0.78 epoch of 656,950 frames, and the old validation loss was still falling at step 3000, so a from-scratch expert will likely underfit. |
| Seeds | At least 3 train seeds per arm (42, 43, 44). Report mean and SD across seeds. Run the paired episode tests within each seed, then pool episode means across seeds. Apply Holm correction. |
| Precision | gate\_patch.py forces SmolVLM2 and the expert to fp32, while the LFM backbone runs in bf16 or fp16 autocast. Record dtype per component in test\_report.json. Compare latency and VRAM across arms only when both ran on the same GPU in the same dtype. |
| Image input | Arm A letterboxes frames to 512x512 for SmolVLM2. Arm B resizes that to 256x256 and pools to 48 visual tokens per camera. This is part of the architecture under test, so record it and do not equalize it. |
| Gate bias init | 4.0 gave a near-identity start from a pretrained expert. From scratch it only biases early training. Keep 4.0 for continuity and add bias 0 as a sensitivity run. |
| Reported cost | Trainable parameters, scratch-trained parameters, inference GFLOPs and peak inference VRAM per arm. |

### 10.4 What the comparison can claim

A vs B differs in backbone depth (16 vs 8 layers), vision encoder, visual token budget, the conditioner (a trained key/value bridge instead of the VLM's native prefix) and the gates. Arm C removes the gate difference only. No arm isolates the conditioner from the LFM backbone, so claims must name the whole SmollerVLA stack as the unit.

## 11. Compatibility Audit: This Guide vs the SmollerVLA Code

The guide's structure fits SmollerVLA, but six items needed changes and two still need a decision. Findings come from smoller\_vla.ipynb (gate\_patch.py, lfm\_backbone\_adapter.py, lfm\_train\_entry.py).

| # | Guide item | What the SmollerVLA code does | Verdict | Change |
| --- | --- | --- | --- | --- |
| 1 | Validation loss = flow-matching velocity MSE (2.2) | The patched forward uses the same x\_t = t\*noise + (1-t)\*actions and target noise - actions, and calls denoise\_step. Training samples t from SmolVLA's sample\_time (a Beta-shaped distribution). | Compatible | Validation uses a uniform grid of 10 timesteps, so it weights t differently from training. Selection is unaffected in practice, but record the difference. |
| 2 | Frozen-backbone feature cache (2.3) | The LFM backbone is frozen, in eval mode, with no augmentation, so its features are deterministic. | Compatible with changes | Cache key extended (2.3). The SmolVLA baseline caches SmolVLM2 prefix key/values instead. |
| 3 | Checkpoint contents (3.4) | LFMAdapter.state\_dict returns the conditioner only, and SmolVLM2 is replaced by parameter-free stubs. | Compatible | Backbone excluded by design. SmolVLM2 tokenizer revision must be pinned (3.4). |
| 4 | Loading best\_valid (4.2 S2) | The patched load\_state\_dict silently skips shape-mismatched tensors and ignores SmolVLM2 keys. | Not compatible | Strict load added (S2). |
| 5 | Version pinning (6) | The architecture is gate\_patch.py (string-anchor edits to installed lerobot) plus runtime class patches. lerobot is installed from the repo head. | Gap | Hash the patch files and patched lerobot files, and pin the lerobot commit (6). |
| 6 | Latency (4.2 S9) | LFM image preprocessing runs on CPU inside the model call. causal\_conv1d falls back to a reference path. The backbone ran bf16 on a T4. | Changed | S9 and the determinism row updated. |
| 7 | Normalization | lerobot normalizes through its pre/post-processor pipeline, with statistics taken from the dataset. | Out of scope | No custom normalization step, per request (R5). Nothing is saved or hashed for it. |
| 8 | Initialization (new, 10) | The notebook warm-starts from lerobot/smolvla\_base. | Not compatible | R8, R9 and the arms in section 10. |
| 9 | Training budget | 4000 steps, about 0.78 epoch. | Inadequate for scratch training | Pilot-based budget (10.3). |
| 10 | Gate bias 4.0 | Rationale was a near-identity start from a pretrained expert. | Moot from scratch | Keep for continuity, add a bias-0 run (10.3). |
| 11 | Dataset | 560 episodes, 656,950 frames, 3 cameras at 640x480 AV1. meta/info.json is stale. | Updated | Split, preconditions and Q1 revised. Run the AV1 decode smoke test. |
| 12 | Complexity (4.2 S10) | FlopCounterMode counts matmul, bmm, conv and sdpa only. | Compatible | Elementwise operations and the CPU image processor are not counted. Say so in the report. |

Open decisions: the training budget in item 9 (it needs the arm A pilot), and the dtype policy in item 6 (match precision across arms, or report the asymmetry).
