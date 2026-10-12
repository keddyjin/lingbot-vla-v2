# H20 selective checkpointing for LingBot-VLA-v2 SFT

Four H20 GPUs can retain the original micro batch of 32, accumulation of 1, and a global batch of 128 while reducing unnecessary activation recomputation. The measured combination uses `LINGBOT_FLEX_BWD_TILE=32x64`, FSDP forward resharding, and selective decoder checkpointing. Teacher models, training losses, trainable modules, optimizer configuration, precision, and data are retained.

## Measured short-run results

All rows use Distributed Muon, the same four `NV_GPU` UUIDs, real RoboTwin data, seed 42, fresh pretrained weights, and 31 updates. The first 10 updates are warmup; updates 11–30 form the 20-update measurement window; update 31 is excluded as terminal. Each end-to-end update includes data fetch, teacher computation, forward/backward, clipping, optimizer and logging, and takes the slowest of all four ranks. Both checkpoint saving and resume are disabled consistently. Measurements are short-run throughput evidence, not final accuracy or long-run convergence validation.

| Run / policy | Mean s/update | Median | p95 | CV | Samples/s | Gain vs fixed original-tile baseline | Allocator peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dist-perf-r2`: original tiles; all GC; reshard | 5.386928178 | 5.384003109 | 5.410700398 | 0.205087% | 23.761223 | baseline | 48.41 / 62.65 |
| `flex32x64-perf-r1`: 32×64 BWD tiles; all GC; reshard | 5.060605264 | 5.057031904 | 5.085876178 | 0.210013% | 25.293417 | 6.448298% | 48.41 / 62.66 |
| `gcaction-perf-r1`: `action-off`; 32×64; reshard | 4.974957772 | 4.974091318 | 4.982381494 | 0.082219% | 25.728862 | 8.280882% | 54.20 / 71.38 |
| `gcaction-textalternate-perf-r1`: `action-off-text-alternate`; 32×64; reshard | 4.645029865 | 4.643597638 | 4.651979270 | 0.140581% | 27.556335 | 15.971874% | 65.42 / 81.03 |
| `gcaction-textoff-perf-r1`: `action-off-text-off`; 32×64; reshard | 4.314818983 | 4.315138167 | 4.321586317 | 0.126682% | 29.665207 | 24.847142% | 76.65 / 91.32 |

The selective policies improve throughput over the tile-only run by 1.721572%, 8.946668%, and 17.284301%, respectively. These local gains are not added to the tile improvement; the last column uses the single fixed baseline. All three selective runs exited successfully with 31 complete four-rank updates and 31 finite loss/GradNorm/LR logs; no obvious short-run divergence was detected. Final logged loss/GradNorm were 0.2041/0.3582, 0.2029/0.3571, and 0.2029/0.3452. This does not verify every gradient tensor or robot success rates.

Peak allocator values are the largest process-lifetime maxima observed across the four ranks, without reset. They include initialization and warmup, are not total GPU memory, and allocated/reserved must not be added. The most aggressive setting approaches 91.32 GiB reserved on a 95.12 GiB H20; it has materially less headroom for other shapes or longer runs. Vision checkpointing remains enabled in every selective setting. Smaller policies provide measured memory/speed tradeoffs rather than assuming the most aggressive policy is universally suitable.

## Default behavior and strict activation

`LINGBOT_GC_POLICY` defaults to `official`. That branch returns before changing any module, so the original GC configuration and other architectures retain their behavior. Selective activation requires `enable_gradient_checkpointing=true`; the helper runs after the original HF checkpoint setup and before TP/EP/FSDP wrapping.

The helper validates exact model paths, 36 Qwen3 text layers, 36 Qwen2 action layers, nonempty Qwen3 vision blocks, expected concrete class names, each layer's existing `gradient_checkpointing=True`, and a callable initialized HF checkpoint function. It validates every group before changing any flag; unsupported policy, architecture, or partially initialized HF setup raises an error rather than applying a partial policy. It logs all enabled/disabled indices. Actual runs contain 24 vision blocks.

| Policy | Action checkpointing | Text checkpointing | Vision checkpointing |
|---|---|---|---|
| `official` | original behavior | original behavior | original behavior |
| `action-off` | all 36 off | all 36 on | all 24 on |
| `action-off-text-alternate` | all 36 off | even indices 0,2,…,34 on; odd indices off | all 24 on |
| `action-off-text-off` | all 36 off | all 36 off | all 24 on |

The implementation also exposes `text-alternate` and `text-off` for separate ablations. Those standalone policies have no measured SFT performance result in this table.

## Gradient, RNG, and numerical boundaries

Disabled layers call the original module forward with autograd enabled and save activations instead of recomputing them. There is no new `no_grad`, detach, parameter freeze, skipped loss, or removed optimizer update. Enabled layers retain the original HF non-reentrant function; its RNG handling, context, and checkpoint parameters are not rewritten. Vision GC and teacher eval/no-grad behavior are unchanged. Current decoder/joint-attention dropout is zero, and flow-matching noise/time is generated before these checkpointed decoder layers.

Saving versus recomputing changes compiler scheduling/fusion and floating-point accumulation. The existing tile change also alters reduction rounding; near-tie MoE routing and repeated updates can amplify small differences. The finite 31-step loss/grad trajectories are therefore stability evidence, not bitwise parity or final-accuracy proof. Any nonfinite loss/gradient or sharp growth requires rejecting the setting rather than attributing it to harmless rounding.

Action MoE forward updates monitoring buffers. Removing action recomputation can change absolute `tokens_per_expert`/sigmoid accumulation and update timing; these counters are not actual sample counts. Current `bias_update_speed=0` keeps the router-bias adjustment zero, and sequence-wise/router-z losses still use original forward router logits. Nonzero bias-update configurations need a new correctness review; the present result must not be generalized to them.

The LR cosine horizon is shortened from 50,000 to 31 in every paired performance run. This controls short-run comparisons but does not validate the long-training schedule. Low CV does not independently prove compiler settling. Original metadata flags for compile settling and loaded-training timing overhead remain unmodified; the idle helper overhead microbenchmark is not an on/off SFT measurement.

## Reproducibility and provenance

Tile-only trials executed functional HEAD `05f9cd1da80e245617e5369259075f5440e3c13e` with the recorded dirty FlexAttention patch; the measured tile implementation was then committed as `0bf4341c8f7619a999a004c5f391d7ea9c2ca238`, author `codex`. The three selective trials executed that parent HEAD with an uncommitted change to `lingbotvla/distributed/torch_parallelize.py`; their exact `dirty-tracked.patch` SHA256 is `c3e3521d580f157577afd03ff09e948f067302884a100e26dd3136cb10b52ef1`. Do not retroactively replace those executed versions with a later selective-checkpoint commit.

The reviewed candidate patch SHA256 is `81e04cb5d67f9ce7def15f14f60708afa292fc01458a916557b5bdd96f5433ca`. Patch-file hashes differ because Git diff serialization/headers differ; each run's HEAD plus retained tracked diff, untracked snapshot, effective/runtime configuration, Docker command, policy-summary log, rank timing JSONL, and raw training log is the execution authority.

Original source: `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`. Environment: Torch 2.8/cu128, Triton 3.4, Transformers 4.57.3. The official 50-task real-data v21 archive is migrated to v3 with scalar values/types and encoded AV1 payloads validated unchanged. Profiler artifacts and analysis are separate from performance timing. No final clean/random robot-success result is claimed.
