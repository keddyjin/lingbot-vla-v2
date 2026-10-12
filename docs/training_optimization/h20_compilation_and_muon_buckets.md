# H20 compilation startup and Distributed Muon bucket tuning

The measured combination retains micro batch 32, accumulation 1, global batch 128, four H20 GPUs selected by `NV_GPU`, all teachers/losses, and the existing precision policy. It uses the already measured Flex BWD32×64 and action/text checkpointing off with vision checkpointing and FSDP forward resharding retained. Isolating a changing Python alignment counter reduces repeated early compilation in the observed run; tuning the existing Distributed Muon bucket field gives a repeatable steady-state improvement.

This document describes completed trials and their exact execution provenance. The counter change and bucket recommendation have not yet been committed as a performance change; the subsequent DeepStack v1 candidate failed CUDA compiled parity and was rejected. Its local precision-v2 repair subsequently passed fixed four-GPU compiled parity and two new SFT trials; the small repeated increment and pending loaded measurement validation are recorded separately. Existing tile and selective checkpoint performance commits remain separate.

## Default-off counter helper and observed startup

`LINGBOT_COMPILE_STEP_COUNTER_HELPER` defaults to zero and is read once at model construction. Off retains the original `self.steps += 1` at the original alignment-forward position. On calls a `torch.compiler.disable` helper at that same position; the helper only increments the Python counter, without Tensor operations, RNG, CUDA work, communication, or changed control of losses. Alignment-disabled behavior and exception ordering remain original. This is not deleting the counter or moving it to optimizer-step boundaries.

On the matched 31-update runs, the clean source had `self.steps` guards and the recompile limit warning during early updates. The helper trial had no such logged warnings; steps 3–8 were 4.3173/4.3013/4.3056/4.3116/4.2991/4.3054 s. Sum of all 31 slowest-rank update times was 186.824878 s versus 203.571363 s; training-command elapsed was 247 versus 262 s. These are observations of an improved early compilation path. Actual shared-cache temperature/hits are unknown; elapsed differences must not all be attributed to the patch or described as a guaranteed cold-start saving.

The helper alone had steady mean 4.303756378 s versus clean 4.309625177 s, a 0.136365% throughput difference close to run noise. It was not committed on that small steady result alone. It can introduce a graph break and change compilation segmentation; warning disappearance does not prove every region is compiled, nor replace a loaded-training timing-overhead check.

## Bucket sweep and selected value

The sole sweep field is `train.dist_muon_layers_per_bucket`; the official default remains 2. It groups adjacent `layers`/`blocks` with the same module prefix. Bucket changes packed communication, two-slot prefetch scheduling and the temporary owner assignment for whole-matrix Muon work. It does not merge different parameters into one orthogonalization matrix, change per-head layouts, alter persistent FSDP/momentum shards, or change LR, momentum, Nesterov, NS5, adjustment, precision or parameter selection.

All following runs retain the same counter-enabled source, tile/GC policy and training controls. Each completes 31 updates: 10 warmup, 20 measured updates 11–30, and an excluded terminal update. End-to-end update time includes data/teacher/forward/backward/clipping/optimizer/logging and uses the slowest of all four ranks.

| Layers per bucket / run | Actual buckets | Mean s/update | p95 | CV | Samples/s | Gain over same-source bucket2 | Allocator peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 / `compilecounter-perf-r1` | 53 | 4.303756378 | 4.309818121 | 0.108315% | 29.741460 | baseline | 76.30 / 90.29 |
| 1 / `muon-bucket1-perf-r1` | 101 | 4.313118973 | 4.321961290 | 0.170825% | 29.676900 | −0.217072% | 76.14 / 92.39 |
| 4 / `muon-bucket4-perf-r1` | 29 | 4.239127439 | 4.245059758 | 0.097242% | 30.194893 | +1.524581% | 76.85 / 91.21 |
| 8 / `muon-bucket8-perf-r1` | 18 | 4.240997770 | 4.246185169 | 0.131697% | 30.181577 | +1.479808% | 78.11 / 92.48 |
| 4 repeat / `muon-bucket4-perf-r2` | 29 | 4.238154466 | 4.243010405 | 0.098571% | 30.201825 | +1.547889% | 76.85 / 91.21 |

The two bucket4 runs pool 40 measured updates: mean **4.238640953 s**, median 4.238996068 s, nearest-rank p95 4.244840466 s, CV 0.098579%, and 30.198359 samples/s. Gain is **1.536234%** over the same-counter bucket2 mean and **27.090929%** over the fixed original Distributed Muon anchor of 5.386928177947993 s. The overall gain includes earlier tile/checkpoint changes and cannot all be assigned to the bucket field or added from rounded local percentages. It supports a repeated observation beyond the 25% stage; 30% has not been reached.

Bucket4 is selected for further model experiments: it matches/slightly exceeds bucket8's measured speed while using less temporary allocator memory and more modest buckets. The small 4-versus-8 timing difference is not asserted statistically significant. Bucket8 is a successful measured alternative, not an additional improvement to add. Bucket1 did not improve throughput and is retained as a rejected configuration result rather than a performance commit.

## Quality and capacity boundaries

All five listed runs have 31 complete four-rank updates, actual 32×1×4 samples, successful exits, and 31 finite loss/GradNorm/LR logs without obvious short-run divergence. Each reports 935 Muon matrices, 737 AdamW parameters and 216 per-head parameters. Bucket count alone changes to 101/53/29/18. The new bucket runs' printed loss/grad/LR values match the counter baseline at the log's precision; this is not bitwise tensor/parameter parity or final accuracy/robot-success validation. Existing teacher models and loss coefficients are retained.

The original FP32/BF16/autocast/TF32 and attention mask/GQA/padding policies are unchanged; this is not a precision tradeoff. Muon's existing internal BF16 Newton–Schulz implementation remains original. Changing temporary owner ranks can change device scheduling, so exact numerical equivalence must not be assumed solely from the bucket label.

Allocator peaks are process-lifetime maxima, include startup/warmup, and have not been reset. They are not total GPU usage, and allocated/reserved are not added. Four H20 GPUs have about 95.12 GiB each; bucket8's 92.48 GiB reserved leaves less headroom for different shapes and longer training. Successful 31-step runs do not guarantee capacity for every workload.

Every trial uses the same short 31-step cosine LR horizon, fresh seed42/weights/data ordering, and no resume/checkpoint saves. This controls the comparison but does not validate a 50,000-step convergence schedule. Save steps/epochs are zero, HF exports false, resume false, and YAML omits `load_checkpoint_path` to use the true Optional[str] default None; explicit YAML null must not be used because upstream parsing can stringify it.

## Provenance and measurement limits

Executed parent HEAD: `73ba46c5e41ce2e089f456377cb0b6201118aac7`. The same uncommitted model change is retained in each trial's `dirty-tracked.patch`, SHA256 `3960f5ec1c11189ba769d8ee779b0a60011763ee65b7d93cbf6fd99d061a76fa`. Reviewed counter patch SHA256: `f59673ea43d246fd98a6d4a47ccdbc3aaeb788667b0edbc7671ea5941653fe28`; candidate model LF SHA256: `add4e6170ba7ec9d81cb344dea37262ec7dd60652b4ac2afc01e5b4af36e8cae`. Original model LF SHA256: `9b3c9a2ad6823f9d58da51b732477785a1e4db0d29db80e030e094c7b605e40c`. Patch serialization hashes differ; HEAD plus each retained run diff/config/command is execution authority.

Runtime controls: counter helper=1, Flex tile=32×64, GC policy=`action-off-text-off`, forward reshard=true, MoE output-size candidate=0, DeepStack indices candidate=0. Each run retains its real git/dirty snapshot, effective/runtime YAML, Docker/GPU UUID metadata, raw training log, rank JSONL and exit status. Do not relabel executed trials with a future commit.

Observed warnings and low CV are not full compiler coverage evidence. The idle CUDA/NFS helper benchmark is not loaded SFT on/off validation. Original `warmup_compile_settled`, `timing_overhead_checked` and quality readiness flags remain unmodified; independent analysis suggestions do not automatically turn them true. Measurement JSONL writing and allocator reads occur after the update endpoint; async drain and total command elapsed are separately recorded.

Nsight partial capture limits, full Torch profile provenance, and any NCU isolated operator capture remain separate from SFT throughput. The NCU operator experiment is not a real training command and is excluded from the SFT time ledger. The DeepStack v2 implementation has separate compiled numerical validation and repeated SFT evidence; rejected candidate results remain historical and are not part of the accepted model implementation.

## DeepStack v1 attempted increment: rejected by CUDA compilation parity

Two real SFT attempts on the counter-enabled bucket4 configuration completed all 31 updates with finite logs:

| Run | Mean s/update | p95 | CV | Training command elapsed |
|---|---:|---:|---:|---:|
| deepstack-perf-r1 | 4.227688087 | 4.235374964 | 0.121845% | 235 s |
| deepstack-perf-r2 | 4.226345365 | 4.232477996 | 0.132029% | 235 s |

The two 20-update windows pool mean 4.227016726 s, median 4.227828969 s, p95 4.234333343 s, CV 0.128027%, and 30.281404 samples/s. The apparent increment over the two bucket4 windows is only 0.274998%; aggregate over the original anchor is 27.440427%. These are retained attempt measurements, **not accepted performance results**, because the required CUDA compiled parity subsequently failed. The supported repeated 25% observation remains the independent bucket4 result of 27.090929%, without this rejected increment.

Executed combined model LF SHA: a5bc92eb8bb5e45feb7a7e9207067f0ef9b4ac5041348ca66fd63dfcff642ec5. Both attempts used parent HEAD73ba46c5e41ce2e089f456377cb0b6201118aac7 plus actual Git diff SHA1cd0c6daba9b29bea6d1be3572c7f7d6c3381634f0c2d5945e9ff8053ec4f4d7. Reviewed incremental DeepStack-on-counter patch SHA: ccf169c28942399aa0ba2e35ed87ebcbe3942ed7a212f7f3e6d260ee2af7d07e. Counter helper=1, DeepStack indices=1, bucket4, tile32×64, action-off-text-off, reshard, MoE-output-size=0, all batch/precision/teachers/losses unchanged as configured. Actual fused numerical semantics were not preserved in the failed mixed-dtype case.

The candidate derives unique row-major indices from the actual boolean visual mask once per forward and reuses them for three DeepStack layers. It uses index_select.clone plus the original dtype cast/add and an in-place index_copy, retains the hidden object and pre-existing aliases, falls back to original HF for noncontiguous inputs, and never caches indices across forwards. CPU validation passed 152 tensor cases plus 8 guards, including output/loss/hidden and all-feature gradients, aliases, mixed dtypes, missing cameras and 1/3 layers. It used no CUDA context and did not exercise torch.compile; that CPU result is insufficient for a compiled CUDA change.

The isolated four-H20 CUDA/compile check then failed on every rank for BF16 hidden with FP32 visual features. Eager HF and compiled HF matched exactly; compiled candidate output relative L2 was about 0.00324, max absolute error 0.015625, hidden gradient relative L2 about 0.00356, FP32 feature gradients about 0.00383, and loss relative error up to about 1.45e-5. Other representative cases passed. Original fixed thresholds were not relaxed.

Generated candidate code directly loaded FP32 features and added them to upcast hidden values before storing BF16, without the original pre-add feature BF16 rounding. PyTorch2.8 Inductor's default fusion can eliminate intermediate low-precision cast boundaries; see [the version-pinned official configuration](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_inductor/config.py#L602). A minimal revision was then prepared to materialize the original dtype-changing .to through a local compiler-disabled helper, preserving its autograd boundary. Same-dtype index operations remain available; no detach/no_grad, global precision override, RNG change, or relaxed threshold is allowed. The revised source must pass CPU and fixed CUDA/compile parity and be remeasured in SFT; v1 timing must not be reused as v2 performance evidence.

All v1 source/worker/cache/reports are retained. Isolated operator validation time is not SFT time. No performance commit should include rejected DeepStack v1; final implementation selection remains pending.

## Precision-v2 index repair: separate repeated evidence

The old v1 failure remains rejected. Model LF source ec9e4280015be612c3bd26e71f502acdb95aaef1d74423c01508c794b8bb1f9e materializes only dtype-changing feature conversion through a local compiler-disabled helper; global precision, RNG and autograd remain original. The actual four-H20 compiled parity-r2 passed all336 output/alias/loss/gradient comparisons with exactly zero error and unchanged thresholds. It used actual helper AST, fixed installed HF source and explicit synthetic operator cases, not real SFT masks or a throughput workload.

Two new31-update SFT trials had means4.224109538146877 and4.225537992751924 seconds. Pooled40 measured updates have mean4.2248237654494005, median4.224301301495871, p954.231323565007187, CV0.094868%, throughput30.297122samples/s. Repeated observed increment over bucket4 controls is0.327048%; aggregate over the fixed original anchor is27.506577%. All31 logs per trial are finite and nine printed loss/grad/LR scalars match the bucket4 control; this does not claim final accuracy or full-tensor training bitwise equivalence. Allocator lifetime peaks76.847224/91.214844GiB are unchanged across the two new trials.

Both executed HEAD73ba plus retained actual dirty diff053a91ed7dd0195179d9fac1f357d5e16a4c9ee9a83754aae8c96f69bfc6c658, with DeepStack1/counter1/bucket4/tile32x64/action-off-text-off/reshard, teacher-batching0 and MoE0. R1/R2 command elapsed235/233seconds. Detailed index implementation, cast/alias boundary, worker/source hashes and rejected v1 history are in [h20_deepstack_indices.md](h20_deepstack_indices.md). Proposed model commit excludes inactive teacher and no-gain Flex warp resource-option candidates. Loaded training instrumentation on/off validation is still running; no original readiness flags are automatically marked true. The0.327% increment is small, not a new5%stage or a statistical-significance claim.

First loaded SFT instrumentation pair completed31 updates per arm with four valid rank observers and finite logs: on max20-update block/20=4.241966163601319 s, off=4.247257724500377 s. The small apparent on-arm advantage is near noise, not negative overhead or a hard upper bound. Off has no ordinary per-update helper timing; its completed real updates and common GPU-drained block are retained without fabricated per-update durations. A reversed off→on pair remains pending. Runtime flags in these actual SFT observer snapshots are matmul precision high and CUDA/cuDNN TF32 true. Original raw readiness flags remain unmodified.
