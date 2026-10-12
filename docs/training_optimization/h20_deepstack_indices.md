# H20 DeepStack indices with preserved compiled cast semantics

The default-off DeepStack index path derives unique row-major indices from the actual boolean visual mask once per Qwen-VL forward and reuses them across three DeepStack layers. On contiguous hidden states it replaces repeated boolean gather/scatter with `index_select(...).clone()`, the original feature dtype/device conversion and addition, then in-place `index_copy_`. It returns the same hidden object, preserves pre-existing aliases and autograd, uses the original HF path for noncontiguous inputs, and never caches indices across forwards. Empty DeepStack does not prepare indices. It changes neither RNG nor teacher/loss definitions.

`LINGBOT_DEEPSTACK_INDICES` defaults to zero. Off retains the original installed HF behavior. The combined model also contains the independently reviewed default-off Python counter helper; its startup observations and Distributed Muon bucket4 recommendation are documented in `h20_compilation_and_muon_buckets.md`.

## Rejected first implementation and local cast repair

The original index candidate passed 152 CPU tensor cases and 8 guards, including output, loss, hidden/feature gradients, aliases, mixed dtypes, missing cameras and noncontiguous fallback. Its two SFT attempts were finite. However, four-H20 compiled parity failed for BF16 hidden states with FP32 visual features: compiled HF matched eager HF, while compiled candidate output relative L2 was approximately 0.00324, max absolute error 0.015625, and gradient relative L2 approximately 0.00356/0.00383. That implementation and its apparent 0.275% speed increment were rejected and never committed.

Generated candidate code eliminated the intermediate feature-to-BF16 rounding before addition. Precision-v2 materializes the original `.to(hidden_states.device, hidden_states.dtype)` in `_cast_deepstack_features_materialized`, decorated with `torch.compiler.disable`, only when feature and hidden dtypes differ. Same-dtype index operations retain the original conversion expression. This local boundary keeps the original autograd cast behavior; it does not detach, disable gradients, change global precision/Inductor cast-emulation settings, or loosen validation thresholds.

The repair has 13 AST/fixture checks, including restoration of the complete previous model AST when the repair is stripped. The previous CPU160 result is retained as evidence of eager index/alias mathematics, not relabeled as a new compiled CUDA result.

## Actual four-H20 compiled parity

`deepstack-gpu-parity-r2` exited zero and passed all four ranks and four representative cases per rank. The worker extracted actual candidate helper ASTs and the fixed installed HF reference. It used batch32, prefix286, hidden2048 and three consecutive layers, with all cameras, missing-camera masks, an empty-visual sample, noncontiguous fallback, BF16/FP32 and both mixed-dtype directions. It compared eager HF, compiled HF and compiled candidate (`fullgraph=False`) output, pre-existing alias, nonlinear loss, hidden gradient and all three feature gradients.

All **336 tensor comparisons were finite and exactly equal**: maximum absolute error, relative L2 and normalized Linf were zero. All pipelines preserved hidden identity and alias. Thresholds were unchanged from the failed run: output/alias/loss 1e-6; FP32 gradient relative L2/Linf 1e-5/1e-4; BF16 gradient 2e-3/8e-3. `inductor_emulate_precision_casts` remained false. The isolated process recorded source matmul precision `high`, CUDA/cuDNN TF32 true, with no global precision override. These flags belong to the operator-validation process and are not automatically a SFT runtime snapshot.

This is explicitly synthetic operator/compile validation, not a captured training mask or a student/teacher training test. It loads no model weights, creates no process group, uses only the four designated H20 UUIDs, and contributes no SFT performance/time claim.

## Actual repeated SFT measurement

Both precision-v2 SFT trials retained four designated H20 GPUs, micro32, accumulation1, global128, fresh seed42, the same real RoboTwin data and weights, all teachers/losses and the original precision configuration. The comparison uses tile32×64, vision checkpointing retained, action/text checkpointing off, FSDP forward resharding, counter helper enabled, Distributed Muon bucket4 and MoE candidate disabled. Teacher batching was disabled in both index trials. Each completes 31 updates, with 10 warmup, 20 measured updates11–30 and an excluded terminal update; no checkpoints or resume.

| Run | Mean seconds/update | Median | p95 (nearest rank) | CV | Samples/s | Command seconds |
|---|---:|---:|---:|---:|---:|---:|
| bucket4 control, two windows | 4.238640953 | 4.238996068 | 4.244840466 | 0.098579% | 30.198359 | 234 + 235 |
| deepstack-v2-perf-r1 | 4.224109538 | 4.223346701 | 4.230351300 | 0.087177% | 30.302244 | 235 |
| deepstack-v2-perf-r2 | 4.225537993 | 4.225053632 | 4.231323565 | 0.099135% | 30.292001 | 233 |
| v2 pooled 40 measured updates | **4.224823765** | 4.224301301 | 4.231323565 | 0.094868% | 30.297122 | — |

The repeated observed incremental throughput gain over the two bucket4 control windows is **0.327048%**; aggregate gain over the fixed original Distributed Muon anchor5.386928177947993 seconds is **27.506577%**. This is a small repeated increment, not a new5% stage or a claim of statistical significance/long-run robustness. The independently repeated bucket4 result already supports the25% short-run observation;30% has not been reached.

Both trials have31 complete four-rank updates with actual32/1/128 and31 complete finite loss/GradNorm/LR logs without obvious short-run divergence. Printed nine loss/grad/LR scalar values match bucket4r2 at log precision. This is not full tensor bitwise training parity or final RoboTwin success-rate validation. Allocator lifetime peaks were76.847224/91.214844GiB allocated/reserved in both trials; these are not total device occupancy and do not guarantee all-input/long-run capacity.

## Execution provenance and limits

Both SFT trials executed parent HEAD `73ba46c5e41ce2e089f456377cb0b6201118aac7` plus retained actual Git diff SHA256 `053a91ed7dd0195179d9fac1f357d5e16a4c9ee9a83754aae8c96f69bfc6c658`. Model LF source SHA256 is `ec9e4280015be612c3bd26e71f502acdb95aaef1d74423c01508c794b8bb1f9e`; the A5→v2 repair patch SHA256 is `7a19a673961699916ee68556d4924a3428ad629b6f08d96a6e13f605a2c3307a`. Execution snapshots also contained default-off teacher and Flex resource-option candidates, inactive in these trials. They are not part of this index result or proposed model-only commit. HEAD plus each run's raw diff/config/command is authoritative; a future commit must not be backfilled into executed records.

Compiled parity pins the installed HF whole-file SHA `685b10cbc48556343df4e159199c5884d02db9ad227640f0f4fbe75ddf0b2a99` and actual worker SHA `7edd8220650dda408c430fd3eb2c93b266cc9a9b06fb5cdde205cdf45c016ad9`. Original v1 source, generated caches, worker and failed reports remain preserved.

End-to-end SFT timing uses the slowest of four ranks and includes data fetch, teacher work, forward/backward, clipping, optimizer and logging. Helper JSONL/allocator reads occur after each recorded update endpoint, and asynchronous drain is separate. Loaded SFT instrumentation on/off validation completed in both execution orders; no raw readiness flags are automatically changed. Low CV does not prove warmup compiler settling or zero timing overhead. Shared cache temperature is unknown.

The identical short31-step cosine horizon controls this comparison but does not validate50,000-step convergence. No published success-rate or final accuracy claim is made. Real SFT command time is tracked separately and remains below the requested seven hours. This document is a commit-ready description of actual evidence; the committed model contains the measured v2 code bytes. Historical runs retain their executed parent HEAD and actual dirty diff.

First loaded SFT instrumentation pair completed31 updates per arm with four valid rank observers and finite logs: on max20-update block/20=4.241966163601319 s, off=4.247257724500377 s. The small apparent on-arm advantage is near noise, not negative overhead or a hard upper bound. Off has no ordinary per-update helper timing; its completed real updates and common GPU-drained block are retained without fabricated per-update durations. A reversed off→on pair also completed: on block/20=4.242313636100152 s, off=4.246731003199239 s, with a similarly small apparent on-arm advantage (0.104018%). Both formal CPU analyses passed full runtime YAML parsing. These short observations do not prove a universal overhead bound. Runtime flags in these actual SFT observer snapshots are matmul precision high and CUDA/cuDNN TF32 true. Original raw readiness flags remain unmodified.
