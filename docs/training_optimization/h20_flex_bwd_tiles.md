# H20 SFT: FP32 FlexAttention backward tiles

The default remains the original kernel options. Set `LINGBOT_FLEX_BWD_TILE=32x64` before starting four-rank training to use the measured H20 configuration. Supported experimental values are `32x32`, `32x64`, and `64x64`; unset or `off` restores the original behavior. The `64x64` option has not been measured in real SFT.

Only backward tile sizes change: M1/N1/M2/N2 = 32/64/64/32. Forward tiles, FP32 inputs, the existing TF32 dot policy, GQA, masks, scaling, padding and cropping are retained. This uses the verified PyTorch 2.8 `bwd_` kernel-option prefix.

Measured on 2026-10-12 with exactly the four allocated H20 GPUs in Docker, Torch 2.8.0+cu128, Transformers 4.57.3, FlashAttention 2.8.3, and the official Distributed Muon SFT configuration. Fixed micro batch 32, accumulation 1, global batch 128; existing gradient checkpointing and forward resharding enabled to fit the four-GPU baseline. Real student, depth/video teachers and all losses are retained. Each fresh run performs 10 warmup, 20 measured and one excluded terminal update; benchmark checkpoint saving is disabled consistently. The short run uses the same compressed 31-step cosine schedule in both arms.

| Configuration | Mean complete update, max of four ranks | CV | Throughput gain vs baseline |
|---|---:|---:|---:|
| Original BWD tiles 16/16/16/16 | 5.386928178 s | 0.2051% | reference |
| BWD 32x32 | 5.110056997 s | 0.1156% | 5.4182% |
| BWD 32x64 | 5.060605264 s | 0.2100% | 6.4483% |

Evidence directories are under `/home/scratch.keddyj_wwfo/robotics-test/LingBotVLA-Test/runs`: `dist-perf-r2`, `flex32x32-perf-r1`, and `flex32x64-perf-r1`. Each preserves source/dirty patch, immutable image ID, UUIDs, launch and effective configurations, full per-rank JSONL, training logs and independent analysis. The measured source was functional parent `05f9cd1da80e245617e5369259075f5440e3c13e` plus the reviewed tile patch; its FlexAttention file SHA256 is `f7e8672efe055af7374584d9f07dfe2cef8eaee4f06bc83274f54474a42cd2b5`. The final commit contains the same measured code bytes.

Four-GPU operator validation (`flex-bwd-parity-r1`) found identical forward outputs and test losses for both tiles; relative L2 gradient errors were at most 3.01e-7. This fixture uses a representative VLA mask, not the full dataset. Actual SFT completed all 31 updates, with finite logged losses/gradient norms/LR and no obvious short-term divergence. Floating-point reduction order changes: maximum logged loss difference was 0.0015 and gradient-norm difference 0.0322 for 32x64; full-model bitwise equality or final accuracy is not claimed.

An independent review of four SFT-generated AOT backward wrappers confirmed actual 32/64/64/32 tiles, unchanged TF32/PRESCALE/GQA and the real shapes (batch32, sequence337 padded384, Q32/KV8 heads, dimension128). Generated backward grid shrank from 120x1x256 to 30x1x256. Nsight Systems and one-update PyTorch Profiler evidence identified the original full FlexAttention backward kernel, rather than a standalone slicing/zeroing operation.

The comparison is one short SFT pair per tile with a repeated stable baseline. CUDA/NFS helper calibration observed about 0.255 ms mean complete instrumentation-call cost in idle contexts; this is not a hard bound for loaded training. Profiler timings and export overhead are excluded from performance comparisons. README reference times are not strictly reproduced because the four-H20 memory settings and measurement conditions differ.
