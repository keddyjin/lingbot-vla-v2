# RTX PRO 5000 Distributed Muon SFT benchmark

Opt-in recipe for the measured 8-GPU PCIe machine, using `NCCL_MIN_CTAS=8`.
The normal training entry and optimizer mathematics are unchanged.

Requirements: repository at `/root/keddyj/LingBotVLA-Test/lingbot-vla-v2-sft-opt-github`,
prepared Docker image/venv, official weights and full RoboTwin clean50 image
dataset beneath the same task root. The fresh session preparation receipts
identify the image, package versions, input revisions and complete file hashes.
This machine uses Torch 2.8.0+cu128 and loaded NCCL 2.27.3.

Run the versioned helpers directly; no copy into `ops/harness` is needed:

```bash
ROOT=/root/keddyj/LingBotVLA-Test
REPO="$ROOT/lingbot-vla-v2-sft-opt-github"
# Comma-separated, eight unique physical UUIDs in the recorded rank order.
bash "$REPO/benchmarks/rtx5k/run_sft.sh" --run-id my-minctas8-r1 \
  --gpu-uuids "$GPU_POOL" --launch
docker wait lingbot-rtx5k-sft-my-minctas8-r1

# Default NCCL comparison with the same recipe, a new run directory each time.
bash "$REPO/benchmarks/rtx5k/run_sft.sh" --run-id my-default-r1 \
  --gpu-uuids "$GPU_POOL" --nccl-min-ctas unset --launch
```

Omit `--launch` to print the Docker plan. Default measurement is 31 updates:
10 warmup, 20 steady, one terminal update. `--warmup-steps` and
`--steady-steps` can extend measurement up to the independent 1000-update
limit. The production `max_steps=50000` learning-rate horizon remains fixed.
Micro batch is 32, accumulation is 1, global batch is 256.
All-layer gradient checkpointing is enabled for capacity; full sharding
after forward remains disabled. The unmodified 4-GPU no-GC configuration
exceeded this machine's memory during the initial forward.

The container entry clears inherited minimum-channel aliases and P2P/protocol
controls, refuses maximum-channel conflicts for MIN4/8, and records the actual canonical
setting. `unset`, `4`, and `8` are the only supported minimum-CTA controls.
There is no peer-validation bypass. No training checkpoint is saved or
resumed. Each run contains source commit/diff, helper hashes, effective config,
input identities, eight rank timings, scalar curves and checkpoint inventory.
The timer includes data fetch, transfers, teachers, model, backward, clipping,
optimizer and logging, and drains CUDA at each complete update boundary.

Measured on eight user-confirmed RTX PRO 5000 GPUs (raw NVML name
`NVIDIA Graphics Device`, SM120, PCI ID `10de:2bb3`, approximately 72 GiB
reported capacity), with two NUMA sockets and PCIe 5.0 x16:

| Control | Repeated steady mean | Pooled samples/s | Throughput gain |
|---|---:|---:|---:|
| Default, three anchors | 5.596255 s | 45.74488 | reference |
| MIN_CTAS=4, two reversed repeats | 5.092872 s | 50.26633 | 9.8841% |
| MIN_CTAS=8, two reversed repeats | 5.018676 s | 51.00947 | 11.5086% |

Baseline/candidate order was default→4→8→default→8→4→default. Each
row pools complete steady update durations; the middle anchor is counted
once. Both MIN8 phases exceed 10% and all adjacent anchor comparisons exceed
10%. MIN8 run CVs are 1.45%/1.64%; these short runs do not establish long-term
convergence or a cross-run confidence interval. Original collective numerical
checks pass on all eight GPUs, and 31-update loss/gradient/LR curves remain
finite. Actual communicator channels become 2/4/8; the environment setting
is a minimum and does not imply that every collective uses exactly 8 CTAs.
P2P/SHM ring routes remain unchanged. Other machines require remeasurement.

Full fresh-session evidence is in `LingBotVLA2.0-Training-Opt-Local-rtx5k`
and the remote task root, especially
`reports/rtx5k-minctas-reverse-repeat-stage-review-r1.{md,json}` and
`reports/rtx5k-minctas-operator-qualification-r1.md`.
