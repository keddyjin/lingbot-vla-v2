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


## ROOT parameter gather overlapped with frozen teachers

The opt-in [ROOT recipe](root_teacher/README.md) defaults to ROOT prefetch on
with `NCCL_MIN_CTAS=8`; its explicit off mode keeps MIN8. The ordinary production
entry defaults to ROOT prefetch off. Starting at update 2, the controller uses
the public asynchronous root FSDP unshard before the existing frozen teachers.
The original compiled model forward consumes the pending gather through its
native FSDP hooks. The first update keeps native lazy initialization.
The original teachers, losses, gradient checkpointing, mixed precision,
clipping, Distributed Muon and learning-rate schedule remain in place.

```bash
bash "$REPO/benchmarks/rtx5k/root_teacher/run_sft.sh" \
  --run-id my-root-min8-on-r1 --gpu-uuids "$GPU_POOL" --launch
docker wait lingbot-rtx5k-sft-my-root-min8-on-r1

# Same MIN8, original teacher scheduling, a fresh run directory.
bash "$REPO/benchmarks/rtx5k/root_teacher/run_sft.sh" \
  --run-id my-root-min8-off-r1 --gpu-uuids "$GPU_POOL" \
  --root-teacher-prefetch off --launch
```

Fresh same-source comparisons use all 260 neural files, the same eight GPUs,
complete clean50 data, official weights, micro batch 32, accumulation 1,
global batch 256, all-layer GC, `full_shard=false` and the original 50,000-step
LR horizon. Each plain short run has 10 warmup, 20 steady and one excluded
terminal update; all selected ranks complete the real batch.

| Direct cumulative comparison | Pooled steady mean | Samples/s |
|---|---:|---:|
| Default NCCL / ROOT off, three fresh anchors | 5.582719 s | 45.85579 |
| MIN8 / ROOT on, two fresh repeats | 4.725878 s | 54.16983 |

Cumulative throughput increases **18.130851%**, while update time decreases
15.348109%; all four adjacent comparisons exceed 15%. This confirms the 15%
stage, with 20% still unconfirmed. A separate actual versioned-entry MIN8
off/on pair measures 5.062968 s to 4.722079 s (**7.219049%** incremental
throughput). These distinct comparisons are not added together. Both sides
of the versioned pair pass 120 independent source/runtime/control checks,
including full 23-file recipe capture, 17 helpers and the new 13-file runtime.

All measured short-run losses, GradNorm and LR remain finite; checkpoint
inventories are empty and resume is disabled. Baseline A/A diagnostic state
differences are preserved under the original thresholds; complete active
gradient equivalence is not established. Nsight capture contains dropped
events, so its observed teacher/gather overlap explains the mechanism without
supplying the throughput claim. Short measurements do not prove long-term
convergence. Bounded longer-run stability is reported separately in the task
root, rather than inferred from these short windows.

Evidence under the remote task root includes
`reports/rtx5k-h9-cumulative-15-stage-human-review-r1.md`,
`reports/rtx5k-versioned-root-recipe-entry-validation-r1.md`,
`reports/rtx5k-root-prefetch-nsys-overlap-r1.md` and
`reports/active2-aa-review-r1/REVIEW.md`.
The frozen `root_teacher` recipe retains its preparation lineage and exact
verified bytes; the measured results above are the implementation evidence.
