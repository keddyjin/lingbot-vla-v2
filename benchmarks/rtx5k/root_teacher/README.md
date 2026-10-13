# Versioned ROOT-teacher + MIN8 SFT candidate recipe

This opt-in recipe prepares the frozen H9 ROOT-teacher prefetch experiment for
eight user-confirmed RTX PRO 5000 GPUs. Both modes use `NCCL_MIN_CTAS=8`.
The wrapper defaults to root prefetch **on** and supports explicit **off/on**.
Actual H9 and versioned-recipe comparisons must establish any incremental gain;
this prepared candidate carries no new performance or parity claim.

Repository location remains
`/root/keddyj/LingBotVLA-Test/lingbot-vla-v2-sft-opt-github`.
Use the prepared image, Torch2.8.0+cu128 / loaded NCCL2.27.3 environment,
official weights, and RoboTwin clean50 image data beneath the authorized task
root. The frozen ROOT patch must already be installed, including trainer d5,
helper349, and original framework8634. Both root modes require this same
installation; the core audit verifies all260 neural source files and all14
root/Torch source pins.

Run directly from its versioned directory; no copy to `ops/harness` is needed:

```bash
ROOT=/root/keddyj/LingBotVLA-Test
REPO="$ROOT/lingbot-vla-v2-sft-opt-github"
# GPU_POOL is eight distinct physical UUIDs in the recorded rank order.
bash "$REPO/benchmarks/rtx5k/root_teacher/run_sft.sh" \
  --run-id root-min8-on-r1 --gpu-uuids "$GPU_POOL" --launch

bash "$REPO/benchmarks/rtx5k/root_teacher/run_sft.sh" \
  --run-id root-min8-off-r1 --gpu-uuids "$GPU_POOL" \
  --root-teacher-prefetch off --launch

# Ordinary3 smoke, a fresh run directory each time.
bash "$REPO/benchmarks/rtx5k/root_teacher/run_sft.sh" \
  --run-id root-min8-on-smoke --gpu-uuids "$GPU_POOL" --kind smoke --launch
```

Omit `--launch` to print a plan. Default perf is31 updates: warmup10, steady20,
one excluded terminal update. `--warmup-steps` and `--steady-steps` support an
independent stop up to1000 updates (for example10+989+1). The original LR
`max_steps=50000` horizon remains fixed. Micro batch32, accumulation1,
global batch256, native DP8 Distributed Muon, all-layer GC and
`full_shard=false` remain unchanged. No checkpoint is saved or resumed.
Only ordinary `--kind smoke|perf` is supported by this wrapper.

MIN8 is fixed; `unset`,4,16,32, protocol/P2P/CPU-affinity/allocator overrides
and other model scheduling candidates cannot be selected through this recipe.
The container entry clears inherited min aliases, requires literal8, rejects
maximum aliases and peer-validation bypass, and records each rank's actual
environment. The wrapper assigns baseline role for root off and candidate role
for root on; `--pair-id` can identify an independently scheduled pair.

The launcher binds its own directory and passes `SFT_HARNESS_DIR` to Docker.
The entry resolves that path beneath the task root before importing Torch and
uses it for source-identity receipts and preparation. `helpers.sha256` locks
the13 H9 helpers plus four unchanged ordinary runtime-review files. Relative
paths are the only changes to the H9 launcher/entry; preparer, auditors,
core260 identities, source spec513, helper349, trainer d5 and framework8634
retain their frozen bytes. `recipe-context.json` records this recipe's lineage
and defaults. H9's inner launcher still defaults to root off; this outer wrapper
always supplies its resolved off/on choice, default on.

For an actual ordinary on3 smoke, the unchanged H9 runtime checker is available:

```bash
python "$REPO/benchmarks/rtx5k/root_teacher/runtime-review/qualify_runtime.py" \
  --run-dir "$ROOT/runs/root-min8-on-smoke" \
  --output "$ROOT/runs/root-min8-on-smoke/root-runtime-qualified-r1.json"
```

The checker covers exact source/context, eight rank/PID/direct-UUID joins,
setup0, skipped update1 and active updates2/3. It supplements the ordinary
analyzer, complete helper manifest, actual exit codes, finite scalar curves,
checkpoint inventory and runtime LR horizon checks. It is deliberately
ordinary-on3-only; a31/1000-update endurance run requires its separately
versioned review scope. First skipped update is not active parity, and cleared
pending state is not a GPU-completion or teacher-overlap timestamp. Full active
gradient/optimizer parity and performance require their own evidence.

Before any performance commit, root must run an actual **versioned ordinary
off/on pair** using this path, preserve its source/helper/config/control
receipts and evaluate the gain within the same MIN8 context. A cumulative
comparison to default MINunset is a separate explicitly identified baseline.
Prepared CPU fixtures and prior unversioned runs do not qualify that pair.
