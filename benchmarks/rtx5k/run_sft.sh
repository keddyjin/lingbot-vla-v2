#!/usr/bin/env bash
# Measured DP8 RTX PRO 5000 recipe. Plans unless --launch is supplied.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT=/root/keddyj/LingBotVLA-Test
RUN_ID='' GPU_POOL='' MIN_CTAS=8 LAUNCH=0 WARMUP=10 STEADY=20
while (($#)); do
  case "$1" in
    --run-id) RUN_ID=${2:?}; shift 2;;
    --gpu-uuids) GPU_POOL=${2:?}; shift 2;;
    --nccl-min-ctas) MIN_CTAS=${2:?}; shift 2;;
    --warmup-steps) WARMUP=${2:?}; shift 2;;
    --steady-steps) STEADY=${2:?}; shift 2;;
    --launch) LAUNCH=1; shift;;
    *) printf 'Unknown recipe argument: %s\n' "$1" >&2; exit 2;;
  esac
done
case "$MIN_CTAS" in unset|4|8) ;; *) printf 'Minimum CTAs must be unset, 4 or 8\n' >&2; exit 2;; esac
(cd "$HERE" && sha256sum -c helpers.sha256 >&2)
ROLE=candidate
[[ "$MIN_CTAS" != unset ]] || ROLE=baseline
args=(--run-id "$RUN_ID" --gpu-uuids "$GPU_POOL" --world-size 8 --optimizer-mode dist
  --kind perf --memory-policy gc --gc-policy official --nccl-min-ctas "$MIN_CTAS"
  --pair-id rtx5k-versioned-nccl --role "$ROLE" --warmup-steps "$WARMUP" --steady-steps "$STEADY"
  --data-manifest "$ROOT/data-preparation/robotwin-clean50-image-training.txt"
  --weights-manifest "$ROOT/artifact-preparation/weights-manifest.json")
[[ "$LAUNCH" == 0 ]] || args+=(--launch)
exec bash "$HERE/rtx5k_launch_sft_run.sh" "${args[@]}"
