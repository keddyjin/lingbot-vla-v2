#!/usr/bin/env bash
# Versioned ROOT+MIN8 candidate; plans unless root scheduler supplies --launch.
set -euo pipefail
readonly HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ROOT=/root/keddyj/LingBotVLA-Test
RUN_ID='' GPU_POOL='' ROOT_PREFETCH=on MIN_CTAS=8 KIND=perf WARMUP=10 STEADY=20 LAUNCH=0
PAIR_ID=rtx5k-versioned-root-min8
while (($#)); do
  case "$1" in
    --run-id) RUN_ID=${2:?}; shift 2;;
    --gpu-uuids) GPU_POOL=${2:?}; shift 2;;
    --root-teacher-prefetch) ROOT_PREFETCH=${2:?}; shift 2;;
    --nccl-min-ctas) MIN_CTAS=${2:?}; shift 2;;
    --kind) KIND=${2:?}; shift 2;;
    --pair-id) PAIR_ID=${2:?}; shift 2;;
    --warmup-steps) WARMUP=${2:?}; shift 2;;
    --steady-steps) STEADY=${2:?}; shift 2;;
    --launch) LAUNCH=1; shift;;
    *) printf 'Unknown root recipe argument: %s\n' "$1" >&2; exit 2;;
  esac
done
case "$ROOT_PREFETCH" in off|on) ;; *) printf 'Root teacher prefetch must be off or on\n' >&2; exit 2;; esac
[[ "$MIN_CTAS" == 8 ]] || { printf 'Versioned root recipe fixes MIN_CTAS to literal8 in both modes\n' >&2; exit 2; }
case "$KIND" in smoke|perf) ;; *) printf 'Ordinary recipe kind must be smoke or perf\n' >&2; exit 2;; esac
[[ "$WARMUP" =~ ^[0-9]+$ && "$STEADY" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid warmup/steady counts\n' >&2; exit 2; }
[[ "$PAIR_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || { printf 'Invalid pair id\n' >&2; exit 2; }
(cd "$HERE" && sha256sum -c helpers.sha256 >&2)
ROLE=candidate
[[ "$ROOT_PREFETCH" != off ]] || ROLE=baseline
args=(--run-id "$RUN_ID" --gpu-uuids "$GPU_POOL" --world-size 8 --optimizer-mode dist
  --kind "$KIND" --memory-policy gc --gc-policy official --nccl-min-ctas 8
  --root-teacher-prefetch "$ROOT_PREFETCH" --fsdp-fwd-prefetch off
  --pair-id "$PAIR_ID" --role "$ROLE" --warmup-steps "$WARMUP" --steady-steps "$STEADY"
  --data-manifest "$ROOT/data-preparation/robotwin-clean50-image-training.txt"
  --weights-manifest "$ROOT/artifact-preparation/weights-manifest.json")
[[ "$LAUNCH" == 0 ]] || args+=(--launch)
exec bash "$HERE/rtx5k_launch_sft_run.sh" "${args[@]}"
