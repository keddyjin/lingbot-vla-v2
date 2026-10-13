#!/usr/bin/env bash
# Root GPU scheduler only. The default prints a plan; --launch starts Docker.
set -euo pipefail
readonly ROOT=/root/keddyj/LingBotVLA-Test
readonly REPO="$ROOT/lingbot-vla-v2-sft-opt-github"
readonly VENV="$ROOT/environment/venv"
readonly HARNESS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
IMAGE_REFERENCE=${SFT_IMAGE:-lmsysorg/sglang:dev}
RUN_ID='' DATA_MANIFEST='' WEIGHTS_MANIFEST='' PAIR_ID='' ROLE=baseline WORLD=8 GPU_POOL=''
MODE=dist KIND=perf MEMORY_POLICY=gc-reshard FLEX_TILE=off GC_POLICY=official
COUNTER=0 INDICES=0 BUCKET=official TEACHER_BATCH=0 MOE_OUTPUT_SIZE=0 BWD_STAGES=off NCCL_PROTO_CONTROL=unset
AFFINITY=off ALLOCATOR=unset P2P_LEVEL=default P2P_DISABLE_CONTROL=unset MIN_CTAS_CONTROL=8
WARMUP=10 STEADY=20 LAUNCH=0 FSDP_FWD_PREFETCH=off ROOT_TEACHER_PREFETCH=off
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
has_text() { if command -v rg >/dev/null 2>&1; then rg -q "$1" "$2"; else grep -q "$1" "$2"; fi; }
while (($#)); do
  case "$1" in
    --run-id) RUN_ID=${2:?}; shift 2;;
    --data-manifest) DATA_MANIFEST=${2:?}; shift 2;;
    --weights-manifest) WEIGHTS_MANIFEST=${2:?}; shift 2;;
    --pair-id) PAIR_ID=${2:?}; shift 2;;
    --role) ROLE=${2:?}; shift 2;;
    --world-size) WORLD=${2:?}; shift 2;;
    --gpu-uuids) GPU_POOL=${2:?}; shift 2;;
    --image) IMAGE_REFERENCE=${2:?}; shift 2;;
    --optimizer-mode) MODE=${2:?}; shift 2;;
    --kind) KIND=${2:?}; shift 2;;
    --memory-policy) MEMORY_POLICY=${2:?}; shift 2;;
    --flex-bwd-tile) FLEX_TILE=${2:?}; shift 2;;
    --flex-bwd-stages) BWD_STAGES=${2:?}; shift 2;;
    --nccl-proto) NCCL_PROTO_CONTROL=${2:?}; shift 2;;
    --p2p-level) P2P_LEVEL=${2:?}; shift 2;;
    --p2p-disable) P2P_DISABLE_CONTROL=${2:?}; shift 2;;
    --nccl-min-ctas) MIN_CTAS_CONTROL=${2:?}; shift 2;;
    --cpu-affinity) AFFINITY=${2:?}; shift 2;;
    --cuda-allocator) ALLOCATOR=${2:?}; shift 2;;
    --gc-policy) GC_POLICY=${2:?}; shift 2;;
    --fsdp-fwd-prefetch) FSDP_FWD_PREFETCH=${2:?}; shift 2;;
    --root-teacher-prefetch) ROOT_TEACHER_PREFETCH=${2:?}; shift 2;;
    --compile-step-counter) COUNTER=${2:?}; shift 2;;
    --deepstack-indices) INDICES=${2:?}; shift 2;;
    --dist-muon-layers-per-bucket) BUCKET=${2:?}; shift 2;;
    --batch-depth-teachers) TEACHER_BATCH=${2:?}; shift 2;;
    --moe-segment-output-size) MOE_OUTPUT_SIZE=${2:?}; shift 2;;
    --warmup-steps) WARMUP=${2:?}; shift 2;;
    --steady-steps) STEADY=${2:?}; shift 2;;
    --launch) LAUNCH=1; shift;;
    *) fail "Unknown argument: $1";;
  esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || fail 'Supply a unique simple run ID.'
[[ "$WORLD" == 4 || "$WORLD" == 8 ]] || fail 'Only world-size 4 or 8 is allowed.'
[[ "$MODE" == dist || "$MODE" == official-muon || "$MODE" == controlled-muon ]] || fail 'Invalid optimizer mode.'
[[ "$ROLE" == baseline || "$ROLE" == candidate ]] || fail 'Invalid role.'
case "$KIND" in smoke|perf|nsys|torch) ;; *) fail 'Invalid run kind.';; esac
case "$MEMORY_POLICY" in official|gc|reshard|gc-reshard) ;; *) fail 'Invalid memory policy.';; esac
case "$FLEX_TILE" in off|32x32|32x64|64x64|16x64|32x128|32x64-w8|32x128-w8) ;; *) fail 'Invalid Flex tile.';; esac
case "$BWD_STAGES" in off|1|2) ;; *) fail 'Invalid Flex backward stages.';; esac
case "$NCCL_PROTO_CONTROL" in unset|Simple) ;; *) fail 'NCCL protocol must be unset or Simple.';; esac
case "$P2P_LEVEL" in default|PHB|SYS) ;; *) fail 'P2P level must be default, PHB or SYS.';; esac
[[ "$P2P_LEVEL" == default || ( "$WORLD" == 8 && "$NCCL_PROTO_CONTROL" == unset ) ]] || fail 'P2P candidate requires eight GPUs and protocol unset.'
case "$P2P_DISABLE_CONTROL" in unset|1) ;; *) fail 'P2P disable must be unset or 1.';; esac
[[ "$P2P_DISABLE_CONTROL" == unset || ( "$WORLD" == 8 && "$NCCL_PROTO_CONTROL" == unset && "$P2P_LEVEL" == default ) ]] || fail 'P2P disable trial requires DP8, protocol unset and default level.'
case "$MIN_CTAS_CONTROL" in 8) ;; *) fail 'H9 root context requires literal minimum CTAs8 in both modes.';; esac
[[ "$WORLD" == 8 && "$NCCL_PROTO_CONTROL" == unset && "$P2P_LEVEL" == default && "$P2P_DISABLE_CONTROL" == unset && "$FSDP_FWD_PREFETCH" == off ]] || fail 'H7 minimum CTA trial requires DP8, protocol unset, default P2P, disable unset and FWD off.'
case "$AFFINITY" in off|gpu-numa) ;; *) fail 'CPU affinity must be off or gpu-numa.';; esac
[[ "$AFFINITY" == off || "$WORLD" == 8 ]] || fail 'NUMA placement requires eight GPUs.'
case "$ALLOCATOR" in unset|expandable) ;; *) fail 'CUDA allocator must be unset or expandable.';; esac
case "$ROOT_TEACHER_PREFETCH" in off|on) ;; *) fail 'Root teacher prefetch must be off or on.';; esac
[[ "$WORLD" == 8 && "$MODE" == dist && "$MEMORY_POLICY" == gc && "$GC_POLICY" == official && "$FSDP_FWD_PREFETCH" == off ]] || fail 'H9 off/on root+MIN8 SFT requires pure gc-only/native DP8/dist.'
[[ "$FLEX_TILE" == off && "$BWD_STAGES" == off && "$COUNTER" == 0 && "$INDICES" == 0 && "$TEACHER_BATCH" == 0 && "$MOE_OUTPUT_SIZE" == 0 && "$BUCKET" == official ]] || fail 'H9 off/on root+MIN8 SFT is an isolated model/system scheduling trial.'
[[ "$NCCL_PROTO_CONTROL" == unset && "$P2P_LEVEL" == default && "$P2P_DISABLE_CONTROL" == unset && "$MIN_CTAS_CONTROL" == 8 && "$AFFINITY" == off && "$ALLOCATOR" == unset ]] || fail 'H9 off/on root+MIN8 SFT fixes system controls.'
case "$FSDP_FWD_PREFETCH" in off|one|pair) ;; *) fail 'FSDP FWD prefetch must be off, one or pair.';; esac
[[ "$FSDP_FWD_PREFETCH" == off || "$WORLD" == 8 ]] || fail 'FSDP FWD prefetch requires DP8.'
case "$GC_POLICY" in official|action-off|text-alternate|text-off|action-off-text-alternate|action-off-text-off|action-off-text-off-vision-alternate) ;; *) fail 'Invalid GC policy.';; esac
for flag in "$COUNTER" "$INDICES" "$TEACHER_BATCH" "$MOE_OUTPUT_SIZE"; do
  [[ "$flag" == 0 || "$flag" == 1 ]] || fail 'Binary optimization flags must be 0 or 1.'
done
[[ "$WARMUP" =~ ^[0-9]+$ && "$STEADY" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid warmup/steady counts.'
WARMUP=$((10#$WARMUP)); STEADY=$((10#$STEADY))
if [[ "$KIND" == smoke ]]; then WARMUP=3; STOP=3;
else
  if [[ "$KIND" == nsys ]]; then STEADY=${LINGBOT_NSYS_CAPTURE_STEPS:-2}; fi
  if [[ "$KIND" == torch ]]; then STEADY=1; fi
  STOP=$((WARMUP + STEADY + 1))
fi
((STOP > 0 && STOP <= 1000)) || fail 'Benchmark independent stop must be at most 1000.'
if [[ "$BUCKET" != official ]]; then
  [[ "$MODE" == dist && "$BUCKET" =~ ^[1-9][0-9]?$ ]] || fail 'Bucket tuning requires Distributed Muon and positive integer.'
  ((BUCKET <= 36)) || fail 'Bucket count setting exceeds 36 actual decoder layers.'
fi
[[ -n "$GPU_POOL" && "$GPU_POOL" != *[[:space:]]* && "$GPU_POOL" != *, ]] || fail 'Supply exact physical GPU UUIDs.'
IFS=',' read -r -a gpu_ids <<< "$GPU_POOL"
[[ ${#gpu_ids[@]} == "$WORLD" ]] || fail 'GPU pool count differs from world size.'
declare -A seen=()
for uuid in "${gpu_ids[@]}"; do
  [[ "$uuid" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ && ! -v "seen[$uuid]" ]] || fail 'Invalid or duplicated GPU UUID.'
  seen[$uuid]=1
done
ROOT_REAL=$(realpath -e -- "$ROOT")
RUN="$ROOT/runs/$RUN_ID"
[[ -n "$DATA_MANIFEST" && "$DATA_MANIFEST" == /* && -f "$DATA_MANIFEST" ]] || fail 'A real existing training manifest is required.'
for path in "$RUN" "$REPO" "$VENV" "$HARNESS" "$DATA_MANIFEST"; do
  case "$(realpath -m -- "$path")/" in "$ROOT_REAL/"*) ;; *) fail 'Execution path escapes authorized ROOT.';; esac
done
if [[ -n "$WEIGHTS_MANIFEST" ]]; then
  [[ "$WEIGHTS_MANIFEST" == /* && -f "$WEIGHTS_MANIFEST" ]] || fail 'Weights manifest must be a real absolute file.'
  case "$(realpath -e -- "$WEIGHTS_MANIFEST")/" in "$ROOT_REAL/"*) ;; *) fail 'Weights manifest escapes ROOT.';; esac
fi
[[ ! -e "$RUN" ]] || fail 'Do not reuse or overwrite an existing run directory.'
[[ ( -f "$VENV/bin/python" || -L "$VENV/bin/python" ) && -f "$REPO/lingbotvla/utils/training_instrumentation.py" ]] || fail 'Missing prepared venv entry or measurement implementation.'
# The Python symlink target belongs to the pinned container image and may not
# exist on the host. Execution/version validation happens inside the container.
[[ $(git --no-optional-locks -C "$REPO" branch --show-current) == keddyj-codex/sft-opt-rtx5k ]] || fail 'Wrong source branch.'
TRAINER="$REPO/tasks/vla/train_lingbotvla.py"
for flag in LINGBOT_BENCHMARK_SKIP_CHECKPOINT LINGBOT_BENCHMARK_STOP_STEPS; do
  has_text "$flag" "$TRAINER" || fail "Functional guard not applied: $flag"
done
if [[ "$FLEX_TILE" != off ]]; then has_text LINGBOT_FLEX_BWD_TILE "$REPO/lingbotvla/models/vla/lingbot_vla/flex_attention.py" || fail 'Review/apply Flex candidate first.'; fi
if [[ "$BWD_STAGES" != off ]]; then has_text LINGBOT_FLEX_BWD_NUM_STAGES "$REPO/lingbotvla/models/vla/lingbot_vla/flex_attention.py" || fail 'Review/apply backward-stage candidate first.'; fi
if [[ "$GC_POLICY" != official ]]; then
  [[ "$MEMORY_POLICY" == gc || "$MEMORY_POLICY" == gc-reshard ]] || fail 'Selective policy requires gradient checkpointing enabled.'
  has_text LINGBOT_GC_POLICY "$REPO/lingbotvla/distributed/torch_parallelize.py" || fail 'Review/apply selective GC candidate first.'
fi
if [[ "$COUNTER" == 1 ]]; then has_text LINGBOT_COMPILE_STEP_COUNTER_HELPER "$REPO/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py" || fail 'Missing counter candidate.'; fi
if [[ "$INDICES" == 1 ]]; then has_text LINGBOT_DEEPSTACK_INDICES "$REPO/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py" || fail 'Missing DeepStack v2 candidate.'; fi
if [[ "$TEACHER_BATCH" == 1 ]]; then has_text LINGBOT_DEPTH_TEACHER_BATCH "$TRAINER" || fail 'Missing teacher batching candidate.'; fi
if [[ "$MOE_OUTPUT_SIZE" == 1 ]]; then has_text LINGBOT_MOE_SEGMENT_OUTPUT_SIZE "$REPO/lingbotvla/ops/triton_moe_loss.py" || fail 'Missing MoE candidate.'; fi
if [[ "$FSDP_FWD_PREFETCH" != off ]]; then has_text LINGBOT_FSDP_FWD_PREFETCH "$REPO/lingbotvla/distributed/torch_parallelize.py" || fail 'Missing reviewed FSDP prefetch candidate.'; fi
[[ -f "$HARNESS/core_source_audit.py" && -f "$HARNESS/core-source-manifest.json" ]] || fail 'Missing strict H7 core byte auditor/manifest.'
[[ -f "$HARNESS/fsdp_prefetch_source_audit.py" ]] || fail 'Missing FSDP prefetch source audit helper.'
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REFERENCE")
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'Existing immutable Docker image is required; scheduler must prepare image first.'
IMAGE_DIGEST=$(docker image inspect --format '{{json .RepoDigests}}' "$IMAGE_REFERENCE")
selected=$(nvidia-smi -i "$GPU_POOL" --query-gpu=uuid,name,driver_version,memory.total,pci.bus_id --format=csv,noheader)
[[ $(printf '%s\n' "$selected" | wc -l) == "$WORLD" ]] || fail 'Physical GPU query count mismatch.'
while IFS=',' read -r uuid name rest; do
  uuid=${uuid//[[:space:]]/}
  [[ -v "seen[$uuid]" ]] || fail 'Unexpected physical GPU UUID mapping.'
done <<< "$selected"
PAIR_ID=${PAIR_ID:-$RUN_ID}
SHORT="$ROOT/t/$(printf '%s' "$RUN_ID" | sha256sum | cut -c1-8)"
VISIBLE=0
for ((rank=1;rank<WORLD;rank++)); do VISIBLE+=",$rank"; done
argv=(docker run --detach --pull=never --name "lingbot-rtx5k-sft-$RUN_ID" --gpus "\"device=$GPU_POOL\"" --read-only --ipc=private --shm-size=32g --log-driver=none
  --mount "type=bind,src=$ROOT,dst=$ROOT" --workdir "$REPO" --user "$(id -u):$(id -g)"
  --tmpfs "$SHORT:rw,nosuid,nodev,size=4g,mode=1777"
  --env "NV_GPU=$GPU_POOL" --env "NVIDIA_VISIBLE_DEVICES=$GPU_POOL" --env "CUDA_VISIBLE_DEVICES=$VISIBLE"
  --env "SFT_HARNESS_DIR=$HARNESS" --env "SFT_WORLD_SIZE=$WORLD" --env "SFT_RUN_DIR=$RUN" --env "SFT_CONTAINER_NAME=lingbot-rtx5k-sft-$RUN_ID"
  --env "TRITON_PTXAS_PATH=$VENV/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"
  --env "SFT_IMAGE_ID=$IMAGE_ID" --env "SFT_IMAGE_REFERENCE=$IMAGE_REFERENCE" --env "SFT_IMAGE_DIGEST=$IMAGE_DIGEST"
  --env "SFT_OPTIMIZER_MODE=$MODE" --env "SFT_KIND=$KIND" --env "SFT_MEMORY_POLICY=$MEMORY_POLICY"
  --env "SFT_NCCL_PROTO_CONTROL=$NCCL_PROTO_CONTROL" --env "SFT_NCCL_P2P_LEVEL_CONTROL=$P2P_LEVEL" --env "SFT_NCCL_P2P_DISABLE_CONTROL=$P2P_DISABLE_CONTROL" --env "SFT_NCCL_MIN_CTAS_CONTROL=$MIN_CTAS_CONTROL"
  --env "LINGBOT_CPU_AFFINITY=$AFFINITY" --env "LINGBOT_CPU_AFFINITY_RECEIPT_DIR=$RUN/numa"
  --env "SFT_CUDA_ALLOCATOR_CONTROL=$ALLOCATOR"
  --env "LINGBOT_FSDP_ROOT_TEACHER_PREFETCH=$ROOT_TEACHER_PREFETCH" --env "LINGBOT_FSDP_ROOT_TEACHER_PREFETCH_RECEIPT_DIR=$RUN/root-teacher-prefetch" --env "LINGBOT_FSDP_ROOT_TEACHER_PREFETCH_SOURCE_SPEC=$HARNESS/source-audit-spec.json"
  --env "LINGBOT_FSDP_FWD_PREFETCH=$FSDP_FWD_PREFETCH" --env "LINGBOT_FSDP_FWD_PREFETCH_RECEIPT_DIR=$RUN/fsdp-forward-prefetch"
  --env "SFT_DATA_MANIFEST=$DATA_MANIFEST" --env "SFT_WEIGHTS_MANIFEST=$WEIGHTS_MANIFEST" --env "SFT_PAIR_ID=$PAIR_ID" --env "SFT_ROLE=$ROLE" --env "SFT_STEADY_STEPS=$STEADY"
  --env "LINGBOT_FLEX_BWD_TILE=$FLEX_TILE" --env "LINGBOT_GC_POLICY=$GC_POLICY" --env "LINGBOT_COMPILE_STEP_COUNTER_HELPER=$COUNTER"
  --env "LINGBOT_FLEX_BWD_NUM_STAGES=$BWD_STAGES"
  --env "LINGBOT_DEEPSTACK_INDICES=$INDICES" --env "SFT_DIST_MUON_LAYERS_PER_BUCKET=$BUCKET"
  --env "LINGBOT_DEPTH_TEACHER_BATCH=$TEACHER_BATCH" --env "LINGBOT_MOE_SEGMENT_OUTPUT_SIZE=$MOE_OUTPUT_SIZE"
  --env LINGBOT_BENCHMARK_SKIP_CHECKPOINT=1 --env "LINGBOT_BENCHMARK_STOP_STEPS=$STOP"
  --env "LINGBOT_MEASURE_JSONL_DIR=$RUN/timings" --env "LINGBOT_MEASURE_RUN_ID=$RUN_ID" --env "LINGBOT_MEASURE_WARMUP_STEPS=$WARMUP" --env LINGBOT_MEASURE_MEMORY=1
  --env "LINGBOT_PROFILE_RANGES=$([[ "$KIND" == nsys || "$KIND" == torch ]] && echo 1 || echo 0)"
  --env "LINGBOT_NSYS_CAPTURE_MODE=${LINGBOT_NSYS_CAPTURE_MODE:-nvtx}" --env "LINGBOT_NSYS_CAPTURE_STEPS=${LINGBOT_NSYS_CAPTURE_STEPS:-2}" --env "LINGBOT_NSYS_CUDA_FLUSH_MS=${LINGBOT_NSYS_CUDA_FLUSH_MS:-100}"
  --env "HOME=$RUN/home" --env USER=root --env LOGNAME=root --env "TMPDIR=$SHORT" --env "TMP=$SHORT" --env "TEMP=$SHORT"
  --env "XDG_CACHE_HOME=$RUN/cache" --env "XDG_CONFIG_HOME=$RUN/config"
  --env "HF_HOME=$ROOT/cache/rtx5k/huggingface" --env "TORCH_HOME=$ROOT/cache/rtx5k/torch"
  --env "TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/rtx5k/inductor" --env "TRITON_CACHE_DIR=$ROOT/cache/rtx5k/triton" --env "CUDA_CACHE_PATH=$ROOT/cache/rtx5k/cuda"
  --env "TORCH_EXTENSIONS_DIR=$RUN/cache/extensions" --env "MPLCONFIGDIR=$RUN/cache/matplotlib" --env "TORCH_KERNEL_CACHE_DIR=$ROOT/cache/rtx5k/torch/kernels"
  --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --entrypoint bash "$IMAGE_ID" -c
  'set +e; bash "$SFT_HARNESS_DIR/rtx5k_sft_container_entry.sh" > "$SFT_RUN_DIR/container.log" 2>&1; code=$?; printf "%s\n" "$code" > "$SFT_RUN_DIR/container-exit-code.txt"; exit "$code"')
if [[ "$LAUNCH" == 0 ]]; then printf '%q ' "${argv[@]}"; printf '\nPlan only; world=%s/global=%s/stop=%s; LR horizon remains source max_steps.\n' "$WORLD" "$((WORLD*32))" "$STOP"; exit 0; fi
while IFS= read -r running; do
  case "$running" in lingbot-rtx5k-sft-*|lingbot-rtx5k-*-parity-*|lingbot-rtx5k-parity-*|lingbot-rtx5k-nccl-*|lingbot-rtx5k-ncu-*|lingbot-sft-*|lingbot-flex-parity-*|lingbot-deepstack-parity-*|lingbot-teacher-parity-*|lingbot-ncu-flex-*) fail "Task GPU container already active: $running";; esac
done < <(docker ps --format '{{.Names}}')
mkdir -p "$RUN/home" "$RUN/cache" "$RUN/config" "$RUN/timings" "$RUN/profiling" "$SHORT"
if [[ -d "$ROOT/cache/rtx5k/inductor" ]]; then printf 'existing shared cache; actual content-key hits unknown\n'; else printf 'new cache directory; actual cold compilation depends on keys\n'; fi > "$RUN/compile-cache-before-launch.txt"
mkdir -p "$ROOT/cache/rtx5k/inductor" "$ROOT/cache/rtx5k/triton" "$ROOT/cache/rtx5k/cuda" "$ROOT/cache/rtx5k/huggingface" "$ROOT/cache/rtx5k/torch/kernels"
printf '%s\n' "$selected" > "$RUN/gpu-selected.csv"
docker image inspect "$IMAGE_ID" > "$RUN/docker-image-inspect.json"
nvidia-smi topo -m > "$RUN/gpu-topology.txt"
printf '%q ' "${argv[@]}" > "$RUN/docker-command.sh"; printf '\n' >> "$RUN/docker-command.sh"
"${argv[@]}" > "$RUN/container-id.txt"
printf 'container=lingbot-rtx5k-sft-%s\nrun=%s\n' "$RUN_ID" "$RUN"
