#!/usr/bin/env bash
set -euo pipefail
readonly ROOT=/root/keddyj/LingBotVLA-Test
readonly PY="$ROOT/environment/venv/bin/python"
readonly REPO="$ROOT/lingbot-vla-v2-sft-opt-github"
readonly HARNESS="${SFT_HARNESS_DIR:?launcher must supply the versioned helper directory}"
case "$(realpath -e -- "$HARNESS")/" in "$ROOT/"*) ;; *) printf 'Helper directory escapes ROOT\n' >&2; exit 2;; esac
# Select before importing Torch/NCCL; baseline is truly unset.
case "${SFT_NCCL_PROTO_CONTROL:-unset}" in
  unset) unset NCCL_PROTO;;
  Simple) export NCCL_PROTO=Simple;;
  *) printf 'NCCL protocol must be unset or Simple\n' >&2; exit 2;;
esac
unset NCCL_P2P_LEVEL NCCL_P2P_DISABLE
case "${SFT_NCCL_P2P_LEVEL_CONTROL:-default}" in
  default) ;;
  PHB|SYS) [[ "${SFT_NCCL_PROTO_CONTROL:-unset}" == unset && "$SFT_WORLD_SIZE" == 8 ]] || exit 2; export NCCL_P2P_LEVEL="$SFT_NCCL_P2P_LEVEL_CONTROL";;
  *) printf 'P2P level must be default, PHB or SYS\n' >&2; exit 2;;
esac
case "${SFT_NCCL_P2P_DISABLE_CONTROL:-unset}" in
  unset) ;;
  1) [[ "${SFT_NCCL_P2P_LEVEL_CONTROL:-default}" == default && "${SFT_NCCL_PROTO_CONTROL:-unset}" == unset && "$SFT_WORLD_SIZE" == 8 ]] || exit 2; export NCCL_P2P_DISABLE=1;;
  *) printf 'P2P disable must be unset or 1\n' >&2; exit 2;;
esac
case "${NCCL_IGNORE_DISABLED_P2P:-0}" in ''|0) ;; *) printf 'Peer validation bypass forbidden\n' >&2; exit 2;; esac
unset NCCL_MIN_CTAS NCCL_MIN_NCHANNELS NCCL_MIN_NRINGS
case "${SFT_NCCL_MIN_CTAS_CONTROL:-unset}" in
  unset) ;;
  4|8) [[ "${SFT_NCCL_PROTO_CONTROL:-unset}" == unset && "${SFT_NCCL_P2P_LEVEL_CONTROL:-default}" == default && "${SFT_NCCL_P2P_DISABLE_CONTROL:-unset}" == unset && "$SFT_WORLD_SIZE" == 8 ]] || exit 2; export NCCL_MIN_CTAS="$SFT_NCCL_MIN_CTAS_CONTROL";;
  *) printf 'Minimum CTAs must be unset, 4 or 8\n' >&2; exit 2;;
esac
[[ "${SFT_NCCL_MIN_CTAS_CONTROL:-unset}" == unset || ( ! -v NCCL_MAX_CTAS && ! -v NCCL_MAX_NCHANNELS && ! -v NCCL_MAX_NRINGS ) ]] || exit 2
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
case "${SFT_CUDA_ALLOCATOR_CONTROL:-unset}" in
  unset) ;;
  expandable) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  *) printf 'CUDA allocator must be unset or expandable\n' >&2; exit 2;;
esac
readonly RUN="$SFT_RUN_DIR"
case "$RUN/" in "$ROOT/runs/"*) ;; *) printf 'Run escapes authorized ROOT/runs\n' >&2; exit 2;; esac
[[ -d "$RUN" ]] || { printf 'Run directory missing\n' >&2; exit 2; }
CURRENT_STAGE=entry_setup
CURRENT_PHASE=preparation
date -u +%FT%TZ > "$RUN/preparation-started-utc.txt"
record_stage() {
  CURRENT_STAGE=$1
  printf '%s\t%s\t%s\tstart\n' "$(date -u +%FT%TZ)" "$CURRENT_PHASE" "$CURRENT_STAGE" >> "$RUN/preparation-lifecycle.tsv"
}
on_error() {
  local failure_status=$1
  trap - ERR
  set +e
  printf '%s\t%s\t%s\tfailed\texit=%s\n' "$(date -u +%FT%TZ)" "$CURRENT_PHASE" "$CURRENT_STAGE" "$failure_status" >> "$RUN/preparation-lifecycle.tsv"
  printf 'phase=%s\nstage=%s\nexit=%s\ntraining_launched=%s\n' "$CURRENT_PHASE" "$CURRENT_STAGE" "$failure_status" "$([[ "$CURRENT_PHASE" == preparation ]] && echo false || echo true)" > "$RUN/entry-failure.txt"
  if [[ "$CURRENT_PHASE" == preparation ]]; then
    date -u +%FT%TZ > "$RUN/preparation-finished-utc.txt"
    printf '%s\n' "$failure_status" > "$RUN/preparation-exit-code.txt"
  fi
  export SFT_FAILURE_STATUS="$failure_status" SFT_FAILURE_STAGE="$CURRENT_STAGE" SFT_FAILURE_PHASE="$CURRENT_PHASE"
  # Only stdlib is required. If the interpreter itself failed, the TSV/text
  # receipt and the outer container exit status still preserve the failure.
  "$PY" -B - <<'PY'
import json,os,pathlib
run=pathlib.Path(os.environ['SFT_RUN_DIR']); path=run/'experiment.json'
try: meta=json.loads(path.read_text()) if path.is_file() else {}
except (ValueError,OSError): meta={}
phase=os.environ['SFT_FAILURE_PHASE']; before_training=phase=='preparation'
meta.setdefault('schema_version',1); meta.setdefault('experiment_id',run.name)
meta.setdefault('intended_workload_kind','real_sft')
meta['status']='preparation_failed' if before_training else 'artifact_finalization_failed'
if before_training: meta['workload_kind']='preparation_validation'
meta['failure']={'phase':phase,'stage':os.environ['SFT_FAILURE_STAGE'],'exit_code':int(os.environ['SFT_FAILURE_STATUS']),
                 'training_launched':not before_training,'receipt':'entry-failure.txt'}
meta['preparation']={'status':'failed' if before_training else 'completed','training_launched':not before_training,
                     'lifecycle_artifact':'preparation-lifecycle.tsv','source_identity_prefix':'entry-source-'}
for field,name in [('git_commit','entry-source-head.txt'),('git_branch','entry-source-branch.txt')]:
    source=run/name
    if source.is_file(): meta['preparation'][field]=source.read_text().strip()
path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
PY
  exit "$failure_status"
}
trap 'on_error "$?"' ERR
record_stage source_identity
cd "$REPO"
export GIT_OPTIONAL_LOCKS=0
git --no-optional-locks -C "$REPO" rev-parse HEAD > "$RUN/entry-source-head.txt"
git --no-optional-locks -C "$REPO" branch --show-current > "$RUN/entry-source-branch.txt"
git --no-optional-locks -C "$REPO" status --porcelain=v1 --untracked-files=all > "$RUN/entry-source-status.txt"
git --no-optional-locks -C "$REPO" diff --binary HEAD > "$RUN/entry-source-dirty.patch"
sha256sum "$REPO/tasks/vla/train_lingbotvla.py" "$REPO/lingbotvla/utils/training_instrumentation.py" "$HARNESS/rtx5k_prepare_sft_run.py" "$HARNESS/rtx5k_sft_container_entry.sh" > "$RUN/entry-source-sha256.txt"
record_stage interpreter_check
[[ -x "$PY" ]] || { printf 'Container cannot execute prepared Python entry\n' >&2; false; }
"$PY" -B -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Prepared task venv requires Python 3.12, got {sys.version}"; print(sys.version)' > "$RUN/interpreter-version.txt"
record_stage runtime_libraries
unset PYTHONPATH PIP_CONSTRAINT PIP_BUILD_CONSTRAINT
export PATH="$ROOT/environment/venv/bin:$PATH"
site_packages=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
runtime_libs=("$site_packages/torch/lib")
# PyAV wheels bundle versioned FFmpeg shared libraries needed by TorchCodec.
# These are task-venv dependencies; keep them ahead of the base-image libraries.
for library in "$site_packages/av.libs" "$site_packages/av/.libs" "$ROOT/environment/ffmpeg/lib"; do
  if [[ -d "$library" ]]; then runtime_libs+=("$library"); fi
done
for library in "$site_packages"/nvidia/*/lib; do
  if [[ -d "$library" ]]; then runtime_libs+=("$library"); fi
done
export LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${runtime_libs[*]}"):/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ptxas="$site_packages/triton/backends/nvidia/bin/ptxas"
if [[ -x "$ptxas" ]]; then export TRITON_PTXAS_PATH="$ptxas"; fi
prep=("$PY" "$HARNESS/rtx5k_prepare_sft_run.py" --run-dir "$SFT_RUN_DIR"
  --world-size "$SFT_WORLD_SIZE" --optimizer-mode "$SFT_OPTIMIZER_MODE" --kind "$SFT_KIND"
  --memory-policy "$SFT_MEMORY_POLICY" --data-manifest "$SFT_DATA_MANIFEST" --pair-id "$SFT_PAIR_ID"
  --role "$SFT_ROLE" --warmup-steps "$LINGBOT_MEASURE_WARMUP_STEPS" --steady-steps "$SFT_STEADY_STEPS")
if [[ -n "${SFT_WEIGHTS_MANIFEST:-}" ]]; then prep+=(--weights-manifest "$SFT_WEIGHTS_MANIFEST"); fi
record_stage prepare_run
"${prep[@]}"
record_stage tool_versions
if [[ "$SFT_KIND" == nsys ]]; then nsys --version > "$SFT_RUN_DIR/profiling/tool-version.txt"; fi
if [[ "$SFT_KIND" == torch ]]; then "$PY" -c 'import torch; print(torch.__version__)' > "$SFT_RUN_DIR/profiling/tool-version.txt"; fi
if [[ -n "${TRITON_PTXAS_PATH:-}" ]]; then "$TRITON_PTXAS_PATH" --version > "$SFT_RUN_DIR/ptxas-version.txt"; fi
record_stage preparation_complete
date -u +%FT%TZ > "$RUN/preparation-finished-utc.txt"
printf '0\n' > "$RUN/preparation-exit-code.txt"
printf '%s\tpreparation\tpreparation_complete\tcompleted\texit=0\n' "$(date -u +%FT%TZ)" >> "$RUN/preparation-lifecycle.tsv"
trap - ERR
CURRENT_PHASE=training
date -u +%FT%TZ > "$SFT_RUN_DIR/training-started-utc.txt"
set +e
bash "$SFT_RUN_DIR/command.sh" > "$SFT_RUN_DIR/train.log" 2>&1
STATUS=$?
set -e
date -u +%FT%TZ > "$SFT_RUN_DIR/training-finished-utc.txt"
printf '%s\n' "$STATUS" > "$SFT_RUN_DIR/training-exit-code.txt"
CURRENT_PHASE=finalization
trap 'on_error "$?"' ERR
record_stage finalization
if [[ "$SFT_KIND" == nsys && -f "$SFT_RUN_DIR/profiling/trace.nsys-rep" ]]; then
  if nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv --output - "$SFT_RUN_DIR/profiling/trace.nsys-rep" > "$SFT_RUN_DIR/profiling/stats.csv" 2> "$SFT_RUN_DIR/profiling/stats-error.log"; then STATS_STATUS=0; else STATS_STATUS=$?; fi
  printf '%s\n' "$STATS_STATUS" > "$SFT_RUN_DIR/profiling/stats-exit-code.txt"
fi
export SFT_TRAINING_STATUS="$STATUS"
"$PY" - <<'PY'
import hashlib,json,os,pathlib,shutil
run=pathlib.Path(os.environ['SFT_RUN_DIR']); path=run/'experiment.json'
meta=json.loads(path.read_text()); status=int(os.environ['SFT_TRAINING_STATUS'])
meta.update(status='completed' if status==0 else 'failed',finished_utc=(run/'training-finished-utc.txt').read_text().strip(),
            training_started_utc=(run/'training-started-utc.txt').read_text().strip(),training_exit_code=status)
meta['preparation']={'status':'completed','training_launched':True,
                     'started_utc':(run/'preparation-started-utc.txt').read_text().strip(),
                     'finished_utc':(run/'preparation-finished-utc.txt').read_text().strip(),
                     'exit_code':int((run/'preparation-exit-code.txt').read_text().strip()),
                     'intent_artifact':'preparation-intent.json','lifecycle_artifact':'preparation-lifecycle.tsv'}
actual=run/'output/lingbotvla_cli.yaml'
if actual.is_file():
    shutil.copyfile(actual,run/'runtime-effective-config.yaml')
    meta['artifacts'].update(runtime_effective_config='runtime-effective-config.yaml',runtime_effective_config_sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
if os.environ['SFT_KIND'] in ('nsys','torch'):
    profile=meta['artifacts']['profiling'][0]
    profile.update(raw_present=(run/profile['raw_artifact']).is_file(),tool_version=(run/'profiling/tool-version.txt').read_text().strip())
payloads=[]
output=run/'output'
if output.exists():
    for candidate in output.rglob('*'):
        if candidate.is_file() and (candidate.suffix in ('.distcp','.pt','.pth','.safetensors','.bin') or candidate.name=='.metadata'):
            payloads.append({'path':str(candidate.relative_to(run)),'size_bytes':candidate.stat().st_size})
meta['checkpoint_inventory']={'payloads':payloads,'payload_bytes':sum(p['size_bytes'] for p in payloads),'unexpected_payload':bool(payloads)}
(run/'checkpoint-inventory.json').write_text(json.dumps(meta['checkpoint_inventory'],indent=2)+'\n')
path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
PY
exit "$STATUS"
