#!/usr/bin/env python3
"""Default-off worker CPU affinity. Import before Torch; no GPU/memory binding."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path('/root/keddyj/LingBotVLA-Test')
POOLS = {0: set(range(0, 64)) | set(range(128, 192)),
         1: set(range(64, 128)) | set(range(192, 256))}
CONTROL = 'LINGBOT_CPU_AFFINITY'
RECEIPTS = 'LINGBOT_CPU_AFFINITY_RECEIPT_DIR'
CURATED_ENV = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'SFT_RUN_DIR',
               'SFT_KIND', 'SFT_WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'NCCL_PROTO',
               'SFT_NCCL_P2P_LEVEL_CONTROL', 'SFT_NCCL_P2P_DISABLE_CONTROL',
               'NCCL_P2P_LEVEL', 'NCCL_P2P_DISABLE', 'NCCL_IGNORE_DISABLED_P2P',
               'SFT_NCCL_MIN_CTAS_CONTROL', 'NCCL_MIN_CTAS', 'NCCL_MIN_NCHANNELS', 'NCCL_MIN_NRINGS',
               'NCCL_MAX_CTAS', 'NCCL_MAX_NCHANNELS', 'NCCL_MAX_NRINGS',
               'LINGBOT_BENCHMARK_SKIP_CHECKPOINT', 'LINGBOT_BENCHMARK_STOP_STEPS',
               'LINGBOT_GC_POLICY', 'LINGBOT_FLEX_BWD_TILE', CONTROL)


def utc():
    return datetime.now(timezone.utc).isoformat()


def policy_from_environment(environment):
    mode = environment.get(CONTROL, 'off')
    if mode not in ('off', 'gpu-numa'):
        raise ValueError(f'{CONTROL} must be off or gpu-numa; got {mode!r}')
    policy = {'mode': mode, 'numa_node': None, 'requested_cpus': None}
    if mode == 'off':
        return policy
    if int(environment.get('WORLD_SIZE', '-1')) != 8 or int(environment.get('LOCAL_WORLD_SIZE', '-1')) != 8:
        raise ValueError('gpu-numa candidate requires exactly one 8-rank node')
    local_rank = int(environment.get('LOCAL_RANK', '-1'))
    if local_rank not in range(8) or int(environment.get('RANK', '-1')) != local_rank:
        raise ValueError('gpu-numa candidate requires RANK == LOCAL_RANK in 0..7')
    if environment.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('This candidate requires reviewed GPU order 0,1,2,3,4,5,6,7')
    node = 0 if local_rank < 4 else 1
    policy.update(numa_node=node, requested_cpus=sorted(POOLS[node]))
    return policy


def numa_maps_summary(text):
    pages = Counter(); kib = Counter(); line_count = 0; policy_counts = Counter()
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 2:
            continue
        line_count += 1; policy_counts[tokens[1]] += 1
        page_kib = next((int(token.split('=', 1)[1]) for token in tokens if token.startswith('kernelpagesize_kB=')), None)
        for token in tokens:
            key, sep, value = token.partition('=')
            if sep and key.startswith('N') and key[1:].isdigit() and value.isdigit():
                pages[key] += int(value)
                if page_kib is not None:
                    kib[key] += int(value) * page_kib
    return {'mapping_lines': line_count, 'policy_line_counts': dict(policy_counts),
            'node_resident_pages': dict(pages), 'node_resident_KiB_known_page_size': dict(kib),
            'limits': 'Snapshot of resident pages; not memory traffic, migration or per-step remote-memory rate.'}


def observe_process(pid, *, os_api=os, proc_root=Path('/proc')):
    row = {'pid': pid, 'observed_utc': utc()}
    try:
        row['actual_cpus'] = sorted(os_api.sched_getaffinity(pid))
    except (AttributeError, OSError) as error:
        row['actual_cpus'] = None; row['affinity_read_error'] = type(error).__name__
    process_dir = proc_root / str(pid)
    try:
        # The command name can contain spaces/parentheses. Fields after its
        # final ')' start at stat field 3; starttime is field 22.
        stat_fields = (process_dir / 'stat').read_text().rpartition(')')[2].split()
        row['process_start_ticks'] = int(stat_fields[19])
    except (OSError, ValueError, IndexError):
        row['process_start_ticks'] = None
    try:
        row['thread_ids'] = sorted(int(path.name) for path in (process_dir / 'task').iterdir())
    except (OSError, ValueError):
        row['thread_ids'] = None
    thread_masks = {}; unreadable_threads = []
    for tid in row['thread_ids'] or []:
        try:
            mask = tuple(sorted(os_api.sched_getaffinity(tid)))
        except (AttributeError, OSError):
            unreadable_threads.append(tid); continue
        thread_masks.setdefault(mask, []).append(tid)
    row['thread_affinity_groups'] = [{'actual_cpus': list(mask), 'thread_ids': tids}
                                      for mask, tids in sorted(thread_masks.items())]
    row['unreadable_thread_affinities'] = unreadable_threads
    try:
        row['numa_maps'] = numa_maps_summary((process_dir / 'numa_maps').read_text())
    except OSError as error:
        row['numa_maps'] = {'available': False, 'read_error': type(error).__name__}
    return row


def apply_policy(policy, *, os_api=os, before=None):
    if policy['mode'] == 'off':
        return {'affinity_changed': False, 'requested_cpus': None}
    requested = set(policy['requested_cpus'])
    allowed = set(os_api.sched_getaffinity(0))
    if not requested.issubset(allowed):
        raise RuntimeError(f'Full socket CPU pool not allowed; missing CPUs {sorted(requested - allowed)}')
    if before is None or before['thread_ids'] is None or len(before['thread_ids']) != 1:
        raise RuntimeError('Bind before importing Torch: exactly one current worker thread must be observable')
    os_api.sched_setaffinity(0, requested)
    actual = set(os_api.sched_getaffinity(0))
    if actual != requested:
        raise RuntimeError(f'Kernel affinity differs from requested full socket pool: {sorted(actual)}')
    return {'affinity_changed': actual != allowed, 'requested_cpus': sorted(requested), 'actual_cpus_after_set': sorted(actual)}


def confined(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError('Candidate path must remain under /root/keddyj/LingBotVLA-Test')
    return resolved


def save_receipt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # One file per worker PID; no cross-rank append or temporary files elsewhere.
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')


def launch_target(target, target_args):
    sys.argv = [str(target), *target_args]
    # Match direct Python script import resolution, including when target is a
    # profiling wrapper. Its own inner runpy/argv changes retain normal behavior.
    if sys.path:
        sys.path[0] = str(target.parent)
    else:
        sys.path.append(str(target.parent))
    runpy.run_path(str(target), run_name='__main__')


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: rtx5k_numa_worker.py TARGET.py [unchanged target arguments...]')
    target = confined(sys.argv[1]); target_args = sys.argv[2:]
    if not target.is_file():
        raise FileNotFoundError(target)
    if RECEIPTS not in os.environ:
        raise ValueError(f'{RECEIPTS} is required for both matched off/on runs')
    receipt_dir = confined(os.environ[RECEIPTS])
    if receipt_dir.is_relative_to(ROOT / 'lingbot-vla-v2-sft-opt-github'):
        raise ValueError('Do not write candidate receipts into the training repository')
    pid = os.getpid(); rank = os.environ.get('RANK', 'unset')
    if not (rank.isdigit() or rank == 'unset'):
        raise ValueError('Invalid RANK receipt filename')
    receipt = receipt_dir / f'rank{rank}-pid{pid}.json'
    payload = {'schema_version': 1, 'created_utc': utc(), 'rank': rank, 'pid': pid,
               'candidate': 'S08 shared full socket CPU pool', 'status': 'entry',
               'environment': {key: os.environ[key] for key in CURATED_ENV if key in os.environ},
               'communication_controls': {**{key + '_is_set': key in os.environ for key in ('NCCL_PROTO', 'NCCL_P2P_LEVEL', 'NCCL_P2P_DISABLE', 'NCCL_MIN_CTAS', 'NCCL_MIN_NCHANNELS', 'NCCL_MIN_NRINGS', 'NCCL_MAX_CTAS', 'NCCL_MAX_NCHANNELS', 'NCCL_MAX_NRINGS')}, **{key: os.environ.get(key) for key in ('NCCL_PROTO', 'NCCL_P2P_LEVEL', 'NCCL_P2P_DISABLE', 'NCCL_MIN_CTAS', 'NCCL_MIN_NCHANNELS', 'NCCL_MIN_NRINGS', 'NCCL_MAX_CTAS', 'NCCL_MAX_NCHANNELS', 'NCCL_MAX_NRINGS')}},
               'default_control': 'off', 'memory_binding': 'none; no mbind/set_mempolicy',
               'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'target': str(target), 'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'target_arguments': target_args, 'before': observe_process(pid)}
    # Capture the existing harness source/shape identity without changing it.
    run_dir = os.environ.get('SFT_RUN_DIR')
    if run_dir:
        experiment = confined(run_dir) / 'experiment.json'
        if experiment.is_file():
            metadata = json.loads(experiment.read_text())
            payload['prepared_source'] = metadata.get('git')
            payload['prepared_batch_and_stop'] = metadata.get('training')
            payload['prepared_gpu_mapping'] = metadata.get('docker', {}).get('cuda_devices')
            payload['prepared_experiment_sha256'] = hashlib.sha256(experiment.read_bytes()).hexdigest()
    save_receipt(receipt, payload)
    try:
        policy = policy_from_environment(os.environ)
        payload['policy'] = policy
        payload['application'] = apply_policy(policy, before=payload['before'])
        payload['after_binding'] = observe_process(pid)
        payload['status'] = 'target_starting'; save_receipt(receipt, payload)
    except BaseException as error:
        payload.update(status='binding_failed', error_type=type(error).__name__, finished_utc=utc())
        save_receipt(receipt, payload)
        raise
    try:
        launch_target(target, target_args)
    except SystemExit as error:
        payload.update(status='target_exit', exit_code=error.code if isinstance(error.code, int) else 0 if error.code is None else 1)
        raise
    except BaseException as error:
        payload.update(status='target_exception', error_type=type(error).__name__)
        raise
    else:
        payload.update(status='target_exit', exit_code=0)
    finally:
        payload.update(finished_utc=utc(), at_exit=observe_process(pid))
        # Keep the target's original exception/exit code if final receipt I/O
        # fails while another exception is already propagating.
        pending_error = sys.exc_info()[0] is not None
        try:
            save_receipt(receipt, payload)
        except OSError as error:
            if not pending_error:
                raise
            print(f'NUMA_RECEIPT_FINALIZE_FAILED {type(error).__name__}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
