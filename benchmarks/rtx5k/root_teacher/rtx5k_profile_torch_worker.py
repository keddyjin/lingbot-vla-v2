#!/usr/bin/env python3
"""Profile one complete real SFT update on rank zero; preserve training math."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

REMOTE_ROOT = Path('/root/keddyj/LingBotVLA-Test')
TARGET = REMOTE_ROOT / 'lingbot-vla-v2-sft-opt-github/tasks/vla/train_lingbotvla.py'


def profile_directory():
    run = Path(os.environ['SFT_RUN_DIR'])
    root = REMOTE_ROOT.resolve()
    if not run.is_absolute() or not run.resolve().is_relative_to(root / 'runs'):
        raise ValueError('SFT_RUN_DIR must be an absolute run directory under authorized ROOT/runs')
    output = (run / 'profiling').resolve()
    if not output.is_relative_to(root / 'runs'):
        raise ValueError('Profiling output escapes authorized ROOT/runs')
    output.mkdir(parents=True, exist_ok=True)
    return output


def metric(event, modern, legacy=None):
    value = getattr(event, modern, None)
    if value is None and legacy:
        value = getattr(event, legacy, None)
    return float(value) if value is not None else None


def export_profile(profile, output, step, metadata, *, complete):
    prefix = f'torch-rank0-step{step}'
    status = dict(metadata, complete_optimizer_update=complete, observation_only=True,
                  time_unit='microseconds', timing_includes_profiler_overhead=True,
                  grouping='operator,input_shapes,top12stackframes')
    try:
        profile.export_chrome_trace(str(output / f'{prefix}.trace.json'))
        events = profile.key_averages(group_by_input_shape=True, group_by_stack_n=12)
        rows = []
        for event in events:
            rows.append({'key': str(event.key), 'count': int(event.count),
                         'cpu_time_total_us': metric(event, 'cpu_time_total'),
                         'self_cpu_time_total_us': metric(event, 'self_cpu_time_total'),
                         'device_time_total_us': metric(event, 'device_time_total', 'cuda_time_total'),
                         'self_device_time_total_us': metric(event, 'self_device_time_total', 'self_cuda_time_total'),
                         'device_type': str(getattr(event, 'device_type', 'unknown')),
                         'input_shapes': getattr(event, 'input_shapes', None),
                         'stack': getattr(event, 'stack', None)})
        rows.sort(key=lambda row: row['self_cpu_time_total_us'] or 0, reverse=True)
        with (output / f'{prefix}.key-averages.json').open('x', encoding='utf-8') as stream:
            json.dump({'metadata': status, 'rows': rows}, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        fields = ['key', 'count', 'cpu_time_total_us', 'self_cpu_time_total_us',
                  'device_time_total_us', 'self_device_time_total_us', 'device_type',
                  'input_shapes', 'stack']
        with (output / f'{prefix}.key-averages.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'input_shapes': json.dumps(row['input_shapes']),
                                 'stack': json.dumps(row['stack'], ensure_ascii=False)})
        status.update(export_status='exported', key_average_rows=len(rows),
                      rows_with_stack=sum(bool(row['stack']) for row in rows))
        status['stack_exports'] = {}
        for name, stack_metric in (('cpu', 'self_cpu_time_total'), ('cuda', 'self_cuda_time_total')):
            path = output / f'{prefix}.{name}-stacks.txt'
            try:
                profile.export_stacks(str(path), stack_metric)
                status['stack_exports'][name] = {'status': 'exported' if path.stat().st_size else 'empty', 'bytes': path.stat().st_size}
            except (AttributeError, RuntimeError, ValueError, AssertionError) as error:
                status['stack_exports'][name] = {'status': 'unsupported_or_unavailable', 'error': repr(error)}
        status['artifacts'] = {path.name: {'bytes': path.stat().st_size,
                                          'sha256': file_sha(path)}
                               for path in output.glob(prefix + '.*') if path.is_file()}
    except Exception as error:
        status.update(export_status='failed', export_error=repr(error))
        raise
    finally:
        with (output / f'{prefix}.status.json').open('x', encoding='utf-8') as stream:
            json.dump(status, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import torch
    import torch.distributed as dist
    from lingbotvla.utils.training_instrumentation import TrainingInstrumentation

    start = int(os.environ['LINGBOT_MEASURE_WARMUP_STEPS']) + 1
    if start < 1 or not os.environ.get('LINGBOT_MEASURE_JSONL_DIR'):
        raise ValueError('Nonnegative warmup and real optimizer-update JSONL measurement are required')
    original_start = TrainingInstrumentation.start_step
    original_finish = TrainingInstrumentation.finish_step
    original_completed = TrainingInstrumentation.completed_update
    capture = {'profiler': None, 'export_attempted': False, 'active': False}

    def start_step(self, step, epoch, epoch_step):
        if step == start:
            torch.cuda.synchronize()
            dist.barrier()
            if self.rank == 0:
                output = profile_directory()
                if any(output.glob(f'torch-rank0-step{start}.*')):
                    raise FileExistsError('Refusing to overwrite an existing Torch profile')
                experiment_path = output.parent / 'experiment.json'
                experiment = json.loads(experiment_path.read_text()) if experiment_path.exists() else {}
                metadata = {'run_id': os.environ['LINGBOT_MEASURE_RUN_ID'], 'rank': 0,
                            'world_size': self.world_size, 'optimizer_step': start,
                            'torch_version': str(torch.__version__),
                            'git_commit': experiment.get('git', {}).get('commit'),
                            'helper_sha256': file_sha(Path(__file__)),
                            'activities': ['CPU', 'CUDA'], 'record_shapes': True,
                            'with_stack': True, 'profile_memory': False,
                            'experimental_config': {'verbose': True, 'enable_cuda_sync_events': True, 'profile_all_threads': True},
                            'scope': 'rank-zero process, complete real optimizer update plus boundary barriers; profiling only'}
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, with_stack=True, profile_memory=False,
                    experimental_config=torch.profiler._ExperimentalConfig(
                        verbose=True, enable_cuda_sync_events=True, profile_all_threads=True))
                capture.update(profiler=profiler, output=output, metadata=metadata)
                profiler.__enter__()
                capture['active'] = True
                profiler.add_metadata('lingbot_run_id', metadata['run_id'])
                profiler.add_metadata('lingbot_git_commit', metadata['git_commit'] or 'unavailable')
                profiler.add_metadata('lingbot_profile_scope', metadata['scope'])
                print(f'TORCH_CAPTURE_START rank=0 optimizer_step={start}', flush=True)
            dist.barrier()
        return original_start(self, step, epoch, epoch_step)

    def completed_update(self, original_step_time_s, *, visual_step, profile_step):
        return original_completed(self, original_step_time_s, visual_step=visual_step,
                                  profile_step=self._range_step == start)

    def finish_step(self, **kwargs):
        step = self._range_step
        result = original_finish(self, **kwargs)
        if step == start:
            dist.barrier()
            if self.rank == 0:
                capture['export_attempted'] = True
                capture['profiler'].__exit__(None, None, None)
                capture['active'] = False
                export_profile(capture['profiler'], capture['output'], start,
                               capture['metadata'], complete=True)
                print(f'TORCH_CAPTURE_STOP rank=0 optimizer_step={start}', flush=True)
            dist.barrier()
        return result

    TrainingInstrumentation.start_step = start_step
    TrainingInstrumentation.completed_update = completed_update
    TrainingInstrumentation.finish_step = finish_step
    try:
        sys.argv[0] = str(TARGET)
        runpy.run_path(str(TARGET), run_name='__main__')
    finally:
        # Preserve a partial trace on a training exception; do not call a new
        # distributed barrier during failure cleanup.
        if capture['active'] and not capture['export_attempted']:
            try:
                capture['profiler'].__exit__(*sys.exc_info())
                export_profile(capture['profiler'], capture['output'], start,
                               capture['metadata'], complete=False)
            except Exception as error:
                print(f'TORCH_PARTIAL_EXPORT_ERROR {error!r}', file=sys.stderr, flush=True)
        TrainingInstrumentation.start_step = original_start
        TrainingInstrumentation.completed_update = original_completed
        TrainingInstrumentation.finish_step = original_finish


if __name__ == '__main__':
    main()
