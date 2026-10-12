#!/usr/bin/env python3
"""Read one real-SFT run; write independent analysis without editing evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

REMOTE_ROOT = Path('/root/keddyj/LingBotVLA-Test')
LOSSES = ('Loss', 'VLA_Loss', 'Depth_Loss', 'Future_Depth_Loss',
          'FutureVideo_Loss', 'SeqWise_Loss', 'RouterZ_Loss')
NUMBER = re.compile(r'[-+]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)', re.I)
STEP = re.compile(r'\bStep (\d+)/(\d+), Epoch (\d+), (.*)')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_path(run, value):
    path = Path(value)
    path = path if path.is_absolute() else run / path
    path = path.resolve()
    if not path.is_relative_to(run):
        raise ValueError(f'Run artifact escapes its directory: {path}')
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value, minimum=1):
    return type(value) is int and value >= minimum


def duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    mean = statistics.mean(values)
    half = len(values) // 2
    first = statistics.median(values[:half]) if half else None
    last = statistics.median(values[-half:]) if half else None
    return {'count': len(values), 'sum_s': sum(values), 'mean_s': mean,
            'median_s': statistics.median(values),
            'p95_s': ordered[max(0, math.ceil(.95 * len(values)) - 1)],
            'p95_method': 'nearest rank', 'min_s': min(values), 'max_s': max(values),
            'population_std_s': statistics.pstdev(values),
            'cv': statistics.pstdev(values) / mean,
            'first_half_median_s': first, 'second_half_median_s': last,
            'half_median_drift_fraction': last / first - 1 if first else None,
            'half_window_note': 'Equal floor(n/2) halves; odd center omitted'}


def log_metrics(path):
    rows, errors, compiler = [], [], []
    previous_step = 0
    for line_number, raw in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        line = ANSI.sub('', raw)
        match = STEP.search(line)
        if match:
            metrics = {}
            for part in match.group(4).split(','):
                pieces = part.strip().split()
                if len(pieces) != 2:
                    continue
                key, token = pieces
                if key in ('StepTime', 'Depth_Forward_Time'):
                    token = token.removesuffix('s')
                if NUMBER.fullmatch(token):
                    metrics[key] = float(token)
            previous_step = int(match.group(1))
            rows.append({'global_step': previous_step, 'steps_per_epoch': int(match.group(2)),
                         'epoch': int(match.group(3)), 'line': line_number, 'metrics': metrics})
        if re.search(r'recompile_limit|cache_size_limit|recompiling function|recompilation triggered', line, re.I):
            compiler.append({'line': line_number, 'after_logged_step': previous_step,
                             'text': line[-600:]})
        if re.search(r'OutOfMemoryError|CUDA out of memory|FloatingPointError|ChildFailedError', line):
            errors.append({'line': line_number, 'text': line[-600:]})
    return rows, errors, compiler


def timestamp(value):
    if not value:
        return None
    return datetime.fromisoformat(value.strip().replace('Z', '+00:00')).timestamp()


def wallclock(run, meta, updates):
    def file_or_meta(name, key):
        path = run / name
        return path.read_text().strip() if path.is_file() else meta.get(key)
    started = file_or_meta('training-started-utc.txt', 'training_started_utc')
    finished = file_or_meta('training-finished-utc.txt', 'finished_utc')
    result = {'training_started_utc': started, 'training_finished_utc': finished,
              'training_command_wallclock_s': None, 'pre_first_update_startup_s': None,
              'after_last_update_shutdown_s': None,
              'sum_max_rank_update_s': sum(row['duration_s'] for row in updates),
              'startup_definition': 'command start to estimated earliest first update timing start',
              'startup_cache_temperature': 'unknown; this is elapsed startup, not proof of a cold cache',
              'clock_limit': 'UTC files have second resolution; update starts estimated from wall timestamp minus perf_counter duration'}
    if started and finished:
        result['training_command_wallclock_s'] = timestamp(finished) - timestamp(started)
    if updates:
        first = updates[0]['rank_timing_records']
        last = updates[-1]['rank_timing_records']
        if started and all(integer(row.get('recorded_at_unix_ns')) for row in first):
            earliest = min(row['recorded_at_unix_ns'] / 1e9 - row['end_to_end_step_time_s'] for row in first)
            result['pre_first_update_startup_s'] = earliest - timestamp(started)
        if finished and all(integer(row.get('recorded_at_unix_ns')) for row in last):
            result['after_last_update_shutdown_s'] = timestamp(finished) - max(row['recorded_at_unix_ns'] / 1e9 for row in last)
    return result


def analyze(run, *, max_cv=.10, max_half_drift=.05, divergence_ratio=5.):
    meta_path = run / 'experiment.json'
    meta = read_json(meta_path)
    artifacts = meta.get('artifacts', {})
    config_path = source_path(run, artifacts.get('effective_config', 'effective_config.json'))
    config = read_json(config_path)
    planned = config.get('train', {})
    issues, warnings = [], []
    if meta.get('schema_version') != 1:
        issues.append('Unsupported experiment schema')
    if meta.get('workload_kind') != 'real_sft':
        issues.append('Workload is not recorded as real SFT')
    world_size = meta.get('training', {}).get('world_size')
    if type(world_size) is not int or world_size not in (4, 8):
        raise ValueError('This RTX task requires recorded world_size 4 or 8')
    global_batch_size = world_size * 32
    BATCH = {'world_size': world_size, 'micro_batch_size': 32,
             'gradient_accumulation_steps': 1, 'global_batch_size': global_batch_size}
    expected_steps = meta.get('training', {}).get('stop_after_steps')
    if not integer(expected_steps) or expected_steps > 1000 or expected_steps > planned.get('max_steps', 0):
        raise ValueError('Invalid independent benchmark stop count or LR horizon')
    if meta.get('training', {}).get('lr_schedule_max_steps') != planned.get('max_steps'):
        issues.append('Recorded LR horizon differs from effective config')
    if planned.get('enable_resume') is not False or planned.get('load_checkpoint_path') is not None:
        issues.append('Fresh no-resume benchmark configuration required')
    for field in ('save_steps', 'save_epochs'):
        if planned.get(field) != 0:
            issues.append(f'Checkpoint writes enabled by {field}')
    for field in ('save_hf_weights', 'async_save_hf_weights'):
        if planned.get(field) is not False:
            issues.append(f'HF checkpoint writes enabled by {field}')
    if not integer(expected_steps):
        raise ValueError('Config max_steps must be a positive integer')
    warmup = meta.get('measurement', {}).get('warmup_steps')
    if not integer(warmup, 0):
        raise ValueError('Manifest warmup_steps must be a nonnegative integer')
    batch_checks = []
    for section_name, section in [('training', meta.get('training', {})),
                                  ('fair_comparison', meta.get('fair_comparison', {})),
                                  ('effective_config.train', planned)]:
        for key, expected in BATCH.items():
            # world_size is supplied by torchrun, accumulation defaults to one.
            if section_name == 'effective_config.train' and key == 'world_size':
                continue
            observed = section.get(key, 1 if key == 'gradient_accumulation_steps' else None)
            good = type(observed) is int and observed == expected
            batch_checks.append({'source': section_name, 'field': key, 'value': observed, 'expected': expected, 'passed': good})
    if not all(row['passed'] for row in batch_checks):
        issues.append('Manifest/config does not match recorded world4-or8/micro32/accum1/global32*world')
    actual_config = None
    runtime_value = artifacts.get('runtime_effective_config')
    if runtime_value:
        runtime_path = source_path(run, runtime_value)
        if runtime_path.is_file():
            try:
                try:
                    import yaml
                except ImportError:
                    # JSON is a YAML subset used by CPU fixtures. Real YAML
                    # still requires PyYAML rather than silently skipping it.
                    actual_config = read_json(runtime_path)
                else:
                    actual_config = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))
                if not isinstance(actual_config, dict) or not isinstance(actual_config.get('train'), dict):
                    raise ValueError('Runtime config has no train mapping')
                runtime_train = actual_config['train']
                if runtime_train.get('max_steps') != planned.get('max_steps'):
                    issues.append('Actual runtime LR horizon differs from the plan')
                if runtime_train.get('enable_resume') is not False or runtime_train.get('load_checkpoint_path') is not None:
                    issues.append('Actual runtime is not a fresh no-resume run')
                if any(runtime_train.get(key) != 0 for key in ('save_steps', 'save_epochs')):
                    issues.append('Actual runtime DCP checkpoint writes enabled')
                if any(runtime_train.get(key) is not False for key in ('save_hf_weights', 'async_save_hf_weights')):
                    issues.append('Actual runtime HF checkpoint writes enabled')
                for key, expected in BATCH.items():
                    observed = actual_config['train'].get(key, 1 if key == 'gradient_accumulation_steps' else None)
                    # Some dumps omit derived world_size; rank timing records verify it.
                    if key == 'world_size' and observed is None:
                        continue
                    good = type(observed) is int and observed == expected
                    batch_checks.append({'source': 'runtime_effective_config.train', 'field': key, 'value': observed, 'expected': expected, 'passed': good})
                    if not good:
                        issues.append(f'Runtime config batch mismatch: {key}')
            except (ValueError, TypeError) as error:
                issues.append(f'Runtime YAML validation failed: {error}')
        else:
            issues.append('Runtime config artifact missing')
    inventory_path = run / 'checkpoint-inventory.json'
    if inventory_path.is_file():
        inventory = read_json(inventory_path)
        if inventory.get('unexpected_payload') is True or inventory.get('payload_bytes', 0) != 0:
            issues.append('Unexpected checkpoint payload in benchmark output')
    else:
        issues.append('Checkpoint inventory receipt is missing')
    expected_hash = artifacts.get('effective_config_sha256')
    if expected_hash and expected_hash != sha(config_path):
        issues.append('Effective config SHA differs from manifest')
    if meta.get('training', {}).get('gradient_checkpointing') != planned.get('enable_gradient_checkpointing'):
        warnings.append('Manifest gradient_checkpointing differs from effective config; use actual memory-policy evidence')
    directory = source_path(run, artifacts.get('timings_dir') or 'timings')
    grouped = {}
    for path in sorted(directory.glob('rank*.jsonl')):
        for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            step, rank = row.get('optimizer_step'), row.get('rank')
            if not integer(step) or type(rank) is not int or rank not in range(world_size):
                raise ValueError(f'{path.name}:{line_number}: invalid step/rank')
            if rank in grouped.setdefault(step, {}):
                raise ValueError(f'{path.name}:{line_number}: duplicate rank/update')
            grouped[step][rank] = row
    updates = []
    for step, ranks in sorted(grouped.items()):
        if set(ranks) != set(range(world_size)):
            issues.append(f'Update {step}: incomplete selected-rank set')
            continue
        rows = [ranks[rank] for rank in range(world_size)]
        expected_phase = 'warmup' if step <= warmup else 'steady'
        good = True
        global_steps = {row.get('global_step') for row in rows}
        if len(global_steps) != 1 or not integer(rows[0].get('global_step')):
            good = False
        if planned.get('enable_resume') is False and rows[0].get('global_step') != step:
            good = False
        for row in rows:
            checks = {'schema_version': 1, 'run_id': meta.get('experiment_id'), 'world_size': world_size, 'data_parallel_size': world_size,
                      'micro_batch_size': 32, 'global_batch_size': global_batch_size, 'accumulation_steps': 1,
                      'micro_batches': 1, 'local_samples': 32, 'samples': global_batch_size, 'batch_complete': True,
                      'boundary': 'before_data_fetch_to_after_logging_and_checkpoint',
                      'timing_clock': 'perf_counter', 'end_sync': True, 'phase': expected_phase}
            for key, expected in checks.items():
                if type(row.get(key)) is not type(expected) or row.get(key) != expected:
                    good = False
            if not duration(row.get('end_to_end_step_time_s')):
                raise ValueError(f'Update {step}: nonfinite/nonpositive timing')
            if not duration(row.get('original_step_time_s')):
                good = False
        for key in ('checkpoint_step', 'terminal_step', 'epoch_terminal_step', 'profile_step'):
            if any(type(row.get(key)) is not bool for row in rows) or len({row.get(key) for row in rows}) != 1:
                good = False
        if not good:
            issues.append(f'Update {step}: inconsistent batch, timing, run identity, ordinal or flags')
        excluded = []
        if expected_phase == 'warmup':
            excluded.append('warmup')
        for flag, policy in [('checkpoint_step', 'checkpoint_steps_policy'), ('terminal_step', 'terminal_steps_policy')]:
            if rows[0].get(flag) and meta.get('measurement', {}).get(policy) == 'exclude':
                excluded.append(flag)
        if any(row.get('profile_step') for row in rows) or meta.get('measurement', {}).get('profiling_enabled') is True:
            excluded.append('profiling')
        times = [row['end_to_end_step_time_s'] for row in rows]
        updates.append({'optimizer_step': step, 'global_step': rows[0].get('global_step'),
                        'phase': expected_phase, 'duration_s': max(times), 'rank_durations_s': times,
                        'slowest_rank': times.index(max(times)), 'batch_verified': good,
                        'global_actual_samples': sum(row['local_samples'] for row in rows) if all(integer(row.get('local_samples')) for row in rows) else None,
                        'excluded_reasons': excluded, 'rank_timing_records': rows})
    for path in sorted(directory.glob('measurement-status-rank-*.json')):
        status = read_json(path)
        if status.get('valid_for_comparison') is not True:
            issues.append(f'Invalid measurement status: {path.name}: {status.get("reason")}')
    if list(grouped) and sorted(grouped) != list(range(1, expected_steps + 1)):
        issues.append('Recorded update ordinals are incomplete or differ from benchmark stop count')
    if len(updates) != expected_steps:
        issues.append('Completed selected-rank updates differ from benchmark stop count')
    stdout = source_path(run, artifacts.get('stdout_log', 'train.log'))
    logged, fatal, compiler = log_metrics(stdout)
    if fatal:
        issues.append('Fatal training errors found in stdout')
    log_steps = [row['global_step'] for row in logged]
    observed_steps = [row['global_step'] for row in updates]
    logs_complete = log_steps == observed_steps and len(logged) == expected_steps
    if not logs_complete:
        issues.append('Missing/duplicate/out-of-order per-update loss/grad log rows')
    finite = bool(logged) and all(all(key in row['metrics'] and math.isfinite(row['metrics'][key])
                     for key in (*LOSSES, 'GradNorm', 'LR')) for row in logged)
    if not finite:
        issues.append('A required loss/grad norm/LR is absent or nonfinite')
    quality = {'logged_update_count': len(logged), 'complete_update_logs': logs_complete,
               'all_logged_losses_gradnorm_lr_finite': finite,
               'loss_and_grad_norm_updates': logged, 'obvious_divergence_flags': [],
               'limitation': 'Rounded globally reduced training logs and grad norm only; no full accuracy evaluation or inspection of every gradient tensor'}
    quality['invalid_or_missing_metrics'] = [
        {'global_step': row['global_step'], 'metric': key, 'line': row['line'],
         'problem': 'missing' if key not in row['metrics'] else 'nonfinite'}
        for row in logged for key in (*LOSSES, 'GradNorm', 'LR')
        if key not in row['metrics'] or not math.isfinite(row['metrics'][key])]
    if any(row['metrics'].get('GradNorm', 0) < 0 for row in logged):
        quality['obvious_divergence_flags'].append('A logged GradNorm is negative')
    for key, limit in [('Loss', divergence_ratio), ('GradNorm', divergence_ratio * 2)]:
        values = [row['metrics'][key] for row in logged if key in row['metrics'] and math.isfinite(row['metrics'][key])]
        if len(values) >= 2:
            count = max(1, len(values) // 3)
            first, last = statistics.median(values[:count]), statistics.median(values[-count:])
            ratio = last / first if first > 0 else None
            quality[key + '_trend'] = {'first_third_median': first, 'last_third_median': last, 'ratio': ratio}
            if ratio is not None and ratio > limit:
                quality['obvious_divergence_flags'].append(f'{key} last/first-third median ratio {ratio:.3g} exceeds {limit:g}')
    ignored = [row['global_step'] for row in logged if row['metrics'].get('Ignore_Batch_Num', 0) != 0]
    if ignored:
        warnings.append(f'Nonzero Ignore_Batch_Num at logged updates {ignored}; inspect data handling')
    codes = {}
    for name in ('training-exit-code.txt', 'container-exit-code.txt'):
        if (run / name).is_file():
            codes[name] = int((run / name).read_text().strip())
    success = meta.get('status') == 'completed' and codes.get('training-exit-code.txt', meta.get('training_exit_code')) == 0
    if any(value != 0 for value in codes.values()):
        success = False
    if not success:
        issues.append('Run not marked completed with zero training exit status')
    steady = [row for row in updates if not row['excluded_reasons'] and row['batch_verified']]
    stats = summary([row['duration_s'] for row in steady])
    minimum = meta.get('measurement', {}).get('minimum_steady_steps')
    enough = integer(minimum) and len(steady) >= minimum
    if not enough:
        warnings.append('No sufficient steady window; smoke/all-warmup run cannot establish performance')
    stable_hint = bool(enough and stats and stats['count'] >= 4 and stats['cv'] <= max_cv
                       and abs(stats['half_median_drift_fraction']) <= max_half_drift
                       and not any(row['after_logged_step'] >= warmup for row in compiler))
    sample_ok = bool(updates) and all(row['batch_verified'] for row in updates) and all(row['passed'] for row in batch_checks)
    quality_ok = success and logs_complete and finite and not quality['obvious_divergence_flags'] and not fatal
    return {'schema_version': 1, 'analysis_kind': 'single-run CPU evidence analysis; no speedup claim',
            'analyzed_utc': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run),
            'experiment_id': meta.get('experiment_id'), 'git': meta.get('git') or {},
            'optimizer': meta.get('training', {}).get('optimizer'), 'memory_policy': meta.get('training', {}).get('memory_policy'),
            'source_hashes': {'experiment.json': sha(meta_path), str(config_path.name): sha(config_path), str(stdout.name): sha(stdout)},
            'successful_exit': success, 'exit_codes': codes,
            'expected_updates': expected_steps, 'world_size': world_size, 'global_batch_size': global_batch_size, 'lr_schedule_max_steps': planned.get('max_steps'), 'complete_all_rank_updates': len(updates),
            'actual_batch_checks': batch_checks, 'actual_samples_per_update_verified': sample_ok,
            'updates': updates, 'warmup_steps': warmup,
            'steady_optimizer_steps': [row['optimizer_step'] for row in steady],
            'excluded_updates': [{'optimizer_step': row['optimizer_step'], 'reasons': row['excluded_reasons']} for row in updates if row['excluded_reasons']],
            'steady_stats': stats,
            'steady_samples_per_second': global_batch_size * len(steady) / stats['sum_s'] if stats else None,
            'steady_window_sufficient': enough, 'statistical_settling_hint': stable_hint,
            'settling_thresholds': {'max_cv': max_cv, 'max_absolute_half_median_drift': max_half_drift},
            'quality': quality, 'compiler_warnings': compiler, 'fatal_errors': fatal,
            'wallclock': wallclock(run, meta, updates), 'issues': issues, 'warnings': warnings,
            'original_validation_flags': meta.get('measurement', {}),
            'suggested_validation_flags': {
                'measurement.actual_samples_per_update_verified': sample_ok and len(updates) == expected_steps,
                'measurement.loss_and_gradients_finite_verified': quality_ok,
                'training.quality_validation_passed': quality_ok,
                'measurement.warmup_compile_settled': None,
                'measurement.timing_overhead_checked': None},
            'validation_notes': ['Suggestions do not edit experiment.json',
                                 'Timing overhead requires independent on/off measurement evidence',
                                 'Low CV/drift supports settling but does not prove absence of compilation; inspect profile/logs',
                                 'A single run never proves an optimization gain; paired run validation remains required',
                                 'Update JSONL write and final asynchronous checkpoint drain are excluded from update timers; total wallclock reported separately']}


def markdown(record):
    stats, wall = record['steady_stats'], record['wallclock']
    lines = [f'# 单次 SFT 分析：{record["experiment_id"]}', '',
             f'commit `{record["git"].get("commit")}`；optimizer `{record["optimizer"]}`。', '',
             f'退出成功：{record["successful_exit"]}；完整全部 rank 更新 {record["complete_all_rank_updates"]}/{record["expected_updates"]}；真实micro32×所选rank×accum1验证：{record["actual_samples_per_update_verified"]}。', '',
             f'全部更新 loss/grad 日志完整：{record["quality"]["complete_update_logs"]}；七项 loss、GradNorm、LR 有限：{record["quality"]["all_logged_losses_gradnorm_lr_finite"]}。', '']
    if stats:
        lines += [f'稳态 {stats["count"]} 步：mean {stats["mean_s"]:.6f}s，median {stats["median_s"]:.6f}s，p95 {stats["p95_s"]:.6f}s（nearest rank），CV {stats["cv"]:.2%}；{record["steady_samples_per_second"]:.4f} samples/s。', '',
                  f'前/后半 median {stats["first_half_median_s"]} / {stats["second_half_median_s"]}s，漂移 {stats["half_median_drift_fraction"]}；统计稳定提示：{record["statistical_settling_hint"]}，不自动证明编译已结束。', '']
    else:
        lines += ['没有可用稳态更新；smoke/全 warmup 只能说明短期可运行性，不能用于性能提升结论。', '']
    lines += [f'训练命令总墙钟 {wall["training_command_wallclock_s"]}s；首更新前启动 {wall["pre_first_update_startup_s"]}s；末更新后退出 {wall["after_last_update_shutdown_s"]}s；各更新最慢 rank 时间和 {wall["sum_max_rank_update_s"]}s。', '',
              '启动时间包含加载/初始化等，不证明 cache 是冷的；UTC 文本为秒精度。逐步计时不含记录写入/末尾异步 drain，总墙钟另报。', '',
              '| update | 所选 rank 完整更新秒数 | max秒 | 排除原因 |', '|---|---|---|---|']
    for row in record['updates']:
        lines.append(f'| {row["optimizer_step"]} | ' + ', '.join(f'{x:.6f}' for x in row['rank_durations_s']) + f' | {row["duration_s"]:.6f} | {", ".join(row["excluded_reasons"]) or "steady"} |')
    lines += ['', '所有原始 rank timing 字段与全部训练标量保存在 analysis.json。以下标志仅建议；原 experiment.json 未改。timing_overhead_checked 与 warmup_compile_settled 留待独立证据，未自动设 true。', '']
    for label, values in [('问题', record['issues']), ('提示', record['warnings']), ('短期发散提示', record['quality']['obvious_divergence_flags'])]:
        lines.append(f'{label}：' + ('；'.join(values) if values else '无'))
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print JSON only; write no analysis files')
    parser.add_argument('--max-cv', type=float, default=.10)
    parser.add_argument('--max-half-drift', type=float, default=.05)
    parser.add_argument('--divergence-ratio', type=float, default=5.)
    args = parser.parse_args()
    run = args.run_dir.resolve(strict=True)
    if not run.is_dir():
        parser.error('--run-dir must be an existing directory')
    local_root = Path(__file__).resolve().parent.parent
    local_allowed = local_root.name == 'LingBotVLA2.0-Training-Opt-Local-rtx5k' and run.is_relative_to(local_root)
    if not run.is_relative_to(REMOTE_ROOT.resolve()) and not local_allowed:
        parser.error('Run must stay inside authorized remote ROOT or this local task folder')
    if any(not math.isfinite(x) or x <= 0 for x in (args.max_cv, args.max_half_drift, args.divergence_ratio)):
        parser.error('Thresholds must be positive finite numbers')
    record = analyze(run, max_cv=args.max_cv, max_half_drift=args.max_half_drift, divergence_ratio=args.divergence_ratio)
    # JSON has no nonstandard NaN/Infinity values; invalid scalar evidence uses null.
    def sanitized(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: sanitized(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitized(item) for item in value]
        return value
    payload = json.dumps(sanitized(record), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.dry_run:
        print(payload, end='')
    else:
        for name, content in [('analysis.json', payload), ('analysis.md', markdown(record))]:
            if (run / name).is_symlink():
                raise ValueError(f'Refusing to write through output symlink: {name}')
            path = source_path(run, name)
            path.write_text(content, encoding='utf-8', newline='\n')
        print(json.dumps({'analysis_json': str(run / 'analysis.json'), 'analysis_md': str(run / 'analysis.md'),
                          'updates': record['complete_all_rank_updates'], 'steady_steps': len(record['steady_optimizer_steps']),
                          'steady_stats': record['steady_stats'], 'issues': record['issues'], 'warnings': record['warnings']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
