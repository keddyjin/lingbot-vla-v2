#!/usr/bin/env python3
"""Prepare a new RTX SFT run in Docker; do not launch training here.

Derived from the historical harness workflow; all execution evidence is new.
The production max_steps remains the LR horizon; an independent environment
stop count bounds this fresh, no-checkpoint benchmark.
"""
from __future__ import annotations

import argparse
from fsdp_prefetch_source_audit import audit_fsdp_forward_prefetch
from core_source_audit import audit_core_source
from root_teacher_source_audit import audit_root_teacher_source, validate_root_teacher_controls
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

import yaml

ROOT = Path('/root/keddyj/LingBotVLA-Test')
REPO = ROOT / 'lingbot-vla-v2-sft-opt-github'
VENV = ROOT / 'environment/venv'
EXPECTED_BRANCH = 'keddyj-codex/sft-opt-rtx5k'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def confined(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve()),
            f'Path escapes authorized ROOT: {path}')
    return path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(REPO), *args])


def differences(a, b, prefix=''):
    result = {}
    for key in sorted(set(a) | set(b)):
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            result.update(differences(a[key], b[key], name))
        elif a.get(key) != b.get(key):
            result[name] = {'reference': a.get(key), 'selected': b.get(key)}
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def weight_inputs(manifest_path=None):
    models = ROOT / 'models'
    student = models / 'lingbot-vla-v2-6b/hf_ckpt'
    qwen = models / 'Qwen3-VL-4B-Instruct'
    paths = {'student': student, 'qwen_processor': qwen,
             'moge_teacher': models / 'moge-2-vitb-normal/model.pt',
             'depth_teacher': models / 'lingbot-vla-v2-6b/depth/model.pt',
             'video_teacher': models / 'lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth',
             'video_config': models / 'lingbot-vla-v2-6b/dino_video/config.yaml'}
    files = [student / 'config.json', student / 'model.safetensors.index.json',
             qwen / 'config.json', qwen / 'tokenizer.json', qwen / 'tokenizer_config.json',
             qwen / 'preprocessor_config.json', qwen / 'video_preprocessor_config.json',
             *[paths[name] for name in ('moge_teacher', 'depth_teacher', 'video_teacher', 'video_config')]]
    index = json.loads((student / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    require(shards and all(Path(name).name == name for name in shards), 'Invalid native student shard index')
    files += [student / name for name in shards]
    for path in files:
        confined(path)
        require(path.is_file() and path.stat().st_size > 0, f'Missing/empty required input: {path}')
    records = [{'path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size,
                'sha256': sha(path) if path.stat().st_size <= 32 * 1024 * 1024 else None,
                'hash_scope': 'actual_small_file' if path.stat().st_size <= 32 * 1024 * 1024 else 'large_file_not_rehashed_per_launch'}
               for path in files]
    evidence = {'validation': 'actual metadata/small-file SHA and large-file sizes; new download receipt required for full hashes',
                'files': records, 'native_student_index_sha256': sha(student / 'model.safetensors.index.json')}
    if manifest_path:
        manifest_path = confined(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get('remote_root', str(ROOT)) == str(ROOT), 'Weights manifest belongs to another task root')
        for row in manifest['files']:
            target = confined(ROOT / row['target_relative_path'])
            require(target.is_file() and target.stat().st_size == row['size_bytes'], f'Manifest size mismatch: {target}')
        evidence.update(manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
                        manifest_claims=manifest, manifest_hashes_reverified_by_launcher=False)
    return paths, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--data-manifest', required=True, type=Path)
    parser.add_argument('--weights-manifest', type=Path)
    parser.add_argument('--world-size', required=True, type=int, choices=(4, 8))
    parser.add_argument('--optimizer-mode', choices=('dist', 'official-muon', 'controlled-muon'), default='dist')
    parser.add_argument('--memory-policy', choices=('official', 'gc', 'reshard', 'gc-reshard'), default='gc-reshard')
    parser.add_argument('--kind', choices=('smoke', 'perf', 'nsys', 'torch'), default='perf')
    parser.add_argument('--pair-id', required=True)
    parser.add_argument('--role', choices=('baseline', 'candidate'), default='baseline')
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--steady-steps', type=int, default=20)
    args = parser.parse_args()
    run, data = confined(args.run_dir), confined(args.data_manifest)
    require(run.is_dir() and not (run / 'experiment.json').exists(), 'Run evidence directory must be fresh')
    # Preserve the intended run and source identity before input/weight/GPU
    # checks; a failed preparation must not masquerade as an executed SFT.
    early_sources = [REPO / 'tasks/vla/train_lingbotvla.py',
                     REPO / 'lingbotvla/utils/training_instrumentation.py',
                     Path(__file__)]
    write_json(run / 'preparation-intent.json', {
        'schema_version': 1, 'workload_kind': 'preparation_validation', 'real_sft_started': False,
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run.name,
        'intended_optimizer_mode': args.optimizer_mode, 'intended_kind': args.kind,
        'requested_world_size': args.world_size, 'requested_global_batch_size': args.world_size * 32,
        'micro_batch_size': 32, 'gradient_accumulation_steps': 1,
        'data_manifest_path': str(data), 'image_id': os.environ.get('SFT_IMAGE_ID'),
        'gpu_uuid_request': os.environ.get('NV_GPU'),
        'git_commit': git('rev-parse', 'HEAD').decode().strip(),
        'git_branch': git('branch', '--show-current').decode().strip(),
        'source_sha256': {str(path): sha(path) for path in early_sources if path.is_file()},
        'benchmark_skip_checkpoint_env': os.environ.get('LINGBOT_BENCHMARK_SKIP_CHECKPOINT'),
        'benchmark_stop_steps_env': os.environ.get('LINGBOT_BENCHMARK_STOP_STEPS')})
    require(data.is_file(), 'Real data manifest is required')
    world, global_batch = args.world_size, args.world_size * 32
    require(int(os.environ['SFT_WORLD_SIZE']) == world, 'Environment world size differs from requested world size')
    require(os.environ.get('LINGBOT_BENCHMARK_SKIP_CHECKPOINT') == '1', 'All task runs require checkpoint skipping')
    warmup = 3 if args.kind == 'smoke' else args.warmup_steps
    steady = int(os.environ.get('LINGBOT_NSYS_CAPTURE_STEPS', '2')) if args.kind == 'nsys' else 1 if args.kind == 'torch' else args.steady_steps
    stop = 3 if args.kind == 'smoke' else warmup + steady + 1
    require(warmup >= 0 and steady > 0 and 0 < stop <= 1000, 'Invalid short benchmark counts')
    require(int(os.environ['LINGBOT_BENCHMARK_STOP_STEPS']) == stop, 'Actual independent stop count differs from plan')
    branch = git('branch', '--show-current').decode().strip()
    require(branch == EXPECTED_BRANCH, f'Unexpected task branch: {branch}')
    image_id = os.environ['SFT_IMAGE_ID']
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image_id) is not None, 'Immutable Docker image ID required')
    trainer = REPO / 'tasks/vla/train_lingbotvla.py'
    trainer_source = trainer.read_text()
    require(all(flag in trainer_source for flag in ('LINGBOT_BENCHMARK_SKIP_CHECKPOINT', 'LINGBOT_BENCHMARK_STOP_STEPS')),
            'Review/apply functional no-save and independent-stop changes before launching')
    gpu_rows = [row.split(',') for row in (run / 'gpu-selected.csv').read_text().splitlines() if row.strip()]
    uuids = [row[0].strip() for row in gpu_rows]
    require(len(uuids) == len(set(uuids)) == world and set(uuids) == set(os.environ['NV_GPU'].split(',')), 'Selected GPU UUID count/mapping mismatch')
    dataset_ids = []
    for line in data.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        robot, value = line.split(maxsplit=1)
        dataset = confined(value)
        info_path = dataset / 'meta/info.json'
        require(robot == 'robotwin' and info_path.is_file(), f'Invalid RoboTwin dataset: {line}')
        info = json.loads(info_path.read_text())
        require(info.get('total_frames', 0) > 0 and info.get('total_episodes', 0) > 0, 'Empty dataset')
        dataset_ids.append({'path': str(dataset), 'info_sha256': sha(info_path), 'frames': info['total_frames'],
                            'episodes': info['total_episodes'], 'codebase_version': info.get('codebase_version')})
    require(dataset_ids and len({row['path'] for row in dataset_ids}) == len(dataset_ids), 'Empty/duplicate dataset manifest')
    paths, weights = weight_inputs(args.weights_manifest)
    dist_source = REPO / 'configs/vla/robotwin/robotwin_dist_muon.yaml'
    source = REPO / ('configs/vla/robotwin/robotwin.yaml' if args.optimizer_mode == 'official-muon' else 'configs/vla/robotwin/robotwin_dist_muon.yaml')
    reference, config = yaml.safe_load(dist_source.read_text()), yaml.safe_load(source.read_text())
    train = config['train']
    require(train['micro_batch_size'] == 32 and train.get('gradient_accumulation_steps', 1) == 1, 'micro32/accum1 must remain original')
    require(type(train['max_steps']) is int and stop <= train['max_steps'], 'Production LR horizon must contain benchmark stop')
    if args.optimizer_mode == 'controlled-muon':
        train['optimizer'] = 'muon'
    optimizer_diff = differences(reference, config)
    memory_reference = yaml.safe_load(yaml.safe_dump(config))
    if args.memory_policy in ('gc', 'gc-reshard'):
        train['enable_gradient_checkpointing'] = True
    if args.memory_policy in ('reshard', 'gc-reshard'):
        train['enable_full_shard'] = True
    memory_diff = differences(memory_reference, config)
    bucket = os.environ.get('SFT_DIST_MUON_LAYERS_PER_BUCKET', 'official')
    if bucket != 'official':
        require(args.optimizer_mode == 'dist' and bucket.isdigit() and 1 <= int(bucket) <= 36, 'Invalid Distributed Muon bucket value')
        train['dist_muon_layers_per_bucket'] = int(bucket)
    config['model'].update(model_path=str(paths['student']), tokenizer_path=str(paths['qwen_processor']))
    config['data'].update(train_path=str(data), robot_config_root=str(REPO / 'configs/robot_configs'))
    train.update(global_batch_size=global_batch, output_dir=str(run / 'output'), enable_resume=False,
                 save_steps=0, save_epochs=0, save_hf_weights=False, async_save_hf_weights=False, enable_profiling=False)
    train.pop('load_checkpoint_path', None)  # Upstream YAML parser can stringify null.
    train['align_params']['depth'].update(moge_path=str(paths['moge_teacher']), morgbd_path=str(paths['depth_teacher']))
    train['align_params']['video'].update(config_path=str(paths['video_config']), ckpt_path=str(paths['video_teacher']))
    prefetch_mode = os.environ.get('LINGBOT_FSDP_FWD_PREFETCH', 'off')
    require(prefetch_mode in ('off', 'one', 'pair'), 'Unknown FSDP FWD prefetch control')
    require(prefetch_mode == 'off' or world == 8, 'FSDP FWD prefetch requires DP8')
    require(os.environ.get('LINGBOT_FSDP_FWD_PREFETCH_RECEIPT_DIR') == str(run / 'fsdp-forward-prefetch'), 'Unexpected FSDP setup receipt directory')
    root_teacher_control=os.environ.get('LINGBOT_FSDP_ROOT_TEACHER_PREFETCH','off')
    require(root_teacher_control in ('off','on'), 'Root teacher prefetch must be off/on')
    require(os.environ.get('LINGBOT_FSDP_ROOT_TEACHER_PREFETCH_RECEIPT_DIR') == str(run/'root-teacher-prefetch'), 'Root teacher receipt directory mismatch')
    if root_teacher_control=='on':
        require(world==8 and args.optimizer_mode=='dist' and args.memory_policy=='gc' and prefetch_mode=='off', 'Root teacher prefetch requires pure native DP8/dist/gc')
    validate_root_teacher_controls(os.environ, world, args.optimizer_mode, args.memory_policy)
    root_teacher_spec_path=confined(os.environ['LINGBOT_FSDP_ROOT_TEACHER_PREFETCH_SOURCE_SPEC'])
    root_teacher_source=audit_root_teacher_source(ROOT,root_teacher_spec_path,root_teacher_control)
    (run/'root-teacher-prefetch-source-spec.json').write_bytes(root_teacher_spec_path.read_bytes())
    write_json(run/'root-teacher-prefetch-source-audit.json',root_teacher_source)
    controls = {'root_teacher_prefetch':root_teacher_control,'fsdp_forward_prefetch': prefetch_mode, 'flex_bwd_tile': os.environ.get('LINGBOT_FLEX_BWD_TILE', 'off'),
                'flex_bwd_num_stages': os.environ.get('LINGBOT_FLEX_BWD_NUM_STAGES', 'off'),
                'selective_checkpoint_policy': os.environ.get('LINGBOT_GC_POLICY', 'official'),
                'compile_step_counter_helper': os.environ.get('LINGBOT_COMPILE_STEP_COUNTER_HELPER', '0') == '1',
                'deepstack_indices': os.environ.get('LINGBOT_DEEPSTACK_INDICES', '0') == '1',
                'batch_depth_teachers': os.environ.get('LINGBOT_DEPTH_TEACHER_BATCH', '0') == '1',
                'moe_segment_output_size': os.environ.get('LINGBOT_MOE_SEGMENT_OUTPUT_SIZE', '0') == '1',
                'dist_muon_layers_per_bucket': train.get('dist_muon_layers_per_bucket'), 'precision_changed': False}
    (run / 'config-source.yaml').write_bytes(source.read_bytes())
    (run / 'launch-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    write_json(run / 'effective_config.json', config)
    write_json(run / 'optimizer-differences.json', optimizer_diff)
    write_json(run / 'memory-policy-differences.json', memory_diff)
    write_json(run / 'weights-identities.json', weights)
    write_json(run / 'dataset-identities.json', dataset_ids)
    (run / 'data-manifest.txt').write_bytes(data.read_bytes())
    (run / 'git-status.txt').write_bytes(git('status', '--porcelain=v1', '--untracked-files=all'))
    (run / 'dirty-tracked.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    untracked = [name for name in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0') if name]
    with tarfile.open(run / 'dirty-untracked.tar.gz', 'w:gz', dereference=False) as archive:
        for name in untracked:
            path = confined(REPO / name)
            archive.add(path, arcname=name, recursive=False)
    import torch
    packages = {}
    for name in ('torch', 'torchvision', 'transformers', 'flash-attn', 'triton', 'numpy', 'pyarrow', 'torchcodec', 'lerobot'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    require(torch.cuda.device_count() == world, 'Docker visible CUDA GPU count mismatch')
    cuda_devices = []
    for rank in range(world):
        props = torch.cuda.get_device_properties(rank)
        raw_uuid = str(props.uuid)
        uuid = raw_uuid if raw_uuid.startswith('GPU-') else 'GPU-' + raw_uuid
        require(uuid.lower() in {value.lower() for value in uuids}, 'CUDA physical UUID differs from selected pool')
        require((props.major, props.minor) == (12, 0), 'Expected Blackwell compute capability 12.0')
        require(70 * 1024**3 <= props.total_memory <= 74 * 1024**3, 'Expected user-confirmed 72 GB GPU capacity')
        cuda_devices.append({'local_rank': rank, 'uuid': uuid, 'raw_cuda_name': props.name,
                             'compute_capability': [props.major, props.minor], 'total_memory_bytes': props.total_memory,
                             'model_identification': 'user-confirmed RTX PRO 5000 72 GB; NVML may report NVIDIA Graphics Device'})
    require(len({row['uuid'].lower() for row in cuda_devices}) == world, 'Duplicate physical CUDA GPU mapping')
    write_json(run / 'cuda-devices.json', cuda_devices)
    actual_nccl = list(torch.cuda.nccl.version())
    require(actual_nccl == [2, 27, 3], f'Pinned NCCL must be 2.27.3, got {actual_nccl}')
    prefetch_audit = audit_fsdp_forward_prefetch(prefetch_mode, REPO, Path(torch.__file__).parent)
    write_json(run / 'fsdp-forward-prefetch-source-audit.json', prefetch_audit)
    versions = {'loaded_nccl_version': actual_nccl, 'packages': packages, 'torch_cuda': torch.version.cuda, 'python': subprocess.check_output([str(VENV / 'bin/python'), '--version']).decode().strip()}
    write_json(run / 'versions.json', versions)
    script = Path(__file__).with_name('rtx5k_profile_sft_worker.py') if args.kind == 'nsys' else Path(__file__).with_name('rtx5k_profile_torch_worker.py') if args.kind == 'torch' else trainer
    affinity = os.environ.get('LINGBOT_CPU_AFFINITY', 'off')
    require(affinity in ('off', 'gpu-numa'), 'Invalid CPU affinity control')
    require(os.environ.get('LINGBOT_CPU_AFFINITY_RECEIPT_DIR') == str(run / 'numa'), 'CPU affinity receipt path mismatch')
    if affinity == 'gpu-numa':
        require(world == 8, 'NUMA placement requires eight ranks')
        # Physical UUID/NUMA mapping from this host's saved preflight receipt.
        expected = ['GPU-945dd068-fcec-e323-fb08-1103398e46e2', 'GPU-e52b9134-dc44-183d-3681-6d5dd478d501',
                    'GPU-2b02967e-9ef9-c7fa-4bc9-cf9e16e3f3b3', 'GPU-c7835be3-1680-8891-8481-b21729de0264',
                    'GPU-10c8f611-e276-21c2-9151-70481371487c', 'GPU-9f634113-d908-b3e2-2abd-d96375d84805',
                    'GPU-4bc4e1ce-a9d1-ec12-4620-a761b5cb6363', 'GPU-ba3c1fa8-f142-ab9e-2eb5-3fd95cafffc0']
        require([row['uuid'].lower() for row in cuda_devices] == [value.lower() for value in expected], 'CUDA UUID order differs from reviewed NUMA mapping')
    wrapper = Path(__file__).with_name('rtx5k_numa_worker.py')
    require(wrapper.is_file(), 'Reviewed CPU affinity wrapper missing')
    controls['cpu_affinity'] = affinity
    allocator = os.environ.get('SFT_CUDA_ALLOCATOR_CONTROL', 'unset')
    require(allocator in ('unset', 'expandable'), 'Invalid allocator control')
    require('PYTORCH_ALLOC_CONF' not in os.environ, 'Unexpected alternative allocator environment')
    require(('PYTORCH_CUDA_ALLOC_CONF' not in os.environ) if allocator == 'unset' else os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True', 'Actual allocator environment differs from plan')
    controls['cuda_allocator'] = allocator
    write_json(run / 'allocator-environment.json', {'control': allocator, 'backend': torch.cuda.memory.get_allocator_backend(),
               'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'PYTORCH_ALLOC_CONF': os.environ.get('PYTORCH_ALLOC_CONF')})
    target = run / script.name if args.kind in ('nsys', 'torch') else script
    command = [str(VENV / 'bin/python'), '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc-per-node={world}', str(run / wrapper.name), str(target), str(run / 'launch-config.yaml')]
    if args.kind == 'nsys':
        capture_mode = os.environ.get('LINGBOT_NSYS_CAPTURE_MODE', 'nvtx')
        flush_ms = int(os.environ.get('LINGBOT_NSYS_CUDA_FLUSH_MS', '100'))
        require(capture_mode in ('nvtx', 'api') and flush_ms in (0, 100), 'Invalid Nsight capture controls')
        require(1 <= steady <= 3, 'Short capture requires one to three updates')
        capture_options = ['--capture-range=nvtx', '--nvtx-capture=lingbot_nsys_capture_v2', '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0'] if capture_mode == 'nvtx' else ['--capture-range=cudaProfilerApi']
        command = ['nsys', 'profile', '--trace=cuda,nvtx', '--cuda-trace-scope=process-tree', '--sample=none', '--cpuctxsw=none', f'--cuda-flush-interval={flush_ms}', '--flush-on-cudaprofilerstop=true', '--wait=all', *capture_options, '--capture-range-end=stop', '--kill=none', '--force-overwrite=false', f'--output={run}/profiling/trace', *command]
    write_json(run / 'command.json', command)
    (run / 'command.sh').write_text(shlex.join(command) + '\n', encoding='utf-8')
    commit = git('rev-parse', 'HEAD').decode().strip()
    core_source = audit_core_source(REPO)
    write_json(run / 'core-source-audit.json', core_source)
    nccl_control = os.environ.get('SFT_NCCL_PROTO_CONTROL','unset')
    require(nccl_control in ('unset','Simple'), 'NCCL protocol control must be unset or Simple')
    require(('NCCL_PROTO' not in os.environ) if nccl_control == 'unset' else os.environ.get('NCCL_PROTO') == 'Simple', 'Actual NCCL_PROTO differs from selected control')
    controls['nccl_proto_control'] = nccl_control
    p2p_control = os.environ.get('SFT_NCCL_P2P_LEVEL_CONTROL', 'default')
    require(p2p_control in ('default', 'PHB', 'SYS'), 'Invalid P2P level control')
    require(p2p_control == 'default' or (world == 8 and nccl_control == 'unset'), 'P2P trial requires DP8 and protocol unset')
    require(('NCCL_P2P_LEVEL' not in os.environ) if p2p_control == 'default' else os.environ.get('NCCL_P2P_LEVEL') == p2p_control, 'Actual P2P level differs from plan')
    p2p_disable = os.environ.get('SFT_NCCL_P2P_DISABLE_CONTROL', 'unset')
    require(p2p_disable in ('unset', '1'), 'Invalid P2P disable control')
    require(p2p_disable == 'unset' or (world == 8 and nccl_control == 'unset' and p2p_control == 'default'), 'P2P disable trial requires DP8, protocol unset, default level')
    require(('NCCL_P2P_DISABLE' not in os.environ) if p2p_disable == 'unset' else os.environ.get('NCCL_P2P_DISABLE') == '1', 'Actual P2P disable differs from plan')
    require(os.environ.get('NCCL_IGNORE_DISABLED_P2P', '0') in ('', '0'), 'Peer validation bypass forbidden')
    controls['nccl_p2p_level_control'] = p2p_control
    write_json(run / 'p2p-environment.json', {'control': p2p_control, 'disable_control': p2p_disable, 'NCCL_P2P_DISABLE_is_set': 'NCCL_P2P_DISABLE' in os.environ, 'NCCL_P2P_DISABLE': os.environ.get('NCCL_P2P_DISABLE'), 'NCCL_P2P_LEVEL_is_set': 'NCCL_P2P_LEVEL' in os.environ, 'NCCL_P2P_LEVEL': os.environ.get('NCCL_P2P_LEVEL'), 'cuda_peer_access_matrix': [[None if i == j else bool(torch.cuda.can_device_access_peer(i,j)) for j in range(world)] for i in range(world)], 'diagonal': 'self/not_queried', 'transport_change_verified': False})
    min_ctas = os.environ.get('SFT_NCCL_MIN_CTAS_CONTROL', 'unset')
    require(min_ctas in ('unset', '8', '16', '32'), 'Invalid minimum CTA control')
    require(world == 8 and nccl_control == 'unset' and p2p_control == 'default' and p2p_disable == 'unset' and prefetch_mode == 'off', 'H7 independent minimum CTA trial requires DP8, default communication and FWD off')
    require(('NCCL_MIN_CTAS' not in os.environ) if min_ctas == 'unset' else os.environ.get('NCCL_MIN_CTAS') == min_ctas, 'Actual minimum CTA control differs from plan')
    require(all(key not in os.environ for key in ('NCCL_MIN_NCHANNELS', 'NCCL_MIN_NRINGS')), 'Legacy minimum channel aliases must be unset')
    require(all(key not in os.environ for key in ('NCCL_MAX_CTAS', 'NCCL_MAX_NCHANNELS', 'NCCL_MAX_NRINGS')), 'Maximum channel overrides mask single-variable trial')
    write_json(run / 'nccl-channel-environment.json', {'control': min_ctas, 'NCCL_MIN_CTAS_is_set': 'NCCL_MIN_CTAS' in os.environ, 'NCCL_MIN_CTAS': os.environ.get('NCCL_MIN_CTAS'), 'legacy_minimum_aliases_unset': True, 'actual_collective_ring_channels_verified': False, 'actual_per_operation_channels_verified': False})
    nccl_environment = {key: os.environ[key] for key in sorted(os.environ) if key.startswith('NCCL_')}
    write_json(run / 'nccl-environment.json', {'control': nccl_control, 'NCCL_PROTO_is_set': 'NCCL_PROTO' in os.environ, 'NCCL_PROTO': os.environ.get('NCCL_PROTO'), 'environment': nccl_environment})
    batch = {'world_size': world, 'micro_batch_size': 32, 'gradient_accumulation_steps': 1, 'global_batch_size': global_batch}
    meta = {'schema_version': 1, 'experiment_id': run.name, 'pair_id': args.pair_id, 'role': args.role,
            'candidate_name': args.optimizer_mode, 'status': 'prepared', 'workload_kind': 'real_sft', 'host': '10.6.142.11',
            'started_utc': datetime.now(timezone.utc).isoformat(), 'optimization_controls': controls,
            'communication_environment': {'nccl_proto_control': nccl_control, 'NCCL_PROTO_is_set': 'NCCL_PROTO' in os.environ, 'NCCL_PROTO': os.environ.get('NCCL_PROTO'), 'NCCL_environment': nccl_environment, 'artifact': 'nccl-environment.json', 'expected_pinned_nccl': '2.27.3', 'loaded_nccl_version': actual_nccl},
            'git': {'commit': commit, 'branch': branch, 'dirty': bool((run / 'git-status.txt').stat().st_size),
                    'patch_sha256': sha(run / 'dirty-tracked.patch'), 'patch_artifact': 'dirty-tracked.patch',
                    'untracked_artifact': 'dirty-untracked.tar.gz', 'untracked_sha256': sha(run / 'dirty-untracked.tar.gz')},
            'docker': {'image_id': image_id, 'image_reference': os.environ['SFT_IMAGE_REFERENCE'], 'image_digest': os.environ.get('SFT_IMAGE_DIGEST', ''),
                       'container_name': os.environ['SFT_CONTAINER_NAME'], 'gpu_request': os.environ['NV_GPU'], 'cuda_visible_devices_inside': os.environ['CUDA_VISIBLE_DEVICES'], 'cuda_devices': cuda_devices},
            'training': {**batch, 'optimizer': train['optimizer'], 'seed': train.get('seed', 42), 'resume': False,
                         'benchmark_skip_checkpoint': True, 'stop_after_steps': stop, 'lr_schedule_max_steps': train['max_steps'],
                         'save_policy': 'no DCP/HF/async/terminal save; fresh pretrained initialization',
                         'gradient_checkpointing': train.get('enable_gradient_checkpointing'), 'quality_validation_passed': False,
                         'memory_policy': {'name': args.memory_policy, 'enable_gradient_checkpointing': train.get('enable_gradient_checkpointing'), 'enable_full_shard': train.get('enable_full_shard')},
                         'allocator_metrics': {'enabled': True, 'peak_scope': 'process_lifetime_no_reset'}},
            'fair_comparison': {**batch, 'gpu_uuids': uuids, 'model_weights_identity': identity(weights),
                                'dataset_manifest_sha256': sha(data), 'dataset_version_identity': identity(dataset_ids),
                                'ordered_batches_identity': f'seed={train.get("seed",42)};fresh;manifest={sha(data)}',
                                'precision_identity': identity({'enable_fp32': train.get('enable_fp32')}),
                                'runtime_identity': identity({'image': image_id, 'versions': versions}),
                                'training_objective_identity': identity({k: v for k, v in train.items() if k not in ('optimizer', 'output_dir', 'enable_gradient_checkpointing', 'enable_full_shard') and not k.startswith(('muon_', 'dist_muon_'))})},
            'measurement': {'warmup_steps': warmup, 'minimum_steady_steps': steady if args.kind == 'perf' else 1,
                            'checkpoint_steps_policy': 'exclude', 'terminal_steps_policy': 'exclude', 'profiling_enabled': args.kind in ('nsys', 'torch'),
                            'timing_semantics': 'max selected-rank GPU-drained complete optimizer-update wall time',
                            'timing_overhead_checked': False, 'warmup_compile_settled': False, 'actual_samples_per_update_verified': False},
            'artifacts': {'effective_config': 'effective_config.json', 'effective_config_sha256': sha(run / 'effective_config.json'),
                          'runtime_effective_config': 'output/lingbotvla_cli.yaml', 'stdout_log': 'train.log', 'timings_dir': 'timings',
                          'config_source': 'config-source.yaml', 'config_source_sha256': sha(source), 'launch_command': 'command.sh', 'profiling': []}}
    meta['preparation'] = {'status': 'prepared', 'training_launched': False,
                           'intent_artifact': 'preparation-intent.json',
                           'lifecycle_artifact': 'preparation-lifecycle.tsv'}
    meta['system_candidate'] = {'cpu_affinity': affinity, 'wrapper_sha256': sha(wrapper),
        'rank_cpu_pools': {'0-3': '0-63,128-191', '4-7': '64-127,192-255'},
        'gpu_numa_order_preflight_verified': affinity == 'gpu-numa',
        'actual_rank_affinity_verified': False, 'actual_dataloader_affinity_verified': False,
        'memory_policy_changed': False, 'scope': 'rank CPU affinity; no mbind/set_mempolicy'}
    meta['optimization_controls'].update(nccl_p2p_level_control=p2p_control, nccl_p2p_disable_control=p2p_disable)
    meta['communication_environment'].update(p2p_disable_control=p2p_disable, NCCL_P2P_DISABLE_is_set='NCCL_P2P_DISABLE' in os.environ, NCCL_P2P_DISABLE=os.environ.get('NCCL_P2P_DISABLE'), p2p_level_control=p2p_control, NCCL_P2P_LEVEL_is_set='NCCL_P2P_LEVEL' in os.environ, NCCL_P2P_LEVEL=os.environ.get('NCCL_P2P_LEVEL'))
    meta['harness_context'] = {'variant':'root-teacher-min8-v9','NCCL_MIN_CTAS':'8',
        'same_MIN8_both_root_modes':True,'same260_core_both_modes':True,'root_helper_source_unchanged':True,
        'context_manifest_sha256':root_teacher_source['context_manifest_sha256'],
        'reference_H8_core_manifest_sha256':'924deda41a409781509153639ec8b2230c9284b8355ca2e70aae973066aecf0c'}
    meta['artifacts']['harness_context_manifest'] = 'root-teacher-context.json'
    meta['core_source_identity'] = core_source
    meta['artifacts']['core_source_audit'] = 'core-source-audit.json'
    meta['optimization_controls']['fsdp_forward_prefetch'] = prefetch_mode
    meta['fsdp_forward_prefetch_environment'] = {'control': prefetch_mode, 'LINGBOT_FSDP_FWD_PREFETCH_is_set': 'LINGBOT_FSDP_FWD_PREFETCH' in os.environ,
        'LINGBOT_FSDP_FWD_PREFETCH': os.environ.get('LINGBOT_FSDP_FWD_PREFETCH'), 'LINGBOT_FSDP_FWD_PREFETCH_RECEIPT_DIR': os.environ.get('LINGBOT_FSDP_FWD_PREFETCH_RECEIPT_DIR'),
        'default_backward_preserved': True, 'actual_rank_schedule_verified': False, 'candidate_framework_installed': prefetch_audit['candidate_framework_installed']}
    meta['root_teacher_prefetch_environment']={'control':root_teacher_control,'LINGBOT_FSDP_ROOT_TEACHER_PREFETCH_is_set':'LINGBOT_FSDP_ROOT_TEACHER_PREFETCH' in os.environ,
        'LINGBOT_FSDP_ROOT_TEACHER_PREFETCH':os.environ.get('LINGBOT_FSDP_ROOT_TEACHER_PREFETCH'),
        'source_spec_sha256':root_teacher_source.get('spec_sha256'),'source_audit_status':root_teacher_source['status'],
        'receipt_dir':os.environ.get('LINGBOT_FSDP_ROOT_TEACHER_PREFETCH_RECEIPT_DIR'),
        'first_update_active':False,'eligible_first_active_update':2,'teacher_precision_gradient_sync_changed':False,
        'active_update_gradient_parity_verified':False}
    meta['artifacts'].update(root_teacher_prefetch_source_audit='root-teacher-prefetch-source-audit.json',root_teacher_prefetch_receipts='root-teacher-prefetch/rank*-pid*.jsonl')
    meta['artifacts'].update(fsdp_prefetch_source_audit='fsdp-forward-prefetch-source-audit.json', fsdp_prefetch_setup_receipts='fsdp-forward-prefetch/rank*-pid*.json')
    meta['optimization_controls']['nccl_min_ctas_control'] = min_ctas
    meta['communication_environment'].update(nccl_min_ctas_control=min_ctas, NCCL_MIN_CTAS_is_set='NCCL_MIN_CTAS' in os.environ, NCCL_MIN_CTAS=os.environ.get('NCCL_MIN_CTAS'))
    meta['artifacts']['nccl_channel_environment'] = 'nccl-channel-environment.json'
    meta['artifacts']['p2p_environment'] = 'p2p-environment.json'
    meta['artifacts'].update(cpu_affinity_receipts='numa/rank*-pid*.json', cpu_affinity_wrapper='rtx5k_numa_worker.py',
                             allocator_environment='allocator-environment.json')
    meta['compile_cache'] = {'shared_paths': {name: os.environ[name] for name in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH')},
                             'before_launch': (run / 'compile-cache-before-launch.txt').read_text().strip(), 'actual_cache_hits': 'unknown'}
    if args.kind in ('nsys', 'torch'):
        raw = 'profiling/trace.nsys-rep' if args.kind == 'nsys' else f'profiling/torch-rank0-step{warmup+1}.trace.json'
        meta['artifacts']['profiling'] = [{'tool': args.kind, 'raw_artifact': raw, 'git_commit': commit,
                                          'git_patch_sha256': meta['git']['patch_sha256'], 'effective_config_sha256': meta['artifacts']['effective_config_sha256'],
                                          'optimizer_steps': list(range(warmup+1, warmup+steady+1)) if args.kind == 'nsys' else [warmup+1],
                                          'report_artifact': 'profiling/analysis.md', 'scope': 'diagnostic run; excluded from ordinary performance comparison'}]
        (run / 'profiling/analysis.md').write_text(f'# Profiling analysis pending\n\nSource commit: {commit}\nDirty patch SHA: {meta["git"]["patch_sha256"]}\n\nInspect actual capture completeness before deriving bottlenecks.\n', encoding='utf-8')
    write_json(run / 'experiment.json', meta)
    for helper in ('rtx5k_prepare_sft_run.py', 'rtx5k_launch_sft_run.sh', 'rtx5k_sft_container_entry.sh', 'rtx5k_analyze_single_run.py', 'rtx5k_profile_sft_worker.py', 'rtx5k_profile_torch_worker.py', 'rtx5k_numa_worker.py', 'fsdp_prefetch_source_audit.py', 'core_source_audit.py', 'core-source-manifest.json', 'root_teacher_source_audit.py', 'source-audit-spec.json', 'root-teacher-context.json'):
        path = Path(__file__).with_name(helper)
        require(path.is_file(), f'Missing reviewed helper: {path}')
        (run / helper).write_bytes(path.read_bytes())
        (run / f'{helper}.sha256').write_text(sha(path) + '\n', encoding='utf-8')
    write_json(run / 'source-file-sha256.json', {str(path.relative_to(REPO)): sha(path) for path in [trainer, REPO / 'lingbotvla/utils/training_instrumentation.py', REPO / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py', REPO / 'lingbotvla/models/vla/lingbot_vla/flex_attention.py', REPO / 'lingbotvla/distributed/torch_parallelize.py', REPO / 'lingbotvla/utils/fsdp_root_teacher_prefetch.py']})
    print(shlex.join(command))


if __name__ == '__main__':
    main()
