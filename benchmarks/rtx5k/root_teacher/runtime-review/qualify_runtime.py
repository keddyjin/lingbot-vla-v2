"""CPU-only ordinary3-smoke root-teacher identity/state qualification; no parity."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re

ROOT='/root/keddyj/LingBotVLA-Test'
REPO=ROOT+'/lingbot-vla-v2-sft-opt-github'
HERE=Path(__file__).resolve().parent
SPEC_SHA='513c5586c8441045b1ed6dea80735ca67fb6689407a8172a7cdc60e08ac4d1ef'
MANIFEST_SHA='73178b3c0462370257d09167d6ee24a5f75528dfcbdc34e2d5e99046addd1060'
TRAINER_SHA='d5f7af2803161800f9590c9b50854b386065776fb3f55968a70dff013a9224c8'
HELPER_SHA='34900378bf7e7c21c7cb3b96a924a2a8a31b83e21ca0078ababa373c865de4aa'
FRAMEWORK_SHA='8634aad0e94869c1daa5016bc7173cc24c2ceacbe507ddce7dc5f7194f2acdb1'
VARIANT='original-core-with-frozen-root-teacher-prefetch-installed'
ARCHIVED_AUDITORS={'core_source_audit.py':'7cd4b9a034a397f5ebb4f109a6ee5255e91bc154e60c6b2f97620141b7a27850',
    'root_teacher_source_audit.py':'9ad4bef1c5ff56a432d8ae2cb013ddccd9ff007060ea73884db64b01ba30e8fe',
    'rtx5k_prepare_sft_run.py':'cb225d4d45ac83bef40c15706e34e5c0f35f5b54f17d7e308427c9401c5b2a31',
    'rtx5k_numa_worker.py':'a7d9f341a0955ee8127036510464e66ea83e6a9f64b2f234ac2d3b360d148168'}
POLICY={'reshard_after_forward':False,'reshard_after_backward':True,'reduce_grads':True,'all_reduce_grads':True,
    'unshard_in_backward':True,'unshard_async_op':False,'parameter_dtype':'torch.bfloat16','reduce_dtype':'torch.float32',
    'output_dtype':'torch.bfloat16','cast_forward_inputs':True}
CONTEXT='root-teacher-min8-v9'
CONTEXT_SHA='ee8f0b8803c1390f3aafddd254c02a0cfa5cf386d03cf4ddac6cf5986f21247a'
NCCL_OVERRIDES=('NCCL_PROTO','NCCL_P2P_LEVEL','NCCL_P2P_DISABLE','NCCL_MIN_CTAS','NCCL_MIN_NCHANNELS',
    'NCCL_MIN_NRINGS','NCCL_MAX_CTAS','NCCL_MAX_NCHANNELS','NCCL_MAX_NRINGS')
LIMITS=[
    'Reads completed artifact bytes only; no Torch import or live GPU/process/source access.',
    'Helper directly recorded actual CUDA UUID; joins its rank/PID to same-PID NUMA and preparer mapping. Timing schema has no PID and is joined by rank/run/update only.',
    'JSONL step0 is setup, step1 is first skipped optimizer update, steps2/3 are active. Zero-based conceptual indices are0/1/2.',
    'Pending cleared proves original CPU hook submission/parameter registration, not a GPU completion timestamp or measured teacher overlap.',
    'Later active after-forward emits flags SHA only; exact frozen helper guards attest normal BF16/requires_grad policy, without independent later tensor-row comparison.',
    'Native EP/TP/PP/CP/Ulysses/offload/train qualifications are exact-source helper guards; setup JSONL does not independently emit their complete runtime values. The ordered actual root DP8 process group is independently recorded.',
    'No target/gradient/optimizer-delta parity, loss convergence, performance gain or complete profiling qualification is claimed.',
]


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value,label):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),label+' must be SHA256')
    return value


def integer(value,label,positive=False):
    require(type(value) is int and (value>0 if positive else value>=0),label+' invalid integer')
    return value


def typed_equal(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(typed_equal(actual[key],value) for key,value in expected.items())
    if isinstance(expected,list):return len(actual)==len(expected) and all(typed_equal(left,right) for left,right in zip(actual,expected))
    return actual==expected


def equal(actual,expected,label):
    # Prevent nested bool/int equality from passing a typed JSON guard.
    require(typed_equal(actual,expected),label+' differs')


def unique_object(pairs):
    result={}
    for name,value in pairs:
        require(name not in result,'Duplicate JSON key: '+name);result[name]=value
    return result


def decode(raw):
    return json.loads(raw,object_pairs_hook=unique_object,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant: '+value)))


def identity(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def schema_flags(rows):
    return [{name:{key:value for key,value in flags.items() if key!='version_counter'} if isinstance(flags,dict) else flags
        for name,flags in row.items()} for row in rows]


def tensor(flags,label):
    require(set(flags)=={'dtype','shape','requires_grad','is_inference','version_counter'},label+' tensor flag keys differ')
    require(flags['dtype'] in ('torch.float32','torch.bfloat16','torch.float16'),label+' unexpected baseline floating dtype')
    require(isinstance(flags['shape'],list) and all(type(size) is int and size>=0 for size in flags['shape']),label+' shape invalid')
    require(type(flags['requires_grad']) is bool,label+' requires_grad missing')
    equal(flags['is_inference'],False,label+' inference tensor forbidden')
    integer(flags['version_counter'],label+' version_counter')


def mapping(rows):
    require(isinstance(rows,list) and len(rows)==8,'Exactly8 prepared GPUs required')
    uuids=[]
    for rank,row in enumerate(rows):
        equal(row.get('local_rank'),rank,'Mapping local_rank')
        uuid=row.get('uuid');require(isinstance(uuid,str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid),'Invalid CUDA UUID')
        equal(row.get('compute_capability'),[12,0],'SM120 mapping')
        memory=integer(row.get('total_memory_bytes'),'GPU capacity',True)
        require(70*1024**3<=memory<=74*1024**3,'Unexpected72GB capacity');uuids.append(uuid.lower())
    require(len(set(uuids))==8,'Duplicate physical GPU UUID')
    return uuids


def status(row,*,root,sharded,pending,label):
    for name,expected in {'is_root':root,'state_training':'IDLE','group_training':'IDLE','is_sharded':sharded,
        'is_unsharded':not sharded,'pending_all_gather':pending,'forward_root_active':False,
        'post_optim_event_present':False,'backward_final_callback_queued':False,
        'explicit_forward_target_count':0,'explicit_backward_target_count':0}.items():equal(row.get(name),expected,label+' '+name)


def context(row):
    equal(row.get('grad_enabled'),True,'Original grad-enabled context')
    equal(row.get('inference_mode_enabled'),False,'No inference allocation context')


def validate_rank(rows,rank,pid,prepared,expected_sources,meta,prepared_sha,prepared_spec_path):
    expected=[('setup',0),('root_process_group',0)]+[(phase,step) for step in (1,2,3) for phase in ('before_teachers','after_original_forward')]
    require(len(rows)==8,'Exactly setup/process-group and6 step records required')
    require([(row.get('phase'),row.get('step')) for row in rows]==expected,'Exact chronological phases/actual step1skip2/3active required')
    stamps=[]
    for row in rows:
        equal(row.get('schema_version'),1,'Runtime schema');equal(row.get('rank'),rank,'Runtime rank');equal(row.get('pid'),pid,'Runtime PID')
        equal(row.get('mode'),'on','Runtime mode');integer(row['step'],'Runtime step')
        stamp=datetime.fromisoformat(row['utc']);require(stamp.tzinfo is not None,'Runtime UTC timestamp must include timezone');stamps.append(stamp)
    require(stamps==sorted(stamps),'Runtime timestamp order differs')
    setup=rows[0];source=setup['source'];context(setup)
    for name in ('root_module_identity','root_state_identity','root_group_identity'):integer(setup.get(name),name,True)
    require(len({setup[name] for name in ('root_module_identity','root_state_identity','root_group_identity')})==3,'Root/module/state/group identities alias')
    equal(setup.get('compiled_wrapper_class'),'OptimizedModule','Canonical compiled wrapper')
    equal(setup.get('root_class'),'FSDPLingbotVlaV2Policy','Actual root class')
    equal(setup.get('policy_before'),POLICY,'Root original exact MP/gradient/reshard policy')
    status(setup['status_before'],root=None,sharded=True,pending=False,label='Fresh root setup')
    for name in ('teacher_math_changed','gradient_sync_changed','precision_changed','clip_changed'):equal(setup.get(name),False,name)
    equal(setup.get('first_step_gate'),'skip step1; original root forward performs state lazy-init; eligible step2+','Lazy gate text')
    equal(setup.get('wait_owner'),'original root FSDP pre_forward -> group.wait_for_unshard -> copy_out Work.wait; no candidate wait','Canonical original wait owner')
    device=setup['actual_device']
    require(set(device)=={'local_rank','uuid','compute_capability','total_memory_bytes'},'Actual runtime device schema differs')
    require(device['uuid'].lower()==prepared[rank]['uuid'].lower(),'Actual rank UUID differs from preparer')
    for name in ('local_rank','compute_capability','total_memory_bytes'):equal(device.get(name),prepared[rank][name],'Actual device '+name)
    require(source['actual_source_sha256']==expected_sources,'Runtime exact14 source map differs')
    equal(source.get('spec_sha256'),SPEC_SHA,'Runtime spec513')
    require(source.get('prepared_cuda_devices')==prepared,'Runtime complete preparer device map differs')
    equal(source.get('prepared_cuda_devices_sha256'),prepared_sha,'Runtime mapping raw SHA')
    require(source.get('spec_path','').startswith(ROOT+'/') and not any(part in ('.','..') for part in source['spec_path'].split('/')),'Runtime spec path escapes ROOT')
    equal(source.get('spec_path'),prepared_spec_path,'Actual runtime/preparer spec path join')
    equal(source.get('git_head'),meta['git']['commit'],'Runtime actual HEAD')
    equal(source.get('git_branch'),meta['git']['branch'],'Runtime branch')
    equal(source.get('runtime_HEAD_equality_required'),False,'Runtime reference HEAD nongate')
    equal(source.get('baseline_source_reference_commit'),'7bb4ef03fb9611aab3ab7885370e6f7e0ed625c8','Runtime original byte reference')
    equal(source.get('dirty_tracked_patch_sha256'),meta['git']['patch_sha256'],'Runtime tracked dirty patch SHA')
    equal(rows[1].get('actual_process_group'),{'world_size':8,'rank':rank,'global_members':list(range(8))},'Ordered native root DP8 process group')
    equal(rows[1].get('qualification_collective_issued'),False,'No qualification collective added')
    count=integer(setup.get('root_parameter_count'),'Root parameter count',True);integer(setup.get('root_parameter_numel'),'Root parameter numel',True)
    original=setup['original_sharded_parameter_flags'];require(isinstance(original,list) and len(original)==count,'Original root flags cardinality differs')
    for index,row in enumerate(original):
        require(set(row)=={'index','sharded_parameter','sharded_data'},'Original sharded row schema differs')
        equal(row.get('index'),index,'Parameter ordinal');tensor(row['sharded_parameter'],'Original parameter');tensor(row['sharded_data'],'Original data')
        require(row['sharded_parameter']['dtype']==row['sharded_data']['dtype'],'Original sharded/data dtype mismatch')
        equal(row['sharded_data']['requires_grad'],False,'Original underlying sharded data no-grad')
    equal(setup.get('original_sharded_schema_sha256'),identity(schema_flags(original)),'Original normal/gradient/precision schema hash')
    for step in (1,2,3):
        before,after=rows[2+2*(step-1):4+2*(step-1)];context(before);context(after)
        equal(before.get('policy_before'),POLICY,'Before root policy');equal(after.get('policy_after'),POLICY,'After root policy')
        status(before['status_before'],root=None if step==1 else True,sharded=True,pending=False,label='Before teacher step'+str(step))
        status(after['status_after'],root=True,sharded=False,pending=False,label='After original forward step'+str(step))
        equal(after.get('explicit_wait_called'),False,'No candidate wait')
        equal(after.get('pending_consumption_scope'),'Original hook finished CPU submission and registered unsharded parameters; not a GPU-completion timestamp','CPU-only pending interpretation')
        digest(after.get('unsharded_parameter_flags_sha256'),'After root flags SHA')
        if step==1:
            equal(before.get('status'),'skipped_first_lazy_init','First optimizer update skipped')
            equal(before.get('unshard_called'),False,'No first-update public unshard')
            equal(before.get('sharded_parameter_flags_sha256'),identity(original),'First-before unchanged sharded flags')
            equal(after.get('status'),'first_original_lazy_init_complete','First original lazy-init completion')
            flags=after.get('first_original_unsharded_parameter_flags');require(isinstance(flags,list) and len(flags)==count,'First actual unsharded flags cardinality differs')
            equal(after['unsharded_parameter_flags_sha256'],identity(flags),'First unsharded flags raw hash')
            for index,(observed,initial) in enumerate(zip(flags,original)):
                require(set(observed)=={'index','sharded_parameter','sharded_data','unsharded_parameter'},'First unsharded flags row differs')
                equal(observed['index'],index,'First unsharded ordinal')
                for key in ('sharded_parameter','sharded_data','unsharded_parameter'):tensor(observed[key],'First '+key)
                equal(observed['unsharded_parameter']['dtype'],'torch.bfloat16','Root actual unsharded MP dtype')
                equal(observed['unsharded_parameter']['requires_grad'],initial['sharded_parameter']['requires_grad'],'Original unsharded gradient mask')
        else:
            equal(before.get('status'),'async_unshard_submitted','Active public async submit')
            equal(before.get('unshard_called'),True,'One public submit');equal(before.get('async_op'),True,'Public async true')
            equal(before.get('handle_class'),'_UnshardHandleImpl','Canonical public handle')
            require(isinstance(before.get('work_class'),str) and bool(before['work_class']),'Pending Work class observation required')
            equal(before.get('policy_after'),POLICY,'After submit policy')
            status(before['status_after_submit'],root=True,sharded=True,pending=True,label='Public submit step'+str(step))
            digest(before.get('sharded_parameter_flags_before_sha256'),'Before submit flags SHA')
            equal(before.get('sharded_parameter_flags_after_submit_sha256'),before['sharded_parameter_flags_before_sha256'],'Submit preserved version/gradient/inference flags')
            tensor(before['gather_output_flags'],'Async gather output')
            equal(before['gather_output_flags']['dtype'],'torch.bfloat16','Original MP gather dtype')
            equal(before['gather_output_flags']['requires_grad'],False,'Gather output no-grad')
            require(type(before.get('pending_event_present')) is bool,'Pending event observation required; either boolean is original API valid')
            equal(before.get('explicit_wait_called'),False,'Candidate does not wait')
            equal(before.get('baseline_step_sync_precedes_insertion'),True,'Original step sync retained')
            equal(after.get('status'),'original_preforward_wait_consumed_pending','Native forward consumed async pending')
            equal(after.get('first_original_unsharded_parameter_flags'),None,'Later full tensor rows absent as original schema')
    return {'rank':rank,'pid':pid,'actual_cuda_uuid':device['uuid'],'setup_step':0,'first_skipped_step':1,
        'active_submitted_consumed_steps':[2,3],'zero_based_optimizer_update_indices':[0,1,2],
        'public_async_submit_count':2,'root_parameter_count':count,'root_module_identity':setup['root_module_identity'],
        'root_state_identity':setup['root_state_identity'],'root_group_identity':setup['root_group_identity'],
        'UUID_evidence':'actual helper torch.cuda device property plus exact same-rank preparer mapping and same-PID ordinary NUMA worker',
        'status':'passed','GPU_completion_timestamp_verified':False,'active_update_gradient_parity_verified':False}


def qualify(run):
    run=Path(run).resolve()
    result={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'status':'failed',
        'runtime_identity_qualified':False,'completed_run_artifacts_examined':True,'full_parity_claim':False,
        'active_update_gradient_parity_verified':False,'performance_qualified':False,'live_GPU_read':False,
        'harness_context_variant':CONTEXT,'qualifier_sha256':sha(Path(__file__)),'scope':'ordinary completed on-mode3-update SFT smoke only',
        'evidence_limits':LIMITS,'artifacts':{},'rank_results':[],'errors':[]}
    def file(name):
        require(isinstance(name,str),'Artifact name must be string');path=(run/name).resolve()
        require(path.is_relative_to(run) and path.is_file(),'Missing/escaped artifact: '+name)
        result['artifacts'][name]={'sha256':sha(path),'bytes':path.stat().st_size};return path
    def read(name):return decode(file(name).read_text(encoding='utf-8-sig'))
    def jsonl(name):
        lines=file(name).read_text(encoding='utf-8-sig').splitlines();require(all(line.strip() for line in lines),'Empty JSONL line: '+name)
        return [decode(line) for line in lines]
    try:
        require(sha(HERE/'expected-source-spec.json')==SPEC_SHA and sha(HERE/'expected-core-manifest.json')==MANIFEST_SHA,'Qualifier expected frozen spec/manifest bytes differ')
        spec=decode((HERE/'expected-source-spec.json').read_text());manifest=decode((HERE/'expected-core-manifest.json').read_text())
        expected_sources={ROOT+'/'+relative:values[0] for relative,values in spec['required_sources'].items()}
        require(len(expected_sources)==14 and all(len(values)==1 for values in spec['required_sources'].values()),'Qualifier exact14 pins required')
        equal(sha(HERE/'expected-context-manifest.json'),CONTEXT_SHA,'Qualifier frozen MIN8 context')
        equal(sha(file('root-teacher-context.json')),CONTEXT_SHA,'Archived frozen MIN8 context')
        meta=read('experiment.json');config=read('effective_config.json')
        for name,value in {'variant':CONTEXT,'NCCL_MIN_CTAS':'8','same_MIN8_both_root_modes':True,
            'same260_core_both_modes':True,'root_helper_source_unchanged':True,'context_manifest_sha256':CONTEXT_SHA,
            'reference_H8_core_manifest_sha256':'924deda41a409781509153639ec8b2230c9284b8355ca2e70aae973066aecf0c'}.items():equal(meta['harness_context'].get(name),value,'H9 context '+name)
        equal(meta.get('experiment_id'),run.name,'Run id');equal(meta.get('status'),'completed','Completed ordinary run')
        equal(meta.get('workload_kind'),'real_sft','Ordinary SFT only, no diagnostic target');equal(meta.get('training_exit_code'),0,'Training exit0')
        equal(meta['preparation'].get('status'),'completed','Preparation completed');equal(meta['preparation'].get('training_launched'),True,'Actual training launched')
        require(re.fullmatch('[0-9a-f]{40}',meta['git']['commit']) is not None,'Actual full HEAD required')
        equal(meta['git']['branch'],'keddyj-codex/sft-opt-rtx5k','Actual branch')
        controls={'root_teacher_prefetch':'on','fsdp_forward_prefetch':'off','flex_bwd_tile':'off','flex_bwd_num_stages':'off',
            'selective_checkpoint_policy':'official','compile_step_counter_helper':False,'deepstack_indices':False,'batch_depth_teachers':False,
            'moe_segment_output_size':False,'dist_muon_layers_per_bucket':2,'precision_changed':False,'cpu_affinity':'off',
            'cuda_allocator':'unset','nccl_proto_control':'unset','nccl_p2p_level_control':'default','nccl_p2p_disable_control':'unset','nccl_min_ctas_control':'8'}
        for name,value in controls.items():equal(meta['optimization_controls'].get(name),value,'Root-only control '+name)
        env=meta['root_teacher_prefetch_environment']
        for name,value in {'control':'on','LINGBOT_FSDP_ROOT_TEACHER_PREFETCH_is_set':True,'LINGBOT_FSDP_ROOT_TEACHER_PREFETCH':'on',
            'source_spec_sha256':SPEC_SHA,'source_audit_status':'passed','first_update_active':False,'eligible_first_active_update':2,
            'teacher_precision_gradient_sync_changed':False,'active_update_gradient_parity_verified':False}.items():equal(env.get(name),value,'Root preparation '+name)
        remote_run=ROOT+'/runs/'+run.name;equal(env.get('receipt_dir'),remote_run+'/root-teacher-prefetch','Runtime receipt scope')
        train=meta['training'];parsed=config['train']
        for name,value in {'world_size':8,'micro_batch_size':32,'gradient_accumulation_steps':1,'global_batch_size':256,
            'optimizer':'dist_muon','resume':False,'benchmark_skip_checkpoint':True,'stop_after_steps':3,'gradient_checkpointing':True}.items():equal(train.get(name),value,'Prepared training '+name)
        equal(train.get('memory_policy'),{'name':'gc','enable_gradient_checkpointing':True,'enable_full_shard':False},'Pure gc-only/full_shardfalse')
        for name,value in {'micro_batch_size':32,'global_batch_size':256,'optimizer':'dist_muon','enable_gradient_checkpointing':True,
            'enable_full_shard':False,'enable_fp32':True,'module_fsdp_enable':True,'use_compile':True,'data_parallel_mode':'fsdp2',
            'enable_resume':False,'save_steps':0,'save_epochs':0,'save_hf_weights':False,'async_save_hf_weights':False,'enable_profiling':False}.items():equal(parsed.get(name),value,'Effective training '+name)
        equal(parsed.get('gradient_accumulation_steps',1),1,'Accum1');equal(parsed.get('load_checkpoint_path'),None,'Fresh no restore')
        for name in ('enable_fsdp_offload','enable_activation_offload'):equal(parsed.get(name,False),False,'No offload '+name)
        for name in ('tensor_parallel_size','expert_parallel_size','pipeline_parallel_size','context_parallel_size','ulysses_parallel_size','data_parallel_replicate_size'):equal(parsed.get(name,1),1,'Native no extra parallel '+name)
        for name in ('world_size','data_parallel_size','data_parallel_shard_size'):require(parsed.get(name) is None or type(parsed[name]) is int and parsed[name]==8,'DP8 effective setting differs '+name)
        align=parsed['align_params'];require(align['depth_loss_weight']>0 and align['future_depth_loss_weight']>0,'Both current/future depth required');equal(align.get('use_future_video'),True,'Future video teacher required')
        equal(meta['measurement'].get('profiling_enabled'),False,'Ordinary smoke, no profiler');equal(meta['measurement'].get('warmup_steps'),3,'Ordinary3 smoke warmup')
        equal(meta['artifacts'].get('effective_config_sha256'),result['artifacts']['effective_config.json']['sha256'],'Effective config SHA')
        runtime_name=meta['artifacts']['runtime_effective_config'];runtime=file(runtime_name)
        equal(sha(runtime),meta['artifacts'].get('runtime_effective_config_sha256'),'Actual runtime YAML bytes SHA')
        for name,expected in [('dirty-tracked.patch',meta['git']['patch_sha256']),('dirty-untracked.tar.gz',meta['git']['untracked_sha256']),
            ('core-source-manifest.json',MANIFEST_SHA),('source-audit-spec.json',SPEC_SHA),('root-teacher-prefetch-source-spec.json',SPEC_SHA),*ARCHIVED_AUDITORS.items()]:equal(sha(file(name)),expected,'Archived source '+name)
        core=read('core-source-audit.json');require(core==meta['core_source_identity'],'Core audit metadata join differs')
        for name,value in {'status':'passed','source_variant':VARIANT,'core_source_file_count':260,'manifest_sha256':MANIFEST_SHA,
            'reference_HEAD_is_runtime_gate':False,'source_variant_applies_to_both_root_modes':True,'core_source_bytes_pinned':True,
            'extra_core_source_files_verified_absent':True,'actual_HEAD':meta['git']['commit'],'actual_branch':meta['git']['branch']}.items():equal(core.get(name),value,'Exact260 core audit '+name)
        equal(core.get('core_source_sha256'),manifest['files'],'Complete exact260 audit');equal(core.get('source_changes_from_baseline'),manifest['source_changes_from_baseline'],'Exact260 delta audit')
        equal(core.get('actual_dirty_diff_sha256'),meta['git']['patch_sha256'],'Core/runtime dirty patch join')
        root_audit=read('root-teacher-prefetch-source-audit.json')
        for name,value in {'harness_context_variant':CONTEXT,'NCCL_MIN_CTAS_context':'8','context_manifest_sha256':CONTEXT_SHA,'root_helper_source_unchanged':True}.items():equal(root_audit.get(name),value,'Root H9 audit context '+name)
        equal(core.get('harness_context_variant'),CONTEXT,'Core H9 context');equal(core.get('NCCL_MIN_CTAS_context'),'8','Core MIN8 context')
        for name,value in {'status':'passed','control':'on','source_variant':VARIANT,'spec_sha256':SPEC_SHA,
            'original_framework_required':FRAMEWORK_SHA,'same_installed_sources_required_in_both_modes':True,
            'runtime_HEAD_equality_required':False,'first_update_active':False,'first_active_update':2,
            'active_update_gradient_parity_verified':False,'full_parity_claim':False}.items():equal(root_audit.get(name),value,'Prepared root audit '+name)
        require(root_audit.get('actual_source_sha256')==expected_sources,'Prepared exact14 source audit differs')
        sources=read('source-file-sha256.json')
        for name,digest_value in [('tasks/vla/train_lingbotvla.py',TRAINER_SHA),('lingbotvla/utils/fsdp_root_teacher_prefetch.py',HELPER_SHA),('lingbotvla/distributed/torch_parallelize.py',FRAMEWORK_SHA)]:equal(sources.get(name),digest_value,'Observed installed source '+name)
        prepared=read('cuda-devices.json');uuids=mapping(prepared);prepared_sha=result['artifacts']['cuda-devices.json']['sha256']
        require(meta['docker']['cuda_devices']==prepared,'Docker prepared map join differs');equal(meta['docker']['cuda_visible_devices_inside'],'0,1,2,3,4,5,6,7','Docker visibility')
        require(sorted(value.lower() for value in meta['docker']['gpu_request'].split(','))==sorted(uuids),'Docker UUID pool differs')
        require(sorted(value.lower() for value in meta['fair_comparison']['gpu_uuids'])==sorted(uuids),'Comparison UUID pool differs')
        versions=read('versions.json');equal(versions['packages'].get('torch'),'2.8.0+cu128','Torch pinned');equal(versions.get('loaded_nccl_version'),[2,27,3],'NCCL loaded pinned')
        checkpoint=meta['checkpoint_inventory']
        for name,value in {'payloads':[],'payload_bytes':0,'unexpected_payload':False}.items():equal(checkpoint.get(name),value,'Checkpoint payload absent '+name)
        equal(meta['system_candidate'].get('cpu_affinity'),'off','Default affinity');equal(meta['system_candidate'].get('wrapper_sha256'),ARCHIVED_AUDITORS['rtx5k_numa_worker.py'],'Numa wrapper source')
        require(meta['communication_environment'].get('loaded_nccl_version')==[2,27,3],'Prepared loaded NCCL identity')
        nccl=read('nccl-environment.json');require(not any(name in nccl['environment'] for name in NCCL_OVERRIDES if name!='NCCL_MIN_CTAS'),'Actual NCCL override masks root+MIN8 trial')
        equal(nccl['environment'].get('NCCL_MIN_CTAS'),'8','Actual prepared MIN8')
        equal(meta['communication_environment']['NCCL_environment'].get('NCCL_MIN_CTAS'),'8','Meta actual MIN8')
        for name,value in {'nccl_min_ctas_control':'8','NCCL_MIN_CTAS_is_set':True,'NCCL_MIN_CTAS':'8'}.items():equal(meta['communication_environment'].get(name),value,'Meta typed MIN8 '+name)
        for name in NCCL_OVERRIDES:
            if name=='NCCL_MIN_CTAS':continue
            require(name not in meta['communication_environment']['NCCL_environment'],'Meta actual NCCL override '+name)
        allocator=read('allocator-environment.json');equal(allocator.get('control'),'unset','Default allocator')
        for name in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):equal(allocator.get(name),None,'Allocator override absent '+name)
        root_files=list((run/'root-teacher-prefetch').glob('*'));require(len(root_files)==8 and all(path.is_file() and path.suffix=='.jsonl' for path in root_files),'Exactly8 JSONL; no setup/error/extra file')
        require(len(list((run/'numa').glob('rank*-pid*.json')))==8,'Exactly8 NUMA PID files')
        require(len(list((run/'timings').glob('rank-*.jsonl')))==8,'Exactly8 timing rank files')
        require(not list((run/'timings').glob('measurement-status-rank-*.json')),'Invalid timing status artifact')
        seen_pids=set()
        for rank in range(8):
            item={'rank':rank,'status':'failed'};result['rank_results'].append(item)
            try:
                candidates=list((run/'root-teacher-prefetch').glob(f'rank{rank:02d}-pid*.jsonl'));require(len(candidates)==1,'Exactly one root runtime PID file per rank')
                rows=jsonl(candidates[0].relative_to(run).as_posix());pid=integer(rows[0].get('pid'),'Runtime PID',True)
                require(pid not in seen_pids,'Duplicate rank PID');seen_pids.add(pid)
                equal(candidates[0].name,f'rank{rank:02d}-pid{pid}.jsonl','Runtime PID filename join')
                item.update(validate_rank(rows,rank,pid,prepared,expected_sources,meta,prepared_sha,root_audit['spec_path']))
                numa=read(f'numa/rank{rank}-pid{pid}.json');equal(numa.get('rank'),str(rank),'NUMA rank');equal(numa.get('pid'),pid,'NUMA PID')
                equal(numa.get('status'),'target_exit','Ordinary worker completed');equal(numa.get('exit_code'),0,'Ordinary worker exit0')
                equal(numa.get('target'),REPO+'/tasks/vla/train_lingbotvla.py','Ordinary trainer only');equal(numa.get('target_sha256'),TRAINER_SHA,'Exact ordinary trainer d5')
                equal(numa.get('worker_sha256'),ARCHIVED_AUDITORS['rtx5k_numa_worker.py'],'Actual wrapper hash')
                require(numa['prepared_source']==meta['git'] and numa['prepared_batch_and_stop']==train and numa['prepared_gpu_mapping']==prepared,'NUMA prepared source/training/GPU joins differ')
                require(numa['policy']=={'mode':'off','numa_node':None,'requested_cpus':None},'Default NUMA policy only');equal(numa.get('memory_binding'),'none; no mbind/set_mempolicy','No memory binding')
                nenv=numa['environment']
                for name,value in {'RANK':str(rank),'LOCAL_RANK':str(rank),'WORLD_SIZE':'8','LOCAL_WORLD_SIZE':'8','SFT_RUN_DIR':remote_run,
                    'SFT_KIND':'smoke','SFT_WORLD_SIZE':'8','CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','LINGBOT_BENCHMARK_SKIP_CHECKPOINT':'1',
                    'LINGBOT_BENCHMARK_STOP_STEPS':'3','LINGBOT_GC_POLICY':'official','LINGBOT_FLEX_BWD_TILE':'off','LINGBOT_CPU_AFFINITY':'off',
                    'SFT_NCCL_P2P_LEVEL_CONTROL':'default','SFT_NCCL_P2P_DISABLE_CONTROL':'unset','SFT_NCCL_MIN_CTAS_CONTROL':'8'}.items():equal(nenv.get(name),value,'Actual NUMA environment '+name)
                for name in NCCL_OVERRIDES:
                    if name=='NCCL_MIN_CTAS':
                        equal(nenv.get(name),'8','Rank actual MIN8 literal');equal(numa['communication_controls'].get(name+'_is_set'),True,'Rank actual MIN8 present');equal(numa['communication_controls'].get(name),'8','Rank actual MIN8 value')
                        continue
                    require(name not in nenv,'Actual rank NCCL env override '+name);equal(numa['communication_controls'].get(name+'_is_set'),False,'Rank actual '+name+' absent');equal(numa['communication_controls'].get(name),None,'Rank actual '+name+' null')
                ticks=[]
                for phase in ('before','after_binding','at_exit'):
                    equal(numa[phase].get('pid'),pid,'Observed process PID '+phase);ticks.append(integer(numa[phase].get('process_start_ticks'),'Process start ticks',True))
                require(len(set(ticks))==1,'Process identity changed across NUMA snapshots')
                times=jsonl(f'timings/rank-{rank:02d}.jsonl');require(len(times)==3,'Three completed optimizer update timings required')
                for step,tick in enumerate(times,1):
                    for name,value in {'schema_version':1,'run_id':run.name,'rank':rank,'local_rank':rank,'world_size':8,'data_parallel_size':8,
                        'global_step':step,'step':step,'optimizer_step':step,'micro_batch_size':32,'global_batch_size':256,'accumulation_steps':1,
                        'optimizer':'dist_muon','micro_batches':1,'local_samples':32,'samples':256,'batch_complete':True,'checkpoint_step':False,
                        'checkpoint_reasons':[],'visual_step':False,'profile_step':False,'end_sync':True,'phase':'warmup',
                        'terminal_step':step==3,'epoch_terminal_step':step==3}.items():equal(tick.get(name),value,'Timing complete update '+name)
                    for name in ('original_step_time_s','end_to_end_step_time_s'):require(type(tick.get(name)) in (int,float) and math.isfinite(tick[name]) and tick[name]>0,'Finite positive timing '+name)
                item['timing_PID_evidence']='Timing schema has no PID; exact rank/run/step join to same-PID NUMA and root runtime records'
            except (ValueError,KeyError,TypeError,AttributeError,OSError) as error:
                item.update(status='failed',error=str(error));result['errors'].append(f'rank{rank}: {error}')
        if not result['errors'] and len(result['rank_results'])==8:result.update(status='passed',runtime_identity_qualified=True,
            actual_step_mapping={'setup':0,'skipped_first':1,'active':[2,3]},public_async_submits_total=16,
            native_original_forward_consumed_total=16,first_skipped_update_is_active_parity=False)
    except (ValueError,KeyError,TypeError,AttributeError,OSError) as error:result['errors'].append(str(error))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();allowed=(Path(ROOT).resolve(),HERE.parents[2])
    for path in (args.run_dir,args.output):require(any(path.resolve()!=base and path.resolve().is_relative_to(base) for base in allowed),'Input/output outside authorized task ROOT')
    result=qualify(args.run_dir);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'runtime_identity_qualified':result['runtime_identity_qualified'],'full_parity_claim':False,'errors':result['errors'],'output':str(args.output)}))
    return 0 if result['runtime_identity_qualified'] else 1


if __name__=='__main__':raise SystemExit(main())
