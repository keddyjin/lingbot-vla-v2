"""Default-off public root FSDP async unshard before unchanged teachers."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/root/keddyj/LingBotVLA-Test')
CONTROL = 'LINGBOT_FSDP_ROOT_TEACHER_PREFETCH'
SPEC = CONTROL + '_SOURCE_SPEC'
RECEIPTS = CONTROL + '_RECEIPT_DIR'


def root_teacher_prefetch_mode(environment=None):
    environment = os.environ if environment is None else environment
    mode = environment.get(CONTROL, 'off').strip().lower()
    if mode not in ('off', 'on'):
        raise ValueError(CONTROL + ' must be off or on')
    return mode


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confined(path):
    path = Path(path).resolve()
    require(path != ROOT.resolve() and path.is_relative_to(ROOT.resolve()), 'Root-prefetch path escapes authorized ROOT')
    return path


def source_audit(environment):
    spec_path = confined(environment[SPEC]); spec = json.loads(spec_path.read_text(encoding='utf-8'))
    require(spec.get('schema_version') == 1 and spec.get('baseline_commit') == '7bb4ef03fb9611aab3ab7885370e6f7e0ed625c8', 'Unknown root-prefetch source spec')
    require(spec.get('candidate') == 'root-teacher-prefetch-rtx5k', 'Wrong candidate source spec')
    repo = 'lingbot-vla-v2-sft-opt-github/'
    wheel = 'environment/venv/lib/python3.12/site-packages/torch/'
    required = {repo + name for name in ('tasks/vla/train_lingbotvla.py', 'lingbotvla/utils/fsdp_root_teacher_prefetch.py',
        'lingbotvla/distributed/torch_parallelize.py', 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
        'lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py', 'lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py',
        'lingbotvla/models/vla/vision_models/module_utils.py')}
    required.update(wheel + 'distributed/fsdp/_fully_shard/' + name for name in ('_fully_shard.py','_fsdp_state.py','_fsdp_param_group.py','_fsdp_param.py','_fsdp_collectives.py'))
    required.add(wheel + '_dynamo/eval_frame.py')
    required.add(wheel + 'distributed/fsdp/__init__.py')
    require(set(spec.get('required_sources', {})) == required, 'Source spec must pin exact trainer/helper/framework/classes/teachers and seven wheel files')
    actual = {}
    for relative, accepted in spec['required_sources'].items():
        path = confined(ROOT / relative); observed = sha(path)
        require(isinstance(accepted, list) and observed in accepted, 'Actual source differs from frozen root-prefetch spec: ' + relative)
        actual[str(path)] = observed
    require(str(Path(__file__).resolve()) in actual, 'Spec did not pin this installed helper')
    import subprocess
    repo_path=ROOT/'lingbot-vla-v2-sft-opt-github'
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-C',str(repo_path),*args])
    head=git('rev-parse','HEAD').decode().strip(); branch=git('branch','--show-current').decode().strip()
    require(len(head)==40 and all(char in '0123456789abcdef' for char in head) and branch=='keddyj-codex/sft-opt-rtx5k', 'Actual task Git commit/branch identity invalid')
    run = confined(environment['SFT_RUN_DIR']); mapping_path = run/'cuda-devices.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    require(len(mapping)==8 and [row.get('local_rank') for row in mapping]==list(range(8)) and len({row.get('uuid','').lower() for row in mapping})==8, 'Actual preparer GPU mapping is not eight unique ordered devices')
    return {'spec_path': str(spec_path), 'spec_sha256': sha(spec_path), 'actual_source_sha256': actual,
            'prepared_cuda_devices':mapping,'prepared_cuda_devices_sha256':sha(mapping_path),
            'git_head':head,'git_branch':branch,'baseline_source_reference_commit':spec['baseline_commit'],
            'runtime_HEAD_equality_required':False,'dirty_tracked_patch_sha256':hashlib.sha256(git('diff','--binary','HEAD')).hexdigest()}


def policy_snapshot(group):
    return {'reshard_after_forward': group._reshard_after_forward, 'reshard_after_backward': group.reshard_after_backward,
        'reduce_grads': group.reduce_grads, 'all_reduce_grads': group.all_reduce_grads,
        'unshard_in_backward': group.unshard_in_backward, 'unshard_async_op': group.unshard_async_op,
        'parameter_dtype': str(group.mp_policy.param_dtype), 'reduce_dtype': str(group.mp_policy.reduce_dtype),
        'output_dtype': str(group.mp_policy.output_dtype), 'cast_forward_inputs': group.mp_policy.cast_forward_inputs}


def tensor_flags(tensor):
    inference=bool(tensor.is_inference())
    return {'dtype':str(tensor.dtype),'shape':list(tensor.shape),'requires_grad':bool(tensor.requires_grad),
            'is_inference':inference,'version_counter':None if inference else int(tensor._version)}


def parameter_flags(group,unsharded=False):
    rows=[]
    for index,param in enumerate(group.fsdp_params):
        row={'index':index,'sharded_parameter':tensor_flags(param.sharded_param),'sharded_data':tensor_flags(param._sharded_param_data)}
        if unsharded:row['unsharded_parameter']=tensor_flags(param.unsharded_param)
        rows.append(row)
    return rows


def schema_flags(rows):
    return [{name:{key:value for key,value in flags.items() if key!='version_counter'} if isinstance(flags,dict) else flags
             for name,flags in row.items()} for row in rows]


def flags_identity(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def group_status(state, group):
    return {'is_root': state._is_root, 'state_training': state._training_state.name,
        'group_training': group._training_state.name, 'is_sharded': group.is_sharded,
        'is_unsharded': group.is_unsharded, 'pending_all_gather': group._all_gather_result is not None,
        'forward_root_active': state._state_ctx.iter_forward_root is not None,
        'post_optim_event_present': state._state_ctx.post_optim_event is not None,
        'backward_final_callback_queued': state._state_ctx.post_backward_final_callback_queued,
        'explicit_forward_target_count': len(state._states_to_forward_prefetch),
        'explicit_backward_target_count': len(state._states_to_backward_prefetch)}


def build_root_teacher_prefetch(model, mode, train, parallel, use_depth_align, use_future_depth, use_future_video,
                                *, torch_module=None, classes=None, source_checker=None, writer=None, environment=None):
    # Off avoids touching the model, framework state, teachers, Torch or files.
    require(mode in ('off', 'on'), 'Unknown root teacher prefetch mode')
    if mode == 'off': return None
    environment = os.environ if environment is None else environment
    try:
        return RootTeacherPrefetch(model, train, parallel, use_depth_align, use_future_depth, use_future_video,
            torch_module=torch_module, classes=classes, source_checker=source_checker, writer=writer, environment=environment)
    except BaseException as error:
        row={'schema_version':1,'utc':datetime.now(timezone.utc).isoformat(),'rank':parallel.global_rank,
            'pid':os.getpid(),'mode':'on','phase':'setup_failed','step':0,'error_type':type(error).__name__,'error':str(error),
            'GPU_unshard_called':False,'full_parity_claim':False}
        if writer is not None:writer(row)
        else:
            # Preserve original exception and never overwrite existing evidence.
            try:
                directory=confined(environment[RECEIPTS]); directory.mkdir(parents=True,exist_ok=True)
                with (directory/f'rank{parallel.global_rank:02d}-pid{os.getpid()}-setup-failed.json').open('x',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
            except (OSError,ValueError,KeyError):pass
        raise


class RootTeacherPrefetch:
    def __init__(self, model, train, parallel, use_depth_align, use_future_depth, use_future_video,
                 *, torch_module=None, classes=None, source_checker=None, writer=None, environment=None):
        self.environment = os.environ if environment is None else environment
        require(root_teacher_prefetch_mode(self.environment) == 'on', 'Actual root-prefetch mode must be on')
        for name, expected in {'LINGBOT_FSDP_FWD_PREFETCH':'off', 'LINGBOT_GC_POLICY':'official',
            'LINGBOT_FLEX_BWD_TILE':'off', 'LINGBOT_FLEX_BWD_NUM_STAGES':'off', 'LINGBOT_DEPTH_TEACHER_BATCH':'0',
            'LINGBOT_TEACHER_METHOD_COMPILE':'off', 'LINGBOT_COMPILE_STEP_COUNTER_HELPER':'0',
            'LINGBOT_DEEPSTACK_INDICES':'0', 'LINGBOT_MOE_SEGMENT_OUTPUT_SIZE':'0'}.items():
            require(self.environment.get(name, expected) == expected, 'Unrelated candidate enabled: ' + name)
        require(self.environment.get('LINGBOT_BENCHMARK_SKIP_CHECKPOINT') == '1', 'Fresh no-checkpoint benchmark required')
        expected_train = {'world_size':8, 'global_rank':parallel.global_rank, 'local_rank':parallel.local_rank,
            'data_parallel_size':8, 'data_parallel_mode':'fsdp2', 'enable_full_shard':False,
            'enable_gradient_checkpointing':True, 'module_fsdp_enable':True, 'enable_fp32':True,
            'enable_fsdp_offload':False, 'enable_activation_offload':False,
            'use_compile':True, 'micro_batch_size':32, 'gradient_accumulation_steps':1, 'global_batch_size':256,
            'enable_resume':False, 'save_steps':0, 'save_epochs':0, 'save_hf_weights':False, 'async_save_hf_weights':False}
        for name, expected in expected_train.items(): require(getattr(train, name, None) == expected, 'Actual root-prefetch train config differs: ' + name)
        require(getattr(train,'load_checkpoint_path',None) is None, 'Resume is forbidden')
        require(parallel.fsdp_enabled and parallel.dp_mode=='fsdp2' and parallel.world_size==8 and parallel.dp_size==8 and parallel.dp_shard_size==8 and parallel.dp_replicate_size==1,
                'Native single-node DP8 required')
        require(not parallel.ep_enabled and not parallel.tp_enabled and parallel.global_rank==parallel.local_rank and parallel.global_rank in range(8), 'EP/TP/multi-node candidate unsupported')
        require(all(getattr(parallel,name,None)==1 for name in ('pp_size','cp_size','ulysses_size')), 'PP/CP/Ulysses candidate unsupported')
        require(use_depth_align is True and use_future_depth is True and use_future_video is True, 'Actual current/future depth and video teacher path required')
        self.torch = torch_module
        if self.torch is None: import torch; self.torch = torch
        require(str(self.torch.__version__)=='2.8.0+cu128', 'Pinned Torch2.8 cu128 required')
        require(self.torch.is_grad_enabled() and not self.torch.is_inference_mode_enabled(), 'Build root prefetch in original grad-enabled/non-inference context')
        if classes is None:
            from torch._dynamo.eval_frame import OptimizedModule
            from torch.distributed.fsdp import FSDPModule
            from torch.distributed.fsdp._fully_shard._fully_shard import _UnshardHandleImpl
            from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
            from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
            from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
            classes={'optimized':OptimizedModule, 'fsdp':FSDPModule, 'state':FSDPState,
                     'group':FSDPParamGroup, 'handle':_UnshardHandleImpl, 'root':LingbotVlaV2Policy}
        self.classes=classes
        require(type(model) is classes['optimized'], 'Exact installed Torch.compile OptimizedModule required')
        self.model=model; self.root=model._orig_mod
        require(isinstance(self.root,classes['fsdp']) and isinstance(self.root,classes['root']) and type(self.root).__name__=='FSDPLingbotVlaV2Policy', 'Exact fully-sharded LingBot policy root required')
        require(self.root.training is True and model.training is True, 'Build controller after original model.train()')
        for name in ('unshard','_get_fsdp_state'):
            require(getattr(getattr(self.root,name,None),'__func__',None) is getattr(classes['fsdp'],name), 'Canonical public FSDP API required')
        self.state=self.root._get_fsdp_state(); self.group=self.state._fsdp_param_group
        require(type(self.state) is classes['state'] and type(self.group) is classes['group'] and self.state._modules==(self.root,) and self.group.modules==(self.root,) and self.group.fsdp_params,
                'Nonempty exact single-root FSDP group required')
        for handle_name, hooks_name, method in (('_pre_forward_hook_handle','_forward_pre_hooks','_pre_forward'),('_post_forward_hook_handle','_forward_hooks','_post_forward')):
            handle=getattr(self.state,handle_name,None); hook=getattr(self.root,hooks_name,{}).get(getattr(handle,'id',None))
            require(getattr(hook,'__self__',None) is self.state and getattr(hook,'__func__',None) is getattr(classes['state'],method), 'Canonical root forward hook missing')
        require(self.root._forward_pre_hooks_with_kwargs.get(self.state._pre_forward_hook_handle.id,False), 'Original root FSDP kwargs hook required')
        require(not any(param._sharded_param_data.is_meta for param in self.group.fsdp_params), 'Root weights must be initialized/loaded')
        require(self.group.device.type=='cuda', 'Actual root must use CUDA')
        require(self.group.device.index==parallel.local_rank, 'Actual root CUDA device index differs from local rank')
        pg=self.group._all_gather_process_group
        self.process_group={'world_size':pg.size(),'rank':pg.rank(),'global_members':self.torch.distributed.get_process_group_ranks(pg)}
        require(self.process_group=={'world_size':8,'rank':parallel.global_rank,'global_members':list(range(8))}, 'Actual root process group is not the ordered full DP8 group')
        self.policy=policy_snapshot(self.group)
        require(self.policy == {'reshard_after_forward':False,'reshard_after_backward':True,'reduce_grads':True,
            'all_reduce_grads':True,'unshard_in_backward':True,'unshard_async_op':False,'parameter_dtype':'torch.bfloat16',
            'reduce_dtype':'torch.float32','output_dtype':'torch.bfloat16','cast_forward_inputs':True}, 'Original BF16 root MP/gradient/reshard policy differs')
        require(self.state._is_root is None and self.group.is_sharded and self.group._all_gather_result is None, 'Fresh untouched pre-first-forward root state required')
        require(not self.state._states_to_forward_prefetch and not self.state._states_to_backward_prefetch, 'Root explicit prefetch controls must stay default')
        self.original_parameter_flags=parameter_flags(self.group)
        require(not any(flags['is_inference'] for row in self.original_parameter_flags for flags in (row['sharded_parameter'],row['sharded_data'])), 'Original sharded root tensors must be normal/version-tracked tensors')
        self.source=(source_audit if source_checker is None else source_checker)(self.environment)
        self.rank=parallel.global_rank; self.pid=os.getpid(); self.step=0; self.awaiting_forward=False; self.pending_handle=None
        self.writer=writer
        if self.writer is None:
            directory=confined(self.environment[RECEIPTS]); directory.mkdir(parents=True,exist_ok=True)
            run=confined(self.environment['SFT_RUN_DIR']); require(directory==run/'root-teacher-prefetch','Receipt directory must match this run')
            self.path=directory/f'rank{self.rank:02d}-pid{self.pid}.jsonl'
            require(not self.path.exists(), 'Refusing to reuse root-prefetch runtime receipt')
        props=self.torch.cuda.get_device_properties(parallel.local_rank)
        uuid=str(props.uuid); uuid=uuid if uuid.lower().startswith('gpu-') else 'GPU-'+uuid
        require(self.torch.cuda.current_device()==parallel.local_rank and (props.major,props.minor)==(12,0), 'Actual rank CUDA SM120 device mismatch')
        self.device={'local_rank':parallel.local_rank,'uuid':uuid,'compute_capability':[props.major,props.minor],'total_memory_bytes':props.total_memory}
        require(70*1024**3<=props.total_memory<=74*1024**3, 'Expected72GB actual CUDA device required')
        mapped=self.source['prepared_cuda_devices'][parallel.local_rank]
        require(mapped.get('local_rank')==parallel.local_rank and mapped.get('uuid','').lower()==uuid.lower() and mapped.get('compute_capability')==[12,0] and mapped.get('total_memory_bytes')==props.total_memory,
                'Actual root prefetch CUDA device differs from preparer rank mapping')
        self.emit('setup', root_module_identity=id(self.root),root_state_identity=id(self.state),root_group_identity=id(self.group),
            compiled_wrapper_class=type(model).__name__,root_class=type(self.root).__name__,source=self.source,
            root_parameter_count=len(self.group.fsdp_params), root_parameter_numel=sum(p._orig_size.numel() for p in self.group.fsdp_params),
            policy_before=self.policy,status_before=group_status(self.state,self.group),
            original_sharded_parameter_flags=self.original_parameter_flags,
            original_sharded_schema_sha256=flags_identity(schema_flags(self.original_parameter_flags)),
            grad_enabled=self.torch.is_grad_enabled(),inference_mode_enabled=self.torch.is_inference_mode_enabled(),
            first_step_gate='skip step1; original root forward performs state lazy-init; eligible step2+',
            wait_owner='original root FSDP pre_forward -> group.wait_for_unshard -> copy_out Work.wait; no candidate wait',
            teacher_math_changed=False,gradient_sync_changed=False,precision_changed=False,clip_changed=False,
            actual_device=self.device,rank_consistency='one single-node torchrun inherited uniform Docker environment; no added rank collective')
        self.emit('root_process_group',actual_process_group=self.process_group,qualification_collective_issued=False)

    def emit(self,phase,**fields):
        row={'schema_version':1,'utc':datetime.now(timezone.utc).isoformat(),'rank':self.rank,'pid':self.pid,'mode':'on','phase':phase,'step':self.step,**fields}
        if self.writer is not None: self.writer(row)
        else:
            with self.path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')

    def check_policy(self):
        require(policy_snapshot(self.group)==self.policy, 'Root MP/gradient/reshard policy changed')
        require(not self.state._states_to_forward_prefetch and not self.state._states_to_backward_prefetch, 'Explicit root FWD/BWD schedules changed')

    def before_teachers(self, step):
        try:return self._before_teachers(step)
        except BaseException as error:
            self.emit('before_teachers_failed',requested_step=step,error_type=type(error).__name__,error=str(error))
            raise

    def _before_teachers(self, step):
        require(type(step) is int and step==self.step+1 and not self.awaiting_forward, 'Exactly one chronological micro32/accum1 forward per update required')
        self.check_policy(); before=group_status(self.state,self.group)
        require(self.torch.is_grad_enabled() and not self.torch.is_inference_mode_enabled(), 'Async root allocation must precede teacher no_grad/inference contexts in grad-enabled code')
        flags_before=parameter_flags(self.group)
        require(schema_flags(flags_before)==schema_flags(self.original_parameter_flags), 'Original root parameter dtype/requires_grad/inference flags changed')
        require(self.root.training and before['state_training']=='IDLE' and before['group_training']=='IDLE' and not before['forward_root_active'] and not before['backward_final_callback_queued'], 'Prefetch only between complete optimizer updates')
        require(not before['post_optim_event_present'], 'External optimizer-event schedules require separate stream proof')
        require(before['is_sharded'] and not before['pending_all_gather'], 'Root must be sharded with no pending gather')
        self.step=step; self.awaiting_forward=True
        if step==1:
            require(before['is_root'] is None, 'First safe gate requires untouched root lazy-init')
            self.emit('before_teachers',status='skipped_first_lazy_init',status_before=before,policy_before=self.policy,unshard_called=False,
                grad_enabled=True,inference_mode_enabled=False,sharded_parameter_flags_sha256=flags_identity(flags_before))
            return
        require(before['is_root'] is True and self.group.comm_ctx is self.state._comm_ctx and hasattr(self.state._comm_ctx,'all_gather_stream'), 'Original first-forward lazy-init/shared CUDA streams not complete')
        self.pending_handle=self.root.unshard(async_op=True)  # the only FSDP mutation/API in this candidate
        require(type(self.pending_handle) is self.classes['handle'] and self.pending_handle._fsdp_param_group is self.group, 'Unexpected public UnshardHandle implementation/group')
        self.check_policy(); pending=self.group._all_gather_result
        require(pending is not None and pending.all_gather_work is not None and self.group.is_sharded, 'Public async unshard did not preserve pending NCCL Work before root forward')
        flags_after=parameter_flags(self.group)
        require(flags_after==flags_before, 'Public async unshard changed sharded parameter dtype/gradient/inference/version flags')
        gather_output=tensor_flags(pending.all_gather_output)
        require(not gather_output['is_inference'] and gather_output['requires_grad'] is False and gather_output['dtype']==self.policy['parameter_dtype'], 'Async gather output must be normal/version-tracked no-grad tensor in original root MP dtype')
        self.emit('before_teachers',status='async_unshard_submitted',status_before=before,status_after_submit=group_status(self.state,self.group),
            policy_before=self.policy,policy_after=policy_snapshot(self.group),unshard_called=True,async_op=True,
            handle_class=type(self.pending_handle).__name__,work_class=type(pending.all_gather_work).__name__,
            grad_enabled=True,inference_mode_enabled=False,sharded_parameter_flags_before_sha256=flags_identity(flags_before),
            sharded_parameter_flags_after_submit_sha256=flags_identity(flags_after),gather_output_flags=gather_output,
            pending_event_present=pending.all_gather_event is not None,
            current_stream_semantics='Torch2.8 async_op=True/IDLE uses current stream for copy-in and NCCL submission; original preforward completes wait',
            explicit_wait_called=False,baseline_step_sync_precedes_insertion=True)

    def after_forward(self,step):
        try:return self._after_forward(step)
        except BaseException as error:
            self.emit('after_forward_failed',requested_step=step,error_type=type(error).__name__,error=str(error))
            raise

    def _after_forward(self,step):
        require(step==self.step and self.awaiting_forward, 'Original root forward must follow teacher prefetch exactly once')
        self.check_policy(); after=group_status(self.state,self.group)
        require(self.torch.is_grad_enabled() and not self.torch.is_inference_mode_enabled(), 'Original root forward must finish in grad-enabled/non-inference context')
        require(after['is_root'] is True and after['is_unsharded'] and not after['pending_all_gather'] and after['state_training']=='IDLE' and after['group_training']=='IDLE' and not after['forward_root_active'], 'Original root pre/post forward did not consume pending unshard')
        flags_after=parameter_flags(self.group,unsharded=True)
        for row,original in zip(flags_after,self.original_parameter_flags):
            parameter=row['unsharded_parameter']
            require(parameter['requires_grad']==original['sharded_parameter']['requires_grad'] and parameter['is_inference'] is False and parameter['dtype']==self.policy['parameter_dtype'],
                    'Original forward unsharded root parameter lost gradient/normal-tensor/original MP attributes')
        self.emit('after_original_forward',status='original_preforward_wait_consumed_pending' if step>1 else 'first_original_lazy_init_complete',
            status_after=after,policy_after=policy_snapshot(self.group),explicit_wait_called=False,
            grad_enabled=True,inference_mode_enabled=False,unsharded_parameter_flags_sha256=flags_identity(flags_after),
            first_original_unsharded_parameter_flags=flags_after if step==1 else None,
            pending_consumption_scope='Original hook finished CPU submission and registered unsharded parameters; not a GPU-completion timestamp')
        self.pending_handle=None; self.awaiting_forward=False
