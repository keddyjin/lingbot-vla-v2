"""H9 CPU-only root-installed source/control audit, strict in both modes."""
import hashlib
import json
from pathlib import Path

ROOT_SPEC_SHA256='513c5586c8441045b1ed6dea80735ca67fb6689407a8172a7cdc60e08ac4d1ef'

CONTEXT_VARIANT='root-teacher-min8-v9'
CONTEXT_MANIFEST_SHA256='ee8f0b8803c1390f3aafddd254c02a0cfa5cf386d03cf4ddac6cf5986f21247a'


def audit_context_manifest():
    path=Path(__file__).with_name('root-teacher-context.json');raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONTEXT_MANIFEST_SHA256:raise ValueError('H9 frozen MIN8 context manifest changed')
    return json.loads(raw)


def validate_root_teacher_controls(environment, world, optimizer_mode, memory_policy):
    audit_context_manifest()
    mode=environment.get('LINGBOT_FSDP_ROOT_TEACHER_PREFETCH','off')
    if mode not in ('off','on'):raise ValueError('H9 root mode must be off/on')
    if (world,optimizer_mode,memory_policy)!=(8,'dist','gc'):raise ValueError('H9 root off/on both require native DP8/dist/gc-only')
    fixed={'LINGBOT_FSDP_FWD_PREFETCH':'off','LINGBOT_GC_POLICY':'official','LINGBOT_FLEX_BWD_TILE':'off',
        'LINGBOT_FLEX_BWD_NUM_STAGES':'off','LINGBOT_COMPILE_STEP_COUNTER_HELPER':'0','LINGBOT_DEEPSTACK_INDICES':'0',
        'LINGBOT_DEPTH_TEACHER_BATCH':'0','LINGBOT_TEACHER_METHOD_COMPILE':'off','LINGBOT_MOE_SEGMENT_OUTPUT_SIZE':'0',
        'SFT_DIST_MUON_LAYERS_PER_BUCKET':'official','SFT_NCCL_PROTO_CONTROL':'unset',
        'SFT_NCCL_P2P_LEVEL_CONTROL':'default','SFT_NCCL_P2P_DISABLE_CONTROL':'unset','SFT_NCCL_MIN_CTAS_CONTROL':'8',
        'LINGBOT_CPU_AFFINITY':'off','SFT_CUDA_ALLOCATOR_CONTROL':'unset'}
    if environment.get('SFT_NCCL_MIN_CTAS_CONTROL')!='8' or environment.get('NCCL_MIN_CTAS')!='8':raise ValueError('H9 both modes require literal planned and actual MIN8')
    for name,value in fixed.items():
        if environment.get(name,value)!=value:raise ValueError('H9 root off/on control differs: '+name)
    for name in ('NCCL_PROTO','NCCL_P2P_LEVEL','NCCL_P2P_DISABLE','NCCL_MIN_NCHANNELS','NCCL_MIN_NRINGS',
                 'NCCL_MAX_CTAS','NCCL_MAX_NCHANNELS','NCCL_MAX_NRINGS','PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):
        if name in environment:raise ValueError('H9 root off/on actual override forbidden: '+name)
    return mode


def audit_root_teacher_source(root,spec_path,mode):
    audit_context_manifest()
    if mode not in ('off','on'):raise ValueError('H9 root mode must be off/on')
    root=Path(root).resolve();spec_path=Path(spec_path).resolve()
    if not spec_path.is_relative_to(root):raise ValueError('Root source spec escapes authorized task ROOT')
    raw=spec_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ROOT_SPEC_SHA256:raise ValueError('H9 root source spec differs from frozen513 bytes')
    spec=json.loads(raw)
    if spec['candidate']!='root-teacher-prefetch-rtx5k' or len(spec['required_sources'])!=14:raise ValueError('H9 exact14 source spec invalid')
    observed={}
    for relative,accepted in spec['required_sources'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError('Root actual source path escapes task ROOT')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual not in accepted:raise ValueError('Actual root-installed source bytes differ: '+relative)
        observed[str(path)]=actual
    return {'status':'passed','control':mode,'source_variant':'original-core-with-frozen-root-teacher-prefetch-installed',
        'spec_sha256':ROOT_SPEC_SHA256,'spec_path':str(spec_path),'actual_source_sha256':observed,
        'original_framework_required':'8634aad0e94869c1daa5016bc7173cc24c2ceacbe507ddce7dc5f7194f2acdb1',
        'same_installed_sources_required_in_both_modes':True,'runtime_HEAD_equality_required':False,
        'reference_source_commit':spec['baseline_commit'],'first_update_active':False,'first_active_update':2,
        'active_update_gradient_parity_verified':False,'full_parity_claim':False,
        'harness_context_variant':CONTEXT_VARIANT,'NCCL_MIN_CTAS_context':'8',
        'context_manifest_sha256':CONTEXT_MANIFEST_SHA256,'root_helper_source_unchanged':True}
