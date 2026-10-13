"""Stdlib source/control audit before SFT; default-off accepts original7bb."""
import hashlib
from pathlib import Path

BASE_FRAMEWORK_SHA='8634aad0e94869c1daa5016bc7173cc24c2ceacbe507ddce7dc5f7194f2acdb1'
CANDIDATE_FRAMEWORK_SHA='e5827cacc5c3d915c014e4fb67416352044e24ec995040ce3c209f296fe56a50'
MODELS_SHA={'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py': 'ec9e4280015be612c3bd26e71f502acdb95aaef1d74423c01508c794b8bb1f9e', 'lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py': 'eb2202bd01916c891b366c90f9bf11a2782f9387031cc24ed08851b3d8f01e96', 'lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py': '2221459c676c27eb70e13edd29d6e25c2c6008a1fc3f0b7c820b8abbbf11aff6'}
TORCH_FSDP_SHA={'_fully_shard.py': '901220efd8b6ea015eebd468a689a8edcf5023cfcc3694159d892950261d8e15', '_fsdp_state.py': '46c55a0b88c364534d8032a14a139a0017a00ddfed1c6b5ffe43cd6a16b3474b', '_fsdp_param_group.py': '2798a87b2631091df738afaff6dfcead38ce0cc421585b4331ecaaefb4e784f5', '_fsdp_param.py': '12e459491589b623158627846f837175eae353e685f691875b5eb6b7884896e1'}


def audit_fsdp_forward_prefetch(mode,repo,torch_package):
    if mode not in ('off','one','pair'):raise ValueError('Unknown LINGBOT_FSDP_FWD_PREFETCH')
    repo=Path(repo);torch_package=Path(torch_package);files={}
    framework=repo/'lingbotvla/distributed/torch_parallelize.py'
    framework_sha=hashlib.sha256(framework.read_text(encoding='utf-8').encode()).hexdigest()
    accepted=(BASE_FRAMEWORK_SHA,CANDIDATE_FRAMEWORK_SHA) if mode=='off' else (CANDIDATE_FRAMEWORK_SHA,)
    if framework_sha not in accepted:raise ValueError('Actual fully_shard framework differs from reviewed base/candidate')
    files[str(framework)]=framework_sha
    for relative,expected in MODELS_SHA.items():
        path=repo/relative;observed=hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        if observed!=expected:raise ValueError('Actual LingBot/Qwen class/math source differs: '+relative)
        files[str(path)]=observed
    for name,expected in TORCH_FSDP_SHA.items():
        path=torch_package/'distributed/fsdp/_fully_shard'/name;observed=hashlib.sha256(path.read_bytes()).hexdigest()
        if observed!=expected:raise ValueError('Actual pinned FSDP API implementation differs: '+name)
        files[str(path)]=observed
    return {'mode':mode,'status':'passed','source_sha256':files,'candidate_framework_installed':framework_sha==CANDIDATE_FRAMEWORK_SHA,
            'default_backward_prefetch_preserved':True,'precision_or_gc_or_gradients_changed':False,'baseline_commit':'7bb4ef03fb9611aab3ab7885370e6f7e0ed625c8'}
