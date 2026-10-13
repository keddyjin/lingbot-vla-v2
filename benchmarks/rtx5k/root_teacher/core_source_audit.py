"""Stdlib strict core-byte audit; actual HEAD/diff are evidence, not reference gates."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

MANIFEST_SHA256 = '73178b3c0462370257d09167d6ee24a5f75528dfcbdc34e2d5e99046addd1060'

def audit_core_source(repo,manifest=None,git_runner=None):
    repo=Path(repo).resolve()
    manifest=Path(manifest) if manifest is not None else Path(__file__).with_name('core-source-manifest.json')
    raw=manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA256: raise ValueError('H9 root-installed core manifest differs from frozen bytes')
    expected=json.loads(raw)
    if expected['source_variant']!='original-core-with-frozen-root-teacher-prefetch-installed': raise ValueError('Unexpected H9 source variant')
    expected_names=set(expected['files'])
    if len(expected_names)!=260: raise ValueError('H9 requires exact260 core files')
    actual_names=set()
    for subdirectory,suffix in [('lingbotvla','.py'),('tasks/vla','.py'),('configs/vla','.yaml')]:
        for path in (repo/subdirectory).rglob('*'+suffix):
            if '__pycache__' in path.parts: continue
            actual_names.add(path.relative_to(repo).as_posix())
    if actual_names!=expected_names: raise ValueError('H9 exact core source set differs; missing='+repr(sorted(expected_names-actual_names))+' extra='+repr(sorted(actual_names-expected_names)))
    observed={}
    for relative,item in expected['files'].items():
        path=(repo/relative).resolve()
        if not path.is_relative_to(repo): raise ValueError('Core source path escapes repository')
        data=path.read_bytes()
        identity={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        if identity!=item: raise ValueError('Actual core source bytes differ from pinned baseline: '+relative)
        observed[relative]=identity
    def git(*arguments):
        return subprocess.check_output(['git','--no-optional-locks','-C',str(repo),*arguments]) if git_runner is None else git_runner(*arguments)
    head=git('rev-parse','HEAD').decode().strip()
    if not re.fullmatch('[0-9a-f]{40}',head): raise ValueError('Missing actual full HEAD')
    branch=git('branch','--show-current').decode().strip()
    if branch!='keddyj-codex/sft-opt-rtx5k': raise ValueError('Unexpected actual source branch')
    diff=git('diff','--binary','HEAD')
    status=git('status','--porcelain=v1','--untracked-files=all')
    return {'schema_version':1,'status':'passed','source_variant':expected['source_variant'],
        'reference_git_commit':expected['reference_git_commit'],'reference_HEAD_is_runtime_gate':False,
        'actual_HEAD':head,'actual_branch':branch,'actual_dirty_diff_sha256':hashlib.sha256(diff).hexdigest(),
        'actual_dirty_diff_bytes':len(diff),'actual_git_status':status.decode(errors='replace'),
        'manifest_sha256':MANIFEST_SHA256,'core_source_sha256':observed,
        'core_source_bytes_pinned':True,'core_source_file_count':260,'source_variant_applies_to_both_root_modes':True,'harness_context_variant':expected['harness_context_variant'],
        'NCCL_MIN_CTAS_context':expected['nccl_min_ctas'],'neural_source_identities_unchanged_from_H8':True,
        'source_changes_from_baseline':expected['source_changes_from_baseline'],'extra_core_source_files_verified_absent':True,'benchmark_only_HEAD_change_allowed':True,
        'full_repository_clean_required':False,'untracked_import_shadowing_verified':False}
