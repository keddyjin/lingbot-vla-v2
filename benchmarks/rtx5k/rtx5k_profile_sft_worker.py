#!/usr/bin/env python3
"""Capture-only candidate: short coordinated NVTX or CUDA API window."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path('/root/keddyj/LingBotVLA-Test')
CAPTURE_LABEL = 'lingbot_nsys_capture_v2'


def plan():
    mode = os.environ.get('LINGBOT_NSYS_CAPTURE_MODE','nvtx')
    count = int(os.environ.get('LINGBOT_NSYS_CAPTURE_STEPS','2'))
    warmup = int(os.environ.get('LINGBOT_MEASURE_WARMUP_STEPS','10'))
    if mode not in ('nvtx','api') or not 1 <= count <= 3 or warmup < 0:
        raise ValueError('Invalid capture-only mode/count/warmup')
    return {'mode':mode,'capture_steps':list(range(warmup+1,warmup+count+1)),
            'capture_label':CAPTURE_LABEL,'capture_controller':'rank_zero_NVTX' if mode == 'nvtx' else 'all_rank_CUDA_APIs',
            'real_training_math_changed':False,'all_rank_coverage_verified':False,'GPU_run_performed':False}


def main():
    options = plan()
    import torch
    import torch.distributed as dist
    from lingbotvla.utils.training_instrumentation import TrainingInstrumentation
    original_start = TrainingInstrumentation.start_step
    original_finish = TrainingInstrumentation.finish_step
    original_completed = TrainingInstrumentation.completed_update
    start,end = options['capture_steps'][0],options['capture_steps'][-1]
    directory = Path(os.environ['LINGBOT_MEASURE_JSONL_DIR']).resolve().parent/'profiling'
    if not directory.is_relative_to(ROOT) or directory == ROOT:
        raise ValueError('Capture receipts must stay under authorized ROOT')
    directory.mkdir(parents=True,exist_ok=True)
    streams = {}

    def event(self,name,step):
        if self.rank not in streams:
            streams[self.rank] = (directory/f'capture-rank-{self.rank:02d}.jsonl').open('x',encoding='utf-8',buffering=1)
        streams[self.rank].write(json.dumps({'event':name,'rank':self.rank,'pid':os.getpid(),
                                           'step':step,'mode':options['mode'],'monotonic_ns':time.monotonic_ns(),
                                           'unix_ns':time.time_ns()})+'\n')

    def start_step(self,step,epoch,epoch_step):
        if step == start:
            torch.cuda.synchronize()
            dist.barrier()
            if options['mode'] == 'nvtx':
                if self.rank == 0:
                    torch.cuda.nvtx.range_push(CAPTURE_LABEL)
            else:
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
            event(self,'capture_start_controller_ready',step)
            # All ranks begin real update only after rank-zero gate opened.
            dist.barrier()
        return original_start(self,step,epoch,epoch_step)

    def completed_update(self,original_step_time_s,*,visual_step,profile_step):
        captured = start <= self._range_step <= end
        return original_completed(self,original_step_time_s,visual_step=visual_step,profile_step=captured)

    def finish_step(self,**kwargs):
        step = self._range_step
        result = original_finish(self,**kwargs)
        if step == end:
            # Finish measurement/NVTX phases first, then drain each device and
            # the boundary collective before any process ends collection.
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            event(self,'capture_end_all_rank_GPU_drained',step)
            if options['mode'] == 'nvtx':
                if self.rank == 0:
                    torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
            event(self,'capture_stop_controller_done',step)
            dist.barrier()
        return result

    TrainingInstrumentation.start_step = start_step
    TrainingInstrumentation.completed_update = completed_update
    TrainingInstrumentation.finish_step = finish_step
    target = ROOT/'lingbot-vla-v2-sft-opt-github/tasks/vla/train_lingbotvla.py'
    sys.argv[0] = str(target)
    try:
        runpy.run_path(str(target),run_name='__main__')
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == '__main__':
    if '--plan' in sys.argv:
        print(json.dumps(plan(),indent=2))
    else:
        main()
