"""Optional local-only SFT profiling ranges and optimizer-step measurements.

This module has no Torch dependency until a requested range/sync is executed.
It does not change losses, gradients, parameter updates, or micro-batch order.
"""

from contextlib import contextmanager, nullcontext
import json
import os
import time


REMOTE_ROOT = "/home/scratch.keddyj_wwfo/robotics-test/LingBotVLA-Test"
TIMING_BOUNDARY = "before_data_fetch_to_after_logging_and_checkpoint"


def _enabled(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _measurement_directory(value):
    """Reject relative paths and symlink escapes from the authorized root."""
    if not os.path.isabs(value):
        raise ValueError("LINGBOT_MEASURE_JSONL_DIR must be an absolute path")
    resolved = os.path.realpath(value)
    root = os.path.realpath(REMOTE_ROOT)
    if os.path.commonpath([root, resolved]) != root:
        raise ValueError(f"Measurement output must stay under {REMOTE_ROOT}")
    return resolved


class TrainingInstrumentation:
    def __init__(self, train_args):
        self.ranges_enabled = _enabled(os.environ.get("LINGBOT_PROFILE_RANGES", "0"))
        self.measure_dir = os.environ.get("LINGBOT_MEASURE_JSONL_DIR", "").strip()
        self.measure_enabled = bool(self.measure_dir)
        self.memory_metrics_enabled = self.measure_enabled and _enabled(
            os.environ.get("LINGBOT_MEASURE_MEMORY", "0")
        )
        self.rank = int(train_args.global_rank)
        self.local_rank = int(train_args.local_rank)
        self.world_size = int(train_args.world_size)
        self.data_parallel_size = int(train_args.data_parallel_size)
        self.micro_batch_size = int(train_args.micro_batch_size)
        self.global_batch_size = int(train_args.global_batch_size)
        self.accumulation_steps = int(train_args.gradient_accumulation_steps)
        self.optimizer = str(train_args.optimizer)
        self._completed_steps = 0
        self._pending = None
        self._start = None
        self._micro_count = 0
        self._local_samples = 0
        self._stream = None
        self._invalid_reason = None
        self._range_step = None
        self._range_micro = None
        if self.measure_enabled:
            self.measure_dir = _measurement_directory(self.measure_dir)
            self.run_id = os.environ.get("LINGBOT_MEASURE_RUN_ID", "").strip()
            if not self.run_id:
                raise ValueError("LINGBOT_MEASURE_RUN_ID is required for timing JSONL")
            self.warmup_steps = int(os.environ.get("LINGBOT_MEASURE_WARMUP_STEPS", "0"))
            if self.warmup_steps < 0:
                raise ValueError("LINGBOT_MEASURE_WARMUP_STEPS must be nonnegative")
            os.makedirs(self.measure_dir, exist_ok=True)
            # A fresh directory prevents accidentally appending another run.
            path = os.path.join(self.measure_dir, f"rank-{self.rank:02d}.jsonl")
            self._stream = open(path, "x", encoding="utf-8", buffering=1)

    def range(self, name):
        if not self.ranges_enabled:
            return nullcontext()
        return self._range(name)

    @contextmanager
    def _range(self, name):
        import torch

        label = f"lingbot/{name}/step={self._range_step}"
        if self._range_micro is not None:
            label += f"/micro={self._range_micro}"
        with torch.profiler.record_function(label):
            torch.cuda.nvtx.range_push(label)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()

    def start_step(self, step, epoch, epoch_step):
        self._range_step = int(step)
        self._range_micro = None
        if not self.measure_enabled:
            return
        if self._pending is not None:
            raise RuntimeError("Previous measurement must finish before the next step")
        # No start synchronize: include data wait and any prior GPU tail naturally.
        self._start = time.perf_counter()
        self._micro_count = 0
        self._local_samples = 0
        self._pending = {
            "schema_version": 1,
            "run_id": self.run_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "data_parallel_size": self.data_parallel_size,
            "global_step": int(step),
            "step": int(step),
            "epoch": int(epoch),
            "epoch_step": int(epoch_step),
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "optimizer": self.optimizer,
            "checkpoint_step": False,
            "checkpoint_reasons": [],
            "epoch_terminal_step": False,
            "terminal_step": False,
            "visual_step": False,
            "profile_step": False,
            "timing_clock": "perf_counter",
            "boundary": TIMING_BOUNDARY,
            "end_sync": True,
            "measurement_write_included": False,
            "async_background_drain_included": False,
        }

    def cancel_step(self):
        # StopIteration did not perform an optimizer update.
        if self.measure_enabled:
            # The previous row may already have been emitted, so a later epoch
            # checkpoint cannot be charged to it without changing the JSONL.
            # Preserve training behavior and invalidate performance comparison.
            self._invalid_reason = "dataloader_exhausted_before_planned_epoch_end"
        self._pending = None
        self._start = None

    def start_micro_batch(self):
        if self._range_micro is None:
            self._range_micro = 0
        else:
            self._range_micro += 1

    def completed_micro_batch(self, micro_batch):
        if not self.measure_enabled:
            return
        # Tensor.shape reads metadata only; no device-to-host synchronization.
        actions = micro_batch["actions"]
        self._local_samples += int(actions.shape[0])
        self._micro_count += 1

    def completed_update(self, original_step_time_s, *, visual_step, profile_step):
        self._range_micro = None
        if not self.measure_enabled:
            return
        self._pending.update({
            "original_step_time_s": float(original_step_time_s),
            "micro_batches": self._micro_count,
            "local_samples": self._local_samples,
            "samples": self._local_samples * self.data_parallel_size,
            "batch_complete": (
                self._micro_count == self.accumulation_steps
                and self._local_samples == self.micro_batch_size * self.accumulation_steps
                and self._local_samples * self.data_parallel_size == self.global_batch_size
            ),
            "visual_step": bool(visual_step),
            "profile_step": bool(profile_step),
        })

    def checkpoint(self, reason):
        if self.measure_enabled and self._pending is not None:
            self._pending["checkpoint_step"] = True
            self._pending["checkpoint_reasons"].append(str(reason))

    def finish_step(self, *, epoch_terminal=False, terminal=False):
        if not self.measure_enabled or self._pending is None:
            return
        if "original_step_time_s" not in self._pending:
            raise RuntimeError("Cannot publish an incomplete optimizer-step measurement")
        import torch

        # One optional sync per optimizer update, never per micro batch. All
        # paired baselines/variants must use the same measurement setting.
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self._completed_steps += 1
        self._pending.update({
            "optimizer_step": self._completed_steps,
            "phase": "warmup" if self._completed_steps <= self.warmup_steps else "steady",
            "end_to_end_step_time_s": elapsed,
            "epoch_terminal_step": bool(epoch_terminal),
            "terminal_step": bool(terminal),
            "recorded_at_unix_ns": time.time_ns(),
        })
        if self.memory_metrics_enabled:
            # Allocator bookkeeping reads only, after the unchanged endpoint.
            # Peaks cover this training process from startup: never reset here.
            # Avoid mem_get_info, extra CUDA work, synchronization, or collectives.
            self._pending["cuda_allocator_memory"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_scope": "process_lifetime_no_reset",
                "captured_after_timing_endpoint": True,
            }
        # This write occurs after the timing endpoint and is stated in metadata.
        self._stream.write(json.dumps(self._pending, ensure_ascii=False) + "\n")
        self._stream.flush()
        self._pending = None
        self._start = None

    def close(self):
        if self._stream is not None:
            self._stream.close()
        if self.measure_enabled and self._invalid_reason is not None:
            path = os.path.join(self.measure_dir, f"measurement-status-rank-{self.rank:02d}.json")
            with open(path, "x", encoding="utf-8") as status_stream:
                json.dump({
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "rank": self.rank,
                    "valid_for_comparison": False,
                    "reason": self._invalid_reason,
                }, status_stream, indent=2)
                status_stream.write("\n")
