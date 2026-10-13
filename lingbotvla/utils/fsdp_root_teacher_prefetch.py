"""Overlap the student root's FSDP2 unshard with teacher forwards."""

import torch


def _require(condition, message):
    if not condition:
        raise RuntimeError(
            f"Root teacher prefetch: {message}; disable with --train.root_teacher_prefetch false."
        )


class RootTeacherPrefetch:
    def __init__(self, model):
        try:
            from torch._dynamo.eval_frame import OptimizedModule
            from torch.distributed.fsdp import FSDPModule
            from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
            from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
            from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
        except ImportError as error:
            raise RuntimeError(
                "Root teacher prefetch requires the Torch FSDP2 runtime; "
                "disable with --train.root_teacher_prefetch false."
            ) from error

        root = model
        while isinstance(root, OptimizedModule):
            root = root._orig_mod
        _require(isinstance(root, FSDPModule), "the student root must be fully_shard FSDP2")
        _require(root.unshard.__func__ is FSDPModule.unshard, "the native public unshard API is required")
        self.root = root
        self.state = root._get_fsdp_state()
        self.group = self.state._fsdp_param_group
        _require(type(self.state) is FSDPState, "the native FSDP2 state is required")
        _require(type(self.group) is FSDPParamGroup and self.group.fsdp_params, "a nonempty root FSDP2 group is required")
        _require(self.state._modules == (root,) and self.group.modules == (root,), "a single student root group is required")
        _require(self.group.device.type == "cuda", "the student root must use CUDA")
        for handle_name, hooks_name, method in (
            ("_pre_forward_hook_handle", "_forward_pre_hooks", "_pre_forward"),
            ("_post_forward_hook_handle", "_forward_hooks", "_post_forward"),
        ):
            handle = getattr(self.state, handle_name)
            hook = getattr(root, hooks_name).get(handle.id)
            _require(
                getattr(hook, "__self__", None) is self.state
                and getattr(hook, "__func__", None) is getattr(FSDPState, method),
                "the original root FSDP2 forward hooks are required",
            )
        self._idle = TrainingState.IDLE
        self._forward_seen = False
        self._awaiting_forward = False
        self._pending_handle = None

    def before_teachers(self):
        _require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), "submit before teacher no_grad/inference contexts")
        _require(self.root.training, "the student root must be in training mode")
        _require(not self._awaiting_forward, "each submission must be followed by the original student forward")
        _require(
            self.state._training_state == self._idle
            and self.group._training_state == self._idle
            and self.state._state_ctx.iter_forward_root is None
            and not self.state._state_ctx.post_backward_final_callback_queued,
            "the previous student forward/backward must be complete",
        )
        _require(self.group._all_gather_result is None, "another root unshard is already pending")
        _require(self.state._state_ctx.post_optim_event is None, "external optimizer event schedules are unsupported")
        if self._forward_seen:
            _require(
                self.state._is_root is True and self.group.comm_ctx is self.state._comm_ctx,
                "the original student forward must initialize the root FSDP2 runtime",
            )
            # The original root pre-forward hook consumes this handle's pending
            # unshard. Keep the handle alive, but do not wait here.
            self._pending_handle = self.root.unshard(async_op=True)
        # Skip the first local forward even when resuming at a later global step.
        self._awaiting_forward = True

    def after_forward(self):
        _require(self._awaiting_forward, "the original student forward must follow before_teachers")
        _require(self.state._is_root is True, "the original forward did not initialize the student root")
        _require(self.group._all_gather_result is None, "the original root pre-forward did not consume the unshard")
        self._pending_handle = None
        self._forward_seen = True
        self._awaiting_forward = False
