"""Default-on decoder-only backward retention using the original public FSDP API.

No filesystem, machine, checkpoint, batch, dtype, sampler or optimizer configuration.
Only the caller's original backward belongs inside the returned context.
"""
import os

ENVIRONMENT = 'LINGBOT_FSDP_DECODER_BACKWARD_RETAIN'


def _fsdp_module_type():
    from torch.distributed.fsdp import FSDPModule
    return FSDPModule


def _require(value, message):
    if not value:
        raise ValueError(message)


def enabled_from_environment():
    value = os.environ.get(ENVIRONMENT, '1')
    _require(value in ('0', '1'), ENVIRONMENT + ' must be 0 or 1')
    return value == '1'


def select_decoders(model):
    base = getattr(model, '_orig_mod', model)
    joint = base.model.qwenvl_with_expert
    language = joint.qwenvl.model.language_model.layers
    action = joint.qwen_expert.model.layers
    _require(len(language) == len(action) == 36, 'Require original 36 language and 36 action decoder modules')
    rows = [(f'model.qwenvl_with_expert.qwenvl.model.language_model.layers.{i}', module) for i, module in enumerate(language)]
    rows += [(f'model.qwenvl_with_expert.qwen_expert.model.layers.{i}', module) for i, module in enumerate(action)]
    _require(len({id(module) for _, module in rows}) == 72, 'Decoder modules must be distinct')
    fsdp_type = _fsdp_module_type()
    groups = set()
    for name, module in rows:
        _require(isinstance(module, fsdp_type), 'Decoder is not an original FSDPModule: ' + name)
        state = module._get_fsdp_state()
        group = state._fsdp_param_group
        _require(group is not None and id(group) not in groups, 'Require 72 distinct original decoder groups')
        groups.add(id(group))
        # Read-only admission. All mutation below uses the two public API methods.
        _require(group.reshard_after_backward is True, 'Original decoder backward policy must be True: ' + name)
        _require(group._reshard_after_forward is False, 'Original decoder forward policy must be False: ' + name)
        _require(callable(module.set_reshard_after_backward) and callable(module.reshard), 'Original public FSDP methods unavailable')
    return rows


class DecoderBackwardRetain:
    def __init__(self, model, *, enabled=None, event_sink=None):
        self.model = model
        self.enabled = enabled_from_environment() if enabled is None else enabled
        _require(type(self.enabled) is bool, 'enabled must be an explicit bool')
        self.event_sink = event_sink
        self.rows = []
        self.changed = []
        self.entered = False
        self.finished = False
        self.cleanup_errors = []

    def _event(self, kind, **values):
        if self.event_sink is not None:
            self.event_sink(kind, **values)

    def _safe_cleanup_event(self, **values):
        try:
            self._event('retention_cleanup', **values)
        except BaseException as error:
            # The original backward/entry exception always has priority.
            self.cleanup_errors.append(dict(stage='event_sink', error_type=type(error).__name__))

    def _restore(self, reshard):
        # Attempt every requested module even when a previous cleanup failed.
        if reshard:
            for name, module in self.changed:
                try:
                    module.reshard()
                except BaseException as error:
                    self.cleanup_errors.append(dict(stage='public_reshard', module=name, error_type=type(error).__name__))
        for name, module in self.changed:
            try:
                module.set_reshard_after_backward(True, recurse=False)
            except BaseException as error:
                self.cleanup_errors.append(dict(stage='restore_public_backward_policy', module=name, error_type=type(error).__name__))
        for name, module in self.changed:
            try:
                group = module._get_fsdp_state()._fsdp_param_group
                _require(group.reshard_after_backward is True, 'Restored backward policy differs')
                if reshard:
                    _require(group.is_sharded is True, 'Public reshard did not restore original sharded registration state')
            except BaseException as error:
                self.cleanup_errors.append(dict(stage='verify_cleanup_state', module=name, error_type=type(error).__name__))

    def __enter__(self):
        _require(not self.entered and not self.finished, 'Context cannot be reused or reentered')
        self.entered = True
        if not self.enabled:
            return self
        self.rows = select_decoders(self.model)
        # Whole-model admission finishes before the first policy mutation.
        self._event('retention_before', enabled=True, selected_modules=[name for name, _ in self.rows])
        try:
            for name, module in self.rows:
                # Include this module before the call so a setter that raises
                # after changing its flag is also restored.
                self.changed.append((name, module))
                module.set_reshard_after_backward(False, recurse=False)
            _require(all(module._get_fsdp_state()._fsdp_param_group.reshard_after_backward is False for _, module in self.rows),
                'Public setter did not activate the requested original decoder policy')
            self._event('retention_active', enabled=True, selected_count=len(self.changed))
        except BaseException:
            self._restore(reshard=False)
            self._safe_cleanup_event(original_exception=True, errors=list(self.cleanup_errors))
            self.finished = True
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            self.finished = True
            return False
        self._restore(reshard=True)
        self.finished = True
        self._safe_cleanup_event(original_exception=exc_value is not None, errors=list(self.cleanup_errors))
        if exc_value is not None:
            # Return False instead of raising exc_value: preserve the original
            # exception object/trace. Cleanup observations remain on this context.
            return False
        if self.cleanup_errors:
            error = RuntimeError('Decoder backward retention cleanup failed; original clip/optimizer must not execute')
            error.cleanup_errors = list(self.cleanup_errors)
            raise error
        return False


def decoder_backward_retain(model, *, enabled=None, event_sink=None):
    return DecoderBackwardRetain(model, enabled=enabled, event_sink=event_sink)
