import flag_gems
from .types import BackendImplKind

_flag_gems_context = None
_flag_gems_context_entered = False

def backend_context_switch(cur_backend):
    """
    Switch backend context based on the current backend.
    """
    global _flag_gems_context, _flag_gems_context_entered
    assert cur_backend is not None, "Current Backend name cannot be None"
    
    if cur_backend == BackendImplKind.VENDOR:
        return

    # check if flagos should be enabled permanently via environment variable
    flag_gems_global_registrar = getattr(flag_gems, 'current_work_registrar', None)
    is_flag_gems_enabled = flag_gems_global_registrar is not None

    # if flagos is enabled permanently, and flagos context is not entered, skip entering flagos context
    if is_flag_gems_enabled and not _flag_gems_context_entered:
        return

    if cur_backend == BackendImplKind.DEFAULT and not _flag_gems_context_entered:
        _flag_gems_context = flag_gems.use_gems()
        _flag_gems_context.__enter__()
        _flag_gems_context_entered = True
        return

    if cur_backend == BackendImplKind.REFERENCE and _flag_gems_context_entered:
        _flag_gems_context.__exit__(None, None, None)
        _flag_gems_context_entered = False
        return
