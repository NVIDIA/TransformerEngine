import os
from contextlib import nullcontext


def gems_context():
    # check if flagos should be enabled permanently via environment variable
    flag_gems_global_registrar = None
    try:
        import flag_gems
        flag_gems_global_registrar = getattr(flag_gems, 'current_work_registrar', None)
    except Exception as e:
        from ...logger_manager import get_logger
        logger = get_logger()
        logger.warning(f"Failed to get flag gems registrar: {e}")
        
    is_flag_gems_global_enabled = flag_gems_global_registrar is not None

    # Check if flagos should be enabled permanently via environment variable
    enable_flagos_permanently = os.getenv("TE_FL_ENABLE_FLAGOS_PERMANENTLY", "false").lower() in ("1", "true", "yes")
    if enable_flagos_permanently and not is_flag_gems_global_enabled:
        flag_gems.enable(record=True, once=True)
        is_flag_gems_global_enabled = True
    
    # Use nullcontext if flag_gems is already globally enabled, otherwise use use_gems() context
    context = nullcontext() if is_flag_gems_global_enabled else flag_gems.use_gems()
    
    return context
