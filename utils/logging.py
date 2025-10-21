import parameters as params

def log_print(*args, level=1, **kwargs):
    """
    Print only if params.VERBOSE >= level.
    Usage: log_print('message', level=2)
    """
    import parameters as params
    debug = getattr(params, 'DEBUG', False)
    verbose = getattr(params, 'VERBOSE', 1)
    if debug or verbose >= level:
        print(*args, **kwargs)
