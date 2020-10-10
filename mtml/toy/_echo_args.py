__doc__ = "Provides the ``echo_args`` function used in ``mtml.toy`` unit tests."

def echo_args(*args, **kwargs):
    """Returns a dict containng all positional and keyword arguments.
    
    The dict has keys ``"args"`` and ``"kwargs"``, corresponding to positional
    and keyword arguments respectively.
    
    :param *args: Positional arguments
    :param **kwargs: Keyword arguments
    :rtype: dict
    """
    return {"args": args, "kwargs": kwargs}