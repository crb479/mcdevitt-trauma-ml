__doc__ = "Model-related utilities."

from functools import wraps
import inspect
import os.path
import pickle
import time


def _persist_pickle(func, enabled = True, target = None, persist_func = None,
                    persist_func_kwargs = None, protocol = None):
    """Persist a model, or any generic Python object, to disk.
    
    Don't call directly. Call through :func:`persist`.
    """
    if func is None:
        raise ValueError("func is None")
    # if no specified target, use current directory the function is located in
    # and use the function's name together with __pickle__ and current time,
    # joined with underscores (we don't like spaces). replace ":" in time with
    # "." since ":" is a reserved character in filesystems.
    if target is None:
        fname = (func.__name__ + "__pickle__" + 
                 "_".join(time.asctime().split()).replace(":", ".") + ".pickle")
        # note that inspect.getfile fails on C/builtin functions!
        target = os.path.dirname(inspect.getfile(func)) + "/" + fname
    # if no persistence function, just use identity
    if persist_func is None:
        persist_func = lambda x: x
    # empty args if None for persist_func_kwargs
    if persist_func_kwargs is None:
        persist_func_kwargs = {}
    # define decorator for persisting object returned by func
    @wraps(func)
    def _persist_dec(*args, **kwargs):
        # evaluate function and get result to pickle
        res = func(*args, **kwargs)
        # if enabled, persist to disk using persist_func and target
        if enabled:
            with open(target, "wb") as wf:
                pickle.dump(persist_func(res, **persist_func_kwargs),
                            wf, protocol = protocol)
        return res
    
    return _persist_dec


def persist_pickle(func = None, enabled = True, target = None,
                   persist_func = None, 
                   persist_func_kwargs = None, protocol = None):
    """Persists a model, or any generic Python object, to disk.
    
    .. warning:: Before you pickle something, please read the
       `Python documentation of the pickle module`__ and familiarize yourself
       with its pros and cons. If possible, use :func:`persist_json` instead
       for plain text, not binary, persistence.
    
    .. __: https://docs.python.org/3/library/pickle.html
    
    :param func: TBA
    :type func: function
    """
    # define new decorator that calls _persist_pickle
    def _wrap_dec(f):
        return _persist_pickle(f, enabled = enabled, target = target,
                               persist_func = persist_func,
                               persist_func_kwargs = persist_func_kwargs,
                               protocol = protocol)
    # if func is None, return _wrap_dec
    if func is None:
        return _wrap_dec
    # else return result
    return _wrap_dec(func)