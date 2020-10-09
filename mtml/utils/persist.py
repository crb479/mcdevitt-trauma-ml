__doc__ = "Function decorators for persisting Python objects to disk."

from functools import wraps
import inspect
import json
import os.path
import pickle
import time


def _persist_pickle(func, enabled = True, target = None, persist_func = None,
                    persist_func_kwargs = None, protocol = None):
    """Decorator to persist function output to disk using :func:`pickle.dump`.
    
    Don't call directly. Call through :func:`persist_pickle`.
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
    """Decorator to persist function output to disk using :func:`pickle.dump`.
    
    .. warning:: Before you pickle something, please read the
       `Python documentation of the pickle module`__ and familiarize yourself
       with its pros and cons. If possible, use :func:`persist_json` instead
       for plain text, not binary, persistence.
    
    .. __: https://docs.python.org/3/library/pickle.html
    
    :param func: Function to be decorated
    :type func: function
    :param enabled: ``True`` to enable pickling, ``False`` to disable pickling.
    :type enabled: bool, optional
    :param target: Filename to pickle results to. If not provided, the filename
        will be the name of the decorated function + ``__pickle__`` + the
        output of :func:`time.asctime` where whitespace is replaced with ``_``
        and ``:`` is replaced with ``.`` with the extension ``.pickle``.
    :type target: str, optional
    :param persist_func: A function that takes in the result returned by
        ``func`` and outputs the Python object to be pickled. For example, if
        ``func`` outputs a tuple and you only want to persist its first element,
        simply set ``persist_func`` to ``lambda x: x[0]``. Must only have one
        positional argument for output of ``func`` and can have keyword args.
    :type persist_func: function
    :oaram persist_func_kwargs: Keyword arguments to pass to ``persist_func``.
    :type persist_func_kwargs: dict, optional
    :param protocol: An integer representing the pickle protocol to use.
        Defaults to :attr:`pickle.DEFAULT_PROTOCOL`.
    :type protocol: int, optional
    :rtype: function
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


def _persist_json(func, enabled = True, target = None, persist_func = None,
                  persist_func_kwargs = None, indent = 4, **dump_kwargs):
    """Decorator to persist function output to disk using :func:`json.dump`.
    
    Don't call directly. Call through :func:`persist_json`.
    """
    if func is None:
        raise ValueError("func is None")
    # if no specified target, use current directory the function is located in
    # and use the function's name together with __pickle__ and current time,
    # joined with underscores (we don't like spaces). replace ":" in time with
    # "." since ":" is a reserved character in filesystems.
    if target is None:
        fname = (func.__name__ + "__json__" + 
                 "_".join(time.asctime().split()).replace(":", ".") + ".json")
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
            with open(target, "w") as wf:
                json.dump(persist_func(res, **persist_func_kwargs),
                          wf, indent = indent, **dump_kwargs)
        return res
    
    return _persist_dec


def persist_json(func = None, enabled = True, target = None,
                 persist_func = None, persist_func_kwargs = None,
                 indent = 4, **dump_kwargs):
    """Decorator to persist function output to disk using :func:`json.dump`.
    
    .. note:: Only a few Python objects can be persisted to disk in JSON format.
       Some compatible objects include dicts, lists, tuples, strings, ints,
       floats, ``True``, ``False``, ``None``, ``np.nan`` or other NaN
       representations. Arbitrary Python objects can be represented as JSON.
    """
    # define new decorator that calls _persist_json
    def _wrap_dec(f):
        return _persist_json(f, enabled = enabled, target = target,
                             persist_func = persist_func,
                             persist_func_kwargs = persist_func_kwargs,
                             indent = indent, **dump_kwargs)
    # if func is None, return _wrap_dec
    if func is None:
        return _wrap_dec
    # else return result
    return _wrap_dec(func)