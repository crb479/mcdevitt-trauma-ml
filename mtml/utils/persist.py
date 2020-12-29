__doc__ = """Function decorators for persisting Python objects to disk.

These should not be used with lambda functions unless ``target`` is specified.
"""

from functools import wraps
import inspect
import json
import math
import pandas as pd
import os.path
import pickle
import time


def _persist_pickle(func, enabled = True, target = None, out_transform = None,
                    out_transform_kwargs = None, protocol = None):
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
    if out_transform is None:
        out_transform = lambda x: x
    # empty args if None for out_transform_kwargs
    if out_transform_kwargs is None:
        out_transform_kwargs = {}
    # define decorator for persisting object returned by func
    @wraps(func)
    def _persist_dec(*args, **kwargs):
        # evaluate function and get result to pickle
        res = func(*args, **kwargs)
        # if enabled, persist to disk using out_transform and target
        if enabled:
            with open(target, "wb") as wf:
                pickle.dump(out_transform(res, **out_transform_kwargs),
                            wf, protocol = protocol)
        return res

    # add _persisted_func attribute to _persist_dec and return
    _persist_dec._persisted_func = func
    return _persist_dec


def persist_pickle(func = None, enabled = True, target = None,
                   out_transform = None, 
                   out_transform_kwargs = None, protocol = None):
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
    :param out_transform: A function that takes in the result returned by
        ``func`` and outputs the Python object to be pickled. For example, if
        ``func`` outputs a tuple and you only want to persist its first element,
        simply set ``out_transform`` to ``lambda x: x[0]``. Must only have one
        positional argument for output of ``func`` and can have keyword args.
    :type out_transform: function
    :oaram out_transform_kwargs: Keyword arguments to pass to ``out_transform``.
    :type out_transform_kwargs: dict, optional
    :param protocol: An integer representing the pickle protocol to use.
        Defaults to :attr:`pickle.DEFAULT_PROTOCOL`.
    :type protocol: int, optional
    :rtype: function
    """
    # define new decorator that calls _persist_pickle
    def _wrap_dec(f):
        return _persist_pickle(
            f, enabled = enabled, target = target,
            out_transform = out_transform, 
            out_transform_kwargs = out_transform_kwargs, protocol = protocol
        )
    # if func is None, return _wrap_dec
    if func is None:
        return _wrap_dec
    # else return result
    return _wrap_dec(func)


def _persist_json(func, enabled = True, target = None, out_transform = None,
                  out_transform_kwargs = None, indent = 4, **dump_kwargs):
    """Decorator to persist function output to disk using :func:`json.dump`.
    
    Don't call directly. Call through :func:`persist_json`.
    """
    if func is None:
        raise ValueError("func is None")
    # if no specified target, use current directory the function is located in
    # and use the function's name together with __json__ and current time,
    # joined with underscores (we don't like spaces). replace ":" in time with
    # "." since ":" is a reserved character in filesystems.
    if target is None:
        fname = (func.__name__ + "__json__" + 
                 "_".join(time.asctime().split()).replace(":", ".") + ".json")
        # note that inspect.getfile fails on C/builtin functions!
        target = os.path.dirname(inspect.getfile(func)) + "/" + fname
    # if no persistence function, just use identity
    if out_transform is None:
        out_transform = lambda x: x
    # empty args if None for out_transform_kwargs
    if out_transform_kwargs is None:
        out_transform_kwargs = {}
    # define decorator for persisting object returned by func
    @wraps(func)
    def _persist_dec(*args, **kwargs):
        # evaluate function and get result to pickle
        res = func(*args, **kwargs)
        # if enabled, persist to disk using out_transform and target
        if enabled:
            with open(target, "w") as wf:
                json.dump(out_transform(res, **out_transform_kwargs),
                          wf, indent = indent, **dump_kwargs)
        return res
    
    # add _persisted_func attribute to _persist_dec and return
    _persist_dec._persisted_func = func
    return _persist_dec


def persist_json(func = None, enabled = True, target = None,
                 out_transform = None, out_transform_kwargs = None,
                 indent = 4, **dump_kwargs):
    """Decorator to persist function output to disk using :func:`json.dump`.
    
    .. note:: Only a few Python objects can be persisted to disk in JSON format.
       Some compatible objects include dicts, lists, tuples, strings, ints,
       floats, ``True``, ``False``, ``None``, ``np.nan`` or other NaN
       representations. Arbitrary Python objects cannot be represented as JSON.

    Refer to :func:`persist_pickle` for parameter details omitted here.

    :param indent: How many spaces to indent JSON output, which is pretty-
        printed by default. Defaults to ``4``. If ``0``, then no pretty-printing
        is performed by :func:`json.dump`.
    :type indent: int, optional
    :param dump_kwargs: Additional keyword args to pass to :func:`json.dump`.
    :rtype: function
    """
    # define new decorator that calls _persist_json
    def _wrap_dec(f):
        return _persist_json(
            f, enabled = enabled, target = target,
            out_transform = out_transform,
            out_transform_kwargs = out_transform_kwargs, 
            indent = indent, **dump_kwargs
        )
    # if func is None, return _wrap_dec
    if func is None:
        return _wrap_dec
    # else return result
    return _wrap_dec(func)


def _df_to_csv_wrapper(df, csv_path, **kwargs):
    """Subroutine wrapper around :func:`pandas.DataFrame.to_csv`.

    :param df: :class:`pandas.DataFrame` to write to disk as CSV file.
    :type df: :class:`pandas.DataFrame`
    :param csv_path: Path to write the CSV file to.
    :type csv_path: str
    :param kwargs: Keyword arguments to pass to :func:`pandas.DataFrame.to_csv`.
    """
    df.to_csv(csv_path, **kwargs)
    

def _persist_csv(func = None, enabled = True, target = None, converter = None,
                converter_kwargs = None, out_transform = None,
                out_transform_kwargs = None):
    """Decorator to persist function output to disk as CSV file.
    
    Don't call directly. Call through :func:`persist_csv`.
    """
    if func is None:
        raise ValueError("func is None")
    # if no specified target, use current directory the function is located in
    # and use the function's name together with __csv__ and current time,
    # joined with underscores (we don't like spaces). replace ":" in time with
    # "." since ":" is a reserved character in filesystems.
    if target is None:
        fname = (func.__name__ + "__csv__" + 
                 "_".join(time.asctime().split()).replace(":", ".") + ".csv")
        # note that inspect.getfile fails on C/builtin functions!
        target = os.path.dirname(inspect.getfile(func)) + "/" + fname
    # if no converter function, use _df_to_csv_wrapper
    if converter is None:
        converter = _df_to_csv_wrapper
    # empty args if None for converter_kwargs
    if converter_kwargs is None:
        converter_kwargs = {}
    # if no persistence function, just use identity
    if out_transform is None:
        out_transform = lambda x: x
    # empty args if None for out_transform_kwargs
    if out_transform_kwargs is None:
        out_transform_kwargs = {}
    # define decorator for persisting object returned by func
    @wraps(func)
    def _persist_dec(*args, **kwargs):
        # evaluate function and get result to pickle
        res = func(*args, **kwargs)
        # if enabled, persist to disk with converter, out_transform, and target
        if enabled:
            converter(out_transform(res, **out_transform_kwargs),
                      target, **converter_kwargs)
        return res

    # add _persisted_func attribute to _persist_dec and return
    _persist_dec._persisted_func = func
    return _persist_dec


def persist_csv(func = None, enabled = True, target = None, converter = None,
                converter_kwargs = None, out_transform = None,
                out_transform_kwargs = None):
    """Decorator to persist function output to disk as CSV file.

    By default, the persisted object is assumed to be a
    :class:`pandas.DataFrame` and ``converter`` will automatically be bound to
    a wrapper around :meth:`pandas.DataFrame.to_csv`. For other objects, a
    custom conversion function is required.

    Refer to :func:`persist_pickle` for parameter details omitted here.

    :param converter: Conversion function to use that will write selected
        function output to disk as a CSV file. The function's signature must
        contain only two positional arguments, which are the object to save as
        CSV and the path to write the CSV file to; all other arguments must be
        keyword arguments. Its return value is ignored. If not provided, then
        the selected output to persist is assumed to be a
        :class:`pandas.DataFrame` and :func:`_df_to_csv_wrapper` is used.
    :type converter: function, optional
    :param converter_kwargs: Dict of keyword arguments to pass to ``converter``.
    :type converter_kwargs: dict, optional
    :rtype: function
    """
    # define new decorator that calls _persist_csv
    def _wrap_dec(f):
        return _persist_csv(
            f, enabled = enabled, target = target,
            out_transform = out_transform, converter = converter,
            converter_kwargs = converter_kwargs,
            out_transform_kwargs = out_transform_kwargs
        )
    # if func is None, return _wrap_dec
    if func is None:
        return _wrap_dec
    # else return result
    return _wrap_dec(func)


def remove_persist(func = None, n = 1):
    """Return the underlying function wrapped by a ``persist`` decorator.

    Each decorator maintains an attribute ``_persisted_func`` that holds the
    function it is currently decorating. :func:`remove_persist` looks for that
    attribute and returns it, effectively "undoing" the effect of the
    ``persist`` decorator. :func:`remove_persist` can remove up to ``n``
    ``persist`` decorators' effects and has no effect if ``func`` does not have
    the attribute that it is looking for.

    :param func: Function to have persistence decorator effect removed from
    :type func: function
    :param n: Maximum number of times :func:`remove_persist` will attempt to
        access the original ``_persisted_func``. Automatically stops if the
        current object is attempt to "unwrap" does not have this attribute.
    :type n: int, optional
    :rtype: function
    """
    # define new decorator that calls _remove_persist
    def _wrap_dec(f):
        return _remove_persist(f, n = n)
    # if func is None, return _wrap_dec
    if func is None:
        return _wrap_dec
    # else return result
    return _wrap_dec(func)


def remove_all_persist(func):
    """Remove all ``persist`` decorator effects and return original function.

    Equivalent to :func:`remove_persist` but with `n = 1000`.

    :rtype: function
    """
    return _remove_persist(func, n = 1000)


def _remove_persist(func, n = 1):
    """Return the underlying function wrapped by a ``persist`` decorator.

    Don't call by itself. Call through :func:`remove_persist`.

    :rtype: function
    """
    # sanity check
    if n < 1:
        raise ValueError("n must be >= 1")
    # until we reach n iterations, keep unrolling
    for _ in range(n):
        # break early if necessary
        if not hasattr(func, "_persisted_func"):
            return func
        # else unroll
        func = func._persisted_func
    # return
    return func