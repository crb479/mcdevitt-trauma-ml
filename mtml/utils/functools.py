__doc__ = "Helper code for use with functions."

from functools import wraps


class EvaluationRecord(dict):
    """Non-mutable :class:`dict` subclass.

    Returned by :func:`return_eval_record` instead of the actual return value of
    the decorated function.
    """
    def __getattr__(self, key):
        return self[key]
    
    def __setattr__(self, key, value):
        raise NotImplementedError("cannot set attributes")

    def __setitem__(self, key, value):
        raise NotImplementedError("cannot set item")

    def __repr__(self):
        item_str = ", ".join([f"{k}={v}" for k, v in self.items()])
        return f"EvaluationRecord({item_str})"


def return_eval_record(f):
    """Returns :class:`EvaluationRecord` containing inputs and outputs of ``f``.

    When ``f`` is invoked with ``args`` and ``kwargs``,
    :func:`return_eval_record` decorates ``f`` so that when it evaluates, it
    returns not its original return value but an :class:`EvaluationRecord`
    containing ``args``, ``kwargs``, and the return value as ``result``. These
    may be accessed as attributes or through key/value indexing (string keys).

    :rtype: function
    """
    @wraps(f)
    def _inner(*args, **kwargs):
        res = f(*args, **kwargs)
        return EvaluationRecord(args = args, kwargs = kwargs, result = res)

    return _inner