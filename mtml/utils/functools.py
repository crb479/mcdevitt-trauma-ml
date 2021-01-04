__doc__ = "Helper code for use with functions."


class EvaluationRecord(dict):
    """Non-mutable :class:`dict` subclass.

    Returned by :func:`return_eval_record` instead of the actual return value of
    the decorated function.
    """
    def __getattr__(self, key):
        # if not in keys, then return object.__getattr__
        if key in self.keys():
            return self[key]
        # object doesn't have __getattr__ as defined in docs but has
        # __getattribute__ interestingly enough
        try:
            return object.__getattr__(self, key)
        except AttributeError:
            pass
        return object.__getattribute__(self, key)
    
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

    def _inner(*args, **kwargs):
        res = f(*args, **kwargs)
        return EvaluationRecord(args = args, kwargs = kwargs, result = res)

    return _inner