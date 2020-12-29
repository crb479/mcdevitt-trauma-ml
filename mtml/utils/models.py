__doc__ = "Miscellaneous model utilities."

import types


def json_safe_get_params(est_cls = None, new_class = False):
    """Make ``get_params`` safe for JSON serialization with nested estimators.

    .. warning::

       Passing ``new_class = False``, the default, will result in the class
       being directly modified. Calling this function multiple times on a class
       results in redefinition and will lead to a :class:`RecursionError` since
       the ``get_params`` method will already have been set to this decorator.
       If not using the decorator on a class *definition*, i.e. wrapping an
       existing class ``A`` with ``json_safe_get_params(A)``, one should not do
       this and instead do ``json_safe_get_params(new_class = True)(A)`` to
       return a new class subclassing ``A``.

    .. warning::
    
       Other objects will **not** be converted to be JSON-friendly. Only objects
       that implement ``get_params`` or ``__call__`` will be converted.

    Replaces the ``get_params`` method of a scikit-learn-compatible estimator
    class with a method that optionally replaces estimators/callables recovered
    as a hyperparameter with their full class name when ``safe = True`` is
    passed. To get the hyperparameters of the estimator as part of the parameter
    dictionary, ``deep = True`` must be passed.

    :param est_cls: A scikit-learn compatible estimator class that must
        implement the ``get_params`` method.
    :type est_cls: type
    :param new_class: ``True`` to return a new class subclassing ``est_cls``.
        This is useful when you want to generate a new class that acts like
        ``est_cls`` but 
    :type new_class: bool, optional
    :rtype: type
    """
    if est_cls is None:

        def _inner_dec(cls):
            return _json_safe_get_params(cls, new_class = new_class)

        return _inner_dec
    
    return _json_safe_get_params(est_cls)


def _json_safe_get_params(est_cls, new_class = False):
    """Make ``get_params`` safe for JSON serialization with nested estimators.

    See docstring for :func:`json_safe_get_params`. Don't call directly, but
    through :func:`json_safe_get_params`.

    :rtype: type
    """
    # if new_class is True, make a new class using the name of est_class +
    # "_json_safe_get_params" appended to the end (ugly, i know).
    if new_class:
        est_cls = types.new_class(
            est_cls.__name__ + "_json_safe_get_params", bases = (est_cls,)
        )
    # prevent RecursionError: if it already has json_safe_get_params, return
    if hasattr(est_cls, "json_safe_get_params"):
        return est_cls
    # bind old get_params as _get_params
    est_cls._get_params = est_cls.get_params
    # modify docstring of _get_params and save it
    doc = est_cls._get_params.__doc__.replace(
        "Get parameters for this estimator.",
        ("Get parameters for this estimator in a JSON-serializable way.\n\n"
         "Estimator objects are replaced with their full class name. if "
         "``safe = True`` is passed. To get estimator hyperparameters, "
         "``get_params`` must be called with ``deep = True``, which will "
         "include the nested estimators' hyperparameters.")
    )
    # new get_params
    def _json_safe_get_params(self, deep = True, safe = False):
        # get results from _get_params
        params = self._get_params(deep = deep)
        # if safe is True, then we need to get rid of estimators
        if safe:
            # loop through keys of dictionary
            for key in params.keys():
                # value in question
                value = params[key]
                # if sklearn estimator (has get_params) or callable
                if hasattr(value, "get_params") or hasattr(value, "__call__"):
                    # if it has the module attribute, then replace estimator
                    # with its full class name
                    if hasattr(value, "__module__"):
                        full_name = value.__module__
                        # if it has the __name__ attribute, attach that, else
                        # try for the name of its class
                        if hasattr(value, "__name__"):
                            full_name += "." + value.__name__
                        else:
                            full_name += "." + value.__class__.__name__
                    # else just use the __repr__ for the object
                    else:
                        full_name = value.__repr__()
                    # update params[key] with full_name
                    params[key] = full_name
        # return params
        return params
    
    # set docstring and bind to get_params
    _json_safe_get_params.__doc__ = doc
    est_cls.get_params = _json_safe_get_params
    # add json_safe_get_params to make it obvious that class was modified to be
    # JSON safe; set this attribute to True (could be used as boolean test)
    est_cls.json_safe_get_params = True
    # return modified class
    return est_cls