__doc__ = "Miscellaneous model utilities."

def json_safe_get_params(est_cls):
    """Make ``get_params`` safe for JSON serialization with nested estimators.

    .. warning:: Other objects will **not** be converted to be JSON-friendly!

    Replaces the ``get_params`` method of a scikit-learn-compatible estimator
    class with a method that optionally replaces estimators recovered as a
    hyperparameter with their full class name when ``safe = True`` is passed. To
    get the hyperparameters of the estimator as part of the parameter
    dictionary, ``deep = True`` must be passed.

    :param est_cls: A scikit-learn compatible estimator class that must
        implement the ``get_params`` method.
    :type est_cls: type
    :rtype: type
    """
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
                # if a sklearn-compatible estimator (has get_params)
                if hasattr(params[key], "get_params"):
                    # replace estimator with its full class name
                    params[key] = str(params[key].__class__).replace(
                        "<class '", ""
                    ).replace("'>", "")
        # return params
        return params
    
    # set docstring and bind to get_params
    _json_safe_get_params.__doc__ = doc
    est_cls.get_params = _json_safe_get_params
    # return modified class
    return est_cls