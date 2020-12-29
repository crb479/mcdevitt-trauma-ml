.. README.rst for joblic_cache

VTE model ``joblib`` cache
==========================

If you are using ``joblib.Memory`` to cache large ``numpy``\ -based objects to
disk during training of VTE models, please set the ``location`` argument for the
``joblib.Memory`` object to be the path to this directory.

Nothing in this directory should *ever* enter version control.