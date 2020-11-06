__doc__ = """Make supervised learning problems from ``VTE_DATA_PREP_PATH``.

The CSV file located at that path is the preprocessed VTE data that can be found
`in the Google Drive`__.

.. __: https://drive.google.com/file/d/1n2cND9y8fdSGmFGN_U03ma4vGOIu0rky/view?
usp=sharing
"""

# pylint: disable=relative-beyond-top-level
from .. import VTE_DATA_PREP_PATH
from .factory import make_slp_from_data


def vte_slp_factory(**kwargs):
    """Genrates supervised learningproblems from the VTE data set.

    A wrapper for :func:`~mtml.data.factory.make_slp_from_data` where the
    ``data_path`` argument is the path to preprocessed VTE data located in
    ``data/files/prep/vte_onlydata_preprocessed.csv``.
    
    See the docstring for :func:`~mtml.data.factory.make_slp_from_data` for
    details on the individual keyword arguments that can be passed.

    :param kwargs: Keyword arguments to pass to
        :func:`~mtml.data.factory.make_slp_from_data`.
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    return make_slp_from_data(VTE_DATA_PREP_PATH, **kwargs)
