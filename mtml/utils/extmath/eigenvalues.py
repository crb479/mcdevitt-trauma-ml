__doc__ = "Miscellaneous eigenvalue computations."

import numpy as np


def n_eigs_pct_trace(eigs, pct = 0.95, presort = True):
    """Returns number of greatest eigenvalues whose sum is >= ``pct`` of trace.

    The trace of a matrix is the sum of its eigenvalues, and oftentimes one is
    interested in how many of the largest eigenvalues are needed such that their
    sum is greater than or equal to a particular percentage of the trace. For
    example, for a positive semidefinite matrix, all eigenvalues are
    nonnegative, if the matrix is a covariance matrix, then the eigenvalues
    normalized by the trace can be interpreted as the "variance explained" by a
    particular orthonormal eigenvector in the eigenbasis returned by PCA.

    :param eigs: Flat array of eigenvalues, in any order, shape ``(n_eigs,)``.
        It is recommended to sort of the eigenvalues in descending order, and if
        so, to then pass ``presort = False`` to prevent copying and sorting.
    :type eigs: :class:`numpy.ndarray`
    :param pct: Float in ``(0, 1]`` giving the percentage of the trace we want
        the sum of the relevant largest eigenvalues to be >= to.
    :type pct: float, optional
    :param presort: Whether to presort the eigenvalues in descending order or
        not (makes a copy of the array). Pass ``False`` if ``eigs`` is already
        known to be sorted in descending order for more efficiency.
    :type presort: bool, optional
    """
    if not isinstance(eigs, np.ndarray):
        raise TypeError("eigs must be numpy.ndarray")
    if pct <= 0 or pct > 1:
        raise ValueError("pct must be in (0, 1]")
    # if presort, copy eigs and sort in descending order
    if presort:
        eigs = eigs.copy()
        eigs.sort()
        eigs = eigs[::-1]
    # trace and running sum of eigenvalues
    trace = eigs.sum()
    eig_sum = 0
    # number of largest eigenvalues needed such that sum >= pct * trace
    n_eigs = 0
    # compute running sum until eig_sum / trace exceeds pct
    for i in range(eigs.shape[0]):
        eig_sum = eig_sum + eigs[i]
        if eig_sum / trace >= pct:
            n_eigs = i + 1
            break
    return n_eigs