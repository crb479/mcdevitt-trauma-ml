__doc__ = "Miscellaneous plotting utilities."

# pylint: disable=import-error
import numpy as np
from matplotlib.axes import Axes

from .extmath.eigenvalues import n_eigs_pct_trace


def normalized_scree_plot(
    ax, eigs, normalize = True, presort = True, pct_trace = None,
    vline_color = "red", vline_kwargs = None, matrix_letter = "C",
    show_legend = True, plot_title = None, plot_marker = "s",
    plot_xticks = None, plot_kwargs = None
):
    """Given an axis and eigenvalues, draw normalized scree plot.

    A scree plot is a plot of the eigenvalues, ordered in descending order. This
    version divides the eigenvalues by their sum (the trace).

    :param ax: Figure's axis to draw the scree plot on
    :type ax: :class:`matplotlib.axes.Axes`
    :param eigs: Vector of eigenvalues, shape ``(n_eigs,)``.
    :type eigs: :class:`numpy.ndarray`
    :param normalize: ``True`` to plot the normalized eigenvalues, i.e. divided
        by the trace (their sum), ``False`` to plot raw eigenvalues.
    :type normalize: bool, optional
    :param presort: ``True`` to sort ``eigs`` in descending order. Set to
        ``False`` for more efficiency if ``eigs`` is properly ordered.
    :type presort: bool, optional
    :param pct_trace: A float in ``(0, 1]`` determining where to draw a vertical
        line in the plot marking the number of eigenvalues needed such that
        their sum is ``pct_trace`` of the trace. Pass ``None`` to omit.
    :type pct_trace: float, optional
    :param vline_color: Color of the vertical line used to mark the number of
        eigenvalues needed to hit ``pct_trace`` of the trace. Ignored if
        ``pct_trace`` is ``None``.
    :type vline_color: str, optional
    :param vline_kwargs: Other keyword args to pass to
        :func:`matplotlib.axes.Axes.axvline`.
    :type vline_kwargs: dict, optional
    :param matrix_letter: The letter used to represent the matrix in the legend
        that is displayed if ``pct_trace`` is not ``None`` and the vertical
        line is drawn. Ignored if ``pct_trace`` is ``None``.
    :type matrix_letter: str, optional
    :param show_legend: ``True`` to show legend describing what the vertical
        line is (marker for number of eigenvalues needed to meet/exceed trace).
    :type show_legend: bool, optional
    :param plot_title: Optional title for the plot.
    :type plot_title: str, optional
    :param plot_marker: Marker used for the scree plot, default ``"s"`` for
        square markers.
    :type plot_marker: str, optional
    :param plot_xticks: Locations for x-axis ticks. If there aren't many
        eigenvalues, it is recommended to pass ``1 + np.arange(eigs.shape[0])``
        so that all the eigenvalue numbers are shown.
    :type plot_xticks: :class:`numpy.ndarray`, optional
    :param plot_kwargs: Othe keyword args to pass to
        :func:`matplotlib.axes.Axes.plot`.
    :type plot_kwargs: dict, optional
    :returns: ``ax``
    :rtype: :class:`matplotlib.axes.Axes`
    """
    if not isinstance(ax, Axes):
        raise TypeError("ax must be matplotlib.axes.Axes")
    if not isinstance(eigs, np.ndarray):
        raise TypeError("eigs must be numpy.ndarray")
    if vline_kwargs is None:
        vline_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    # get number of eigenvalues
    n_eigs = eigs.shape[0]
    # if presort, then sort in descending order
    if presort:
        eigs = eigs.copy()
        eigs.sort()
        eigs = eigs[::-1]
    # trace (assumes normalized)
    trace = 1
    # if normalize, compute trace
    if normalize:
        trace = eigs.sum()
    # plot the normalized eigenvalues
    ax.plot(
        1 + np.arange(n_eigs), eigs / trace, marker = plot_marker,
        **plot_kwargs
    )
    # if plot_xticks is not None, set x axis ticks
    if plot_xticks is not None:
        ax.set_xticks(plot_xticks)
    # if pct_trace is not None
    if pct_trace is not None:
        if pct_trace <= 0 or pct_trace > 1:
            raise ValueError("pct_trace must be in (0, 1]")
        # compute number of eigenvalues needed to meet pct_trace of trace
        n_pct_trace = n_eigs_pct_trace(eigs, pct = pct_trace, presort = False)
        # plot vertical line indicating last eigenvalue needed to explain
        # n_pct_trace of eigenvalues' sum (trace)
        ax.axvline(
            x = n_pct_trace, color = vline_color,
            label = (
                r"$ k = " + str(n_pct_trace) + r"$, $ \frac{1}{\mathrm{tr}"
                r"(\mathbf{" + matrix_letter + r"})}\sum_{i = 1}^k\lambda_i "
                r"\geq " + str(pct_trace) + r" $"
            )
        )
        # show legend if show_legend is True
        if show_legend:
            ax.legend()
    # set axis labels and axis title (if not None)
    ax.set_xlabel("eigenvalue number")
    ax.set_ylabel("percent of trace")
    if plot_title is not None:
        ax.set_title(plot_title)
    # return Axes
    return ax