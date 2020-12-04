__doc__ = """Contains data transformations used on VTE data.

These transformations must be compatible with the call signature of
:func:`mtml.data.vte.vte_slp_factory`, or more generally, the call signature of
:func:`mtml.data.factory.make_slp_from_data`.
"""


def replace_hdl_tot_chol_with_ratio(df):
    """Replace ``tot_cholesterol_result`` and ``hdl_result`` with their ratio.

    Recommended domain-specific feature engineering step for VTE data.

    :param df: :class:`pandas.DataFrame` containing the VTE data.
    :type df: :class:`pandas.DataFrame`
    :rtype: :class:`pandas.DataFrame`
    """
    # get ratio of total cholesterol and HDL
    ratio = df["tot_cholesterol_result"].values / df["hdl_result"].values
    # drop both original columns and add new column as tot_chol_over_hdl
    df.drop(columns = ["tot_cholesterol_result", "hdl_result"], inplace = True)
    df = df.assign(tot_chol_over_hdl = ratio)
    # done, so return
    return df