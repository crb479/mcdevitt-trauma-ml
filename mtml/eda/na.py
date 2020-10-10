__doc__ = "Module in ``mtml.eda`` subpackage for dealing with missing data."

import pandas as pd


def na_count_by_col(df, percentage = False, sort_cols = "keep"):
    """Compute count of missing values in ``df``, by column.
    
    Missing values are determined by :func:`pandas.isna`.
    
    :param df: A :class:`pandas.DataFrame` that might have missing values.
    :type df: :class:`pandas.DataFrame`
    :param percentage: ``True`` to return percentage of missing values,
        ``False`` (default) to return counts of missing values.
    :type percentage: bool, optional
    :param sort_cols: Order to sort columns when returning statistics. Defaults
        to ``"keep"`` to maintain original column order, ``"ascending"`` to
        rank from least to most missing, ``"descending"`` to rank from most to
        least missing.
    :type sort_cols: str, optional
    :returns: :class:`pandas.Series` with an index consisting of all the columns
        names in ``df`` indexing counts or percentages of missing values.
    :rtype: :class:`pandas.Series`
    """
    if df is None:
        raise ValueError("df is None")
    # if percentage, use df.shape[0] as deflator, else set to 1 for counts. also
    # adjust the name of the final returned Series as well.
    if percentage:
        deflator = df.shape[0]
        col_name = "percent"
    else:
        deflator = 1
        col_name = "count"
    # get whether each element is NA or not, sum over rows + set name
    na_series = pd.isna(df).sum(axis = 0) / deflator
    na_series.name = col_name
    # sort by index, depending on sort_cols, and then return
    if sort_cols == "keep":
        pass
    elif sort_cols == "ascending":
        na_series.sort_values(ascending = True, inplace = True)
    elif sort_cols == "descending":
        na_series.sort_values(ascending = False, inplace = True)
    else:
        raise ValueError(f"unknown value sort_cols value {sort_cols}")
    return na_series