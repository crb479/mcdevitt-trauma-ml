__doc__ = "Simple data preprocessing routines."

import numpy as np
import pandas as pd
import re

trauma_score_cols = (
    "ais head1", "ais face2", "ais chest3", "ais abdomen4", "ais extremity5",
    "ais skin6", "iss", "APACHE-2", "GCS (ED arrival)"
)
"Labels for trauma score columns to drop from data set."

disease_comorb_cols = (
    "hiv", "aids", "hepatic failure", "immuno-suppression", "asthma", "copd",
    "ild", "other chronic lung disease", "cad", "chf", "esrd dialysis",
    "cirrhosis", "diabetes", "malignancy"
)
"Pre-existing disease and comorbidity column labels."

mangled_dup_re = r"^.+\.[0-9]+$"
r"Regular expression for matching ``pandas``\ -mangled duplicate columns."


def drop_mangled_dups(df, columns = None, inplace = False):
    """Drop duplicate columns from a CSV file loaded into a pandas DataFrame.
    
    When a CSV file is loaded into a :class:`~pandas.DataFrame` using
    :func:`~pandas.read_csv`, duplicate columns are typically mangled. I.e. for
    some column ``X``, duplicates are named ``X.1``, ``X.2``, etc. This function
    returns a :class:`~pandas.DataFrame` with these columns dropped.
    
    .. warning::
    
       Since duplicate columns are assumed to be mangled in the ``X.1``, ``X.2``
       format, if ``df`` has non-duplicated column labels in this format, they
       will get dropped unless column labels are explicitly provided to the
       ``columns`` parameter.
    
    :param df: :class:`~pandas.DataFrame` to drop mangled duplicated cols from.
    :type df: :class:`pandas.DataFrame`
    :param columns: A list of string column labels for which mangled duplicate
        columns will be dropped. If ``None``, the method will drop mangled
        duplicates for all columns in ``df``.
    :type columns: iterable, optional
    :param inplace: ``True`` to perform operation in place, in which a reference
        to the original ``df`` is returned. Defaults to ``False`` to return a
        modified copy of ``df``.
    :type inplace: bool, optional
    :rtype: :class:`~pandas.DataFrame`
    """
    if df is None:
        raise ValueError("df is None")
    # compile regex
    _mdre = re.compile(mangled_dup_re)
    # find all duplicated columns
    drop_cols = []
    for col in df.columns:
        if _mdre.match(col):
            drop_cols.append(col)
    # if columns is None, drop all columns that match _mdre
    if columns is None:
        df_copy = df.drop(columns = drop_cols, inplace = inplace)
        if inplace:
            return df
        return df_copy
    # else only drop the columns in columns that are duplicated
    sub_drop_cols = []
    for dcol in drop_cols:
        if ".".join(dcol.split(".")[:-1]) in columns:
            sub_drop_cols.append(dcol)
    df_copy = df.drop(columns = sub_drop_cols, inplace = inplace)
    if inplace:
        return df
    return df_copy


def yes_no_binarize(df, columns = None, case_sensitive = True, inplace = False):
    """Replaces ``"Yes"`` and ``"No"`` with ``1`` and ``0``, respectively.
    
    Can specify a subset of columns to perform the replacement on using the
    ``columns`` parameter. 
    
    :param df:
    :type df: :class:`~pandas.DataFrame`
    :param columns: Column labels to perform substitution on. Default ``None``
        to apply to entire data set.
    :type columns: iterable, optional
    :param inplace: ``True`` to perform operation in place, in which a reference
        to the original ``df`` is returned. Defaults to ``False`` to return a
        modified copy of ``df``.
    :type inplace: bool, optional
    :returns: ``df`` with ``"Yes"`` and ``"No"`` replaced with ``1`` and ``0``
        respectively, case sensitivity depending on the ``case_sensitive``
        parameter. Note this will be a new :class:`~pandas.DataFrame` if
        ``inplace`` is ``False``, else it will be the original reference ``df``.
    :rtype: :class:`~pandas.DataFrame`
    """
    if df is None:
        raise ValueError("df is None")
    # yes and no strings
    yes, no = "Yes", "No"
    # if case_sensitive is False, then replace with regex
    if case_sensitive == False:
        yes, no = r"^[yY][eE][sS]$", r"^[nN][oO]"
    # if columns is None, perform substitution on whole DataFrame
    if columns is None:
        if inplace:
            df.replace(to_replace = yes, value = 1, inplace = True)
            df.replace(to_replace = no, value = 0, inplace = True)
            return df
        # else return copy
        return df.replace(
            to_replace = yes, value = 1, inplace = False
        ).replace(to_replace = no, value = 0, inplace = False)
    # else perform substitution on the columns only
    # deep copy if inplace is False
    if inplace == False:
        df_copy = df.copy(deep = True)
    else:
        df_copy = df
    for col in columns:
        df_copy.loc[:, col].replace(to_replace = yes, value = 1, inplace = True)
        df_copy.loc[:, col].replace(to_replace = no, value = 0, inplace = True)
    return df_copy