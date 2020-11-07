__doc__ = """Make supervised learning problems from ``SF_NUM_DATA_PREP_PATH``.

``SF_NUM_DATA_PREP_PATH`` corresponds to the
``data/prep/sf_trauma_data_num.csv`` data file, which is a miminally
preprocessed version of the raw data located in
``data/raw/sf_trauma_data_num_raw.csv``.

Current format is ``[(cls)(reg)]_[input_data_description]_[problem_type]``,
where ``cls`` is for classification, ``reg`` is for regression. ``problem_type``
for example could be ``mort`` for predicting mortality at discharge, ``mof``
for predicting multiple organ failure, ``trauma`` for predicting whether
someone has suffered significant trauma or not (>15 ISS score), etc.

``input_data_description`` must consist of the following quantifiers, separated
by an underscore: ``demo`` for patient demographics, ``prem`` for the single
feature indicating preexisting disease/comorbidity, ``vitals`` for patient
vitals data, and ``lab`` for patient lab panel data. These correspond to
the headers in ``SF Hospital trauma data-v2.xlsx``
`located in the Google Drive`__.

.. __: https://drive.google.com/drive/folders/1VyFHmTdDq-yMMvj_CPfEcV60Jvb70-RL
   ?usp=sharing
"""
import _io
import math
import numpy as np
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .. import SF_NUM_DATA_PREP_PATH, SF_DISEASE_COMORB_COLS, SF_VITALS_COLS
from .factory import make_slp_from_data


def sf_slp_factory(**kwargs):
    """Generates supervised learning problems from SF trauma data set.

    A wrapper for :func:`~mtml.data.factory.make_slp_from_data` where the
    ``data_path`` argument is the path to the preprocessed numerical SF trauma
    data located in ``data/files/prep/sf_trauma_data_num.csv``.
    
    See the docstring for :func:`~mtml.data.factory.make_slp_from_data` for
    details on the individual keyword arguments that can be passed.

    :param kwargs: Keyword arguments to pass to
        :func:`~mtml.data.factory.make_slp_from_data`.
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    return make_slp_from_data(SF_NUM_DATA_PREP_PATH, **kwargs)


def make_trauma_feature(ser):
    """Creates 0/1 trauma feature from :class:`pandas.Series` of ISS scores.
    
    Note that NaN values are handled correctly, i.e. passed unaffected.
    
    :param ser: A :class:`pandas.Series` of ISS scores
    :type ser: :class:`pandas.Series`
    :rtype: :class:`numpy.ndarray`
    """
    # lambda to perform conversion
    _conv = lambda x: 1 if x > 15 else (np.nan if np.isnan(x) else 0)
    return np.array(tuple(map(_conv, ser.values)))


def cls_vitals_trauma(test_size = 0.25, shuffle = True, dropna = False,
                      na_axis = 0, na_how = "any", na_thresh = None,
                      na_subset = None, random_state = None):
    """Return vitals data from ``SF_NUM_DATA_PREP_PATH`` to classify trauma.
    
    Trauma is indicated when ISS score is greater than 15.

    :param test_size: Float in ``[0, 1]`` to represent fraction of points to
        use for validation data or an int to represent number of points to use
        for validation data. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: float or int, optional
    :param shuffle: Whether or not to shuffle the data before splitting,
        defaults to ``True``. ``False`` leaves split data arranged in ascending
        index order.
    :type shuffle: bool, optional
    :param dropna: ``True`` to drop missing values. Default ``False``.
    :type dropna: bool, optional
    :param na_axis: ``axis`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_axis: int, optional
    :param na_how: ``how`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_how: str, optional
    :param na_thresh: ``thresh`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_thresh: int, optional
    :param na_subset: ``subset`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_subset: array-like, optional
    :param random_state: Int seed or :class:`~numpy.random.RandomState` to use
        to make multiple calls to this method reproducible. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type random_state: int or :class:`~numpy.random.RandomState`, optional
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    # use factory method
    return sf_slp_factory(
        inputs = SF_VITALS_COLS, targets = ["iss"], 
        target_transform = make_trauma_feature, dropna = dropna,
        na_axis = na_axis, na_how = na_how, na_thresh = na_thresh,
        na_subset = na_subset, test_size = test_size, shuffle = shuffle,
        random_state = random_state
    )


def cls_vitals_mort(test_size = 0.25, shuffle = True, dropna = False,
                      na_axis = 0, na_how = "any", na_thresh = None,
                      na_subset = None, random_state = None):
    """Return vitals data from ``SF_NUM_DATA_PREP_PATH`` to classify mortality.
    
    Mortality corresponds to mortality at discharge.

    :param test_size: Float in ``[0, 1]`` to represent fraction of points to
        use for validation data or an int to represent number of points to use
        for validation data. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: float or int, optional
    :param shuffle: Whether or not to shuffle the data before splitting,
        defaults to ``True``. ``False`` leaves split data arranged in ascending
        index order.
    :type shuffle: bool, optional
    :param dropna: ``True`` to drop missing values. Default ``False``.
    :type dropna: bool, optional
    :param na_axis: ``axis`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_axis: int, optional
    :param na_how: ``how`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_how: str, optional
    :param na_thresh: ``thresh`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_thresh: int, optional
    :param na_subset: ``subset`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_subset: array-like, optional
    :param random_state: Int seed or :class:`~numpy.random.RandomState` to use
        to make multiple calls to this method reproducible. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type random_state: int or :class:`~numpy.random.RandomState`, optional
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    return sf_slp_factory(
        inputs = SF_VITALS_COLS, targets = ["mortality at disch"],
        dropna = dropna, na_axis = na_axis, na_how = na_how,
        na_thresh = na_thresh, na_subset = na_subset, test_size = test_size,
        shuffle = shuffle, random_state = random_state
    )


def cls_vitals_mof(test_size = 0.25, shuffle = True, dropna = False,
                   na_axis = 0, na_how = "any", na_thresh = None,
                   na_subset = None, random_state = None):
    """Return vitals data from ``SF_NUM_DATA_PREP_PATH`` to classify MOF.
    
    MOF = multiple organ failure.

    :param test_size: Float in ``[0, 1]`` to represent fraction of points to
        use for validation data or an int to represent number of points to use
        for validation data. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type test_size: float or int, optional
    :param shuffle: Whether or not to shuffle the data before splitting,
        defaults to ``True``. ``False`` leaves split data arranged in ascending
        index order.
    :type shuffle: bool, optional
    :param dropna: ``True`` to drop missing values. Default ``False``.
    :type dropna: bool, optional
    :param na_axis: ``axis`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_axis: int, optional
    :param na_how: ``how`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_how: str, optional
    :param na_thresh: ``thresh`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_thresh: int, optional
    :param na_subset: ``subset`` parameter of :meth:`pandas.DataFrame.dropna`.
    :type na_subset: array-like, optional
    :param random_state: Int seed or :class:`~numpy.random.RandomState` to use
        to make multiple calls to this method reproducible. Passed directly to
        :func:`~sklearn.model_selection.train_test_split`.
    :type random_state: int or :class:`~numpy.random.RandomState`, optional
    :returns: A 4-tuple ``(X_train, X_test, y_train, y_test)`` of
        :class:`numpy.ndarray` objects
    :rtype: tuple
    """
    return sf_slp_factory(
        inputs = SF_VITALS_COLS, targets = ["mof"], dropna = dropna, 
        na_axis = na_axis, na_how = na_how, na_thresh = na_thresh,
        na_subset = na_subset, test_size = test_size, shuffle = shuffle, 
        random_state = random_state
    )


def sf_domain_impute(target = None):
    """Using our domain-specific knowledge, impute the raw SF trauma data set.

    Imputes the data in ``data/files/prep/sf_trauma_data_num.csv`` using our
    domain-specific knowledge. Drops columns APACHE-2, hr0_lctate, albumin,
    day1_bilirubin, day1_urine output total, hr0_fibrinogen, all comorbidities
    (ex. aids, hiv, asthma, etc.). Drops rows where smoking status, race, or
    latino features are missing and any rows where iss, mof, or mortality at
    disch (the target columns) are missing. Imputes bmi, weight kg, and height
    cm if two out of those three values are present in a row, else skip.

    :param target: Optional file name or file object to save the imputed data
        set to. If ``None``, then no writing to disk is performed.
    :type target: str or , optional
    :returns: The data set preprocessed using our domain-specific knowledge.
    :rtype: :class:`pandas.DataFrame`
    """
    # read the SF trauma data
    df = pd.read_csv(SF_NUM_DATA_PREP_PATH)
    # drop rows if smoking, blood type, race, latino missing or if we are
    # missing any of the 3 outcomes we care about: iss (trauma score), mof
    # (multiple organ failure), mortality at discharge, or if any rows don't
    # contain protein C or D-dimer observations
    df.dropna(
        subset = [
            "smoking status", "race", "latino", "iss", "mof",
            "mortality at disch", "hr0_Protein C", "hr0_D-Dimer"
        ], inplace = True
    )
    # drop rows if missing the 3 outcomes we care about: iss (trauma score),
    # mof (multiple organ failure), mortality at discharge
    df.dropna(subset = ["iss", "mof", "mortality at disch"], inplace = True)
    # drop columns: all comobidities, APACHE-2, lactate, albumin, bilirubin, 
    # urine output, fibrinogen
    df.drop(
        columns = list(SF_DISEASE_COMORB_COLS) + [
            "APACHE-2", "hr0_lactate", "albumin", "day1_bilirubin",
            "day1_urine output total (ml)", "hr0_fibrinogen"
        ], inplace = True
    )
    # what to do with ABG? hr0_pH, hr0_paCO2, hr0_paO2, hr0_HCO3
    # weight, height (cm), bmi imputation
    whb = ["weight kg", "height cm", "bmi"]
    # impute BMI rows (bmi = mass / height ** 2; height in METERS). note we use
    # df.index.values since some of the rows were dropped.
    for i in df.index.values:
        # reference to the weight in kg, height in cm, bmi
        whb_vals = df.loc[i, whb]
        # if there is a single NA value, impute
        if whb_vals.isna().sum() == 1:
            # if weight is na, weight = bmi * (height / 100) ** 2
            if np.isnan(whb_vals[0]):
                df.loc[i, whb[0]] = whb_vals[2] * (whb_vals[1] ** 2) / 1e4
            # else if height is na, height = sqrt(mass / bmi) * 100
            elif np.isnan(whb_vals[1]):
                df.loc[i, whb[1]] = math.sqrt(whb_vals[0] / whb_vals[2]) * 100
            # else if bmi is na, bmi = mass / (height / 100) ** 2
            elif np.isnan(whb_vals[2]):
                df.loc[i, whb[2]] = whb_vals[0] / (whb_vals[1] ** 2) / 1e4
        # else just ignore, can't impute if 2 of 3 are missing
    # do nothing if target is None
    if target is None:
        pass
    # else write to new file
    else:
        df.to_csv(target, index = False)
    # return modified DataFrame
    return df