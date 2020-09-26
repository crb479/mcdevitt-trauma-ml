__doc__ = "Fixtures that can be imported for use with ``pytest`` tests."

import pandas as pd
import pytest

from .. import sf_num_data_prep_path as _sf_num_data_prep_path


@pytest.fixture(scope = "session")
def sf_num_data_prep_path():
    "Returns absolute path to ``data/prep/sf_trauma_data_num.csv``."
    return _sf_num_data_prep_path


@pytest.fixture(scope = "session")
def sf_num_data_prep_data(sf_num_data_prep_path):
    "Returns :class:`~pandas.DataFrame` from ``sf_num_data_prep_path``."
    return pd.read_csv(sf_num_data_prep_path)