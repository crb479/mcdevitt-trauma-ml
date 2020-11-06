__doc__ = "Fixtures that can be imported for use with ``pytest`` tests."

import pandas as pd
import pytest

# pylint: disable=relative-beyond-top-level
from .. import SF_NUM_DATA_PREP_PATH


@pytest.fixture(scope = "session")
def sf_num_data_prep_path():
    "Returns absolute path to ``data/files/prep/sf_trauma_data_num.csv``."
    return SF_NUM_DATA_PREP_PATH


@pytest.fixture(scope = "session")
def sf_num_data_prep_data(sf_num_data_prep_path):
    "Returns :class:`~pandas.DataFrame` from ``SF_NUM_DATA_PREP_PATH``."
    return pd.read_csv(SF_NUM_DATA_PREP_PATH)