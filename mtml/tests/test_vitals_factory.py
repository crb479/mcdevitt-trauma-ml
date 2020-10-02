__doc__ = "Test ``mtml.data.sf_trauma`` module used for generating data."

# need to import all or else pytest will miss some fixtures
from .fixtures import *

import pytest

from .. import vitals_cols
from ..data.sf_trauma import cls_vitals_mof, cls_vitals_mort, cls_vitals_trauma

@pytest.mark.parametrize("seed", [7])
@pytest.mark.parametrize("func", [cls_vitals_trauma, cls_vitals_mort,
                                  cls_vitals_mof])
def test_factory_vitals_with_na(func, seed):
    """Test functions using vitals data as input data, keeping NA data.
    
    Check shape, column names, output.
    """
    # seed for reproducibility
    X_train, X_val, y_train, y_val = func(random_state = seed)
    # check columns
    assert list(X_train.columns) == list(X_val.columns)
    assert list(X_train.columns) == list(vitals_cols)
    # check shapes
    assert X_train.shape == (1120, 9)
    assert y_train.shape == (1120,)
    assert X_val.shape == (374, 9)
    assert y_val.shape == (374,)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]


@pytest.mark.parametrize("seed", [7])
@pytest.mark.parametrize("func", [cls_vitals_trauma, cls_vitals_mort,
                                  cls_vitals_mof])
def test_factory_vitals_without_na(func, seed):
    """Test functions using vitals data as input data, dropping any NA data.
    
    Check shape, column names, output.
    """
    # seed for reproducibility
    X_train, X_val, y_train, y_val = func(dropna = True, random_state = seed)
    # check columns
    assert list(X_train.columns) == list(X_val.columns)
    assert list(X_train.columns) == list(vitals_cols)
    # check shapes (note the data set is much smaller)
    # using different functions gives slightly different NA droppage
    assert X_train.shape == (240, 9) or X_train.shape == (242, 9)
    assert y_train.shape == (240,) or y_train.shape == (242,)
    assert X_val.shape == (81, 9)
    assert y_val.shape == (81,)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    