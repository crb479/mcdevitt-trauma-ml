__doc__ = "Test ``mtml.data.sf_trauma`` module used for generating data."

# need to import all or else pytest will miss some fixtures
from .fixtures import *

import pytest

from .. import vitals_cols
from ..data.sf_trauma import cls_vitals_mof, cls_vitals_mort, cls_vitals_trauma

@pytest.mark.parametrize("seed", [7])
@pytest.mark.parametrize("func", [cls_vitals_trauma, cls_vitals_mort,
                                  cls_vitals_mof])
def test_vitals_factory(func, seed):
    """Test data sets created using :func:`mtml.data.sf_trauma.vitals_factory`.
    
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
    