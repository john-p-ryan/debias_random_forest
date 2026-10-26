import numpy as np
import pytest

from debiasforest import simulate


@pytest.fixture(scope="session")
def f1_data():
    X, y, f = simulate("f1", 120, sigma=1.0, rng=11)
    Xt, _, ft = simulate("f1", 300, sigma=1.0, rng=12)
    return X, y, Xt, ft
