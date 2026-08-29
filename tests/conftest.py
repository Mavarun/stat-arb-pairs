import pytest

from tests._synth import make_cointegrated_pair


@pytest.fixture
def coint_pair():
    y, x, beta, intercept = make_cointegrated_pair()
    return y, x, beta, intercept
