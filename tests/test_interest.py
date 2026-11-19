import numpy as np
import pytest

from pricing_engine.assumptions.interest import FlatRate, RatesByYear, interest_from_config


def test_flat_rate():
    np.testing.assert_array_equal(FlatRate(0.055).rates(3), [0.055, 0.055, 0.055])


def test_rates_by_year():
    np.testing.assert_array_equal(RatesByYear((0.03, 0.04, 0.05)).rates(2), [0.03, 0.04])


def test_rates_by_year_too_short():
    with pytest.raises(ValueError):
        RatesByYear((0.03,)).rates(2)


@pytest.mark.parametrize("bad", [FlatRate, lambda r: RatesByYear((r,))])
def test_rate_at_or_below_minus_one_rejected(bad):
    with pytest.raises(ValueError):
        bad(-1.0)


def test_from_config():
    assert isinstance(interest_from_config(0.06), FlatRate)
    assert isinstance(interest_from_config([0.05, 0.06]), RatesByYear)
