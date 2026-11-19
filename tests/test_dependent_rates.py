"""Dependent rates. Expected values from DHW only."""

import numpy as np
import pytest

from pricing_engine.assumptions.basis import dependent_rates
from pricing_engine.assumptions.mortality import MortalityTable
from pricing_engine.config_loader import config_path


@pytest.fixture(scope="module")
def susm():
    return MortalityTable.from_csv(config_path("mortality/susm.csv"))


# --- Gate: DHW Example 13.1, p. 527 (whole life, w = 5%) ---

def test_wl_year5_age64(susm):
    q = susm.q_vector(60, 5)
    p0d, p0w = dependent_rates(q, 0.05)
    p00 = 1 - p0d - p0w
    assert p0d[4] == pytest.approx(0.00529, abs=5e-6)
    assert p0w[4] == pytest.approx(0.04974, abs=5e-6)
    assert p00[4] == pytest.approx(0.94498, abs=5e-6)


def test_wl_year1_age60(susm):
    p0d, p0w = dependent_rates(susm.q_vector(60, 1), 0.05)
    assert (1 - p0d - p0w)[0] == pytest.approx(0.94677, abs=5e-6)


# --- Gate: term, no lapse (DHW Table 13.3) ---

def test_term_no_lapse_is_one_minus_q():
    q = MortalityTable.from_csv(config_path("mortality/term_profit.csv")).q_vector(60, 10)
    p0d, p0w = dependent_rates(q)  # w defaults to 0
    np.testing.assert_array_equal(p0w, 0.0)
    assert (1 - p0d - p0w)[1] == pytest.approx(0.989)  # age 61


# --- Unit behaviour ---

def test_decrements_sum_to_one():
    q = np.array([0.01, 0.02, 0.5])
    p0d, p0w = dependent_rates(q, [0.05, 0.075, 0.1])
    p00 = 1 - p0d - p0w
    assert np.all(p00 >= 0)
    np.testing.assert_allclose(p0d + p0w + p00, 1.0)


def test_full_lapse_leaves_no_one_in_force():
    p0d, p0w = dependent_rates([0.02], 1.0)  # e.g. WL year 30: everyone surrenders
    assert p0w[0] == pytest.approx(0.98)
    assert (1 - p0d - p0w)[0] == pytest.approx(0.0)


def test_lapse_vector_by_year():
    _, p0w = dependent_rates([0.1, 0.2], [0.05, 0.075])
    np.testing.assert_allclose(p0w, [0.9 * 0.05, 0.8 * 0.075])


def test_unknown_timing_rejected():
    with pytest.raises(ValueError):
        dependent_rates([0.01], 0.05, timing="mid_year")


@pytest.mark.parametrize("w", [-0.1, 1.1])
def test_bad_lapse_rejected(w):
    with pytest.raises(ValueError):
        dependent_rates([0.01], w)
