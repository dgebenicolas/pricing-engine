"""Whole life product. Expected values from DHW Example 13.1 (pp. 524-527)."""

from dataclasses import replace

import numpy as np
import pytest

from pricing_engine.config_loader import config_path
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.base import Product
from pricing_engine.products.whole_life import WholeLife


@pytest.fixture
def product():
    return WholeLife.from_yaml(config_path("products/whole_life_dhw.yaml"))


@pytest.fixture
def mp():
    return ModelPoint.from_yaml(config_path("model_points/wl_dhw.yaml"))


def test_is_a_product(product):
    assert isinstance(product, Product)


def test_term_runs_to_omega(product, mp):
    assert product.term(mp, omega=130) == 71     # 130 - 60 + 1


def test_h_vector_dhw_p524(product):
    h = product.h_vector(15)
    np.testing.assert_array_equal(h[:4], 0.0)                       # years 1-4
    np.testing.assert_allclose(h[4:13], np.arange(1, 10) / 10)      # years 5-13: 0.1 .. 0.9
    np.testing.assert_array_equal(h[13:], 0.9)                      # years 14+


def test_h_vector_shorter_than_list(product):
    np.testing.assert_array_equal(product.h_vector(3), 0.0)


def test_cash_value_year5_dhw(product, mp):
    # 5V = 7504.02 (DHW p. 526); other reserves irrelevant to this check.
    V = np.zeros(72)
    V[5] = 7504.02
    s = product.policy_schedule(mp, V)
    assert s.surrender_benefit[4] == pytest.approx(750.402)          # 0.1 * 7504.02
    assert 0.04974 * s.surrender_benefit[4] == pytest.approx(37.32, abs=0.005)  # Table 13.8 ESB


def test_schedule_shape(product, mp):
    s = product.policy_schedule(mp, np.zeros(72))
    assert s.n == 71
    np.testing.assert_array_equal(s.premium, 2300.0)
    np.testing.assert_array_equal(s.death_benefit, 100_000.0)
    np.testing.assert_array_equal(s.maturity_benefit, 0.0)


def test_cash_value_uses_year_end_reserve(product, mp):
    V = np.arange(72, dtype=float)     # tV = t, so CV_t = h_t * t
    s = product.policy_schedule(mp, V)
    t = np.arange(1, 72)
    np.testing.assert_allclose(s.surrender_benefit, product.h_vector(71) * t)


def test_needs_reserves(product, mp):
    with pytest.raises(ValueError):
        product.policy_schedule(mp)


def test_needs_premium(product, mp):
    with pytest.raises(ValueError):
        product.policy_schedule(replace(mp, premium=None), np.zeros(72))


def test_rejects_fixed_term(product, mp):
    with pytest.raises(ValueError):
        product.term(replace(mp, term=10), omega=130)


@pytest.mark.parametrize("bad", [[], [0.1, 1.5], [-0.1]])
def test_bad_cash_value_pct_rejected(bad):
    with pytest.raises(ValueError):
        WholeLife(bad)


def test_from_dict():
    p = WholeLife.from_dict({"cash_value_pct": [0, 0.5]})
    np.testing.assert_array_equal(p.h_vector(4), [0, 0.5, 0.5, 0.5])
