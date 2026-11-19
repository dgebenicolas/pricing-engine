"""Term product. Expected values from DHW Sec. 13.3 (p. 509)."""

from dataclasses import replace

import numpy as np
import pytest

from pricing_engine.config_loader import config_path
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.base import Product
from pricing_engine.products.term_life import TermLife


@pytest.fixture
def mp():
    return ModelPoint.from_yaml(config_path("model_points/term_dhw.yaml"))


def test_is_a_product():
    assert isinstance(TermLife(), Product)


def test_term_is_mp_term(mp):
    assert TermLife().term(mp, omega=69) == 10   # term CSVs end at age 69


def test_dhw_schedule(mp):
    s = TermLife().policy_schedule(mp)
    assert s.n == 10
    np.testing.assert_array_equal(s.premium, 1500.0)
    np.testing.assert_array_equal(s.death_benefit, 100_000.0)
    np.testing.assert_array_equal(s.surrender_benefit, 0.0)
    np.testing.assert_array_equal(s.maturity_benefit, 0.0)


def test_reserves_ignored(mp):
    a = TermLife().policy_schedule(mp)
    b = TermLife().policy_schedule(mp, V=np.arange(11.0))
    np.testing.assert_array_equal(a.premium, b.premium)
    np.testing.assert_array_equal(a.death_benefit, b.death_benefit)


def test_premium_follows_model_point(mp):
    s = TermLife().policy_schedule(replace(mp, premium=1575.21))
    np.testing.assert_array_equal(s.premium, 1575.21)


def test_runs_past_omega_rejected(mp):
    with pytest.raises(ValueError):
        TermLife().term(mp, omega=68)


def test_missing_term_rejected(mp):
    with pytest.raises(ValueError):
        TermLife().term(replace(mp, term=None), omega=130)


def test_missing_premium_rejected(mp):
    with pytest.raises(ValueError):
        TermLife().policy_schedule(replace(mp, premium=None))
