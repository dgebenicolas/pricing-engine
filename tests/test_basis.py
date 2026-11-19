"""Basis bundling + the four DHW bases loaded from YAML. Expected values from DHW only."""

import numpy as np
import pytest

from pricing_engine.assumptions.basis import Basis
from pricing_engine.config_loader import config_path


def load(name):
    return Basis.from_yaml(config_path(f"bases/{name}.yaml"))


# --- Term, DHW Sec. 13.3 ---

def test_term_profit_basis():
    b = load("term_profit")
    premiums = np.full(10, 1500.0)
    np.testing.assert_allclose(b.q(60, 10), 0.01 + 0.001 * np.arange(10))
    np.testing.assert_array_equal(b.i(10), 0.055)
    np.testing.assert_array_equal(b.w(10), 0.0)
    assert b.initial_expense(premiums) == pytest.approx(700.0)        # p. 510
    np.testing.assert_allclose(b.renewal_expenses(premiums), 52.5)    # p. 510


def test_term_profit_survival_t2():
    p0d, p0w = load("term_profit").dependent_rates(60, 10)
    assert (1 - p0d - p0w)[1] == pytest.approx(0.989)                 # Table 13.3


def test_term_reserve_basis():
    b = load("term_reserve")
    np.testing.assert_allclose(b.q(60, 10), 0.011 + 0.001 * np.arange(10))
    np.testing.assert_array_equal(b.i(10), 0.04)                      # p. 512
    assert b.initial_expense([1500.0]) == 0.0
    np.testing.assert_array_equal(b.renewal_expenses(np.full(10, 1500.0)), 0.0)


# --- Whole life, DHW Example 13.1 ---

def test_wl_profit_basis():
    b = load("wl_profit")
    premiums = np.full(71, 2300.0)
    np.testing.assert_array_equal(b.i(71), 0.06)                      # p. 525
    assert b.initial_expense(premiums) == 1800.0
    np.testing.assert_allclose(premiums - b.renewal_expenses(premiums), 2200.0)


def test_wl_profit_dependent_rates_p527():
    p0d, p0w = load("wl_profit").dependent_rates(60)   # to omega
    p00 = 1 - p0d - p0w
    assert len(p0d) == 71
    assert p00[0] == pytest.approx(0.94677, abs=5e-6)                 # age 60
    assert p0w[4] == pytest.approx(0.04974, abs=5e-6)                 # age 64
    assert p00[4] == pytest.approx(0.94498, abs=5e-6)
    assert p00[29] == pytest.approx(0.0)                              # year 30: all surrender


def test_wl_reserve_basis_has_no_lapse():
    b = load("wl_reserve")
    p0d, p0w = b.dependent_rates(60)
    np.testing.assert_array_equal(p0w, 0.0)
    assert (1 - p0d)[4] == pytest.approx(0.99471, abs=5e-6)           # 1 - q64
    np.testing.assert_array_equal(b.i(71), 0.05)


def test_same_mortality_both_wl_bases():
    np.testing.assert_array_equal(load("wl_profit").q(60), load("wl_reserve").q(60))


def test_unknown_timing_rejected():
    b = load("term_profit")
    with pytest.raises(ValueError):
        Basis(b.mortality, b.interest, lapse_timing="mid_year")


# --- from_dict: edited values, mortality passed in ---

def test_from_dict_matches_from_yaml():
    from pricing_engine.config_loader import load_yaml
    yaml_basis = load("term_profit")
    cfg = load_yaml(config_path("bases/term_profit.yaml"))
    b = Basis.from_dict(cfg, yaml_basis.mortality)
    np.testing.assert_array_equal(b.i(10), yaml_basis.i(10))
    assert b.initial_expense([1500.0]) == yaml_basis.initial_expense([1500.0])


def test_from_dict_with_edits():
    m = load("term_profit").mortality
    b = Basis.from_dict({"interest": 0.06, "expenses": {"initial_fixed": 500}}, m)
    np.testing.assert_array_equal(b.i(3), 0.06)
    assert b.initial_expense([1500.0]) == 500.0
    np.testing.assert_array_equal(b.w(3), 0.0)                  # lapse left out -> none
