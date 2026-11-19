"""Step 9 gate: no shock = base NPV (74.13, DHW p. 520); adverse shocks
lower NPV. Only directions are asserted: DHW prints no shocked values."""

import numpy as np
import pytest

from pricing_engine.analysis.profit_test import npv
from pricing_engine.analysis.sensitivity import (
    SHOCKS, scenario, scenarios, shock, spider, tornado,
)
from pricing_engine.assumptions.basis import Basis
from pricing_engine.config_loader import config_path
from pricing_engine.engine.projection import run
from pricing_engine.engine.reserves import FPTReserve, NetPremiumReserve
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.term_life import TermLife
from pricing_engine.products.whole_life import WholeLife

R = 0.10
BASE = 74.13


def basis(name):
    return Basis.from_yaml(config_path(f"bases/{name}.yaml"))


TERM_MP = ModelPoint.from_yaml(config_path("model_points/term_dhw.yaml"))
TERM_RES, TERM_PROF = basis("term_reserve"), basis("term_profit")


def term_npv(profit_basis):
    return npv(run(TERM_MP, TermLife(), NetPremiumReserve(), TERM_RES, profit_basis), R)


# --- gate ---

@pytest.mark.parametrize("field", list(SHOCKS))
def test_no_shock_is_base(field):
    _, neutral = SHOCKS[field]
    assert term_npv(shock(TERM_PROF, field, neutral)) == pytest.approx(BASE, abs=0.01)


def test_higher_mortality_lowers_npv():
    assert term_npv(shock(TERM_PROF, "mortality", 1.1)) < BASE


def test_higher_expenses_lower_npv():
    assert term_npv(shock(TERM_PROF, "expenses", 1.1)) < BASE


def test_lower_earned_rate_lowers_npv():
    assert term_npv(shock(TERM_PROF, "interest", -0.01)) < BASE


# --- what a shock does to the basis ---

def test_shock_values():
    np.testing.assert_allclose(shock(TERM_PROF, "mortality", 1.1).q(60, 2), [0.011, 0.0121])
    np.testing.assert_allclose(shock(TERM_PROF, "interest", -0.01).i(2), [0.045, 0.045])
    E = shock(TERM_PROF, "expenses", 2.0)
    assert E.initial_expense([1500.0]) == pytest.approx(2 * 700)   # 2 x (400 + 20% x 1500)


def test_original_basis_untouched():
    shock(TERM_PROF, "mortality", 1.5)
    assert TERM_PROF.q(60, 1)[0] == 0.010


def test_rates_capped_at_one():
    wl = basis("wl_profit")
    assert shock(wl, "mortality", 3.0).q(60).max() == 1.0          # q130 = 1 stays 1
    assert shock(wl, "lapse", 2.0).w(40).max() == 1.0              # year-30 lapse of 1 stays 1


def test_missing_assumption_unchanged():
    assert shock(TERM_PROF, "lapse", 2.0) is TERM_PROF             # term has no lapse


def test_bad_inputs():
    with pytest.raises(ValueError):
        shock(TERM_PROF, "premium", 1.1)
    with pytest.raises(ValueError):
        shock(TERM_PROF, "mortality", -1.0)                        # scaled: no negatives
    shock(TERM_PROF, "interest", -0.01)                            # shifted: negative is fine


# --- scenario: several shocks together ---

def test_scenario_stacks_shocks():
    b = scenario(TERM_PROF, {"mortality": 1.1, "expenses": 1.1, "interest": -0.01})
    np.testing.assert_allclose(b.q(60, 1), [0.011])
    np.testing.assert_allclose(b.i(1), [0.045])
    assert b.initial_expense([1500.0]) == pytest.approx(770)


def test_scenario_empty_is_base():
    assert scenario(TERM_PROF, {}) is TERM_PROF


def test_scenarios_table():
    T = scenarios(term_npv, TERM_PROF, {
        "adverse": {"mortality": 1.1, "expenses": 1.1, "interest": -0.01},
        "favorable": {"mortality": 0.9, "expenses": 0.9, "interest": 0.01},
    })
    assert list(T.scenario) == ["base", "adverse", "favorable"]
    v = dict(zip(T.scenario, T.value))
    assert v["base"] == pytest.approx(BASE, abs=0.01)
    assert v["adverse"] < v["base"] < v["favorable"]


# --- tornado ---

def test_tornado():
    T = tornado(term_npv, TERM_PROF,
                {"mortality": (0.9, 1.1), "interest": (-0.01, 0.01), "expenses": (0.9, 1.1)})
    assert list(T.columns) == ["field", "low_size", "high_size", "base", "low", "high", "swing"]
    np.testing.assert_allclose(T.base, BASE, atol=0.01)
    assert T.swing.is_monotonic_decreasing
    np.testing.assert_allclose(T.swing, (T.high - T.low).abs())


def test_tornado_whole_life_same_code():
    product = WholeLife.from_yaml(config_path("products/whole_life_dhw.yaml"))
    m = ModelPoint.from_yaml(config_path("model_points/wl_dhw.yaml"))
    res = basis("wl_reserve")
    T = tornado(lambda b: npv(run(m, product, FPTReserve(), res, b), R), basis("wl_profit"),
                {"mortality": (0.9, 1.1), "lapse": (0.5, 1.5)})
    assert T.base[0] == pytest.approx(2938.26, abs=0.01)


# --- spider ---

def test_spider():
    sizes = {"mortality": [0.8, 0.9, 1.0, 1.1, 1.2], "interest": [-0.02, 0.0, 0.02]}
    T = spider(term_npv, TERM_PROF, sizes)
    assert len(T) == 8 and list(T.columns) == ["field", "size", "value"]
    mort = T[T.field == "mortality"]
    assert mort.value.is_monotonic_decreasing                      # more deaths, less profit
    assert mort[mort["size"] == 1.0].value.item() == pytest.approx(BASE, abs=0.01)
    assert T[(T.field == "interest") & (T["size"] == 0.0)].value.item() == pytest.approx(BASE, abs=0.01)
