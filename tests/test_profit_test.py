"""Step 6 gate: profit measures and premium solver. DHW pages cited."""

from dataclasses import replace

import numpy as np
import pytest

from pricing_engine.analysis.profit_test import (
    epv_premiums, irr, npv, profit_margin, solve_premium,
)
from pricing_engine.assumptions.basis import Basis
from pricing_engine.config_loader import config_path
from pricing_engine.engine.projection import run
from pricing_engine.engine.reserves import (
    FPTReserve, NetPremiumReserve, NoReserve, ZeroizedReserve,
)
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.term_life import TermLife
from pricing_engine.products.whole_life import WholeLife

R = 0.10   # risk discount rate, DHW Ch. 13


def basis(name):
    return Basis.from_yaml(config_path(f"bases/{name}.yaml"))


TERM_MP = ModelPoint.from_yaml(config_path("model_points/term_dhw.yaml"))
TERM_RES, TERM_PROF = basis("term_reserve"), basis("term_profit")


def term_ledger(m=TERM_MP, method=NetPremiumReserve(), reserve_basis=TERM_RES):
    return run(m, TermLife(), method, reserve_basis, TERM_PROF)


@pytest.fixture(scope="module")
def L():
    return term_ledger()


# --- Term with net premium reserves (pp. 519-521) ---

def test_npv(L):
    assert npv(L, R) == pytest.approx(74.13, abs=0.01)


def test_irr(L):
    assert irr(L) == pytest.approx(0.124, abs=0.0005)


def test_profit_margin(L):
    assert epv_premiums(L, R) == pytest.approx(9684.5, abs=0.05)      # p. 521
    assert profit_margin(L, R) == pytest.approx(0.0077, abs=0.00005)


# --- Effect of reserves: p. 522 ---

def test_npv_without_reserves():
    assert npv(term_ledger(method=NoReserve()), R) == pytest.approx(270.39, abs=0.01)


def test_reserves_cost_nothing_when_r_equals_earned_rate(L):
    # Reserves earn 5.5%; discounting at 5.5% removes the cost of capital.
    assert npv(L, 0.055) == pytest.approx(248.30, abs=0.01)
    assert npv(term_ledger(method=NoReserve()), 0.055) == pytest.approx(248.30, abs=0.01)


# --- Premium solver: p. 521 ---

@pytest.fixture(scope="module")
def P_solved():
    return solve_premium(TERM_MP, term_ledger, lambda led: profit_margin(led, R), 0.05)


def test_solver_premium(P_solved):
    assert P_solved == pytest.approx(1575.21, abs=0.005)


def test_solver_results(P_solved):
    L5 = term_ledger(replace(TERM_MP, premium=P_solved))
    assert npv(L5, R) == pytest.approx(508.50, abs=0.01)
    assert epv_premiums(L5, R) == pytest.approx(10170.03, abs=0.01)
    np.testing.assert_allclose(
        L5.signature,
        [-715.0, 197.7, 201.5, 203.9, 204.9, 204.5, 202.6, 199.4, 194.6, 188.4, 180.8],
        atol=0.05)
    assert irr(L5) == pytest.approx(0.250, abs=0.0005)


# --- Zeroized reserves: Tables 13.6, 13.7 (pp. 522-523) ---

@pytest.fixture(scope="module")
def zero_V():
    return ZeroizedReserve().reserves(TERM_MP, 10, TERM_PROF)


def test_table_13_6(zero_V):
    np.testing.assert_allclose(
        zero_V[3:10], [247.62, 494.78, 658.32, 732.63, 711.42, 587.65, 353.45], atol=0.005)
    np.testing.assert_array_equal(zero_V[:3], 0.0)    # 2V = -78.17 floored, then 1V, 0V
    assert zero_V[10] == 0.0


@pytest.fixture(scope="module")
def zero_L():
    return term_ledger(method=ZeroizedReserve(), reserve_basis=TERM_PROF)


def test_table_13_7(zero_L):
    np.testing.assert_allclose(
        zero_L.profit, [-700.00, 527.11, 427.11, 82.47] + [0.0] * 7, atol=0.01)
    np.testing.assert_allclose(
        zero_L.signature, [-700.00, 527.11, 422.84, 80.74] + [0.0] * 7, atol=0.01)


def test_zeroized_npv(zero_L):
    assert npv(zero_L, R) == pytest.approx(189.31, abs=0.01)    # pp. 523-524


# --- Whole life, non-par: regression values (not printed in DHW) ---

@pytest.fixture(scope="module")
def wl():
    product = WholeLife.from_yaml(config_path("products/whole_life_dhw.yaml"))
    reserve, profit = basis("wl_reserve"), basis("wl_profit")
    return lambda m: run(m, product, FPTReserve(), reserve, profit)


def test_wl_regression(wl):
    m = ModelPoint.from_yaml(config_path("model_points/wl_dhw.yaml"))
    L = wl(m)
    assert npv(L, R) == pytest.approx(2938.26, abs=0.01)
    assert profit_margin(L, R) == pytest.approx(0.189, abs=0.0005)
    assert irr(L) == pytest.approx(0.495, abs=0.0005)
    P = solve_premium(m, wl, lambda led: profit_margin(led, R), 0.05)
    assert P == pytest.approx(1949.34, abs=0.005)


# --- general behaviour ---

def test_irr_not_found_raises():
    L = term_ledger(replace(TERM_MP, premium=1000))
    L = L.assign(signature=-L.signature.abs())        # all losses: NPV never 0
    with pytest.raises(ValueError):
        irr(L)


def test_solver_unreachable_target():
    with pytest.raises(ValueError):
        solve_premium(TERM_MP, term_ledger, lambda led: profit_margin(led, R), 5.0)
