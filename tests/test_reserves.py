"""Reserves. Every expected value is from DHW (page/table cited) or a regression value."""

import numpy as np
import pytest

from pricing_engine.assumptions.basis import Basis
from pricing_engine.config_loader import config_path
from pricing_engine.engine.reserves import (
    FPTReserve, NetPremiumReserve, epv_factors, net_premium, policy_values,
)
from pricing_engine.model_point import ModelPoint


def basis(name):
    return Basis.from_yaml(config_path(f"bases/{name}.yaml"))


@pytest.fixture(scope="module")
def wl_reserve():
    return basis("wl_reserve")


@pytest.fixture(scope="module")
def term_reserve():
    return basis("term_reserve")


TERM_MP = ModelPoint(issue_age=60, sum_insured=100_000, premium=1500, term=10)
WL_MP = ModelPoint(issue_age=60, sum_insured=100_000, premium=2300)


# --- epv_factors: SUSM at 5% ---

@pytest.mark.parametrize("age, A, a", [
    (60, 0.29028, None),        # Table D.4
    (61, 0.30243, 14.6491),     # Table D.4
    (64, 0.34113, 13.8363),     # p. 526
])
def test_epv_whole_life_dhw(wl_reserve, age, A, a):
    A_x, a_x = epv_factors(wl_reserve.q(age), 0.05)
    assert A_x == pytest.approx(A, abs=5e-6)
    if a is not None:
        assert a_x == pytest.approx(a, abs=5e-5)


def test_epv_identity_whole_life(wl_reserve):
    A, a = epv_factors(wl_reserve.q(60), 0.05)
    d = 0.05 / 1.05
    assert A + d * a == pytest.approx(1.0)            # A_x = 1 - d a_x


def test_epv_by_hand_two_years():
    # q = [0.1, 0.2], i = 10%: A = 0.1/1.1 + 0.9*0.2/1.21, a = 1 + 0.9/1.1
    A, a = epv_factors([0.1, 0.2], 0.10)
    assert A == pytest.approx(0.1 / 1.1 + 0.9 * 0.2 / 1.21)
    assert a == pytest.approx(1 + 0.9 / 1.1)


def test_epv_rate_by_year_matches_flat():
    q = [0.1, 0.2, 0.3]
    assert epv_factors(q, [0.05, 0.05, 0.05]) == pytest.approx(epv_factors(q, 0.05))


# --- policy_values: the recursion itself ---

def test_policy_values_one_year_by_hand():
    V = policy_values(S=1000, P=50, E=0, q=[0.1], i=0.1)
    assert V[1] == 0.0
    assert V[0] == pytest.approx(0.1 * 1000 / 1.1 - 50)


def test_policy_values_floor():
    V = policy_values(S=1000, P=500, E=0, q=[0.1, 0.1], i=0.1, floor=0.0)
    assert np.all(V >= 0)


# --- Term: net premium reserve, DHW Sec. 13.3 ---

def test_term_net_premium_p512(term_reserve):
    P = NetPremiumReserve().net_premiums(TERM_MP, 10, term_reserve)
    np.testing.assert_allclose(P, 1447.63, atol=0.005)


TABLE_13_2 = [0.00, 410.05, 740.88, 988.90, 1150.10,
              1219.94, 1193.37, 1064.74, 827.76, 475.45, 0.00]


def test_term_reserves_table_13_2(term_reserve):
    V = NetPremiumReserve().reserves(TERM_MP, 10, term_reserve)
    assert len(V) == 11
    np.testing.assert_allclose(V, TABLE_13_2, atol=0.005)


def test_term_net_premium_matches_function(term_reserve):
    q = term_reserve.q(60, 10)
    assert net_premium(100_000, q, 0.04) == pytest.approx(1447.63, abs=0.005)


# --- Whole life: FPT, DHW Example 13.1 ---

def test_fpt_premiums_p526(wl_reserve):
    P = FPTReserve().net_premiums(WL_MP, 71, wl_reserve)
    q60 = wl_reserve.q(60, 1)[0]
    assert P[0] == pytest.approx(100_000 * q60 / 1.05)            # year 1: S v q60
    np.testing.assert_allclose(P[1:], 2064.47, atol=0.005)       # P* (p. 526)


@pytest.fixture(scope="module")
def wl_V(wl_reserve):
    return FPTReserve().reserves(WL_MP, 71, wl_reserve)


def test_fpt_shape_and_ends(wl_V):
    assert len(wl_V) == 72
    assert wl_V[0] == 0.0 and wl_V[1] == 0.0                      # FPT: 0V = 1V = 0
    assert wl_V[-1] == 0.0                                        # 71V = 0 at omega


def test_fpt_gate_values(wl_V):
    assert wl_V[3] == pytest.approx(3644.88, abs=0.005)           # p. 526
    assert wl_V[4] == pytest.approx(5548.04, abs=0.005)           # p. 526 (5548)
    assert wl_V[5] == pytest.approx(7504.02, abs=0.005)           # regression value


@pytest.mark.parametrize("t, V", [
    (2, 1795), (3, 3645), (4, 5548), (5, 7504),
    (27, 58149), (28, 60361), (29, 62518),
])
def test_fpt_table_13_8_column_2(wl_V, t, V):
    assert wl_V[t] == pytest.approx(V, abs=0.5)                   # Table 13.8 (rounded)


def test_fpt_prospective_formula_p526(wl_V, wl_reserve):
    # tV = S A_{60+t} - P* a_{60+t} for t >= 2
    P_star = 2064.47
    for t in (2, 4, 10, 30):
        A, a = epv_factors(wl_reserve.q(60 + t), 0.05)
        assert wl_V[t] == pytest.approx(100_000 * A - P_star * a, abs=0.1)
