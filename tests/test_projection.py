"""Projection ledger. Expected values from DHW tables (pages cited)."""

import numpy as np
import pytest

from pricing_engine.assumptions.basis import Basis
from pricing_engine.config_loader import config_path
from pricing_engine.engine.projection import project, run
from pricing_engine.engine.reserves import FPTReserve, NetPremiumReserve
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.term_life import TermLife
from pricing_engine.products.whole_life import WholeLife


def basis(name):
    return Basis.from_yaml(config_path(f"bases/{name}.yaml"))


def mp(name):
    return ModelPoint.from_yaml(config_path(f"model_points/{name}.yaml"))


# --- Table 13.1 (p. 511): term, no reserves ---

def test_table_13_1_no_reserves():
    m = mp("term_dhw")
    sched = TermLife().policy_schedule(m)
    ledger = project(sched, np.zeros(11), basis("term_profit"), m.issue_age)
    np.testing.assert_allclose(ledger.expenses, [700] + [52.50] * 10, atol=0.005)
    np.testing.assert_allclose(ledger.interest[1:], 79.61, atol=0.005)
    np.testing.assert_allclose(ledger.edb[1:], np.arange(1000, 2000, 100), atol=0.005)
    np.testing.assert_allclose(
        ledger.profit,
        [-700.00, 527.11, 427.11, 327.11, 227.11, 127.11,
         27.11, -72.89, -172.89, -272.89, -372.89],
        atol=0.005,
    )


# --- Tables 13.3 (p. 513) and 13.4 (p. 516): term with net premium reserves ---

@pytest.fixture(scope="module")
def term_ledger():
    return run(mp("term_dhw"), TermLife(), NetPremiumReserve(),
               basis("term_reserve"), basis("term_profit"))


def test_table_13_3(term_ledger):
    L = term_ledger
    np.testing.assert_allclose(
        L.reserve_bf[1:],
        [0.00, 410.05, 740.88, 988.90, 1150.10, 1219.94, 1193.37, 1064.74, 827.76, 475.45],
        atol=0.005)
    np.testing.assert_allclose(
        L.interest[1:],
        [79.61, 102.17, 120.36, 134.00, 142.87, 146.71, 145.25, 138.17, 125.14, 105.76],
        atol=0.005)
    np.testing.assert_allclose(L.edb[1:], np.arange(1000, 2000, 100), atol=0.005)
    np.testing.assert_allclose(
        L.reserve_cf,
        [0.00, 405.95, 732.73, 977.04, 1135.15, 1202.86, 1175.47, 1047.70, 813.69, 466.89, 0.00],
        atol=0.005)
    # Pr and Pi use atol = 1 cent: DHW computed Pr1 from 1V rounded to 410.05
    # (121.163 -> 121.16); the exact 1V = 410.0473 gives 121.166.
    np.testing.assert_allclose(
        L.profit,
        [-700.00, 121.16, 126.99, 131.70, 135.26, 137.61, 138.68, 138.41, 136.72, 133.52, 128.71],
        atol=0.01)


def test_table_13_4_signature(term_ledger):
    np.testing.assert_allclose(
        term_ledger.signature,
        [-700.00, 121.16, 125.72, 128.95, 130.84, 131.39,
         130.56, 128.35, 124.75, 119.76, 113.37],
        atol=0.01)


def test_term_has_no_surrender_or_maturity(term_ledger):
    assert (term_ledger.esb == 0).all() and (term_ledger.eeb == 0).all()


# --- Table 13.8 (p. 525): whole life, FPT, same engine code ---

@pytest.fixture(scope="module")
def wl_ledger():
    product = WholeLife.from_yaml(config_path("products/whole_life_dhw.yaml"))
    return run(mp("wl_dhw"), product, FPTReserve(), basis("wl_reserve"), basis("wl_profit"))


#        t   (t-1)V  P-E   I     EDB   ECV    EtV    Pr_t-
TABLE_13_8 = [
    (1,      0, 2200,  132,  340,     0,     0, 1992),
    (2,      0, 2200,  132,  379,     0,  1699,  254),
    (3,   1795, 2200,  240,  423,     0,  3448,  364),
    (4,   3645, 2200,  351,  473,     0,  5246,  477),
    (5,   5548, 2200,  465,  529,    37,  7091,  556),
    (6,   7504, 2200,  582,  591,    95,  8983,  618),
    (28, 58149, 2200, 3621, 7224,  3780, 51800, 1166),
    (29, 60361, 2200, 3754, 8080,  3879, 53157, 1199),
    (30, 62518, 2200, 3883, 9033, 52901,     0, 6667),
]


@pytest.mark.parametrize("row", TABLE_13_8, ids=lambda r: f"t{r[0]}")
def test_table_13_8_columns_2_to_8(wl_ledger, row):
    t, V_bf, p_less_e, I, edb, ecv, ev, pr = row
    r = wl_ledger.iloc[t]
    assert r.reserve_bf == pytest.approx(V_bf, abs=0.5)
    assert r.premium - r.expenses == pytest.approx(p_less_e, abs=0.5)
    assert r.interest == pytest.approx(I, abs=0.5)
    assert r.edb == pytest.approx(edb, abs=0.5)
    assert r.esb == pytest.approx(ecv, abs=0.5)
    assert r.reserve_cf == pytest.approx(ev, abs=0.5)
    assert r.profit == pytest.approx(pr, abs=0.5)


def test_wl_row0_and_gate_values(wl_ledger):
    assert wl_ledger.profit[0] == pytest.approx(-1800)            # Table 13.8
    assert wl_ledger.interest[4] == pytest.approx(350.69, abs=0.005)  # p. 526
    assert wl_ledger.profit[5] == pytest.approx(555.68, abs=0.005)    # regression value


def test_wl_nobody_in_force_after_year_30(wl_ledger):
    assert len(wl_ledger) == 72                                   # t = 0..71
    np.testing.assert_allclose(wl_ledger.signature[31:], 0.0, atol=1e-9)


# --- general behaviour ---

def test_V_length_checked():
    m = mp("term_dhw")
    sched = TermLife().policy_schedule(m)
    with pytest.raises(ValueError):
        project(sched, np.zeros(10), basis("term_profit"), m.issue_age)


def test_maturity_benefit_paid_to_survivors():
    from pricing_engine.products.base import PolicySchedule
    sched = PolicySchedule(
        premium=np.array([100.0]), death_benefit=np.array([1000.0]),
        surrender_benefit=np.zeros(1), maturity_benefit=np.array([1000.0]))
    L = project(sched, np.zeros(2), basis("term_reserve"), 60)    # q60 = 0.011, no lapse
    assert L.eeb[1] == pytest.approx(0.989 * 1000)
