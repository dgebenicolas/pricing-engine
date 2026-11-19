"""Profit measures (DHW Sec. 13.5) and the premium solver (Sec. 13.6).

Every measure reads only the ledger built by engine.projection; the risk
discount rate r is an input here, never part of a basis.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pricing_engine.model_point import ModelPoint


def _discount(r: float, t: np.ndarray) -> np.ndarray:
    """v_r^t = (1 + r)^-t."""
    return (1 + r) ** -np.asarray(t, dtype=float)


def npv(ledger: pd.DataFrame, r: float) -> float:
    """NPV = sum_t Pi_t v_r^t."""
    return float(np.sum(ledger.signature * _discount(r, ledger.t)))


def epv_premiums(ledger: pd.DataFrame, r: float) -> float:
    """sum_t P_t * (t-1)p00 * v_r^(t-1): premium for year t is paid at t - 1."""
    t = ledger.t.to_numpy()
    return float(np.sum(ledger.premium * ledger.inforce * _discount(r, t - 1)))


def profit_margin(ledger: pd.DataFrame, r: float) -> float:
    """NPV / EPV of premiums, both at the risk discount rate (Eq. 13.3)."""
    return npv(ledger, r) / epv_premiums(ledger, r)


def irr(ledger: pd.DataFrame, lo: float = -0.99, hi: float = 10.0) -> float:
    """j with NPV at rate j = 0 (Eq. 13.2). Assumes one root in [lo, hi]."""
    f_lo, f_hi = npv(ledger, lo), npv(ledger, hi)
    if np.sign(f_lo) == np.sign(f_hi):
        raise ValueError(f"no sign change in NPV between {lo} and {hi}; IRR not found")
    return float(brentq(lambda j: npv(ledger, j), lo, hi))


def solve_premium(
    mp: ModelPoint,
    ledger_for: Callable[[ModelPoint], pd.DataFrame],
    measure: Callable[[pd.DataFrame], float],
    target: float,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Premium P with measure(ledger_for(mp with premium P)) = target (Sec. 13.6).

    ledger_for reruns the whole pipeline, e.g. lambda m: run(m, product, ...).
    Default search range: 0.0001% to 100% of the sum insured.
    """
    lo = 1e-6 * mp.sum_insured if lo is None else lo
    hi = mp.sum_insured if hi is None else hi

    def gap(P: float) -> float:
        return measure(ledger_for(replace(mp, premium=P))) - target

    g_lo, g_hi = gap(lo), gap(hi)
    if np.sign(g_lo) == np.sign(g_hi):
        raise ValueError(f"target not reached for premiums in [{lo}, {hi}]")
    return float(brentq(gap, lo, hi, xtol=1e-8))
