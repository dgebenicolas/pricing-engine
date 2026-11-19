"""Projection: the profit test ledger (DHW Eq. 13.1, p. 516) and run().

Every rate in project() comes from the PROFIT basis. The reserve basis is
used only inside the reserve method, called from run().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing_engine.assumptions.basis import Basis
from pricing_engine.engine.reserves import ReserveMethod
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.base import PolicySchedule, Product


def project(sched: PolicySchedule, V: np.ndarray, basis: Basis, issue_age: int) -> pd.DataFrame:
    """One row per time t = 0..n. Row 0 holds pre-contract costs only.

    Pr_0 = -E_0 - 0V
    Pr_t = (t-1)V + P_t - E_t + I_t - EDB_t - ESB_t - EEB_t - E_tV
    Pi_t = (t-1)p00 * Pr_t,  Pi_0 = Pr_0
    """
    n = sched.n
    V = np.asarray(V, dtype=float)
    if len(V) != n + 1:
        raise ValueError(f"V needs n + 1 = {n + 1} values (t = 0..n), got {len(V)}")

    P = sched.premium
    p0d, p0w = basis.dependent_rates(issue_age, n)
    p00 = 1 - p0d - p0w

    E0 = basis.initial_expense(P)                 # paid at t = 0
    E = basis.renewal_expenses(P)                 # paid at start of each year
    V_bf, V_cf = V[:-1], V[1:]                    # (t-1)V and tV for t = 1..n
    interest = basis.i(n) * (V_bf + P - E)
    edb = p0d * sched.death_benefit
    esb = p0w * sched.surrender_benefit
    eeb = p00 * sched.maturity_benefit
    reserve_cf = p00 * V_cf
    profit = V_bf + P - E + interest - edb - esb - eeb - reserve_cf
    inforce = np.concatenate([[1.0], np.cumprod(p00)[:-1]])   # (t-1)p00 for t = 1..n

    def with_t0(x0: float, x: np.ndarray) -> np.ndarray:
        return np.concatenate([[x0], x])

    pr0 = -E0 - V[0]
    return pd.DataFrame({
        "t": np.arange(n + 1),
        "reserve_bf": with_t0(0.0, V_bf),
        "premium": with_t0(0.0, P),
        "expenses": with_t0(E0, E),
        "interest": with_t0(0.0, interest),
        "edb": with_t0(0.0, edb),
        "esb": with_t0(0.0, esb),
        "eeb": with_t0(0.0, eeb),
        "reserve_cf": with_t0(V[0], reserve_cf),  # row 0: cost of setting up 0V
        "profit": with_t0(pr0, profit),           # Pr_t, profit vector
        "inforce": with_t0(1.0, inforce),         # (t-1)p00; 1 for row 0
        "signature": with_t0(pr0, inforce * profit),  # Pi_t, profit signature
    })


def run(
    mp: ModelPoint,
    product: Product,
    reserve_method: ReserveMethod,
    reserve_basis: Basis,
    profit_basis: Basis,
) -> pd.DataFrame:
    """V (reserve basis) -> schedule (product) -> ledger (profit basis)."""
    omega = min(reserve_basis.mortality.omega, profit_basis.mortality.omega)
    n = product.term(mp, omega)
    V = reserve_method.reserves(mp, n, reserve_basis)
    sched = product.policy_schedule(mp, V)
    return project(sched, V, profit_basis, mp.issue_age)
