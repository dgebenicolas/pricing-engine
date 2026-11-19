"""Reserves: EPV factors, net premiums and the policy value recursion.

Only the RESERVE basis is ever passed in here. Arrays are by policy year:
q[k], i[k] belong to year k + 1 (age x + k). V runs t = 0..n.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from pricing_engine.assumptions.basis import Basis
from pricing_engine.model_point import ModelPoint


# --- building blocks ---

def _per_year(x: ArrayLike, n: int) -> np.ndarray:
    """Scalar -> n copies; array -> checked to have length n."""
    a = np.broadcast_to(np.asarray(x, dtype=float), (n,))
    return a.astype(float)


def epv_factors(q: ArrayLike, i: ArrayLike) -> tuple[float, float]:
    """A and a-due over the years covered by q (n = len(q)).

        A = sum_k v^(k+1) * kp * q_k       (1 paid at end of year of death)
        a = sum_k v^k     * kp             (1 paid at start of each year alive)

    q to omega (last q = 1) gives whole life A_x, a_x; q for n years gives
    the n-year term insurance A^1_{x:n} and annuity-due a_{x:n}.
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    i = _per_year(i, n)
    kp = np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])   # kp[k] = prob. alive at start of year k+1
    v_end = np.cumprod(1 / (1 + i))                         # v^(k+1)
    v_start = np.concatenate([[1.0], v_end[:-1]])           # v^k
    A = float(np.sum(v_end * kp * q))
    a = float(np.sum(v_start * kp))
    return A, a


def net_premium(S: float, q: ArrayLike, i: ArrayLike) -> float:
    """Level net premium P = S * A / a-due over the years covered by q."""
    A, a = epv_factors(q, i)
    return S * A / a


def policy_values(
    S: ArrayLike, P: ArrayLike, E: ArrayLike, q: ArrayLike, i: ArrayLike,
    floor: float | None = None,
) -> np.ndarray:
    """Backward recursion from nV = 0 (DHW Ch. 7):

        (t-1)V = (q_t * S_t + p_t * tV) / (1 + i_t) - (P_t - E_t)

    S, P, E, i: scalar or one value per year. floor=0 gives zeroized reserves.
    Returns V[0..n].
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    S, P, E, i = (_per_year(x, n) for x in (S, P, E, i))
    V = np.zeros(n + 1)
    for t in range(n, 0, -1):                 # t = n, n-1, ..., 1
        k = t - 1                             # array index of policy year t
        V[t - 1] = (q[k] * S[k] + (1 - q[k]) * V[t]) / (1 + i[k]) - (P[k] - E[k])
        if floor is not None:
            V[t - 1] = max(V[t - 1], floor)
    return V


# --- reserve methods (plug-ins) ---

class ReserveMethod(Protocol):
    """Anything that returns reserves V[0..n] for a model point."""

    def reserves(self, mp: ModelPoint, n: int, basis: Basis) -> np.ndarray: ...


class NoReserve:
    """No reserves held: V = 0 throughout (DHW Table 13.1)."""

    def reserves(self, mp: ModelPoint, n: int, basis: Basis) -> np.ndarray:
        return np.zeros(n + 1)


class NetPremiumReserve:
    """Level net premium policy values (DHW p. 512)."""

    def net_premiums(self, mp: ModelPoint, n: int, basis: Basis) -> np.ndarray:
        P = net_premium(mp.sum_insured, basis.q(mp.issue_age, n), basis.i(n))
        return np.full(n, P)

    def reserves(self, mp: ModelPoint, n: int, basis: Basis) -> np.ndarray:
        q, i = basis.q(mp.issue_age, n), basis.i(n)
        return policy_values(mp.sum_insured, self.net_premiums(mp, n, basis), 0.0, q, i)


class FPTReserve:
    """Full preliminary term (DHW p. 526): year 1 is one-year term, then a
    net premium reserve for a life one year older: P* = S A_{x+1} / a_{x+1}."""

    def net_premiums(self, mp: ModelPoint, n: int, basis: Basis) -> np.ndarray:
        S = mp.sum_insured
        q, i = basis.q(mp.issue_age, n), basis.i(n)
        year1 = S * q[0] / (1 + i[0])                 # S v q_x: cost of 1 year of cover
        P_star = net_premium(S, q[1:], i[1:])         # S A_{x+1} / a_{x+1}
        return np.concatenate([[year1], np.full(n - 1, P_star)])

    def reserves(self, mp: ModelPoint, n: int, basis: Basis) -> np.ndarray:
        q, i = basis.q(mp.issue_age, n), basis.i(n)
        V = policy_values(mp.sum_insured, self.net_premiums(mp, n, basis), 0.0, q, i)
        # 0V and 1V are zero by construction; the recursion leaves rounding noise.
        tol = 1e-8 * mp.sum_insured
        if abs(V[0]) > tol or abs(V[1]) > tol:
            raise ArithmeticError(f"FPT gave 0V={V[0]}, 1V={V[1]}; expected ~0")
        V[0] = V[1] = 0.0
        return V


class ZeroizedReserve:
    """Smallest reserves that stop any negative profit after t = 0 (DHW Sec. 13.7).

    Same recursion with the GROSS premium, renewal expenses and floor = 0.
    Pass the PROFIT basis as this method's basis. Uses p = 1 - q, so it
    assumes no lapses or surrender benefits (true for the DHW term example).
    """

    def reserves(self, mp: ModelPoint, n: int, basis: Basis) -> np.ndarray:
        if mp.premium is None:
            raise ValueError("zeroized reserves need the gross premium")
        P = np.full(n, mp.premium)
        E = basis.renewal_expenses(P)
        q, i = basis.q(mp.issue_age, n), basis.i(n)
        return policy_values(mp.sum_insured, P, E, q, i, floor=0.0)
