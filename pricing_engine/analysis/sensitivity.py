"""Sensitivity: shock PROFIT-basis assumptions, rerun, compare (step 9).

The reserve basis stays fixed: a shock is a change in experience, not in
how the company sets reserves. Shocks wrap the existing assumption objects
(decorator pattern), so any Mortality / Interest / Lapse / Expenses
implementation can be shocked without changing it.

Size of a shock:
    mortality, lapse, expenses: scale (multiply), no shock = 1.0
    interest:                    shift (add),     no shock = 0.0
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pricing_engine.assumptions.basis import Basis
from pricing_engine.assumptions.expenses import Expenses
from pricing_engine.assumptions.interest import Interest
from pricing_engine.assumptions.lapse import Lapse
from pricing_engine.assumptions.mortality import Mortality


# --- wrappers: same interface, shocked values ---

@dataclass(frozen=True)
class ScaledMortality:
    base: Mortality
    factor: float

    @property
    def omega(self) -> int:
        return self.base.omega

    def q_vector(self, issue_age: int, n: int | None = None) -> np.ndarray:
        return np.minimum(self.factor * self.base.q_vector(issue_age, n), 1.0)


@dataclass(frozen=True)
class ShiftedInterest:
    """Every year's rate + shift (a parallel shift of the curve)."""
    base: Interest
    shift: float

    def rates(self, n: int) -> np.ndarray:
        return self.base.rates(n) + self.shift


@dataclass(frozen=True)
class ScaledLapse:
    base: Lapse
    factor: float

    def w_vector(self, n: int) -> np.ndarray:
        return np.minimum(self.factor * self.base.w_vector(n), 1.0)


@dataclass(frozen=True)
class ScaledExpenses:
    base: Expenses
    factor: float

    def initial(self, premiums: ArrayLike) -> float:
        return self.factor * self.base.initial(premiums)

    def renewal(self, premiums: ArrayLike) -> np.ndarray:
        return self.factor * self.base.renewal(premiums)


# field -> (wrapper, size meaning "no shock")
SHOCKS = {
    "mortality": (ScaledMortality, 1.0),
    "interest": (ShiftedInterest, 0.0),
    "lapse": (ScaledLapse, 1.0),
    "expenses": (ScaledExpenses, 1.0),
}


def shock(basis: Basis, field: str, size: float) -> Basis:
    """New basis with one assumption shocked (scaled, or shifted for interest).

    A missing assumption (lapse or expenses = None) is zero; scaling zero
    gives zero, so the basis comes back unchanged.
    """
    if field not in SHOCKS:
        raise ValueError(f"unknown field {field!r}; choose from {list(SHOCKS)}")
    wrapper, neutral = SHOCKS[field]
    if neutral == 1.0 and size < 0:
        raise ValueError(f"{field} is scaled: size must be >= 0")
    current = getattr(basis, field)
    if current is None:
        return basis
    return replace(basis, **{field: wrapper(current, size)})


def scenario(basis: Basis, moves: dict[str, float]) -> Basis:
    """Several shocks at once, e.g. {"mortality": 1.1, "interest": -0.01}."""
    for field, size in moves.items():
        basis = shock(basis, field, size)
    return basis


# --- tables for the app to plot ---

def tornado(
    evaluate: Callable[[Basis], float],
    basis: Basis,
    shocks: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """One row per field: metric at the low and high size, sorted by swing.

    evaluate maps a profit basis to a number (e.g. Run.evaluate).
    swing = |high - low|, biggest first (the top bar of the tornado).
    """
    base = evaluate(basis)
    rows = []
    for field, (lo, hi) in shocks.items():
        low = evaluate(shock(basis, field, lo))
        high = evaluate(shock(basis, field, hi))
        rows.append({"field": field, "low_size": lo, "high_size": hi,
                     "base": base, "low": low, "high": high, "swing": abs(high - low)})
    return (pd.DataFrame(rows)
            .sort_values("swing", ascending=False, ignore_index=True))


def spider(
    evaluate: Callable[[Basis], float],
    basis: Basis,
    sizes: dict[str, list[float]],
) -> pd.DataFrame:
    """One row per (field, size): the metric with only that field shocked."""
    rows = []
    for field, values in sizes.items():
        for s in values:
            rows.append({"field": field, "size": s, "value": evaluate(shock(basis, field, s))})
    return pd.DataFrame(rows)


def scenarios(
    evaluate: Callable[[Basis], float],
    basis: Basis,
    named: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """One row per named scenario (all its moves together), base first."""
    rows = [{"scenario": "base", "value": evaluate(basis)}]
    for name, moves in named.items():
        rows.append({"scenario": name, "value": evaluate(scenario(basis, moves))})
    return pd.DataFrame(rows)
