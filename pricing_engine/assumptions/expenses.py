"""Expenses: initial (time 0, E_0) and renewal (start of each year, E_t).

E_0 is paid at t = 0 and earns no interest; E_t is paid with the premium
at the start of year t (DHW p. 510).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class Expenses(Protocol):
    """Anything that can price E_0 and E_1..E_n from the premium vector."""

    def initial(self, premiums: ArrayLike) -> float: ...

    def renewal(self, premiums: ArrayLike) -> np.ndarray: ...


@dataclass(frozen=True)
class PremiumExpenses:
    """Fixed amounts plus percentages of premium.

    E_0 = initial_fixed + initial_pct * P_1            (P_1 = first premium)
    E_t = renewal_fixed + renewal_pct * P_t  for each year a premium is paid
    """

    initial_fixed: float = 0.0
    initial_pct: float = 0.0
    renewal_fixed: float = 0.0
    renewal_pct: float = 0.0

    def __post_init__(self) -> None:
        if min(self.initial_fixed, self.initial_pct, self.renewal_fixed, self.renewal_pct) < 0:
            raise ValueError("expense parameters must be >= 0")

    def initial(self, premiums: ArrayLike) -> float:
        first = float(np.asarray(premiums, dtype=float)[0])
        return self.initial_fixed + self.initial_pct * first

    def renewal(self, premiums: ArrayLike) -> np.ndarray:
        p = np.asarray(premiums, dtype=float)
        return np.where(p > 0, self.renewal_fixed, 0.0) + self.renewal_pct * p


def expenses_from_config(cfg: dict | None) -> Expenses | None:
    """YAML mapping of the four PremiumExpenses fields; null means no expenses."""
    if cfg is None:
        return None
    return PremiumExpenses(**{k: float(v) for k, v in cfg.items()})
