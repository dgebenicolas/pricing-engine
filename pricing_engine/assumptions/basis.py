"""Basis: one complete set of assumptions (mortality, interest, lapse, expenses).

A pricing run uses two: the reserve basis (reserves only) and the profit
basis (everything that projects policies). The risk discount rate is an
analysis input, not part of a basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from pricing_engine.assumptions.expenses import Expenses, expenses_from_config
from pricing_engine.assumptions.interest import Interest, interest_from_config
from pricing_engine.assumptions.lapse import Lapse, lapse_from_config
from pricing_engine.assumptions.mortality import Mortality, MortalityTable
from pricing_engine.config_loader import load_yaml

TIMINGS = ("year_end",)


def dependent_rates(
    q: ArrayLike, w: ArrayLike = 0.0, timing: str = "year_end"
) -> tuple[np.ndarray, np.ndarray]:
    """Turn independent q (death) and w (lapse) into dependent p0d, p0w.

    year_end (DHW p. 526): deaths happen through the year, lapses only at
    year end, so only survivors can lapse:
        p0d = q,   p0w = (1 - q) * w,   p00 = 1 - p0d - p0w (caller).
    w can be a scalar (same every year) or one value per policy year.
    """
    if timing not in TIMINGS:
        raise ValueError(f"unknown lapse timing {timing!r}; supported: {TIMINGS}")
    q = np.asarray(q, dtype=float)
    w = np.broadcast_to(np.asarray(w, dtype=float), q.shape)
    if np.any((w < 0) | (w > 1)):
        raise ValueError("every lapse rate w must be in [0, 1]")
    p0d = q.copy()
    p0w = (1 - q) * w
    return p0d, p0w


@dataclass(frozen=True)
class Basis:
    mortality: Mortality
    interest: Interest
    lapse: Lapse | None = None          # None = no lapses (w = 0)
    expenses: Expenses | None = None    # None = no expenses (net premium basis)
    lapse_timing: str = "year_end"

    def __post_init__(self) -> None:
        if self.lapse_timing not in TIMINGS:
            raise ValueError(f"unknown lapse timing {self.lapse_timing!r}")

    # --- per-year vectors, policy years 1..n ---

    def q(self, issue_age: int, n: int | None = None) -> np.ndarray:
        return self.mortality.q_vector(issue_age, n)

    def i(self, n: int) -> np.ndarray:
        return self.interest.rates(n)

    def w(self, n: int) -> np.ndarray:
        return np.zeros(n) if self.lapse is None else self.lapse.w_vector(n)

    def dependent_rates(self, issue_age: int, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = self.q(issue_age, n)
        return dependent_rates(q, self.w(len(q)), self.lapse_timing)

    def initial_expense(self, premiums: ArrayLike) -> float:
        return 0.0 if self.expenses is None else self.expenses.initial(premiums)

    def renewal_expenses(self, premiums: ArrayLike) -> np.ndarray:
        p = np.asarray(premiums, dtype=float)
        return np.zeros_like(p) if self.expenses is None else self.expenses.renewal(p)

    # --- config ---

    @classmethod
    def from_dict(cls, cfg: dict, mortality: Mortality) -> Basis:
        """Build a Basis from a dict (YAML contents or app edits).

        The mortality table is passed in ready-made (default CSV or an
        upload), so cfg's own 'mortality' entry, if any, is ignored.
        """
        return cls(
            mortality=mortality,
            interest=interest_from_config(cfg["interest"]),
            lapse=lapse_from_config(cfg.get("lapse")),
            expenses=expenses_from_config(cfg.get("expenses")),
            lapse_timing=cfg.get("lapse_timing", "year_end"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Basis:
        """Read a YAML file; its mortality path is relative to the YAML file."""
        path = Path(path)
        cfg = load_yaml(path)
        mortality = MortalityTable.from_csv(path.parent / cfg["mortality"])
        return cls.from_dict(cfg, mortality)
