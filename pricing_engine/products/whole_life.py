"""Whole life insurance, non-par (DHW Example 13.1, p. 524).

Sum insured S paid at the end of the year of death, level premium P
annually in advance for life, cash value CV_t = h_t * tV on surrender
at the end of year t. Runs to omega: n = omega - issue_age + 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pricing_engine.config_loader import load_yaml
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.base import PolicySchedule, Product


class WholeLife(Product):

    def __init__(self, cash_value_pct: list[float]) -> None:
        """cash_value_pct[k] = h for policy year k + 1; the last value repeats."""
        h = np.asarray(cash_value_pct, dtype=float)
        if h.ndim != 1 or len(h) == 0:
            raise ValueError("cash_value_pct must be a non-empty list")
        if np.any((h < 0) | (h > 1)):
            raise ValueError("every cash value percentage must be in [0, 1]")
        self.cash_value_pct = h

    @classmethod
    def from_yaml(cls, path: str | Path) -> WholeLife:
        return cls.from_dict(load_yaml(path))

    @classmethod
    def from_dict(cls, cfg: dict) -> WholeLife:
        return cls(cash_value_pct=cfg["cash_value_pct"])

    def h_vector(self, n: int) -> np.ndarray:
        """h_t for years 1..n, repeating the last given value."""
        h = self.cash_value_pct
        if n <= len(h):
            return h[:n].copy()
        return np.concatenate([h, np.full(n - len(h), h[-1])])

    def term(self, mp: ModelPoint, omega: int) -> int:
        if mp.term is not None:
            raise ValueError("whole life runs to omega; mp.term must be None")
        if mp.issue_age > omega:
            raise ValueError(f"issue age {mp.issue_age} is past omega = {omega}")
        return omega - mp.issue_age + 1

    def policy_schedule(self, mp: ModelPoint, V: np.ndarray | None = None) -> PolicySchedule:
        if mp.premium is None:
            raise ValueError("policy_schedule needs a premium; solve for it first")
        if V is None:
            raise ValueError("whole life needs reserves V: cash value = h_t * tV")
        V = np.asarray(V, dtype=float)
        n = len(V) - 1                                   # V runs t = 0..n
        if n < 1:
            raise ValueError("V must hold at least 0V and 1V")
        return PolicySchedule(
            premium=np.full(n, mp.premium),              # P every year, for life
            death_benefit=np.full(n, mp.sum_insured),    # S on death in any year
            surrender_benefit=self.h_vector(n) * V[1:],  # CV_t = h_t * tV (year-end)
            maturity_benefit=np.zeros(n),                # no maturity benefit
        )
