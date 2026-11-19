"""Product interface: turn a model point into a year-by-year benefit/premium schedule.

The engine only reads PolicySchedule arrays, so it never needs to know
which product it is projecting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pricing_engine.model_point import ModelPoint


@dataclass(frozen=True)
class PolicySchedule:
    """Contract cash flows for policy years t = 1..n (index 0 = year 1)."""

    premium: np.ndarray            # P_t, paid at the start of year t if in force
    death_benefit: np.ndarray      # paid at the end of year t on death in year t
    surrender_benefit: np.ndarray  # CV_t, paid at the end of year t on surrender
    maturity_benefit: np.ndarray   # paid at the end of year t to survivors

    def __post_init__(self) -> None:
        arrays = (self.premium, self.death_benefit, self.surrender_benefit, self.maturity_benefit)
        n = len(self.premium)
        if n < 1:
            raise ValueError("schedule needs at least one policy year")
        for a in arrays:
            if a.ndim != 1 or len(a) != n:
                raise ValueError("all schedule arrays must be 1-D with the same length")
            if np.any(a < 0):
                raise ValueError("schedule amounts must be >= 0")

    @property
    def n(self) -> int:
        """Number of policy years."""
        return len(self.premium)


class Product(ABC):
    """Every product must say how long it runs and what it pays each year."""

    @abstractmethod
    def term(self, mp: ModelPoint, omega: int) -> int:
        """Number of policy years n. omega = last age of the mortality table."""

    @abstractmethod
    def policy_schedule(self, mp: ModelPoint, V: np.ndarray | None = None) -> PolicySchedule:
        """Schedule for years 1..n. V = reserves t = 0..n, for products whose
        benefits depend on them (e.g. cash value = h_t * tV); others ignore it."""
