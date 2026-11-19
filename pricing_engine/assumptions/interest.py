"""Interest: the rate earned (or used for reserves) in each policy year."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Interest(Protocol):
    """Anything that can return i for policy years 1..n."""

    def rates(self, n: int) -> np.ndarray: ...


@dataclass(frozen=True)
class FlatRate:
    """Same rate every year (DHW Ch. 13)."""

    rate: float

    def __post_init__(self) -> None:
        if self.rate <= -1:
            raise ValueError("interest rate must be > -1")

    def rates(self, n: int) -> np.ndarray:
        return np.full(n, self.rate, dtype=float)


@dataclass(frozen=True)
class RatesByYear:
    """One rate per policy year, e.g. forward rates from a yield curve (DHW p. 510)."""

    by_year: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.by_year) == 0 or any(r <= -1 for r in self.by_year):
            raise ValueError("need at least one rate, each > -1")

    def rates(self, n: int) -> np.ndarray:
        if n > len(self.by_year):
            raise ValueError(f"asked for {n} years, only {len(self.by_year)} rates given")
        return np.array(self.by_year[:n], dtype=float)


def interest_from_config(cfg: float | list[float]) -> Interest:
    """A number gives FlatRate; a list gives RatesByYear."""
    if isinstance(cfg, (int, float)):
        return FlatRate(float(cfg))
    return RatesByYear(tuple(float(r) for r in cfg))
