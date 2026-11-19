"""Lapse: the independent surrender rate w for each policy year.

w_t is the probability that a policy still in force at the end of year t
(after deaths) surrenders then. dependent_rates() turns it into p0w.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Lapse(Protocol):
    """Anything that can return w for policy years 1..n."""

    def w_vector(self, n: int) -> np.ndarray: ...


@dataclass(frozen=True)
class LapseBand:
    """Rate for policy years from_year..to_year inclusive; to_year=None means 'and after'."""

    from_year: int
    to_year: int | None
    rate: float


@dataclass(frozen=True)
class BandedLapse:
    """Piecewise-constant lapse rates, e.g. DHW Example 13.1 (p. 525)."""

    bands: tuple[LapseBand, ...]

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("need at least one lapse band")
        expected_start = 1
        for k, b in enumerate(self.bands):
            if b.from_year != expected_start:
                raise ValueError(f"band {k} starts at year {b.from_year}, expected {expected_start}")
            if not 0 <= b.rate <= 1:
                raise ValueError("every lapse rate must be in [0, 1]")
            if b.to_year is None:
                if k != len(self.bands) - 1:
                    raise ValueError("only the last band can be open-ended")
            else:
                if b.to_year < b.from_year:
                    raise ValueError(f"band {k} ends before it starts")
                expected_start = b.to_year + 1

    @property
    def last_year(self) -> int | None:
        """Last year covered, or None if the final band is open-ended."""
        return self.bands[-1].to_year

    def w_vector(self, n: int) -> np.ndarray:
        if self.last_year is not None and n > self.last_year:
            raise ValueError(f"lapse rates only cover years 1-{self.last_year}, asked for {n}")
        w = np.empty(n, dtype=float)
        for b in self.bands:
            start = b.from_year - 1                       # year 1 -> index 0
            stop = n if b.to_year is None else min(b.to_year, n)
            w[start:stop] = b.rate
        return w


def lapse_from_config(cfg: list[dict] | None) -> Lapse | None:
    """YAML list of {from_year, to_year (optional), rate}; null means no lapses."""
    if cfg is None:
        return None
    return BandedLapse(tuple(
        LapseBand(int(b["from_year"]), b.get("to_year"), float(b["rate"])) for b in cfg
    ))
