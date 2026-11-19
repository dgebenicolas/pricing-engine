"""Mortality: a table of one-year death probabilities q_x by integer age.

The engine only ever asks for q by policy year; where q came from is invisible.
Policy year t (t = 1..n) uses age issue_age + t - 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

import numpy as np
import pandas as pd


class Mortality(Protocol):
    """Anything that can return q for policy years 1..n and knows its last age."""

    @property
    def omega(self) -> int: ...

    def q_vector(self, issue_age: int, n: int | None = None) -> np.ndarray: ...


@dataclass(frozen=True)
class MortalityTable:
    ages: np.ndarray  # consecutive integer ages, youngest first
    q: np.ndarray     # q[k] = q at ages[k]

    def __post_init__(self) -> None:
        if len(self.ages) == 0 or len(self.ages) != len(self.q):
            raise ValueError("ages and q must be non-empty and the same length")
        if np.any(np.diff(self.ages) != 1):
            raise ValueError("ages must be consecutive integers")
        if np.any((self.q < 0) | (self.q > 1)):
            raise ValueError("every q must be in [0, 1]")

    @classmethod
    def from_csv(cls, source: str | Path | IO) -> MortalityTable:
        """Read a CSV with columns 'age' and 'q'.

        source: a file path, or an open file (e.g. a Streamlit upload).
        """
        df = pd.read_csv(source)
        missing = {"age", "q"} - set(df.columns)
        if missing:
            raise ValueError(f"mortality CSV is missing column(s): {sorted(missing)}")
        return cls(ages=df["age"].to_numpy(dtype=int), q=df["q"].to_numpy(dtype=float))

    @property
    def omega(self) -> int:
        """Last age in the table."""
        return int(self.ages[-1])

    def q_vector(self, issue_age: int, n: int | None = None) -> np.ndarray:
        """q for policy years 1..n, i.e. ages issue_age .. issue_age + n - 1.

        n=None runs to the end of the table: n = omega - issue_age + 1.
        """
        start = issue_age - int(self.ages[0])
        if start < 0 or issue_age > self.omega:
            raise ValueError(f"issue age {issue_age} outside table {self.ages[0]}-{self.omega}")
        if n is None:
            n = self.omega - issue_age + 1
        if n < 1 or start + n > len(self.q):
            raise ValueError(f"n={n} from age {issue_age} runs past omega={self.omega}")
        return self.q[start:start + n].copy()
