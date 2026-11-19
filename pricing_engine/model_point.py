"""ModelPoint: the data describing one representative policy. No maths here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pricing_engine.config_loader import load_yaml


@dataclass(frozen=True)
class ModelPoint:
    issue_age: int              # x, age at entry
    sum_insured: float          # S, paid at the end of the year of death
    premium: float | None = None  # P, annual premium in advance; None = to be solved
    term: int | None = None       # n in years; None = whole of life (to omega)

    def __post_init__(self) -> None:
        if self.issue_age < 0:
            raise ValueError("issue_age must be >= 0")
        if self.sum_insured <= 0:
            raise ValueError("sum_insured must be > 0")
        if self.premium is not None and self.premium <= 0:
            raise ValueError("premium must be > 0 (or None to solve for it)")
        if self.term is not None and self.term < 1:
            raise ValueError("term must be >= 1 (or None for whole life)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelPoint:
        return cls.from_dict(load_yaml(path))

    @classmethod
    def from_dict(cls, cfg: dict) -> ModelPoint:
        """issue_age, sum_insured, premium (optional), term (optional)."""
        return cls(
            issue_age=int(cfg["issue_age"]),
            sum_insured=float(cfg["sum_insured"]),
            premium=None if cfg.get("premium") is None else float(cfg["premium"]),
            term=None if cfg.get("term") is None else int(cfg["term"]),
        )
