"""Wiring: run config -> ready objects -> evaluate(profit_basis).

Top layer: the only module that imports products, engine and analysis
together. No actuarial maths here, only lookups and plumbing.

Two stages, so the app can edit in between:
    load_run("term_dhw")  -> plain dict (numbers, lists, paths)
    build_run(cfg)        -> Run (real objects, ready to price)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from pricing_engine.analysis.profit_test import irr, npv, profit_margin
from pricing_engine.assumptions.basis import Basis
from pricing_engine.assumptions.mortality import MortalityTable
from pricing_engine.config_loader import config_path, load_yaml
from pricing_engine.engine.projection import run
from pricing_engine.engine.reserves import (
    FPTReserve, NetPremiumReserve, NoReserve, ReserveMethod, ZeroizedReserve,
)
from pricing_engine.model_point import ModelPoint
from pricing_engine.products.base import Product
from pricing_engine.products.term_life import TermLife
from pricing_engine.products.whole_life import WholeLife

# --- lookup tables: config word -> code ---

PRODUCTS = {"term_life": TermLife, "whole_life": WholeLife}

RESERVE_METHODS = {
    "none": NoReserve,
    "net_premium": NetPremiumReserve,
    "fpt": FPTReserve,
    "zeroized": ZeroizedReserve,       # set reserve_basis = profit basis (DHW Sec. 13.7)
}

METRICS = {                            # every metric called as metric(ledger, r)
    "npv": npv,
    "profit_margin": profit_margin,
    "irr": lambda ledger, r: irr(ledger),   # IRR needs no discount rate
}


def _pick(table: dict, name: str, what: str):
    if name not in table:
        raise ValueError(f"unknown {what} {name!r}; choose from {list(table)}")
    return table[name]


# --- stage 1: files -> plain dict ---

def _basis_dict(name: str) -> dict:
    """A basis YAML as a dict, with its mortality CSV path made absolute."""
    path = config_path(name)
    cfg = load_yaml(path)
    cfg["mortality"] = (path.parent / cfg["mortality"]).resolve()
    return cfg


def load_run(name: str) -> dict:
    """Read config/runs/<name>.yaml and pull in every file it points to."""
    cfg = load_yaml(config_path(f"runs/{name}.yaml"))
    product_file = cfg.get("product_config")
    return {
        "product": cfg["product"],
        "product_config": load_yaml(config_path(product_file)) if product_file else {},
        "model_point": load_yaml(config_path(cfg["model_point"])),
        "reserve_method": cfg["reserve_method"],
        "reserve_basis": _basis_dict(cfg["reserve_basis"]),
        "profit_basis": _basis_dict(cfg["profit_basis"]),
        "risk_discount": float(cfg["risk_discount"]),
        "metric": cfg["metric"],
        "shocks": {field: tuple(s) for field, s in (cfg.get("shocks") or {}).items()},
        "spider": {field: list(s) for field, s in (cfg.get("spider") or {}).items()},
        "scenarios": {name: dict(m) for name, m in (cfg.get("scenarios") or {}).items()},
    }


# --- stage 2: plain dict -> objects ---

def _build_basis(cfg: dict) -> Basis:
    """cfg['mortality'] is a CSV path, or a table already built (e.g. an upload)."""
    m = cfg["mortality"]
    table = MortalityTable.from_csv(m) if isinstance(m, (str, Path)) else m
    return Basis.from_dict(cfg, table)


@dataclass(frozen=True)
class Run:
    model_point: ModelPoint
    product: Product
    reserve_method: ReserveMethod
    reserve_basis: Basis
    profit_basis: Basis
    risk_discount: float
    metric: Callable[[pd.DataFrame, float], float]
    shocks: dict[str, tuple[float, float]]          # tornado
    spider: dict[str, list[float]]
    scenarios: dict[str, dict[str, float]]

    def ledger(self, profit_basis: Basis | None = None,
               model_point: ModelPoint | None = None) -> pd.DataFrame:
        """The ledger; either input can be swapped (shocked basis, trial premium)."""
        return run(
            self.model_point if model_point is None else model_point,
            self.product,
            self.reserve_method,
            self.reserve_basis,
            self.profit_basis if profit_basis is None else profit_basis,
        )

    def measure(self, ledger: pd.DataFrame) -> float:
        return self.metric(ledger, self.risk_discount)

    def evaluate(self, profit_basis: Basis) -> float:
        """profit basis -> metric. This is what tornado() calls."""
        return self.measure(self.ledger(profit_basis=profit_basis))


def build_run(cfg: dict) -> Run:
    return Run(
        model_point=ModelPoint.from_dict(cfg["model_point"]),
        product=_pick(PRODUCTS, cfg["product"], "product")(**cfg["product_config"]),
        reserve_method=_pick(RESERVE_METHODS, cfg["reserve_method"], "reserve method")(),
        reserve_basis=_build_basis(cfg["reserve_basis"]),
        profit_basis=_build_basis(cfg["profit_basis"]),
        risk_discount=float(cfg["risk_discount"]),
        metric=_pick(METRICS, cfg["metric"], "metric"),
        shocks=dict(cfg["shocks"]),
        spider=dict(cfg["spider"]),
        scenarios=dict(cfg["scenarios"]),
    )
