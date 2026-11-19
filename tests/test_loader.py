"""Loader: default run YAMLs reproduce the DHW gates; edits flow through."""

import io

import pytest

from pricing_engine.analysis.profit_test import solve_premium
from pricing_engine.analysis.sensitivity import scenarios, spider, tornado
from pricing_engine.app.loader import build_run, load_run
from pricing_engine.assumptions.mortality import MortalityTable
from pricing_engine.config_loader import config_path


# --- defaults ---

def test_term_default():
    r = build_run(load_run("term_dhw"))
    assert r.evaluate(r.profit_basis) == pytest.approx(74.13, abs=0.01)    # p. 520


def test_wl_default():
    r = build_run(load_run("wl_dhw"))
    assert r.evaluate(r.profit_basis) == pytest.approx(2938.26, abs=0.01)  # regression value


def test_load_run_is_plain_data():
    cfg = load_run("term_dhw")
    assert cfg["model_point"] == {"issue_age": 60, "sum_insured": 100000,
                                  "premium": 1500, "term": 10}
    assert cfg["profit_basis"]["interest"] == 0.055
    assert cfg["profit_basis"]["mortality"].is_file()                      # absolute CSV path
    assert cfg["shocks"]["mortality"] == (0.9, 1.1)


# --- edits, as the app will make them ---

def test_edit_metric():
    cfg = load_run("term_dhw")
    cfg["metric"] = "profit_margin"
    r = build_run(cfg)
    assert r.evaluate(r.profit_basis) == pytest.approx(0.0077, abs=0.00005)  # p. 521
    cfg["metric"] = "irr"
    r = build_run(cfg)
    assert r.evaluate(r.profit_basis) == pytest.approx(0.124, abs=0.0005)    # p. 520


def test_edit_reserve_method_to_zeroized():
    cfg = load_run("term_dhw")
    cfg["reserve_method"] = "zeroized"
    cfg["reserve_basis"] = cfg["profit_basis"]                             # Sec. 13.7
    r = build_run(cfg)
    assert r.evaluate(r.profit_basis) == pytest.approx(189.31, abs=0.01)   # pp. 523-524


def test_edit_premium():
    cfg = load_run("term_dhw")
    cfg["model_point"]["premium"] = 1575.21
    cfg["metric"] = "profit_margin"
    r = build_run(cfg)
    assert r.evaluate(r.profit_basis) == pytest.approx(0.05, abs=0.00005)  # p. 521


def test_uploaded_mortality_table():
    cfg = load_run("term_dhw")
    upload = io.BytesIO(config_path("mortality/term_profit.csv").read_bytes())
    cfg["profit_basis"]["mortality"] = MortalityTable.from_csv(upload)     # object, not path
    r = build_run(cfg)
    assert r.evaluate(r.profit_basis) == pytest.approx(74.13, abs=0.01)


def test_unknown_names_raise():
    for key in ("product", "reserve_method", "metric"):
        cfg = load_run("term_dhw")
        cfg[key] = "nonsense"
        with pytest.raises(ValueError, match="choose from"):
            build_run(cfg)


# --- analysis plugs straight in ---

def test_tornado_from_config():
    r = build_run(load_run("wl_dhw"))
    T = tornado(r.evaluate, r.profit_basis, r.shocks)
    assert set(T.field) == {"mortality", "interest", "lapse", "expenses"}
    assert T.base[0] == pytest.approx(2938.26, abs=0.01)


def test_spider_and_scenarios_from_config():
    r = build_run(load_run("term_dhw"))
    S = spider(r.evaluate, r.profit_basis, r.spider)
    assert len(S) == 15                                                    # 3 fields x 5 sizes
    T = scenarios(r.evaluate, r.profit_basis, r.scenarios)
    assert list(T.scenario) == ["base", "adverse", "favorable"]
    assert T.value[0] == pytest.approx(74.13, abs=0.01)


def test_solve_premium_from_config():
    cfg = load_run("term_dhw")
    cfg["metric"] = "profit_margin"
    r = build_run(cfg)
    P = solve_premium(r.model_point, lambda m: r.ledger(model_point=m), r.measure, 0.05)
    assert P == pytest.approx(1575.21, abs=0.005)                          # p. 521
