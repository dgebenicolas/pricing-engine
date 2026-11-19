"""Streamlit UI: a thin layer over app.loader and analysis. No maths here.

Run from the repo root:
    streamlit run pricing_engine/app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import streamlit as st
import yaml

from pricing_engine.analysis.profit_test import solve_premium
from pricing_engine.analysis.sensitivity import scenarios, spider, tornado
from pricing_engine.app.loader import METRICS, PRODUCTS, RESERVE_METHODS, build_run, load_run
from pricing_engine.assumptions.mortality import MortalityTable
from pricing_engine.config_loader import CONFIG_DIR

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1, 2
FORMATS = {"npv": "{:,.2f}", "profit_margin": "{:.2%}", "irr": "{:.2%}"}
UPLOAD = "Upload your own…"


# --- small helpers (widgets only) ---

def presets() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "runs").glob("*.yaml"))


def mortality_files() -> dict[str, Path]:
    return {p.name: p for p in sorted((CONFIG_DIR / "mortality").glob("*.csv"))}


def yaml_box(label: str, value, key: str, height: int = 160):
    """Show a value as editable YAML text; return what the user left there."""
    text = st.text_area(label, yaml.safe_dump(value, sort_keys=False), key=key, height=height)
    return yaml.safe_load(text)


def basis_editor(title: str, cfg: dict, key: str) -> dict:
    """Mortality table picker + upload, and the rest of the basis as YAML."""
    st.subheader(title)
    files = mortality_files()
    names = list(files) + [UPLOAD]
    current = Path(cfg["mortality"]).name
    choice = st.selectbox("Mortality table", names,
                          index=names.index(current) if current in names else 0, key=f"{key}_m")
    if choice == UPLOAD:
        f = st.file_uploader("CSV with columns age, q (last row = ω)", type="csv", key=f"{key}_up")
        mortality = MortalityTable.from_csv(f) if f else cfg["mortality"]
    else:
        mortality = files[choice]
    rest = {k: v for k, v in cfg.items() if k != "mortality"}
    edited = yaml_box("Interest, lapse, expenses", rest, f"{key}_y", height=260) or {}
    return {**edited, "mortality": mortality}


# --- page ---

st.set_page_config(page_title="Pricing engine", layout="wide")
st.title("Life insurance pricing & profit test")
st.caption("Defaults reproduce DHW Ch. 13. Edit anything; results update.")

preset = st.sidebar.selectbox("Preset", presets())
cfg = load_run(preset)
k = preset                                   # widget keys per preset, so switching resets them

with st.sidebar:
    st.header("Run")
    cfg["product"] = st.selectbox("Product", list(PRODUCTS),
                                  index=list(PRODUCTS).index(cfg["product"]), key=f"{k}_prod")
    cfg["reserve_method"] = st.selectbox(
        "Reserve method", list(RESERVE_METHODS),
        index=list(RESERVE_METHODS).index(cfg["reserve_method"]), key=f"{k}_rm")
    same_basis = st.checkbox("Reserve basis = profit basis (use for zeroized)", key=f"{k}_same")
    cfg["metric"] = st.selectbox("Metric (solver & sensitivity)", list(METRICS),
                                 index=list(METRICS).index(cfg["metric"]), key=f"{k}_met")
    cfg["risk_discount"] = st.number_input("Risk discount rate r", value=cfg["risk_discount"],
                                           step=0.005, format="%.4f", key=f"{k}_r")

    st.header("Model point")
    mp = cfg["model_point"]
    mp["issue_age"] = st.number_input("Issue age x", value=int(mp["issue_age"]), step=1, key=f"{k}_x")
    mp["sum_insured"] = st.number_input("Sum insured S", value=float(mp["sum_insured"]),
                                        step=1000.0, key=f"{k}_S")
    mp["premium"] = st.number_input("Annual premium P", value=float(mp["premium"] or 0.0),
                                    step=10.0, format="%.2f", key=f"{k}_P")
    whole = st.checkbox("Whole of life (term to ω)", value=mp.get("term") is None, key=f"{k}_wl")
    mp["term"] = None if whole else st.number_input(
        "Term n (years)", value=int(mp.get("term") or 10), min_value=1, step=1, key=f"{k}_n")

    st.header("Product settings")
    cfg["product_config"] = yaml_box("YAML (empty for term)", cfg["product_config"] or {},
                                     f"{k}_pc", height=100) or {}

tab_bases, tab_results, tab_solver, tab_sens = st.tabs(
    ["Bases", "Results", "Premium solver", "Sensitivity"])

with tab_bases:
    left, right = st.columns(2)
    with left:
        cfg["profit_basis"] = basis_editor("Profit basis", cfg["profit_basis"], f"{k}_pb")
    with right:
        if same_basis:
            st.subheader("Reserve basis")
            st.info("Using the profit basis.")
        else:
            cfg["reserve_basis"] = basis_editor("Reserve basis", cfg["reserve_basis"], f"{k}_rb")
    if same_basis:
        cfg["reserve_basis"] = cfg["profit_basis"]

with tab_sens:
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg["shocks"] = yaml_box("Tornado: field: [low, high]", cfg["shocks"], f"{k}_sh") or {}
    with c2:
        cfg["spider"] = yaml_box("Spider: field: [sizes]", cfg["spider"], f"{k}_sp") or {}
    with c3:
        cfg["scenarios"] = yaml_box("Scenarios: name: {field: size}", cfg["scenarios"],
                                    f"{k}_sc") or {}
    st.caption("mortality, lapse, expenses multiply (1 = no shock); "
               "interest adds to the rate (0 = no shock, -0.01 = one point lower).")

try:
    run = build_run(cfg)
    ledger = run.ledger()
except (ValueError, KeyError, TypeError, ArithmeticError) as e:
    st.error(f"Can't run this configuration: {e}")
    st.stop()

with tab_results:
    cols = st.columns(len(METRICS))
    for col, (name, metric) in zip(cols, METRICS.items()):
        try:
            col.metric(name, FORMATS[name].format(metric(ledger, run.risk_discount)))
        except ValueError:
            col.metric(name, "n/a")
    st.subheader("Profit signature Πₜ")
    st.altair_chart(
        alt.Chart(ledger).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("t:O", title="t"),
            y=alt.Y("signature:Q", title="Πₜ"),
            tooltip=["t", alt.Tooltip("signature:Q", format=",.2f")]),
        width="stretch")
    st.subheader("Ledger (Eq. 13.1)")
    st.dataframe(ledger.round(2), hide_index=True, width="stretch")

with tab_solver:
    target = st.number_input(f"Target {cfg['metric']}", value=0.05, step=0.01,
                             format="%.4f", key=f"{k}_tgt")
    if st.button("Solve premium", key=f"{k}_solve"):
        try:
            P = solve_premium(run.model_point, lambda m: run.ledger(model_point=m),
                              run.measure, target)
            st.success(f"Premium P = {P:,.2f} gives {cfg['metric']} = {target}")
        except ValueError as e:
            st.error(str(e))

with tab_sens:
    try:
        T = tornado(run.evaluate, run.profit_basis, cfg["shocks"])
        S = spider(run.evaluate, run.profit_basis, cfg["spider"])
        C = scenarios(run.evaluate, run.profit_basis, cfg["scenarios"])
    except (ValueError, TypeError, ArithmeticError) as e:
        st.error(f"Can't run sensitivity: {e}")
        st.stop()

    st.subheader(f"Tornado: {cfg['metric']}")
    bars = T.melt(id_vars=["field", "base", "swing"], value_vars=["low", "high"],
                  var_name="shock", value_name="value")
    order = list(T.field)                                   # biggest swing on top
    tornado_chart = alt.Chart(bars).mark_bar(cornerRadius=4).encode(
        y=alt.Y("field:N", sort=order, title=None),
        x=alt.X("value:Q", title=cfg["metric"]),
        x2="base:Q",
        color=alt.Color("shock:N", scale=alt.Scale(domain=["low", "high"], range=[BLUE, ORANGE]),
                        title="size"),
        tooltip=["field", "shock", alt.Tooltip("value:Q", format=",.2f")])
    base_rule = alt.Chart(T.head(1)).mark_rule(color="gray").encode(x="base:Q")
    st.altair_chart(tornado_chart + base_rule, width="stretch")
    st.dataframe(T.round(4), hide_index=True, width="stretch")

    st.subheader("Spider")
    st.caption("One panel per assumption, since scale and shift sizes are on different axes.")
    spider_chart = alt.Chart(S).mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
        x=alt.X("size:Q", title="shock size"),
        y=alt.Y("value:Q", title=cfg["metric"]),
        tooltip=["field", "size", alt.Tooltip("value:Q", format=",.2f")],
    ).properties(width=260, height=200).facet(facet="field:N", columns=4).resolve_scale(x="independent")
    st.altair_chart(spider_chart)

    st.subheader("Scenarios")
    st.altair_chart(
        alt.Chart(C).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            y=alt.Y("scenario:N", sort=list(C.scenario), title=None),
            x=alt.X("value:Q", title=cfg["metric"]),
            tooltip=["scenario", alt.Tooltip("value:Q", format=",.2f")]),
        width="stretch")
    st.dataframe(C.round(4), hide_index=True, width="stretch")
