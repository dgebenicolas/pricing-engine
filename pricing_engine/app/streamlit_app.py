"""Streamlit UI: a thin layer over app.loader and analysis. No maths here.

Every widget starts from the preset's config dict and writes the user's
edit back into it; then build_run(cfg) does all the work. Rates are shown
as percentages for readability and converted back (display only).

Run from the repo root:
    streamlit run pricing_engine/app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from pricing_engine.analysis.profit_test import solve_premium
from pricing_engine.analysis.sensitivity import SHOCKS, scenarios, spider, tornado
from pricing_engine.app.loader import METRICS, PRODUCTS, RESERVE_METHODS, build_run, load_run
from pricing_engine.assumptions.mortality import MortalityTable
from pricing_engine.config_loader import CONFIG_DIR

# Categorical palette, fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE = PALETTE[0], PALETTE[1]
GRAY = "#8a8984"                                       # neutral: the base case
FIELD_COLORS = dict(zip(["mortality", "interest", "lapse", "expenses"],
                        [PALETTE[0], PALETTE[2], PALETTE[6], PALETTE[3]]))  # one colour per assumption, always
UPLOAD = "Upload your own…"

# --- display names (labels only) ---

PRODUCT_NAMES = {"term_life": "Term life", "whole_life": "Whole life"}
RESERVE_NAMES = {"none": "No reserves", "net_premium": "Net premium",
                 "fpt": "Full preliminary term (FPT)", "zeroized": "Zeroized"}
METRIC_NAMES = {"npv": "NPV", "profit_margin": "Profit margin", "irr": "IRR"}
METRIC_FORMATS = {"npv": "{:,.2f}", "profit_margin": "{:.2%}", "irr": "{:.2%}"}
SHOCK_NAMES = {"mortality": "Mortality ×", "interest": "Interest ± % pts",
               "lapse": "Lapse ×", "expenses": "Expenses ×"}
PRODUCT_SETTINGS = {   # config key -> (label, help, shown as %)
    "cash_value_pct": ("Cash value by policy year (% of reserve)",
                       "Paid on surrender at the end of year t: h_t × tV. "
                       "The last row applies to every later year.", True),
}


def name_of(names: dict):
    return lambda key: names.get(key, key)


# --- small input helpers ---

def pct_input(label: str, value: float, key: str, where=st, help: str | None = None) -> float:
    """Show a rate as % (0.055 -> 5.5); return it as a decimal again."""
    shown = where.number_input(label, value=round(float(value) * 100, 8), step=0.1,
                               format="%.3f", key=key, help=help)
    return shown / 100


def by_year_table(values: list[float], column: str, key: str, as_pct: bool) -> list[float]:
    """Editable one-column table, one row per policy year."""
    scale = 100 if as_pct else 1
    df = pd.DataFrame({"Year": range(1, len(values) + 1), column: [v * scale for v in values]})
    out = st.data_editor(df, num_rows="dynamic", hide_index=True, disabled=["Year"],
                         key=key, width="stretch")
    return [float(v) / scale for v in out[column].dropna()]


# --- assumption editors ---

def mortality_picker(current, key: str):
    """Pick a CSV in config/mortality or upload one; preview q by age."""
    files = {p.name: p for p in sorted((CONFIG_DIR / "mortality").glob("*.csv"))}
    names = list(files) + [UPLOAD]
    now = Path(current).name if isinstance(current, (str, Path)) else UPLOAD
    choice = st.selectbox("Mortality table", names,
                          index=names.index(now) if now in names else 0, key=f"{key}_m",
                          help="CSV with columns age, q. The last row is ω (set q = 1 there).")
    chosen = files.get(choice, current)
    if choice == UPLOAD:
        f = st.file_uploader("CSV with columns age, q", type="csv", key=f"{key}_up")
        if f is not None:
            try:
                chosen = MortalityTable.from_csv(f)
            except (ValueError, KeyError) as e:
                st.error(f"Can't use this file: {e}")
    try:
        table = chosen if isinstance(chosen, MortalityTable) else MortalityTable.from_csv(chosen)
    except (ValueError, KeyError, TypeError):
        return chosen
    st.caption(f"Ages {int(table.ages[0])}–{table.omega}  ·  ω = {table.omega}")
    preview = pd.DataFrame({"age": table.ages, "q": table.q})
    st.altair_chart(
        alt.Chart(preview).mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X("age:Q", title="age"),
            y=alt.Y("q:Q", title="q (log scale)", scale=alt.Scale(type="log")),
            tooltip=["age", alt.Tooltip("q:Q", format=".6f")],
        ).properties(height=140),
        width="stretch")
    return chosen


def interest_editor(value, key: str):
    by_year = isinstance(value, list)
    mode = st.radio("Interest rate", ["Flat", "By policy year"], index=int(by_year),
                    horizontal=True, key=f"{key}_imode",
                    help="Profit basis: rate earned on cash flows and reserves. "
                         "Reserve basis: rate used to discount.")
    if mode == "Flat":
        return pct_input("Rate (%)", value[0] if by_year else value, f"{key}_i")
    rates = by_year_table(value if by_year else [value], "Rate %", f"{key}_iy", as_pct=True)
    st.caption("Needs a rate for every policy year.")
    return rates


def lapse_editor(value, key: str):
    if st.toggle("No lapses", value=value is None, key=f"{key}_nolapse"):
        return None
    bands = value or [{"from_year": 1, "to_year": None, "rate": 0.05}]
    df = pd.DataFrame({
        "From year": [b["from_year"] for b in bands],
        "To year": [b.get("to_year") for b in bands],
        "Lapse rate %": [b["rate"] * 100 for b in bands],
    })
    out = st.data_editor(df, num_rows="dynamic", hide_index=True, key=f"{key}_lapse",
                         width="stretch", column_config={
        "From year": st.column_config.NumberColumn(min_value=1, step=1, required=True),
        "To year": st.column_config.NumberColumn(
            min_value=1, step=1, help="Leave blank on the last row: applies to every later year."),
        "Lapse rate %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5,
                                                      required=True),
    })
    st.caption("Lapses happen at year end, so only survivors lapse: p0w = (1 − q) · w.")
    rows = out.dropna(subset=["From year", "Lapse rate %"])
    return [{"from_year": int(r["From year"]),
             "to_year": None if pd.isna(r["To year"]) else int(r["To year"]),
             "rate": float(r["Lapse rate %"]) / 100} for _, r in rows.iterrows()]


def expenses_editor(value, key: str) -> dict:
    e = value or {}
    left, right = st.columns(2)
    return {
        "initial_fixed": left.number_input(
            "Initial: fixed ($)", value=float(e.get("initial_fixed", 0)), min_value=0.0,
            step=50.0, key=f"{key}_e0f", help="Paid once at t = 0, before the first premium."),
        "initial_pct": pct_input(
            "Initial: % of first premium", e.get("initial_pct", 0), f"{key}_e0p", where=right),
        "renewal_fixed": left.number_input(
            "Renewal: fixed ($ per year)", value=float(e.get("renewal_fixed", 0)), min_value=0.0,
            step=10.0, key=f"{key}_erf", help="Paid with each premium, start of every year."),
        "renewal_pct": pct_input(
            "Renewal: % of each premium", e.get("renewal_pct", 0), f"{key}_erp", where=right),
    }


def basis_editor(title: str, cfg: dict, key: str, full: bool) -> dict:
    st.subheader(title)
    out = dict(cfg)
    out["mortality"] = mortality_picker(cfg["mortality"], key)
    out["interest"] = interest_editor(cfg["interest"], key)
    if full:
        st.markdown("**Lapses**")
        out["lapse"] = lapse_editor(cfg.get("lapse"), key)
        st.markdown("**Expenses**")
        out["expenses"] = expenses_editor(cfg.get("expenses"), key)
    else:
        st.caption("Reserves use only mortality and interest.")
    return out


def product_settings_editor(cfg: dict, key: str) -> dict:
    if not cfg:
        st.caption("No settings for this product.")
        return {}
    out = {}
    for name, value in cfg.items():
        label, help_text, as_pct = PRODUCT_SETTINGS.get(name, (name.replace("_", " "), None, False))
        if isinstance(value, list):
            st.markdown(label, help=help_text)
            out[name] = by_year_table(value, "%" if as_pct else "value", f"{key}_{name}", as_pct)
        else:
            out[name] = st.number_input(label, value=float(value), help=help_text,
                                        key=f"{key}_{name}")
    return out


# --- sensitivity editors (interest shown in % points, others as factors) ---

def is_shift(field: str) -> bool:
    return SHOCKS[field][1] == 0.0


def to_ui(field: str, size: float) -> float:
    return size * 100 if is_shift(field) else size


def from_ui(field: str, size: float) -> float:
    return float(size) / 100 if is_shift(field) else float(size)


def field_column(**kw):
    return st.column_config.SelectboxColumn("Assumption", options=list(SHOCKS),
                                            required=True, **kw)


def tornado_editor(shocks: dict, key: str) -> dict:
    df = pd.DataFrame([{"Assumption": f, "Low": to_ui(f, lo), "High": to_ui(f, hi)}
                       for f, (lo, hi) in shocks.items()], columns=["Assumption", "Low", "High"])
    out = st.data_editor(df, num_rows="dynamic", hide_index=True, key=key, width="stretch",
                         column_config={"Assumption": field_column()})
    return {r.Assumption: (from_ui(r.Assumption, r.Low), from_ui(r.Assumption, r.High))
            for r in out.dropna().itertuples()}


def spider_editor(sizes: dict, key: str) -> dict:
    df = pd.DataFrame([{"Assumption": f, "Sizes": ", ".join(f"{to_ui(f, s):g}" for s in v)}
                       for f, v in sizes.items()], columns=["Assumption", "Sizes"])
    out = st.data_editor(df, num_rows="dynamic", hide_index=True, key=key, width="stretch",
                         column_config={"Assumption": field_column(),
                                        "Sizes": st.column_config.TextColumn(
                                            help="Comma-separated, e.g. 0.5, 1, 1.5")})
    result = {}
    for r in out.dropna().itertuples():
        try:
            result[r.Assumption] = [from_ui(r.Assumption, x) for x in r.Sizes.split(",") if x.strip()]
        except ValueError:
            st.error(f"Spider sizes for {r.Assumption} must be numbers separated by commas.")
    return result


def scenarios_editor(named: dict, key: str) -> dict:
    cols = ["Scenario"] + [SHOCK_NAMES[f] for f in SHOCKS]
    df = pd.DataFrame([{"Scenario": name,
                        **{SHOCK_NAMES[f]: to_ui(f, m[f]) if f in m else None for f in SHOCKS}}
                       for name, m in named.items()], columns=cols)
    out = st.data_editor(df, num_rows="dynamic", hide_index=True, key=key, width="stretch",
                         column_config={SHOCK_NAMES[f]: st.column_config.NumberColumn()
                                        for f in SHOCKS})
    result = {}
    for _, r in out.iterrows():
        if pd.isna(r["Scenario"]) or not str(r["Scenario"]).strip():
            continue
        result[str(r["Scenario"])] = {f: from_ui(f, r[SHOCK_NAMES[f]])
                                      for f in SHOCKS if not pd.isna(r[SHOCK_NAMES[f]])}
    return result


# --- page ---

st.set_page_config(page_title="Pricing engine", layout="wide")
st.title("Life insurance pricing & profit test")
st.markdown(
    "Defaults reproduce Dickson, Hardy & Waters, *Actuarial Mathematics for Life Contingent "
    "Risks*, Ch. 13: the 10-year term policy (Sec. 13.3–13.6) and whole life (Example 13.1). "
    "Change anything; results update instantly.")

presets = sorted(p.stem for p in (CONFIG_DIR / "runs").glob("*.yaml"))
preset = st.sidebar.selectbox("Preset", presets)
cfg = load_run(preset)
k = preset                                   # widget keys per preset, so switching resets them

with st.sidebar:
    st.header("1 · Policy")
    cfg["product"] = st.selectbox("Product", list(PRODUCTS), format_func=name_of(PRODUCT_NAMES),
                                  index=list(PRODUCTS).index(cfg["product"]), key=f"{k}_prod")
    mp = cfg["model_point"]
    mp["issue_age"] = st.number_input("Issue age x", value=int(mp["issue_age"]), min_value=0,
                                      step=1, key=f"{k}_x")
    mp["sum_insured"] = st.number_input("Sum insured S ($)", value=float(mp["sum_insured"]),
                                        min_value=1.0, step=1000.0, key=f"{k}_S",
                                        help="Paid at the end of the year of death.")
    mp["premium"] = st.number_input("Annual premium P ($)", value=float(mp["premium"] or 0.0),
                                    step=10.0, format="%.2f", key=f"{k}_P",
                                    help="Paid at the start of each policy year.")
    whole = st.checkbox("Whole of life (to the end of the table)", value=mp.get("term") is None,
                        key=f"{k}_wl")
    mp["term"] = None if whole else st.number_input(
        "Term n (years)", value=int(mp.get("term") or 10), min_value=1, step=1, key=f"{k}_n")
    cfg["product_config"] = product_settings_editor(cfg["product_config"] or {}, f"{k}_pc")

    st.header("2 · Reserves & measures")
    cfg["reserve_method"] = st.selectbox(
        "Reserve method", list(RESERVE_METHODS), format_func=name_of(RESERVE_NAMES),
        index=list(RESERVE_METHODS).index(cfg["reserve_method"]), key=f"{k}_rm")
    same_basis = st.checkbox("Reserve on the profit basis", key=f"{k}_same",
                             help="Tick for zeroized reserves (DHW Sec. 13.7).")
    cfg["metric"] = st.selectbox("Measure for solver & sensitivity", list(METRICS),
                                 format_func=name_of(METRIC_NAMES),
                                 index=list(METRICS).index(cfg["metric"]), key=f"{k}_met")
    cfg["risk_discount"] = pct_input("Risk discount rate r (%)", cfg["risk_discount"], f"{k}_r",
                                     help="Hurdle rate used to discount profits. Not a basis.")

tab_bases, tab_results, tab_solver, tab_sens = st.tabs(
    ["Assumptions", "Results", "Premium solver", "Sensitivity"])

with tab_bases:
    left, right = st.columns(2)
    with left:
        cfg["profit_basis"] = basis_editor("Profit basis (what we expect to happen)",
                                           cfg["profit_basis"], f"{k}_pb", full=True)
    with right:
        if same_basis:
            st.subheader("Reserve basis")
            st.info("Using the profit basis.")
            cfg["reserve_basis"] = cfg["profit_basis"]
        else:
            cfg["reserve_basis"] = basis_editor("Reserve basis (prudent, for reserves only)",
                                                cfg["reserve_basis"], f"{k}_rb", full=False)

with tab_sens:
    st.caption("Mortality, lapse and expenses are multiplied (1 = no change, 1.1 = +10%). "
               "Interest is moved in percentage points (0 = no change, −1 = one point lower). "
               "Only the profit basis is shocked; reserves stay as they are.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Tornado**: one assumption at a time, low and high")
        cfg["shocks"] = tornado_editor(cfg["shocks"], f"{k}_sh")
    with c2:
        st.markdown("**Spider**: one assumption at several sizes")
        cfg["spider"] = spider_editor(cfg["spider"], f"{k}_sp")
    with c3:
        st.markdown("**Scenarios**: several assumptions together (blank = unchanged)")
        cfg["scenarios"] = scenarios_editor(cfg["scenarios"], f"{k}_sc")

try:
    run = build_run(cfg)
    ledger = run.ledger()
except (ValueError, KeyError, TypeError, ArithmeticError) as e:
    st.error(f"Can't run this configuration: {e}")
    st.stop()

metric_name = METRIC_NAMES[cfg["metric"]]

with tab_results:
    cols = st.columns(len(METRICS))
    summary = []                                       # rows for the profit test CSV
    for col, (name, metric) in zip(cols, METRICS.items()):
        try:
            value = metric(ledger, run.risk_discount)
            col.metric(METRIC_NAMES[name], METRIC_FORMATS[name].format(value))
        except ValueError:
            value = None
            col.metric(METRIC_NAMES[name], "n/a")
        summary.append({"measure": METRIC_NAMES[name], "value": value})
    st.caption(f"NPV and profit margin at r = {run.risk_discount:.2%}.")
    summary += [{"measure": "Risk discount rate r", "value": run.risk_discount},
                {"measure": "Premium P", "value": run.model_point.premium},
                {"measure": "Sum insured S", "value": run.model_point.sum_insured},
                {"measure": "Issue age x", "value": run.model_point.issue_age}]
    d1, d2 = st.columns(2)
    d1.download_button("Download ledger (CSV)", ledger.to_csv(index=False),
                       file_name=f"{preset}_ledger.csv", mime="text/csv", key=f"{k}_dl_ledger")
    d2.download_button("Download profit test summary (CSV)",
                       pd.DataFrame(summary).to_csv(index=False),
                       file_name=f"{preset}_profit_test.csv", mime="text/csv",
                       key=f"{k}_dl_summary")
    st.subheader("Profit signature Πₜ")
    HALF_CENT = 0.005                                            # shows as 0.00 anyway
    active = ledger.index[ledger.signature.abs() >= HALF_CENT]
    last = int(ledger.t[active.max()]) if len(active) else 0
    shown = ledger[ledger.t <= last]                             # chart and table stop here
    n_years = int(ledger.t.max())
    if last < n_years:
        left = float(ledger.inforce[ledger.t == last + 1].iloc[0])   # share in force after t = last
        if left == 0:
            st.info(f"All policies have ended by t = {last}: everyone has died or lapsed "
                    f"(e.g. a 100% lapse rate that year). Later years are all zero, so they're "
                    f"hidden here; the ledger CSV keeps all {n_years} years.")
        else:
            st.info(f"After t = {last}, Πₜ is below half a cent every year: only {left:.2g} "
                    f"of the original policies ({left:.4%}) are still in force. Those years are "
                    f"hidden here; the ledger CSV keeps all {n_years} years.")
    st.altair_chart(
        alt.Chart(shown).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("t:O", title="t"),
            y=alt.Y("signature:Q", title="Πₜ"),
            tooltip=["t", alt.Tooltip("signature:Q", format=",.2f")]),
        width="stretch")
    st.subheader("Ledger (DHW Eq. 13.1)")
    st.dataframe(shown.round(2), hide_index=True, width="stretch")

with tab_solver:
    st.markdown(f"Find the premium that gives a target **{metric_name}** "
                "(change the measure in the sidebar).")
    is_rate = cfg["metric"] != "npv"
    target = st.number_input(f"Target {metric_name}" + (" (%)" if is_rate else " ($)"),
                             value=5.0 if is_rate else 0.0, step=0.5 if is_rate else 100.0,
                             key=f"{k}_{cfg['metric']}_tgt")   # new box per measure
    target = target / 100 if is_rate else target
    if st.button("Solve premium", key=f"{k}_solve"):
        try:
            P = solve_premium(run.model_point, lambda m: run.ledger(model_point=m),
                              run.measure, target)
            shown = METRIC_FORMATS[cfg["metric"]].format(target)
            st.success(f"Premium P = {P:,.2f} gives {metric_name} = {shown}")
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

    if len(T):
        st.subheader(f"Tornado: {metric_name}")
        def size_text(field, size):                   # "×1.1" or "−1 pt" next to each bar
            return f"{to_ui(field, size):+g} pt" if is_shift(field) else f"×{size:g}"

        bars = pd.DataFrame([
            {"field": r.field, "shock": side, "value": value, "base": r.base,
             "label": size_text(r.field, size)}
            for r in T.itertuples()
            for side, value, size in (("low", r.low, r.low_size), ("high", r.high, r.high_size))])
        base = float(T.base.iloc[0])
        reach = float((bars.value - base).abs().max()) * 1.25 or 1.0   # widest bar + room for labels
        x_scale = alt.Scale(domain=[base - reach, base + reach], zero=False)   # centred on base
        y = alt.Y("field:N", sort=list(T.field), title=None,               # biggest swing on top
                  scale=alt.Scale(paddingInner=0.45), axis=alt.Axis(labelFontSize=13))
        layer = alt.Chart(bars).encode(y=y)
        tornado_bars = layer.mark_bar(cornerRadius=4).encode(
            x=alt.X("value:Q", title=metric_name, scale=x_scale),
            x2="base:Q",
            color=alt.Color("shock:N", title="shock",
                            scale=alt.Scale(domain=["low", "high"], range=[BLUE, ORANGE])),
            tooltip=["field", "shock", "label", alt.Tooltip("value:Q", format=",.4f")])
        left_labels = layer.transform_filter("datum.value < datum.base").mark_text(
            align="right", dx=-6, fontSize=12).encode(x="value:Q", text="label:N")
        right_labels = layer.transform_filter("datum.value >= datum.base").mark_text(
            align="left", dx=6, fontSize=12).encode(x="value:Q", text="label:N")
        base_rule = alt.Chart(T.head(1)).mark_rule(color=GRAY, strokeWidth=2).encode(x="base:Q")
        chart = (tornado_bars + left_labels + right_labels + base_rule).properties(
            width=720, height=70 * len(T))                  # taller rows, narrower chart
        st.altair_chart(chart)
        st.caption(f"Grey line = base {metric_name} "
                   f"({METRIC_FORMATS[cfg['metric']].format(base)}). Biggest swing on top.")

    if len(S):
        st.subheader(f"Spider: {metric_name}")
        st.caption("One panel per assumption (factors and % points use different scales).")
        S_ui = S.assign(size=[to_ui(f, s) for f, s in zip(S.field, S["size"])])
        st.altair_chart(
            alt.Chart(S_ui).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
                x=alt.X("size:Q", title="shock size"),
                y=alt.Y("value:Q", title=metric_name),
                color=alt.Color("field:N", legend=None,           # panel title names it
                                scale=alt.Scale(domain=list(FIELD_COLORS),
                                                range=list(FIELD_COLORS.values()))),
                tooltip=["field", "size", alt.Tooltip("value:Q", format=",.4f")],
            ).properties(width=260, height=200)
             .facet(facet=alt.Facet("field:N", title=None), columns=4)
             .resolve_scale(x="independent"))

    st.subheader(f"Scenarios: {metric_name}")
    names = list(C.scenario)                              # "base" first, then in config order
    colours = [GRAY] + [PALETTE[i % len(PALETTE)] for i in range(len(names) - 1)]
    fmt = ",.2f" if cfg["metric"] == "npv" else ".2%"
    scen = alt.Chart(C).encode(
        y=alt.Y("scenario:N", sort=names, title=None),
        x=alt.X("value:Q", title=metric_name),
        tooltip=["scenario", alt.Tooltip("value:Q", format=fmt)])
    scen_bars = scen.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("scenario:N", legend=None,
                        scale=alt.Scale(domain=names, range=colours)))
    scen_labels = scen.mark_text(align="left", dx=6).encode(text=alt.Text("value:Q", format=fmt))
    st.altair_chart(scen_bars + scen_labels, width="stretch")
