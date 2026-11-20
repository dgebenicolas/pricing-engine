"""Streamlit app smoke tests: each preset renders and shows the DHW numbers.

The defaults pass through every widget (rates shown as %, tables for lapse
and cash values), so matching DHW also proves nothing is lost on the way.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "pricing_engine" / "app" / "streamlit_app.py")


def ledger_rows(at):
    return next(len(d.value) for d in at.dataframe if "signature" in d.value.columns)


def start(preset):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.selectbox[0].select(preset).run()
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


@pytest.mark.parametrize("preset, npv", [("term_dhw", "74.13"), ("wl_dhw", "2,938.26")])
def test_preset_renders(preset, npv):
    at = start(preset)
    assert not at.exception and not at.error
    assert metrics(at)["NPV"] == npv


def test_term_margin_and_irr():
    m = metrics(start("term_dhw"))
    assert m["Profit margin"] == "0.77%"       # DHW p. 521
    assert m["IRR"].startswith("12.4")         # DHW p. 520: 12.4% (app shows 12.43%)


def test_solver_button():
    at = start("term_dhw")
    at.selectbox(key="term_dhw_met").select("profit_margin").run()
    at.button[0].click().run()                 # default target 5 %
    assert "1,575.21" in at.success[0].value   # DHW p. 521


def test_interest_typed_as_percent():
    at = start("term_dhw")
    at.number_input(key="term_dhw_pb_i").set_value(4.5).run()   # 5.5 % -> 4.5 %
    npv = float(metrics(at)["NPV"].replace(",", ""))
    assert npv < 74.13 and not at.error


def test_bad_input_shows_error_not_crash():
    at = start("term_dhw")
    at.number_input(key="term_dhw_x").set_value(80).run()      # 80 + 10 yrs: past table end (69)
    assert at.error and not at.exception


def test_download_buttons_present():
    at = start("term_dhw")
    labels = [b.label for b in at.get("download_button")]
    assert "Download ledger (CSV)" in labels
    assert "Download profit test summary (CSV)" in labels


def test_wl_chart_stops_when_everyone_has_left():
    at = start("wl_dhw")
    assert any("t = 30" in i.value for i in at.info)            # 100% lapse in year 30
    assert ledger_rows(at) == 31                     # t = 0..30 shown


def test_term_not_trimmed():
    at = start("term_dhw")
    assert not any("have ended" in i.value for i in at.info)
    assert ledger_rows(at) == 11                     # t = 0..10

