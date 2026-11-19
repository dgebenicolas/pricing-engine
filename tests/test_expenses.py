import numpy as np
import pytest

from pricing_engine.assumptions.expenses import PremiumExpenses, expenses_from_config


def test_term_dhw_p510():
    e = PremiumExpenses(initial_fixed=400, initial_pct=0.20, renewal_pct=0.035)
    premiums = np.full(10, 1500.0)
    assert e.initial(premiums) == pytest.approx(700.0)        # 400 + 0.2 * 1500
    np.testing.assert_allclose(e.renewal(premiums), 52.5)     # 0.035 * 1500


def test_whole_life_dhw_p525():
    e = PremiumExpenses(initial_fixed=1800, renewal_fixed=100)
    premiums = np.full(30, 2300.0)
    assert e.initial(premiums) == 1800.0
    np.testing.assert_allclose(premiums - e.renewal(premiums), 2200.0)  # Table 13.8 col 3


def test_no_fixed_renewal_when_no_premium():
    e = PremiumExpenses(renewal_fixed=100, renewal_pct=0.05)
    np.testing.assert_allclose(e.renewal([1000.0, 0.0]), [150.0, 0.0])


def test_negative_parameter_rejected():
    with pytest.raises(ValueError):
        PremiumExpenses(initial_fixed=-1)


def test_from_config():
    assert expenses_from_config(None) is None
    assert expenses_from_config({"initial_fixed": 1800}) == PremiumExpenses(initial_fixed=1800)
