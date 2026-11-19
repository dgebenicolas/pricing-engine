from dataclasses import FrozenInstanceError, replace

import pytest

from pricing_engine.config_loader import config_path
from pricing_engine.model_point import ModelPoint


def test_term_dhw_from_yaml():
    mp = ModelPoint.from_yaml(config_path("model_points/term_dhw.yaml"))
    assert mp == ModelPoint(issue_age=60, sum_insured=100_000, premium=1500, term=10)


def test_wl_dhw_from_yaml():
    mp = ModelPoint.from_yaml(config_path("model_points/wl_dhw.yaml"))
    assert mp == ModelPoint(issue_age=60, sum_insured=100_000, premium=2300, term=None)


def test_premium_and_term_optional():
    mp = ModelPoint(issue_age=60, sum_insured=100_000)
    assert mp.premium is None and mp.term is None


def test_replace_premium_for_solver():
    mp = ModelPoint(60, 100_000, 1500, 10)
    assert replace(mp, premium=1575.21).premium == 1575.21
    assert mp.premium == 1500  # original unchanged


def test_frozen():
    mp = ModelPoint(60, 100_000, 1500, 10)
    with pytest.raises(FrozenInstanceError):
        mp.premium = 1600


@pytest.mark.parametrize("kwargs", [
    dict(issue_age=-1, sum_insured=100_000),
    dict(issue_age=60, sum_insured=0),
    dict(issue_age=60, sum_insured=100_000, premium=0),
    dict(issue_age=60, sum_insured=100_000, term=0),
])
def test_bad_values_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelPoint(**kwargs)


def test_replace_is_also_validated():
    with pytest.raises(ValueError):
        replace(ModelPoint(60, 100_000, 1500, 10), premium=-5)


def test_from_dict():
    mp = ModelPoint.from_dict({"issue_age": 45, "sum_insured": 250_000, "term": 20})
    assert mp == ModelPoint(issue_age=45, sum_insured=250_000, premium=None, term=20)
