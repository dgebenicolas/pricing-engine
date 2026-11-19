import numpy as np
import pytest

from pricing_engine.model_point import ModelPoint
from pricing_engine.products.base import PolicySchedule, Product


def schedule(n=3, **override):
    arrays = dict(
        premium=np.full(n, 100.0),
        death_benefit=np.full(n, 1000.0),
        surrender_benefit=np.zeros(n),
        maturity_benefit=np.zeros(n),
    )
    arrays.update(override)
    return PolicySchedule(**arrays)


# --- PolicySchedule ---

def test_schedule_n():
    assert schedule(n=10).n == 10


def test_schedule_length_mismatch_rejected():
    with pytest.raises(ValueError):
        schedule(n=3, death_benefit=np.full(4, 1000.0))


def test_schedule_negative_rejected():
    with pytest.raises(ValueError):
        schedule(n=3, premium=np.array([100.0, -1.0, 100.0]))


def test_schedule_empty_rejected():
    with pytest.raises(ValueError):
        schedule(n=0)


def test_schedule_must_be_1d():
    with pytest.raises(ValueError):
        schedule(n=2, premium=np.ones((2, 1)))


# --- Product ABC ---

def test_product_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Product()


def test_incomplete_product_rejected():
    class NoSchedule(Product):
        def term(self, mp, omega):
            return 1

    with pytest.raises(TypeError):
        NoSchedule()


def test_minimal_product_works():
    class Dummy(Product):
        def term(self, mp, omega):
            return mp.term

        def policy_schedule(self, mp, V=None):
            return schedule(n=self.term(mp, omega=130))

    mp = ModelPoint(issue_age=60, sum_insured=1000, premium=100, term=3)
    sched = Dummy().policy_schedule(mp)
    assert sched.n == 3
