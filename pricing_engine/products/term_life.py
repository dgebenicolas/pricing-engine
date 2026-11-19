"""Level term insurance: the validation fixture (DHW Sec. 13.3, p. 509).

Sum insured S paid at the end of the year of death, level premium P
annually in advance, no surrender value, no maturity benefit.
"""

from __future__ import annotations

import numpy as np

from pricing_engine.model_point import ModelPoint
from pricing_engine.products.base import PolicySchedule, Product


class TermLife(Product):

    def term(self, mp: ModelPoint, omega: int) -> int:
        if mp.term is None:
            raise ValueError("term insurance needs mp.term")
        last_age = mp.issue_age + mp.term - 1          # age in the final policy year
        if last_age > omega:
            raise ValueError(f"term runs to age {last_age}, past omega = {omega}")
        return mp.term

    def policy_schedule(self, mp: ModelPoint, V: np.ndarray | None = None) -> PolicySchedule:
        # V is ignored: term benefits do not depend on reserves.
        if mp.premium is None:
            raise ValueError("policy_schedule needs a premium; solve for it first")
        if mp.term is None:
            raise ValueError("term insurance needs mp.term")
        n = mp.term
        return PolicySchedule(
            premium=np.full(n, mp.premium),             # P every year
            death_benefit=np.full(n, mp.sum_insured),   # S on death in any year
            surrender_benefit=np.zeros(n),              # no cash value
            maturity_benefit=np.zeros(n),               # nothing paid at expiry
        )
