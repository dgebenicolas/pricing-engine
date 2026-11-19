"""Step 2 gate: mortality tables. Expected values from DHW only."""

import numpy as np
import pytest

from pricing_engine.assumptions.mortality import MortalityTable
from pricing_engine.config_loader import config_path


@pytest.fixture(scope="module")
def susm():
    return MortalityTable.from_csv(config_path("mortality/susm.csv"))


def epv_whole_life(q, i):
    """A_x and a-due_x for whole life at the first age of q, by backward recursion.

    Kept in the test on purpose: assumptions/ must not import engine/.
    A_x = v q_x + v p_x A_{x+1},  a_x = 1 + v p_x a_{x+1},  closed with q_omega = 1.
    """
    v = 1 / (1 + i)
    A, a = 0.0, 0.0
    for qx in q[::-1]:
        A = v * qx + v * (1 - qx) * A
        a = 1 + v * (1 - qx) * a
    return A, a


# --- Gate: SUSM against DHW Table D.4 (i = 5%) ---

def test_susm_A60(susm):
    A, _ = epv_whole_life(susm.q_vector(60), 0.05)
    assert A == pytest.approx(0.29028, abs=5e-6)


def test_susm_A61_and_a61(susm):
    A, a = epv_whole_life(susm.q_vector(61), 0.05)
    assert A == pytest.approx(0.30243, abs=5e-6)
    assert a == pytest.approx(14.6491, abs=5e-5)


def test_susm_q_values_p527(susm):
    q = susm.q_vector(60, 5)
    assert q[0] == pytest.approx(0.003398, abs=5e-7)  # age 60
    assert q[4] == pytest.approx(0.00529, abs=5e-6)   # age 64


def test_susm_closes_at_130(susm):
    assert susm.omega == 130
    q = susm.q_vector(60)
    assert len(q) == 130 - 60 + 1
    assert q[-1] == 1.0


# --- Gate: term tables, DHW Sec. 13.3 ---

def test_term_profit_formula():
    q = MortalityTable.from_csv(config_path("mortality/term_profit.csv")).q_vector(60, 10)
    t = np.arange(10)
    np.testing.assert_allclose(q, 0.01 + 0.001 * t)


def test_term_reserve_formula():
    q = MortalityTable.from_csv(config_path("mortality/term_reserve.csv")).q_vector(60, 10)
    t = np.arange(10)
    np.testing.assert_allclose(q, 0.011 + 0.001 * t)


# --- Unit behaviour ---

def test_q_vector_slices_by_age():
    tbl = MortalityTable(ages=np.array([50, 51, 52]), q=np.array([0.1, 0.2, 1.0]))
    np.testing.assert_array_equal(tbl.q_vector(51), [0.2, 1.0])
    np.testing.assert_array_equal(tbl.q_vector(50, 2), [0.1, 0.2])


def test_q_vector_returns_copy():
    tbl = MortalityTable(ages=np.array([50, 51]), q=np.array([0.1, 1.0]))
    tbl.q_vector(50)[0] = 99
    assert tbl.q[0] == 0.1


@pytest.mark.parametrize("age, n", [(49, 1), (52, None), (50, 3), (50, 0)])
def test_q_vector_out_of_range(age, n):
    tbl = MortalityTable(ages=np.array([50, 51]), q=np.array([0.1, 1.0]))
    with pytest.raises(ValueError):
        tbl.q_vector(age, n)


@pytest.mark.parametrize("ages, q", [
    ([50, 52], [0.1, 1.0]),   # gap in ages
    ([50, 51], [0.1, 1.5]),   # q > 1
    ([50], [0.1, 0.2]),       # length mismatch
])
def test_bad_table_rejected(ages, q):
    with pytest.raises(ValueError):
        MortalityTable(ages=np.array(ages), q=np.array(q))


# --- from_csv with an open file (what a Streamlit upload gives) ---

def test_from_csv_accepts_uploaded_file(susm):
    import io
    data = config_path("mortality/susm.csv").read_bytes()
    uploaded = MortalityTable.from_csv(io.BytesIO(data))
    np.testing.assert_array_equal(uploaded.q, susm.q)          # identical to reading the path
    assert uploaded.omega == 130


def test_from_csv_missing_column():
    import io
    with pytest.raises(ValueError, match="q"):
        MortalityTable.from_csv(io.StringIO("age,qx\n60,0.01\n"))


def test_from_csv_bad_table_rejected():
    import io
    with pytest.raises(ValueError):                             # gap in ages
        MortalityTable.from_csv(io.StringIO("age,q\n60,0.01\n62,0.02\n"))
