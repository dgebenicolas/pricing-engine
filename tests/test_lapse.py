import numpy as np
import pytest

from pricing_engine.assumptions.lapse import BandedLapse, LapseBand, lapse_from_config

DHW_EX_13_1 = [  # p. 525
    {"from_year": 1, "to_year": 9, "rate": 0.05},
    {"from_year": 10, "to_year": 29, "rate": 0.075},
    {"from_year": 30, "rate": 1.0},
]


def test_dhw_example_13_1_bands():
    w = lapse_from_config(DHW_EX_13_1).w_vector(30)
    np.testing.assert_array_equal(w[0:9], 0.05)     # years 1-9
    np.testing.assert_array_equal(w[9:29], 0.075)   # years 10-29
    assert w[29] == 1.0                             # year 30


def test_open_ended_band_extends():
    w = lapse_from_config(DHW_EX_13_1).w_vector(71)  # WL from 60 to omega = 130
    np.testing.assert_array_equal(w[29:], 1.0)


def test_shorter_n_truncates():
    np.testing.assert_array_equal(lapse_from_config(DHW_EX_13_1).w_vector(3), 0.05)


def test_closed_bands_cannot_extend():
    lapse = BandedLapse((LapseBand(1, 5, 0.05),))
    with pytest.raises(ValueError):
        lapse.w_vector(6)


def test_none_config_means_no_lapse():
    assert lapse_from_config(None) is None


@pytest.mark.parametrize("bands", [
    (LapseBand(2, 5, 0.05),),                          # does not start at year 1
    (LapseBand(1, 5, 0.05), LapseBand(7, None, 0.1)),  # gap (year 6)
    (LapseBand(1, None, 0.05), LapseBand(2, 3, 0.1)),  # open band not last
    (LapseBand(1, 5, 1.5),),                           # rate > 1
    (LapseBand(1, 0, 0.05),),                          # ends before it starts
])
def test_bad_bands_rejected(bands):
    with pytest.raises(ValueError):
        BandedLapse(bands)
