import os
import sys
import pytest
import numpy as np
from filterpicker import filterpicker as FP

# ---------------------------------------------

def test_simple():
    """
    This test is the repetition of the script one, asserting the right behaviour
    """
    errors = []
    #
    fpp = FP.FilterPicker(0.008, np.loadtxt('./tests/fg_sac.npa'),
                          filter_window=200, longterm_window=100, t_up=20,
                          threshold_1=20, threshold_2=10)
    pidx, punc, pfrq = fpp.run()
    #
    try:
        assert pytest.approx(24.304, 0.001) == pidx[0]
    except AssertionError:
        errors.append("Erroneous PickTime")
    #
    try:
        assert pytest.approx(0.008, 0.001) == punc[0]
    except AssertionError:
        errors.append("Erroneous PickUncert")
    #
    try:
        assert pytest.approx(1.0, 0.1) == pfrq[0]
    except AssertionError:
        errors.append("Erroneous PickFreq")    
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))    
