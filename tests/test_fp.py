import pytest
import numpy as np
from filterpicker import filterpicker as FP

# ---------------------------------------------


def test_simple_sactrace():
    """
    This test is a script repetition, asserting the right behaviour
    INPUT TRACE: ObsPy read function "Z" trace
    """
    errors = []
    #
    fpp = FP.FilterPicker(0.008, np.loadtxt('./tests/fg_sac.npa'),
                          filter_window=1.6, longterm_window=3.2, t_up=0.16,
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
        assert pytest.approx(2.0, 0.1) == pfrq[0]
    except AssertionError:
        errors.append("Erroneous PickFreq")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_simple_optrace():
    """
    This test is a script repetition, asserting the right behaviour
    INPUT TRACE: ObsPy read function "Z" trace
    """
    errors = []
    #
    fpp = FP.FilterPicker(0.01, np.loadtxt('./tests/fg_op.npa'),
                          filter_window=1.6, longterm_window=3.2, t_up=0.16,
                          threshold_1=10, threshold_2=10)
    pidx, punc, pfrq = fpp.run()
    # [4.7  4.93] [0.03 0.01] [5. 2.]
    #
    if len(pidx) != 2:
        errors.append("Pick IDX list is incomplete")
    if len(punc) != 2:
        errors.append("Pick UNC list is incomplete")
    if len(pfrq) != 2:
        errors.append("Pick FRQ list is incomplete")
    #
    try:
        assert pytest.approx(4.7, 0.001) == pidx[0]
    except AssertionError:
        errors.append("Erroneous PickTime -> First")
    # --------- First
    try:
        assert pytest.approx(0.03, 0.001) == punc[0]
    except AssertionError:
        errors.append("Erroneous PickUncert -> First")
    #
    try:
        assert pytest.approx(5.0, 0.1) == pfrq[0]
    except AssertionError:
        errors.append("Erroneous PickFreq -> First")
    # --------- Second
    try:
        assert pytest.approx(4.93, 0.001) == pidx[1]
    except AssertionError:
        errors.append("Erroneous PickTime -> Second")
    #
    try:
        assert pytest.approx(0.01, 0.001) == punc[1]
    except AssertionError:
        errors.append("Erroneous PickUncert -> Second")
    #
    try:
        assert pytest.approx(2.0, 0.1) == pfrq[1]
    except AssertionError:
        errors.append("Erroneous PickFreq -> Second")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_optrace_cf():
    """
    This test is a script repetition, asserting the right behaviour
    INPUT TRACE: ObsPy read function "Z" trace
    """
    errors = []
    #
    fpp = FP.FilterPicker(0.01, np.loadtxt('./tests/fg_op.npa'),
                          filter_window=1.6, longterm_window=3.2, t_up=0.16,
                          threshold_1=10, threshold_2=10)
    pidx, punc, pfrq = fpp.run()

    # [4.7  4.93] [0.03 0.01] [5. 2.]
    fpp_cf = fpp.get_evaluation_function()
    fpp_cf_ref = np.loadtxt('./tests/fg_op_cf.npa')

    #
    try:
        assert pytest.approx(fpp_cf) == fpp_cf_ref
    except AssertionError:
        errors.append("Carachteristic Functions doesn't match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_sactrace_cf():
    """
    This test is a script repetition, asserting the right behaviour
    INPUT TRACE: ObsPy read function "Z" trace
    """
    errors = []
    #
    fpp = FP.FilterPicker(0.008, np.loadtxt('./tests/fg_sac.npa'),
                          filter_window=1.6, longterm_window=3.2, t_up=0.16,
                          threshold_1=20, threshold_2=10)
    pidx, punc, pfrq = fpp.run()
    fpp_cf = fpp.get_evaluation_function()
    fpp_cf_ref = np.loadtxt('./tests/fg_sac_cf.npa')
    #
    try:
        assert pytest.approx(fpp_cf) == fpp_cf_ref
    except AssertionError:
        errors.append("Carachteristic Functions doesn't match")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
