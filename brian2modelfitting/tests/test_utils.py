'''
Test the modelfitting utils
'''
import numpy as np
from brian2modelfitting import callback_text, callback_setup, make_dic
from brian2modelfitting.modelfitting.utils import ProgressBar
from numpy.testing.utils import assert_equal


def test_callback_setup():
    pass


def test_ProgressBar():
    pass


def test_callback_setup():
    pass


def test_make_dic():
    names = ['a', 'b']
    values = [1, 2]
    result_dic = make_dic(names, values)

    assert isinstance(result_dic, dict)
    assert_equal(result_dic, {'a': 1, 'b': 2})
