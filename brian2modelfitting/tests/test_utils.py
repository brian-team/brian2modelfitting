'''
Test the modelfitting utils
'''
import sys
import tqdm
import numpy as np
from brian2modelfitting import (callback_text, callback_setup, make_dic,
                               callback_none)
from brian2modelfitting.modelfitting.utils import ProgressBar
from numpy.testing.utils import assert_equal


def test_callback_text(capsys):
    callback_text({'a':1}, [1.2], [1, 2, 3], 3)
    c, _ = capsys.readouterr()
    assert_equal(c, "Round 3: fit {'a': 1} with error: 1.2\n")


def test_callback_none():
    c = callback_none({'a':1}, [1.2], [1, 2, 3], 3)
    assert isinstance(c, type(None))


def test_ProgressBar():
    pb = ProgressBar(toolbar_width=10)
    assert_equal(pb.toolbar_width, 10)
    assert isinstance(pb.t, tqdm._tqdm.tqdm)
    pb({'a':1}, [1.2], [1, 2, 3], 3)


def test_callback_setup():
    c = callback_setup('text', 10)
    assert c == callback_text

    c = callback_setup('progressbar', 10)
    assert isinstance(c, ProgressBar)

    c = callback_setup(None, 10)
    assert callable(c)
    x = c({'a':1}, [1.2], [1, 2, 3], 3)
    assert x is None

    def callback(res, errors, parameters, k):
        return parameters

    c = callback_setup(callback, 10)
    assert callable(c)
    x = c({'a':1}, [1.2], [1, 2, 3], 3)
    assert_equal(x, [1, 2, 3])


def test_make_dic():
    names = ['a', 'b']
    values = [1, 2]
    result_dic = make_dic(names, values)

    assert isinstance(result_dic, dict)
    assert_equal(result_dic, {'a': 1, 'b': 2})
