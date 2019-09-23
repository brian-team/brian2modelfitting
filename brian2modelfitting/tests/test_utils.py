'''
Test the modelfitting utils
'''
import tqdm
from brian2modelfitting import (callback_text, callback_setup, make_dic,
                               callback_none)
from brian2modelfitting.utils import ProgressBar
from numpy.testing.utils import assert_equal


def test_callback_text(capsys):
    callback_text([1, 2, 3], [1.2, 2.3, 0.1], {'a':3}, 0.1, 2)
    c, _ = capsys.readouterr()
    assert_equal(c, "Round 2: fit {'a': 3} with error: 0.1\n")


def test_callback_none():
    c = callback_none([1, 2, 3], [1.2, 2.3, 0.1], {'a':3}, 0.1, 2)
    assert isinstance(c, type(None))


def test_ProgressBar():
    pb = ProgressBar(toolbar_width=10)
    assert_equal(pb.toolbar_width, 10)
    assert isinstance(pb.t, tqdm.tqdm)
    pb([1, 2, 3], [1.2, 2.3, 0.1], {'a':3}, 0.1, 2)


def test_callback_setup():
    c = callback_setup('text', 10)
    assert c == callback_text

    c = callback_setup('progressbar', 10)
    assert isinstance(c, ProgressBar)

    c = callback_setup(None, 10)
    assert callable(c)
    x = c([1, 2, 3], [1.2, 2.3, 0.1], {'a':3}, 0.1, 2)
    assert x is None

    def callback(params, errors, best_params, best_error, index):
        return params

    c = callback_setup(callback, 10)
    assert callable(c)
    x = c([1, 2, 3], [1.2, 2.3, 0.1], {'a':3}, 0.1, 2)
    assert_equal(x, [1, 2, 3])


def test_make_dic():
    names = ['a', 'b']
    values = [1, 2]
    result_dic = make_dic(names, values)

    assert isinstance(result_dic, dict)
    assert_equal(result_dic, {'a': 1, 'b': 2})
