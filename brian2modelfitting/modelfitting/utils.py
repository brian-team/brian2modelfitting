from tqdm.autonotebook import tqdm
from types import FunctionType


def callback_text(res, errors, parameters, k):
    print("Round {}: fit {} with error: {}".format(k, res, min(errors)))

def callback_none(res, errors, parameters, k):
    pass

class ProgressBar(object):
    """Setup for tqdm progress bar in Fitter"""
    def __init__(self, toolbar_width=10):
        self.toolbar_width = toolbar_width
        self.t = tqdm(total=toolbar_width)

    def __call__(self, res, errors, parameters, k):
        self.t.update(1)


def callback_setup(set_type, n_rounds):
    """
    Helper function for callback setup in Fitter, loads option:
    'text', 'progressbar' or custion FunctionType
    """
    if set_type == 'text':
        callback = callback_text
    elif set_type == 'progressbar':
        callback = ProgressBar(n_rounds)
    elif type(set_type) is FunctionType:
        callback = set_type
    elif set_type is None:
        callback = callback_none
    else:
        raise TypeError("callback has to be a str ('text' or 'progressbar'),\
                         callable or None")

    return callback


def make_dic(names, values):
    """Create dictionary based on list of strings and 2D array"""
    result_dict = {name: value for name, value in zip(names, values)}

    return result_dict
