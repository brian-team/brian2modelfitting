from tqdm.autonotebook import tqdm
from types import FunctionType


def callback_text(params, errors, best_params, best_error, index):
    """Default callback print-out for Fitters"""
    param_str = ', '.join([f"{p}={v!s}" for p, v in sorted(best_params.items())])
    print(f"Round {index}: Best parameters {param_str} (error: {best_error!s})")


def callback_none(params, errors, best_params, best_error, index):
    """Non-verbose callback"""
    pass


class ProgressBar(object):
    """Setup for tqdm progress bar in Fitter"""
    def __init__(self, total=None, **kwds):
        self.t = tqdm(total=total, **kwds)

    def __call__(self, params, errors, best_params, best_error, index):
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
    elif set_type is None:
        callback = callback_none
    elif type(set_type) is FunctionType:
        callback = set_type
    else:
        raise TypeError("callback has to be a str ('text' or 'progressbar'), "
                        "callable or None")

    return callback


def make_dic(names, values):
    """Create dictionary based on list of strings and 2D array"""
    result_dict = {name: value for name, value in zip(names, values)}

    return result_dict
