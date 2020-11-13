from types import FunctionType

from brian2.units.fundamentalunits import Quantity
from tqdm.autonotebook import tqdm



def callback_text(params, errors, best_params, best_error, index, additional_info):
    """Default callback print-out for Fitters"""
    params = []
    for p, v in sorted(best_params.items()):
        if isinstance(v, Quantity):
            params.append(f'{p}={v.in_best_unit(precision=2)}')
        else:
            params.append(f'{p}={v:.2g}')
    param_str = ', '.join(params)
    if isinstance(best_error, Quantity):
        best_error_str = best_error.in_best_unit(precision=2)
    else:
        best_error_str = f'{best_error:.2g}'
    round = f'Round {index}: '
    if (additional_info and
            'metric_weights' in additional_info and
            len(additional_info['metric_weights'])>1):
        errors = []
        for weight, error, varname in zip(additional_info['metric_weights'],
                                          additional_info['best_errors'],
                                          additional_info['output_var']):
            if isinstance(error, Quantity):
                errors.append(f'{weight!s}×{error.in_best_unit(precision=2)} ({varname})')
            else:
                errors.append(f'{weight!s}×{error:.2g} ({varname})')
        error_sum = ' + '.join(errors)
        print(f"{round}Best parameters {param_str}\n"
              f"{' '*len(round)}Best error: {best_error_str} = {error_sum}")
    else:
        print(f"{round}Best parameters {param_str}\n"
              f"{' '*len(round)}Best error: {best_error_str} ({additional_info['output_var'][0]})")


def callback_none(params, errors, best_params, best_error, index, additional_info):
    """Non-verbose callback"""
    pass


class ProgressBar(object):
    """Setup for tqdm progress bar in Fitter"""
    def __init__(self, total=None, **kwds):
        self.t = tqdm(total=total, **kwds)

    def __call__(self, params, errors, best_params, best_error, index,
                 additional_info):
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
