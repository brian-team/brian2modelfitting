from types import FunctionType

from brian2 import have_same_dimensions
from brian2.units.fundamentalunits import Quantity
from tqdm.auto import tqdm


def _format_quantity(v, precision=3):
    if isinstance(v, Quantity):
        return f'{v.in_best_unit(precision=precision)}'
    else:
        return f'{v:.{precision}g}'


def callback_text(params, errors, best_params, best_error, index,
                  additional_info):
    """Default callback print-out for Fitters"""
    params = []
    for p, v in sorted(best_params.items()):
        params.append(f'{p}={_format_quantity(v)}')
    param_str = ', '.join(params)
    round = f'Round {index}: '
    if len(additional_info.get('objective_errors', [])) > 1:
        best_error_str = _format_quantity(best_error, precision=4)
        errors = []
        for error, normed_error, varname in zip(additional_info['objective_errors'],
                                                additional_info['objective_errors_normalized'],
                                                additional_info['output_var']):
            if not have_same_dimensions(error, normed_error) or error != normed_error:
                raw_error_str = f', unnormalized error: {_format_quantity(error)}'
            else:
                raw_error_str = ''
            errors.append(f'{_format_quantity(normed_error)} ({varname}{raw_error_str})')
        error_sum = ' + '.join(errors)
        print(f"{round}Best parameters {param_str}\n"
              f"{' '*len(round)}Best error: {best_error_str} = {error_sum}")
    else:
        print(f"{round}Best parameters {param_str}")
        if 'objective_errors_normalized' in additional_info:
            best_error_normed = _format_quantity(additional_info['objective_errors_normalized'][0])
            best_error_raw = _format_quantity(additional_info['objective_errors'][0])
            if (not have_same_dimensions(additional_info['objective_errors_normalized'][0],
                                         additional_info['objective_errors'][0]) or
                    best_error_normed != best_error_raw):
                print(f"{' ' * len(round)}Best error: {best_error_normed} ({additional_info['output_var'][0]}, "
                      f"unnormalized error: {best_error_raw})")
            else:
                print(f"{' ' * len(round)}Best error: {best_error_normed} ({additional_info['output_var'][0]})")
        else:
            best_error_str = _format_quantity(best_error, precision=4)
            print(f"{' ' * len(round)}Best error: {best_error_str} ({additional_info['output_var'][0]})")


def callback_none(params, errors, best_params, best_error, index,
                  additional_info):
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
        raise TypeError('callback has to be a str (`text` or `progressbar`), '
                        'allable or None')
    return callback


def make_dic(names, values):
    """Create dictionary based on list of strings and 2D array"""
    result_dict = {name: value for name, value in zip(names, values)}
    return result_dict
