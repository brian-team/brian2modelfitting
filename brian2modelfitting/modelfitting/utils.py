from tqdm.autonotebook import tqdm


def callback_text(res, errors, parameters, k):
    print("Round {}: fit {} with error: {}".format(k, res, min(errors)))


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
    else:
        raise TypeError("callback has to be a str ('text' or 'progressbar') or\
                         callable")

    return callback


def make_dic(names, values):
    """Create dictionary based on list of strings and 2D array"""
    result_dict = {name: value for name, value in zip(names, values)}

    return result_dict
