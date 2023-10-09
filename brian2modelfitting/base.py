"""
Basic functions such as input handling, shared between fitting and inference classes.
"""
from collections.abc import Mapping
import numbers

from brian2 import get_logger
from brian2.core.namespace import get_local_namespace
from brian2.core.functions import Function
from brian2.devices import RuntimeDevice, get_device, device
from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
from brian2.units.fundamentalunits import (Quantity,
                                           DIMENSIONLESS,
                                           fail_for_dimension_mismatch)

from .simulator import RuntimeSimulator, CPPStandaloneSimulator

logger = get_logger(__name__)


def get_full_namespace(additional_namespace, level=0):
    # Get the local namespace with all the values that could be relevant
    # in principle -- by filtering things out, we avoid circular loops
    namespace = {key: value
                 for key, value in get_local_namespace(level=level + 1).items()
                 if (not key.startswith('_') and
                     isinstance(value, (Quantity, numbers.Number, Function)))}
    namespace.update(additional_namespace)

    return namespace


def handle_input_args(input_dict_or_arr, input_var, model):
    """
    Handle the input argument of the fit/inference methods.

    Parameters
    ----------
    input_dict_or_arr : dict or `np.ndarray`, optional
        A dictionary mapping the name of the input variable to the input.
        Note that only a single input is currently supported.
        When used together with the ``input_var`` argument (deprecated),
        this can also be a single array.
    input_var : str, optional
        The name of the input variable. Deprecated, use the ``input_dict``
        argument instead.
    model : `~brian2.equations.equations.Equations`
        The equations of the model to fit/infer.

    Returns
    -------
    input_arr, input_var : (`~numpy.ndarray`, str)
        The input array and the name of the input variable.
    """
    # Support deprecated legacy syntax of input_var + input or the new
    # syntax with a dictionary as input
    if input_var is not None:
        logger.warn("Use the 'input' argument with a dictionary instead "
                    "of giving the name as 'input_var'",
                    name_suffix='deprecated_input_var')
        if isinstance(input_dict_or_arr, Mapping) and input_var not in input_dict_or_arr:
            raise ValueError("Name given as 'input_var' and key in "
                             "'input' dictionary do not match.")
    else:
        if not isinstance(input_dict_or_arr, Mapping):
            raise TypeError("'input' argument has to be a dictionary "
                            "mapping the name of the input variable to the "
                            "input.")
        if len(input_dict_or_arr) > 1:
            raise NotImplementedError("Only a single input is currently "
                                      "supported.")
        input_var = list(input_dict_or_arr.keys())[0]

    if isinstance(input_dict_or_arr, Mapping):
        input_arr = input_dict_or_arr[input_var]
    else:
        input_arr = input_dict_or_arr

    if input_var != "spikes" and input_var not in model.identifiers:
        raise NameError(f"{input_var} is not an identifier in the model")
    return input_arr, input_var


def handle_output_args(output_dict, output_vars, model):
    """
    Handle the output argument of the fit/inference methods.

    Parameters
    ----------
    output_dict : dict, optional
        A dictionary mapping the name of the output variables to the output.
    output_vars : list of str, optional
        The name of the output variables. Deprecated, use the ``output_dict``
        argument instead.
    model : `~brian2.equations.equations.Equations`
        The equations of the model to fit/infer.

    Returns
    -------
    outputs, output_var : (list, str)
        The output list and the name of the output variables.
    """
    # Support deprecated legacy syntax of output_var + output or the new
    # syntax with a dictionary as output
    if output_vars is not None:
        logger.warn("Use the 'output' argument with a dictionary instead "
                    "of giving the name as 'output_var'",
                    name_suffix='deprecated_output_var')
        if isinstance(output_vars, str):
            output_vars = [output_vars]

        if isinstance(output_dict, Mapping):
            if set(output_vars) != set(output_dict.keys()):
                raise ValueError("Names given as 'output_var' and keys "
                                 "in 'output' dictionary do not match.")
            output_list = list(output_dict.values())
        elif not isinstance(output_dict, list):
            output_list = [output_dict]
        else:
            output_list = output_dict
    else:
        if not isinstance(output_dict, Mapping):
            raise TypeError("'output' argument has to be a dictionary "
                            "mapping the name of the input variable to the "
                            "input.")
        output_vars = list(output_dict.keys())
        output_list = list(output_dict.values())

    for o_var in output_vars:
        if o_var != 'spikes' and o_var not in model.names:
            raise NameError(f"{o_var} is not a model variable")

    return output_list, output_vars


def output_dims(output_list, output_var, model):
    """
    Verify the output dimensions.

    Parameters
    ----------
    output_list : list of `~numpy.ndarray`
        List of output values, for each variable in ``output_var``.
    output_var : list of str
        List of output variable names.
    model : `~brian2.equations.equations.Equations`
        The equations of the model to fit/infer.

    Returns
    -------
    dims : list of `~brian2.units.fundamentalunits.Dimension`
        The dimensions of the output variables.
    """
    dims = []
    for o_var, out in zip(output_var, output_list):
        if o_var == 'spikes':
            dims.append(DIMENSIONLESS)
        else:
            dims.append(model[o_var].dim)
            fail_for_dimension_mismatch(out, dims[-1],
                                        'The provided target values '
                                        '("output") need to have the same '
                                        'units as the variable '
                                        '{}'.format(o_var))
    return dims


def input_equations(input_var, input_dim):
    """
    Define the input variable for the equations of the model.

    Parameters
    ----------
    input_arr : `~numpy.ndarray`
        The input array.
    input_var : str
        The name of the input variable.

    Returns
    -------
    input_eqs : str
        The equations setting the input variable.
    """
    input_dim = '1' if input_dim is DIMENSIONLESS else repr(input_dim)
    input_eqs = "{} = input_var(t, i % n_traces) : {}".format(input_var,
                                                              input_dim)
    return input_eqs


def output_equations(output_var, output_dim):
    """
    Make the output variables available within the model equations. This
    can be useful for approaches that couple the system to the target values.

    Parameters
    ----------
    output_var : list of str
        The names of the output variables.
    output_dim : list of `~brian2.units.fundamentalunits.Dimension`
        The dimensions of the output variables.

    Returns
    -------
    output_eqs : str
        The equations setting a variable named ``<output_variable>_target``.
    """
    counter = 0
    output_eqs = []
    for o_var, o_dim in zip(output_var, output_dim):
        if o_var != 'spikes':
            counter += 1
            # For approaches that couple the system to the target values,
            # provide a convenient variable
            output_expr = f'output_var_{counter}(t, i % n_traces)'
            output_dim = ('1' if o_dim is DIMENSIONLESS
                          else repr(o_dim))
            output_eqs.append("{}_target = {} : {}".format(o_var,
                                                           output_expr,
                                                           output_dim))
    return "\n".join(output_eqs)


def handle_param_init(param_init, model):
    """
    Verify that the parameters given in ``param_init`` are valid.

    Parameters
    ----------
    param_init : dict or None
        A dictionary mapping the name of the parameters to their initial value.
    model : `~brian2.equations.equations.Equations`
        The equations of the model to fit/infer.

    Returns
    -------
    param_init : dict
        The dictionary mapping the name of the parameters to their initial values.
    """
    if not param_init:
        param_init = {}
    for param, val in param_init.items():
        if not (param in model.diff_eq_names or
                param in model.parameter_names):
            raise ValueError(f"{param} is not a model variable or a "
                             "parameter in the model")
    return param_init


def setup_fit():
    """
    Function sets up simulator in one of the two available modes: runtime
    or standalone. The `.Simulator` that will be used depends on the currently
    set `.Device`. In the case of `.CPPStandaloneDevice`, the device will also
    be reset if it has already run a simulation.

    Returns
    -------
    simulator : `.Simulator`
    """
    simulators = {
        CPPStandaloneDevice: CPPStandaloneSimulator(),
        RuntimeDevice: RuntimeSimulator()
    }
    if isinstance(get_device(), CPPStandaloneDevice):
        if device.has_been_run is True:
            build_options = dict(device.build_options)
            get_device().reinit()
            get_device().activate(**build_options)
    simulator = [sim for dev, sim in simulators.items()
                 if isinstance(get_device(), dev)]
    assert len(simulator) == 1, f"Found {len(simulator)} simulators for device {get_device().__class__.__name__}"
    return simulator[0]
