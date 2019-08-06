from brian2 import (NeuronGroup, TimedArray, Equations, get_device, Network,
                    StateMonitor, SpikeMonitor, device)
from brian2.devices import reinit_devices
from .modelfitting import setup_fit, setup_neuron_group, get_spikes


def generate_fits(model=None,
                  params=None,
                  input=None,
                  input_var=None,
                  output_var=None,
                  dt=None,
                  method=None,
                  reset=None, refractory=False, threshold=None,
                  param_init=None):
    """
    Generate instance of best fits for predicted parameters and all of the
    traces

    Parameters
    ----------
    model : `~brian2.equations.Equations` or string
        The equations describing the model.
    params : dict
        Predicted parameters
    input : input data as a 2D array
    input_var : string
        Input variable name.
    output_var : string
        Output variable name or 'spikes' to reproduce spike time.
    dt : time step
    method: string, optional
        Integration method
    param_init: dict
        Dictionary of variables to be initialized with the value

    Returns
    -------
    fits: array
        Traces of output varaible or spike times
    """
    if get_device().__class__.__name__ == 'CPPStandaloneDevice':
        device.has_been_run = False
        reinit_devices()
        # device.reinint()
        # device.activate()

    simulator = setup_fit(model, dt, param_init, input_var, None)

    parameter_names = model.parameter_names
    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt
    n_neurons = Ntraces

    input_traces = TimedArray(input.transpose(), dt=dt)
    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                              ' + "% s" % repr(input_unit))

    neurons = setup_neuron_group(model, n_neurons, method, threshold, reset,
                                 refractory, param_init,
                                 input_var=input_traces,
                                 output_var=output_var,
                                 Ntraces=Ntraces)

    if output_var == 'spikes':
        monitor = SpikeMonitor(neurons, record=True, name='monitor')
    else:
        monitor = StateMonitor(neurons, output_var, record=True,
                               name='monitor')

    network = Network(neurons, monitor)
    simulator.initialize(network)

    simulator.run(duration, params, parameter_names)

    if output_var == 'spikes':
        fits = get_spikes(simulator.network['monitor'])
    else:
        fits = getattr(simulator.network['monitor'], output_var)

    return fits
