import os
import abc
from numpy import atleast_1d
from brian2 import device, NeuronGroup


def initialize_parameter(variableview, value):
    """initliazie parameter variable in static file, returns Dummy device"""
    variable = variableview.variable
    array_name = device.get_array_name(variable)
    static_array_name = device.static_array(array_name, value)
    device.main_queue.append(('set_by_array', (array_name,
                                               static_array_name,
                                               False)))
    return static_array_name


def initialize_neurons(params_names, neurons, params):
    """
    initialize each parameter for NeuronGroup returns dictionary of Dummy
    devices
    """
    params_init = dict()

    for name in params_names:
        params_init[name] = initialize_parameter(neurons.__getattr__(name),
                                                 params[name])
    return params_init


def run_again():
    """re-run the NeuronGroup on cpp file"""
    device.run(device.project_dir, with_output=False, run_args=[])


def set_parameter_value(identifier, value):
    """change parameter value in cpp file"""
    atleast_1d(value).tofile(os.path.join(device.project_dir,
                                          'static_arrays',
                                          identifier))


def set_states(init_dict, values):
    """set parameters values in the file for the NeuronGroup"""
    for obj_name, obj_values in values.items():
        set_parameter_value(init_dict[obj_name], obj_values)


class Simulation(object):
    """
    Simluation class created to perform a simulation for fit_traces
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """pass"""
        pass

    @abc.abstractmethod
    def initialize(self, network):
        """
        Prepares the simulation for running

        Parameters
        ----------
        network: Network initialized instance
            consisting of NeuronGroup named 'neurons' and a Monitor named
            'monitor'
        """
        pass

    @abc.abstractmethod
    def run(self, duration, params, params_names):
        """
        Restores the network, sets neurons to required parameters and runs
        the simulation

        Parameters
        ----------
        duration: simulation duration [ms]
        params: dict
            parameters to be set
        params_names: list strings
            names of parameters to set the dictionary
        """
        pass


class RuntimeSimulation(Simulation):
    """Simulation class created for use with RuntimeDevice"""
    def initialize(self, network):
        if network['neurons'] is NeuronGroup:
            raise Exception("Network needs to have a NeuronGroup 'neurons'")

        self.network = network
        self.network.store()

    def run(self, duration, params, params_names):
        self.network.restore()
        self.network['neurons'].set_states(params, units=False)
        self.network.run(duration, namespace={})


class CPPStandaloneSimulation(Simulation):
    """Simulation class created for use with CPPStandaloneDevice"""
    def initialize(self, network):
        if not network['neurons']:
            raise Exception("Network needs to have a NeuronGroup 'neurons'")

        self.network = network

    def run(self, duration, params, params_names):
        """
        Simulation has to be run in two stages in order to initalize the
        code generaion
        """
        if not device.has_been_run:
            self.params_init = initialize_neurons(params_names,
                                                  self.network['neurons'],
                                                  params)
            self.network.run(duration)

        else:
            set_states(self.params_init, params)
            run_again()
