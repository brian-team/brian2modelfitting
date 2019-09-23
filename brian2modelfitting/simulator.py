import os
import abc
from numpy import atleast_1d
from brian2 import device, NeuronGroup


def initialize_parameter(variableview, value):
    """initialize parameter variable in static file, returns Dummy device"""
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


class Simulator(metaclass=abc.ABCMeta):
    """
    Simluation class created to perform a simulation for fit_traces
    """
    @abc.abstractmethod
    def initialize(self, network, var_init, name):
        """
        Prepares the simulation for running

        Parameters
        ----------
        network: `~brian2.core.network.Network`
            Network consisting of a `~brian2.groups.neurongroup.NeuronGroup`
            named ``neurons`` and a monitor named ``monitor``.
        var_init: dict
            dictionary to initialize the variable states
        name: `str`, optional
            name of the network
        """
        pass

    @abc.abstractmethod
    def run(self, duration, params, params_names):
        """
        Restores the network, sets neurons to required parameters and runs
        the simulation

        Parameters
        ----------
        duration: `~brian2.units.fundamentalunits.Quantity`
            Simulation duration
        params: dict
            parameters to be set
        params_names: list[str]
            names of parameters to set the dictionary
        """
        pass


class RuntimeSimulator(Simulator):
    """Simulation class created for use with RuntimeDevice"""
    def initialize(self, network, var_init, name='neurons'):
        if network[name] is NeuronGroup:
            raise Exception("Network needs to have a NeuronGroup 'neurons'")

        self.network = network
        self.var_init = var_init
        self.network.store()

    def run(self, duration, params, params_names, name='neurons'):
        self.network.restore()
        self.network[name].set_states(params, units=False)
        if self.var_init is not None:
            for k, v in self.var_init.items():
                self.network[name].__setattr__(k, v)

        self.network.run(duration, namespace={})


class CPPStandaloneSimulator(Simulator):
    """Simulation class created for use with CPPStandaloneDevice"""
    def initialize(self, network, var_init, name='neurons'):
        if not network[name]:
            raise Exception("Network needs to have a NeuronGroup 'neurons'")

        self.network = network
        self.var_init = var_init

    def run(self, duration, params, params_names, name='neurons'):
        """
        Simulation has to be run in two stages in order to initalize the
        code generaion
        """
        if not device.has_been_run:
            self.params_init = initialize_neurons(params_names,
                                                  self.network[name],
                                                  params)
            if self.var_init is not None:
                for k, v in self.var_init.items():
                    self.network[name].__setattr__(k, v)

            self.network.run(duration, namespace={})

        else:
            set_states(self.params_init, params)
            run_again()
