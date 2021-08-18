Advanced Features
=================

This part of documentation lists other features provided alongside or inside
`~brian2modelfitting.fitter.Fitter` and `~brian2modelfitting.inferencer.Inferencer`
objects to allow users easier and a more flexible development when working on
their own problems.

.. contents::
    :local:
    :depth: 1

Parameters initialization
-------------------------

Whilst running `~brian2modelfitting.fitter.Fitter` or `~brian2modelfitting.inferencer.Inferencer`,
the user is able to pass the values of the parameters and variables that will
be used as initial conditions when solving the differential equations defined
in the neuron model.

Initial conditions should be passed by using an additional dictionary to the
constructor:

.. code:: python

  init_conds = {'v': -30*mV}


.. code:: python

  fitter = TraceFitter(..., param_init = init_conds)

or

.. code:: python

  inferencer = Inferencer(..., param_init=init_conds)

Restart
-------
By default any `~brian2modelfitting.fitter.Fitter` object works in continuous
optimization mode between run, where all of the parameters drawn are being
evaluated.

By setting the ``restart`` argument in `~brian2modelfitting.fitter.Fitter.fit()`
to ``True``, the user can restart the optimizer and the optimization will
start from scratch.

Used by Fitter optimizer and metric can only be changed when the flat is
``True``.

The previously outlined ``restart`` argument is used in the similar fashion
in `~brian2modelfitting.inferencer.Inferencer.infer()` method. It is set to
``False`` by default, and each following re-call of the method will result
in the multi-round inference. If the user wants amortized inference without
using any knowledge from the previous round of optimization instead, the
``restart`` argument should be set to ``True``.

Multi-objective optimization
----------------------------
In the case of `~brian2modelfitting.fitter.Fitter` classes, it is possible to
fit more than one output variable at the same time by combining the errors for
each variable. To do so, the user can specify several output variables during
the initialization as follows:

.. code:: python

  fitter = TraceFitter(...,
                       output={'x': target_x,
                               'y': target_y})

If the fitter function uses a single metric, it is applied to both variables.

.. note::
 
  This approach requires that the resulting error has the same units for all
  variables, i.e., it would not be possible to use the same `.MSEMetric` on
  variables with different units, since the errors cannot be simply added up.

As a more general solution, the user can specify a metric for each variable
and utilize their normalization arguments to make the units compatible (most
commonly by turning both errors into dimensionless quantities). The
normalization also defines the relative weights of all errors. For example, if
the variable ``x`` has dimensions of mV and the variable ``y`` is
dimensionless, the following metrics can be used to make an error of 10 mV in
``x`` to be weighed as much as an error of 0.1 in ``y``

.. code:: python

  metrics = {'x': MSEMetric(normalization=10*mV),
             'y': MSEMetric(normalization=0.1)}

This has to be passed as the ``metric`` argument of the `~brian2modelfitting.fitter.Fitter.fit`
function.

In the case of the `~brian2modelfitting.inferencer.Inferencer` class,
switching from a single- to multi-objective optimization is seamless. The user
has to provide multiple output variables during the initialization process the
same way as for `~brian2modelfitting.fitter.Fitter` classes:

.. code:: python

  inferencer = Inferencer(...,
                          output={'x': target_x,
                                  'y': target_y})

Later, during the inference process, the user has to define feautres for each
output variable as follows:

.. code:: python

  posterior = inferencer.infer(...,
                               features={'x': list_of_features_for_x,
                                         'y': list_of_features_for_y})

If the user prefers automatic feature extraction, the ``features`` argument
should not be defined (it should stay set to None).

.. warning::
 
  If the user chooses to define a list of features for extracting the summary
  features, it is important to keep in mind that the total number of features
  will be increased as many times as there are output variables set for
  multi-objective optimization.

Callback function
-----------------

To visualize the progress of the optimization we provided few possibilities of
the feedback inside the `~brian2modelfitting.fitter.Fitter`.

The 'callback' input provides few default options, updated in each round:
 - ``'text'`` (default) - prints out the parameters of the best fit and
   corresponding error;
 - ``'progressbar'`` - uses ``tqdm.autonotebook`` to provide a progress bar;
 - ``None`` - non-verbose;

as well as **customized feedback option**. User can provide
a *callable* (i.e., a function), that ensures either returning an output or
printout. If callback returns ``True``, the fitting execution will be
interrupted.

User gets four arguments to customize over:
 - ``params`` - set of parameters from current round;
 - ``errors`` - set of errors from current round;
 - ``best_params`` - best parameters globally, from all rounds;
 - ``best_error`` - best parameters globally, from all rounds;
 - ``index`` - index of current round.

An example callback function:

.. code:: python

  def callback_fun(params, errors, best_params, best_error, index):
      print('index {} errors minimum: {}'.format(index, min(errors)))

  ...

  fitter = TraceFitter(...)
  result, error  = fitter.fit(..., callback=callback_fun)

OnlineTraceFitter
-----------------
  
`~brian2modelfitting.fitter.OnlineTraceFitter` was created to work with long
traces or large-scale optimization problems. This `~brian2modelfitting.fitter.Fitter`
class uses online mean square error as a metric.
When the `~brian2modelfitting.fitter.OnlineTraceFitter.fit()` method is called
there is no need of specifying a metric, which is by default set to None.
The errors are instead calculated with `~brian2.groups.group.Group.run_regularly`
for each simulation.
  
.. code:: python
  
  fitter = OnlineTraceFitter(model=model,
                             input={'I': inp_traces},
                             output={'v': out_traces},
                             dt=0.1*ms,
                             n_samples=5)
  
  result, error = fitter.fit(optimizer=optimizer,
                             n_rounds=1,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area])
  
Reference the target values in the equations
--------------------------------------------
  
A model can refer to the target output values within the equations. For
example, if the membrane potential trace *v* (i.e. `output_var='v'`) is used
for the optimization, equations can refer to the target trace as `v_target`.
This allows adding a coupling term such as: `coupling*(v_target - v)` to
the equation that corresponds to state variable `v`, pulling the trajectory
towards the correct solution.

Generate Traces
---------------

`~brian2modelfitting.fitter.Fitter` and `~brian2modelfitting.inferencer.Inferencer`
classes allow the user can to generate the traces with optimized parameters.

For a quick access to best fitted set of parameters `~brian2modelfitting.fitter.Fitter`
classes provide ready to use functions:

 - `~brian2modelfitting.fitter.TraceFitter.generate_traces` inside `~brian2modelfitting.fitter.TraceFitter`;
 - `~brian2modelfitting.fitter.SpikeFitter.generate_spikes` inside `~brian2modelfitting.fitter.SpikeFitter`.

These functions can be called after the fitting procedure is finalized in the
following manner, without any input arguments:

.. code:: python

    fitter = TraceFitter(...)
    results, error = fitter.fit(...)
    traces = fitter.generate_traces()

.. code:: python

    fitter = SpikeFitter(...)
    results, error = fitter.fit(...)
    spikes = fitter.generate_spikes()

On the other hand, since the `~brian2modelfitting.inferencer.Inferencer` class
is able to perform the inference of the unknown parameter distribution by
utilizing output traces and spike trains simultaneously, ``generate_traces``
is used for both.

Once the approximated posterior distribution is built, the user is allowed to
call ``generate_traces`` on `~brian2modelfitting.inferencer.Inferencer`
object. If only one output variable is used for the optimization of the
parameters, the user does not have to specifiy output variable in the 
``generate_traces`` method through ``output_var`` argument. If, for example,
the multi-objective optimization is performed by using both output traces and
spike trains and the user is interested in only times of spike events,
``output_var`` should be set to ``'spike'``. Otherwise, if the user specifies
a list of names or the ``output_var`` is not specified, a dictionary with keys
set to output variable names and with their respective values, will be
returned instead.


Customize the ``generate`` method for `~brian2modelfitting.fitter.Fitter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create traces for other parameters, or generate traces after the spike
train fitting, user can call the `~brian2modelfitting.fitter.Fitter.generate`
method, which takes in the following arguments:

.. code:: python

  fitter.generate(params=..., output_var=..., param_init=..., level=0)

where ``params`` should be a dictionary of parameters for which we generate
the traces; ``output_var`` provides an option to pick one or more variables
for visualization; with ``param_init``, the user is able to define the initial
values for differential equations in the model; and ``level`` allows for
specification of the namespace level from which we are able to get the
constant parameters of the model.

If ``output_var`` is the name of a single variable name (or the special name
``'spikes'``), a single `~.Quantity` (for variables) or a list of spikes time
arrays (for ``'spikes'``) will be returned. If a list of names is provided,
then the result is a dictionary with all the results.

.. code:: python

    fitter = TraceFitter(...)
    results, error = fitter.fit(...)
    traces = fitter.generate(output_var=['v', 'h', 'n', 'm'])
    v_trace = traces['v']
    h_trace = traces['h']

Results
-------

`~brian2modelfitting.fitter.Fitter` classes store all of the parameters
used by the optimizer as well as the corresponding errors. To retrieve them
you can call the `~brian2modelfitting.fitter.Fitter.results`.


.. code:: python

  fitter = TraceFitter(...)
  ...
  traces = fitter.generate_traces()

.. code:: python

  fitter = SpikeFitter(...)
  ...
  results = fitter.results(format='dataframe')


Results can be returned in one of the following formats:

 - ``'list'`` (default) - returns a list of dictionaries with corresponding
   parameters (including units) and errors;
 - ``'dict'`` - returns a dictionary of arrays with corresponding parameters
   (including units) and errors;
 - ``'dataframe'`` - returns a `~pandas.DataFrame` (without units).

The use of units (only relevant for formats ``'list'`` and ``'dict'``) can be
switched on or off with the ``use_units`` argument. If it is not specified, it
will default to the value used during the initialization of the `Fitter`
(which itself defaults to ``True``).

Example output:
~~~~~~~~~~~~~~~
- ``format='list'``:

.. code:: python

  [{'gl': 80.63365773 * nsiemens, 'g_kd': 66.00430921 * usiemens, 'g_na': 145.15634566 * usiemens, 'errors': 0.00019059452295872703},
   {'gl': 83.29319947 * nsiemens, 'g_kd': 168.75187749 * usiemens, 'g_na': 130.64547027 * usiemens, 'errors': 0.00021434415430605653},
   ...]


- ``format='dict'``:

.. code:: python

  {'g_na': array([176.4472297 , 212.57019659, ...]) * usiemens,
   'g_kd': array([ 43.82344525,  54.35309635, ...]) * usiemens,
   'gl': array([ 69.23559876, 134.68463669, ...]) * nsiemens,
   'errors': array([1.16788502, 0.5253008 , ...])}


- ``format='dataframe'``:

.. code:: python

     g_na      gl            g_kd      errors
  0  0.000280  8.870238e-08  0.000047  0.521425
  1  0.000192  1.121861e-07  0.000118  0.387140
  ...


Posterior distribution analysis
-------------------------------

Unlike `~brian2modelfitting.fitter.Fitter` classes, the `~brian2modelfitting.inferencer.Inferencer`
class does not keep track of all parameter values. Rather, it stores all
training data for neural density estimator which will later be used for
building the posterior distribution of each unknown parameter. Thus, the `~brian2modelfitting.inferencer.Inferencer`
does not returns best-fit values and corresponding errors, but the entire
posterior distribution that can be used to draw samples from, compute
descriptive statistics of parameters, analyize pairwise relationship between
each to parameters, etc.

There are three methods that enable the comprehensive analysis of the
posterior:

- `~brian2modelfitting.inferencer.Inferencer.pairplot` - returns axes of drawn
  samples from the posterior in a 2-dimenstional grid with marginals and
  pairwise marginals. Using this method, the user is able to inspect the
  relationship for all combinations of distributions for each parameter;
- `~brian2modelfitting.inferencer.Inferencer.conditional_pairplot` -
  visualizes the conditional pairplot;
- `~brian2modelfitting.inferencer.Inferencer.conditional_corrcoeff` - returns
  the correlation matrix of a distribution conditioned with the user-specified
  condition.

To see this in action, go to our tutorial page and learn how to use each of
these methods.

Standalone mode
---------------

Just like with regular Brian 2 scripts, all computations in the toolbox can be
performed in ``Runtime`` mode (default) or ``Standalone`` mode. For details,
please check the official Brian 2 documentation: https://brian2.readthedocs.io/en/stable/user/computation.html

To enable the ``Standalone`` mode, and to allowthe source code generation to
C++ code, add the following code right after Brian 2 is imported, but before
the simulation code:

.. code:: python

  set_device('cpp_standalone')

Important notes:
~~~~~~~~~~~~~~~~

.. warning::

  In the ``Standalone`` mode, a single script should not contain multiple
  `~brian2modelfitting.fitter.Fitter` or `~brian2modelfitting.inferencer.Inferencer`
  classes. Please, use separate scripts.

Note that the generation of traces or spikes via `~brian2modelfitting.fitter.Fitter.generate`
will always use runtime mode, even when the fitting procedure uses standalone mode.

Embedding network for automatic feature extraction
--------------------------------------------------

If the ``features`` argument of the `~brian2modelfitting.inferencer.Inferencer`
class is not defined, automatic feature extraction from the given output
traces will occur. By default, this is done by using the multi-layer
perceptron that is trained in parallel with the neural density estimator of
choice during the inference process. If the user wants to specify their own
custom embedding network, it is possible to do so by creating a neural
network by using ``PyTorch`` library and passing the instance of that neural
network as an additional keyword argument as follows:

.. code:: python
  
  import torch
  from torch import nn
  
  ...

  class CustomEmbeddingNet(nn.Module):

      def __init__(self, in_features, out_features, ...):
          ...

      def forward(self, x):
          ...

  
  in_features = out_traces.shape[1]
  out_features = ...
  embedding_net = CustomEmbeddingNet(in_features, out_features, ...)

  ...

  inferencer = Inferencer(...)
  inferencer.infer(...,
                   inference_kwargs={'embedding_net': embedding_net})

GPU usage for inference
-----------------------

It is possible to use the GPU for training the sdensity estimator. It is enough
to specify the ``sbi_device`` to ``'gpu'`` or ``'cuda'``.  Otherwise, if not
specified, or if set to ``'cpu'``, training will be done by using the CPU.

.. note::

  For default density estimators that are used either for SNPE, SNLE and SNRE,
  there are no significant speed-ups expected if the training is translocated
  to the GPU.

It is, however, possible to achieve a significant speed-up if the custom
embedding network relies on convolutions to extract feautres. Such operations
are known to achieve improvement in compuation time multifold.
