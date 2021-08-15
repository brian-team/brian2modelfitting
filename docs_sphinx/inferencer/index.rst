Inferencer
==========

Unlike more traditional inverse identification procedures that rely either on
gradient or gradient-free methods, the ``Inferencer`` class supports
simulation-based inference that has been established as a powerful alternative
approach.

The simulation-based inference is data-driven procedure supported by the
`sbi <https://www.mackelab.org/sbi/>`_, ``PyTorch``-based toolbox by Macke lab.

In general, this method yields twofold improvement over point-estimate fitting
procedures:

#. Simulation-based inference acts as if the actual statistical inference is
   performed, even in cases of extremly complex models with untractable
   likelihood function. Thus, instead of returning a single set of optimal
   parameters, it results in the approximated posterior distribution over
   unknown parameters. This is achieved by training a neural density estimator,
   details of which will be explained in depth later in the documentation.
#. Simulation-based inference uses prior system knowledge sparsely, using 
   only the most important features to identify mechanistic models that are 
   consistent with the recordings. This is achieved either by providing the 
   predifend set of features, or by automatically extraciting summary features 
   by using deep neural networks which is trained in parallel with neural 
   density estimator. 

The ``Inferencer`` class, in its core, is a fancy wrapper around the ``sbi``
package where the focus is put on inferring the unknown parameters of the
single-cell neuron models defined in Brian 2 simulator.

Neural density estimator
------------------------

There are three main estimation techniques supported in ``sbi`` that the user
can take the full control over seamlesly by using the ``Inferencer``:

#. sequential neural posterior estimation (SNPE)
#. sequential neural likelihood estimation (SNLE)
#. sequential neural ratio estimator (SNRE)

Simulation-based inference workflow
----------------------------------- 

The inferencer procedure is defined via three main steps:

#. step.
   Prior over unknown parameters needs to be defined, where the simplest
   choice would be uniform distribution given lower and upper bound
   (currently, this is only prior distribution supported through
   ``brian2modelfitting`` toolbox).
   After that, simulated data are generated given a mechanistic model with
   unknown parameters set as constants.
   Instead of taking the full output of the model, the neural network takes
   in summary data statistics of the output, e.g. instead of voltage trace as
   the output from a neuron model, we would feed a neural network with
   relevant electrophysiology features that outline the gist of the output
   sufficiently well.
#. step.
   A neural network learns association between the summary data statistics
   and unknown parameters (given the prior distribution over parameters).
   The learning method is heavily dependent on the choice of the inference
   technique.
#. step.
   The trained neural network is applied to the empirical data to infer
   posterior distribution over unknown parameters.  Optionally, this process
   can be repeated by using the trained posterior distribution over parameters
   as the prior distribution proposal for a refined optimization.

Implementation
--------------

Go to `the tutorial section <https://brian2modelfitting.readthedocs.io/en/stable/introduction/tutorial_sbi.html>`_
in  for the in-depth implementation analysis.