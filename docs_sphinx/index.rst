brian2modelfitting
==================

The package `.brian2modelfitting` is a tool for parameter fitting of neuron
models in the `Brian 2 simulator <https://brian2.readthedocs.org>`_.

Please report bugs at the `github issue tracker <https://github.com/brian-team/brian2modelfitting/issues>`_
or at the `Brian discussion forum <https://groups.google.com/forum/#!forum/briansupport>`_. The latter is
also a could place to discuss feature requests or potential contributions.


Model fitting
-------------

The ``brian2modelfitting`` toolbox offers allows the user to perform data driven
optimization for custom neuronal models specified with
`Brian 2 <https://brian2.readthedocs.org>`_.

The toolbox allows the user to find the best fit of the parameters for recorded traces and
spike trains. Just like Brian itself, the Model Fitting Toolbox is designed to
be easy to use and save time through automatic parallelization of the
simulations using code generation.


Contents
--------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   introduction/index
   optimizer/index
   metric/index
   features/index
   examples/index

API reference
-------------
.. toctree::
   :maxdepth: 5
   :titlesonly:

   api/brian2modelfitting

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
