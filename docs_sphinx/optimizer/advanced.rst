Custom Optimizer
================

To use a different back-end optimization library, user can provide a
custom class that inherits from provided abstract class:

.. code:: python

   Optimizer()


Follows `ask()/tell()` interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
User can plug in different optimization tool, as long as it follows ```ask() / tell```
interface. Abstract ``class Optimizer`` prepared for different back-end libraries.
All of the optimizer specific arguments have to be provided upon
optimizers initialization.


```ask() / tell``` interface in optimizer class:

.. code:: python

  parameters = optimizer.ask()

  errors = simulator.run(parameters)

  optimizer.tell(parameters, errors)
  results = optimizer.recommend()
