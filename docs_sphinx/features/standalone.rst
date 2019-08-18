Standalone mode
===============

Just like with regular Brian script, modelfitting computations can be performed in
``Runtime`` mode (default) or ``Standalone`` mode.
<https://brian2.readthedocs.io/en/stable/user/computation.html>

To enable this mode, add the following line after your Brian import, but before your simulation code:

.. code:: python

  set_device('cpp_standalone')
