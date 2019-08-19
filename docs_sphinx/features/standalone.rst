Standalone mode
===============

Just like with regular Brian script, modelfitting computations can be performed in
``Runtime`` mode (default) or ``Standalone`` mode.
(https://brian2.readthedocs.io/en/stable/user/computation.html)

To enable this mode, add the following line after your Brian import, but before your simulation code:

.. code:: python

  set_device('cpp_standalone')


Important notes:
----------------

 .. warning::
     One script can not be used to initialize multiple Fitters, use separate scripts!

 .. warning::
     To use ``fitter.generate()`` user has to reinitialize the device, which
     causes the device to reset and disables the possibility for further fitting or
     retrieving information from fitter monitors.


To reinitialize the device add additional piece of code before calling ``generate()``:

.. code:: python

    device.reinit()
    device.activate()

    fitter.generate_traces()
