.. _getting-started:

Getting started
===============

Installation
------------

``lvmsurveysim`` is not currently pip-installable. Instead, you'll need to get or clone the code from the `GitHub repo <https://github.com/sdss/lvmsurveysim>`__. Once you get the code, simply run ::

    python setup.py install

This will install ``lvmsurveysim`` and all its dependencies. To check that everything worked, and from inside a Python terminal, try importing ::

    >>> import lvmsurveysim
    >>> lvmsurveysim.__version__
    0.1.0

If you don't get an error then you are probably ready to start using the code.

Additionally you may want to check out `lvmcore <https://github.com/sdss/lvmcore>`__, which contains the target file (although this is not necessary to run a simulation). Simply download the code and make the environment variable ``$LVMCORE_DIR`` point to the directory.


Running a simulation
--------------------

To run a simulation you will need a target file. If you don't provide one, ``lvmsurveysim`` will try to load ``$LVMCORE_DIR/surveydesign/targets.yaml``. You can check the :ref:`target-defining` section.

The following code runs a simple simulation

.. code-block:: python
    :linenos:

    from lvmsurveysim.schedule import ObservingPlan, Scheduler
    from lvmsurveysim.target import TargetList

    # Creates a list of targets/
    targets = TargetList(target_file='targets.yaml')

    # Creates observing plans for APO and LCO for the range 2021-2025.
    apo_plan = ObservingPlan(2459216, 2461041, observatory='APO')
    lco_plan = ObservingPlan(2459216, 2461041, observatory='LCO')

    # Creates an Scheduler instance and runs the simulation

    scheduler = Scheduler(targets, observing_plans=[apo_plan, lco_plan])
    scheduler.run(progress_bar=True)
    scheduler.save('lvmsurveysim_results.fits', overwrite=True)

Note that in line 5 we provide the name of a file with the targets we want to observe. If that parameter is not provided the ``lvmcore`` target file will be used.

A more complicated example, with some plotting of the inputs and outputs can be found `here <https://gist.github.com/albireox/3e88a206f557af98ae1e4de9ecc338c4>`__.
