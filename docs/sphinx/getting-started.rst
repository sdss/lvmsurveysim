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

Note that the code requires Python 3.6+ to run.


Running a simulation
--------------------

To run a simulation you will need a target file. If you don't provide one, ``lvmsurveysim`` will try to load ``$LVMCORE_DIR/surveydesign/targets.yaml``. You can check the :ref:`target-defining` section.

The following code runs a simple simulation

.. code-block:: python
    :linenos:

    from lvmsurveysim.schedule import ObservingPlan, Simulator, TileDB
    from lvmsurveysim.target import TargetList
    import matplotlib.pyplot as plt

    # Creates a list of targets/
    targets = TargetList(target_file='./targets.yaml')

    # Create tile database and save
    tiledb = TileDB(targets)
    tiledb.tile_targets()
    tiledb.save('lco_tiledb', fits=True, overwrite=True)

    # Alternatively, load a previously tiled survey from disk:
    tiledb = TileDB.load('lco_tiledb', fits=True)

    # Creates observing plans for LCO for the range sep 2021 - jun 2025.
    lco_plan = ObservingPlan(2459458, 2460856, observatory='LCO') # baseline

    # Creates an Simulator instance and runs the simulation
    sim = Simulator(tiledb, observing_plan=lco_plan, verbos_level=1)

    # Run the simulation
    sim.run(progress_bar=True)

    # evaluate the results:
    sim.print_statistics()
    sim.plot_survey('LCO', use_groups=True)
    sim.plot_survey('LCO', use_groups=True, cumulative=True)
    sim.plot_survey('LCO', lst=True, use_groups=True)
    sim.plot(fast=True) # footprint
    sim.plot_airmass(tname='ALL', group=True, norm=True)
    sim.plot_shadow_height(tname='ALL', group=True, norm=True, cumulative=True, linear_log=True)


Note that in line 6 we provide the name of a file with the targets we want to observe. If that parameter is not provided the ``lvmcore`` target file will be used.

`sim.run <lvmsurveysim.schedule.scheduler.Simulator.run>` uses the default parameters for the simulation defined in the ``scheduler`` of the `configuration file <https://github.com/sdss/lvmsurveysim/blob/master/lvmsurveysim/etc/lvmsurveysim_defaults.yaml>`__.
