.. _lvmsurveysim-changelog:

Changelog
=========

This file documents the main changes to the ``lvmsurveysim`` code.

* Better save and restore of `.Scheduler` objects.
* Improved plotting of the results of the scheduler.
* Implemented tiling of regions using an hexagonal, monolithic IFU. This tiling mode is now the default one.
* New `~.Scheduler.print_statistics`, `~.Scheduler.get_target_time`, and `~.Scheduler.plot_survey` methods.
* Implemented zenith avoidance, multiple exposures per pointing, minimum exposures per visit, and other improvements in the `.Scheduler` class.
* :release:`0.1.0 <2019-03-13>`
* Initial version.
