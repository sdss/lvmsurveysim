
.. _target:

lvmsurveysim.target
===================


.. _target-defining:

Defining a target
-----------------

A `.Target` is defined as a `.Region` on the sky with a name and additional methods for tiling and plotting. Regions on the sky can be of `circular <.CircularRegion>`, `elliptical <.EllipticalRegion>`, `polygonal <.PolygonalRegion>`, or `rectangular <.RectangularRegion>` shape.

While targets and regions can be initialised programatically, it is usually more convenient to define them in a YAML file. For example, the following text defines an elliptical and polygonal region

.. code-block:: yaml

    LMC:
        coords: [79.5135633, -68.5271292]
        region_type: circle
        frame: icrs
        region_params:
            r: 4.0
        priority: 30
        observatory: LCO
        telescope: LVM-160
        max_airmass: 2.00
        min_shadowheight: 1.0
        exptime: 1200
        n_exposures: 9
        min_exposures: 3
        min_moon_dist: 45
        max_lunation: 0.25
        overhead: 1.1
        tiling_strategy: center_first
        group: ["MCs"]

    ORION_SPARSE:
        coords: [206.42, -17.74]
        region_type: circle
        frame: galactic
        region_params:
            r: 19.5
        priority: 9
        observatory: BOTH
        telescope: LVM-160
        max_airmass: 1.75
        min_shadowheight: 500.0
        exptime: 900
        n_exposures: 1
        min_exposures: 1
        min_moon_dist: 60
        max_lunation: 1.0
        overhead: 1.1
        tiling_strategy: center_first
        sparse: 5
        group: ["ORI"]

    MW1:
    coords: [315.0, 0.0]
    region_type: rectangle
    frame: galactic
    region_params:
        width: 150.0
        height: 16.0
        pa: 0
    priority: 10
    observatory: LCO
    telescope: LVM-160
    max_airmass: 1.75
    min_shadowheight: 1000.0
    exptime: 900
    n_exposures: 1
    min_exposures: 1
    min_moon_dist: 60
    max_lunation: 1.0
    overhead: 1.1
    tiling_strategy: lowest_airmass
    group: ["MW"]

In all cases we need to define the ``region_type`` and the coordinate ``frame`` (either ``icrs`` or ``galactic``) in which the coordinates are written. For ``M33`` we specify the ``coords`` of the centre of the ellipse and define the ``region_params`` with the major and minor axis lengths and the parallactic angle. All values must be in degrees. For ``MW2`` we provide a list of coordinates with all the vertices of the polygon. We also define the ``priority`` of the target for scheduling (higher priority means it is more likely to be observed) and the telescope we want to use to observe it.

More examples of regions can be seen `here <https://github.com/sdss/lvmcore/blob/master/surveydesign/targets.yaml>`__.

Targets can then be loaded by calling `.Target.from_list` ::

    m33 = Target.from_list('M33', target_file='my_targets.yaml')

Or we can load all the targets in the file ::

    targets = TargetList(target_file='my_targets.yaml')

If the file is not specified in either case, ``lvmsurveysim`` defaults to ``$LVMCORE_DIR/surveydesign/targets.yaml``.


.. _target-reference:

target
------

.. automodule:: lvmsurveysim.target
    :members: _VALID_FRAMES

.. automodule:: lvmsurveysim.target.target
    :members: Target, TargetList
    :show-inheritance:
    :undoc-members:
    :private-members:


regions
-------

.. automodule:: lvmsurveysim.target.region
    :members: Region, EllipticalRegion, CircularRegion, PolygonalRegion, RectangularRegion, OverlapRegion, region_factory
    :show-inheritance:
    :undoc-members:
