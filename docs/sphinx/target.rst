
.. _target:

lvmsurveysim.target
===================


.. _target-defining:

Defining a target
-----------------

A `.Target` is defined as a `.Region` on the sky with a name and additional methods for tiling and plotting. Regions on the sky can be of `circular <.CircularRegion>`, `elliptical <.EllipticalRegion>`, `polygonal <.PolygonalRegion>`, or `rectangular <.RectangularRegion>` shape.

While targets and regions can be initialised programatically, it is usually more convenient to define them in a YAML file. For example, the following text defines an elliptical and polygonal region

.. code-block:: yaml

    M33:
        coords: [23.462100, 30.659942]
        region_type: ellipse
        frame: icrs
        region_params:
            a: 1.16
            b: 0.666
            pa: 0
        priority: 2
        telescope: LVM-1m

    MW2:
        coords: [[229.26093667,  -8.49621056],
                 [214.06408833, -51.3662725],
                 [227.68629528, -30.87828389]]
        region_type: polygon
        frame: galactic
        priority: 1
        telescope: LVM-160

In both cases we need to define the ``region_type`` and the coordinate ``frame`` (either ``icrs`` or ``galactic``) in which the coordinates are written. For ``M33`` we specify the ``coords`` of the centre of the ellipse and define the ``region_params`` with the major and minor axis lengths and the parallactic angle. All values must be in degrees. For ``MW2`` we provide a list of coordinates with all the vertices of the polygon. We also define the ``priority`` of the target for scheduling (higher priority means it is more likely to be observed) and the telescope we want to use to observe it.

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
