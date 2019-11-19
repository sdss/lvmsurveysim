#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-03-10
# @Filename: scheduler.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-09-25 15:20:08

import itertools
import os
import warnings

import astropy
import cycler
import matplotlib.pyplot as plt
import numpy
import shapely.vectorized
from matplotlib import animation

import lvmsurveysim.target
import lvmsurveysim.utils.spherical
from lvmsurveysim import IFU, config, log
from lvmsurveysim.exceptions import LVMSurveySimError, LVMSurveySimWarning
from lvmsurveysim.schedule.plan import ObservingPlan
from lvmsurveysim.utils.plot import __MOLLWEIDE_ORIGIN__, get_axes, transform_patch_mollweide, convert_to_mollweide
import time
try:
    import mpld3
except ImportError:
    mpld3 = None


numpy.seterr(invalid='raise')


__all__ = ['Scheduler', 'AltitudeCalculator']

__ZENITH_AVOIDANCE__ = config['scheduler']['zenith_avoidance']
__DEFAULT_TIME_STEP__ = config['scheduler']['timestep']

def polygon_perimeter(x, y, n=1.0, min_points=5):
    """ x and y are numpy type arrays. Function returns perimiter values every n-degree in length"""
    x_perimeter = numpy.array([])
    y_perimeter = numpy.array([])
    for x1,x2,y1,y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        # Calculate the length of a segment, hopefully in degrees
        print("%f,%f to %f,%f"%(x1,y1,x2,y2))
        dl = ((x2-x1)**2 + (y2-y1)**2)**0.5

        n_dl = numpy.max([int(dl/n), min_points])
        
        if x1 != x2:
            m = (y2-y1)/(x2-x1)
            b = y2 - m*x2

            interp_x = numpy.linspace(x1, x2, num=n_dl, endpoint=True)
            interp_y = interp_x * m + b
        
        else:
            interp_x = numpy.full(n_dl, x1)
            interp_y = numpy.linspace(y1,y2, n_dl, endpoint=True)

        x_perimeter = numpy.append(x_perimeter, interp_x)
        y_perimeter = numpy.append(y_perimeter, interp_y)
    return(x_perimeter, y_perimeter)

class AltitudeCalculator(object):
    """Calculate the altitude of a constant set of objects at some global JD,
    or at a unique jd per object.

    This is for efficiency reasons. The intermediate cos/sin arrays of the
    coordinates are cached.

    All inputs are in degrees. The output is in degrees.

    """

    def __init__(self, ra, dec, lon, lat):

        self.ra = numpy.deg2rad(numpy.atleast_1d(ra))
        self.dec = numpy.deg2rad(numpy.atleast_1d(dec))
        assert len(ra) == len(dec), 'ra and dec must have the same length.'
        self.sindec = numpy.sin(self.dec)
        self.cosdec = numpy.cos(self.dec)
        self.lon = lon   # this stays in degrees
        self.sinlat = numpy.sin(numpy.radians(lat))
        self.coslat = numpy.cos(numpy.radians(lat))

    def __call__(self, jd=None, lst=None):
        """Object caller.

        Parameters
        ----------
        jd : float or ~numpy.ndarray
            Scalar or array of JD values. If array, it needs to be the same
            length as ``ra``, ``dec``.
        lst : float or ~numpy.ndarray
            Scalar or array of Local Mean Sidereal Time values, in hours.
            If array, it needs to be the same length as ``ra``, ``dec``.
            Either ``jd`` is provided, this parameter is ignored.

        Returns
        -------
        altitude : `float` or `~numpy.ndarray`
            An array of the same size of the inputs with the altitude of the
            targets at ``jd`` or ``lst``, in degrees.

        """

        if jd is not None:
            dd = jd - 2451545.0
            lmst_rad = numpy.deg2rad(
                (280.46061837 + 360.98564736629 * dd +
                 # 0.000388 * (dd / 36525.)**2 +   # 0.1s / century, can be neglected here
                 self.lon) % 360)
        else:
            lmst_rad = numpy.deg2rad((lst * 15) % 360.)

        cosha = numpy.cos(lmst_rad - self.ra)
        sin_alt = (self.sindec * self.sinlat +
                   self.cosdec * self.coslat * cosha)

        return numpy.rad2deg(numpy.arcsin(sin_alt))


class Scheduler(object):
    """Schedules a list of targets following and observing plan.

    Parameters
    ----------
    targets : ~lvmsurveysim.target.target.TargetList
        The `~lvmsuveysim.target.target.TargetList` object with the list of
        targets to schedule.
    observing_plans : list of `.ObservingPlan` or None
        A list with the `.ObservingPlan` to use (one for each observatory).
        If `None`, the list will be created from the ``observing_plan``
        section in the configuration file.
    ifu : ~lvmsurveysim.ifu.IFU
        The `~lvmsurveysim.ifu.IFU` to use. Defaults to the one from the
        configuration file.
    remove_overlap : bool
        If set, removes pointings in regions that overlap with other regions
        with higher priority.

    Attributes
    ----------
    overlap : dict
        A dictionary with the masks of pointing overlaps between regions, and
        the global no overlap mask for each region.
    pointings : dict
        A dictionary with the pointings for each one of the targets in
        ``targets``. It is the direct result of calling
        `TargetList.get_tiling
        <lvmsurveysim.target.target.TargetList.get_tiling>`.
    schedule : ~astropy.table.Table
        An astropy table with the results of the scheduling. Includes
        information about the JD of each observation, the target observed,
        the index of the pointing in the target tiling, coordinates, etc.

    """

    def __init__(self, targets, observing_plans=None, ifu=None, remove_overlap=True, verbos_level=0):

        if observing_plans is None:
            observing_plans = self._create_observing_plans()

        assert isinstance(observing_plans, (list, tuple)), \
            'observing_plans must be a list of ObservingPlan instances.'

        for op in observing_plans:
            assert isinstance(op, ObservingPlan), \
                'one of the items in observing_plans is not an instance of ObservingPlan.'

        self.observing_plans = observing_plans

        self.targets = targets
        self.ifu = ifu or IFU.from_config()

        self.pointings = targets.get_tiling(ifu=self.ifu, to_frame='icrs')
        self.tile_priorities = targets.get_tile_priorities()
        self.tiling_type = 'hexagonal'
        self.verbos_level = verbos_level

        # Calculate overlap but don't apply the masks
        self.overlap = self.get_overlap()

        # Remove pointings that overlap with other regions.
        if remove_overlap:

            for ii in self.pointings:
                tname = self.targets[ii].name

                # Remove the overlapping tiles from the pointings and
                # remove their tile priorities.
                self.pointings[ii] = self.pointings[ii][
                    self.overlap[tname]['global_no_overlap']]
                self.tile_priorities[ii] = self.tile_priorities[ii][
                    self.overlap[tname]['global_no_overlap']]

                if len(self.pointings[ii]) == 0:
                    warnings.warn(f'target {tname} completely overlaps with other '
                                  'targets with higher priority.', LVMSurveySimWarning)

        self.schedule = None

    def __repr__(self):

        return (f'<Scheduler (observing_plans={len(self.observing_plans)}, '
                f'n_target={len(self.pointings)})>')

    def get_overlap(self):
        """Returns a dictionary of masks with the overlap between regions."""

        overlap = {}

        # Sort priorities.
        s = sorted(self.pointings)

        # Create an array of pointing to priority, one per target
        priorities = numpy.array([self.targets[idx].priority for idx in s])

        # Save the names ... why not
        names = numpy.array([self.targets[idx].name for idx in s])

        sorted_indices = numpy.argsort(priorities)[::-1]

        # Initialise the overlap dictionaries. Set the global_no_overlap to
        # True for all the pointings in the target tiling
        for idx in s:
            name = self.targets[idx].name
            overlap[name] = {}
            overlap[name]['global_no_overlap'] = numpy.ones(len(self.pointings[idx]),
                                                            dtype=numpy.bool)

        #import spherical geometry routine to use for calculating polygons in spherical coordinates
        from spherical_geometry import polygon as spherical_geometry_polygon

        for i_i, i in enumerate(sorted_indices[:-1]):
            # i has the highest priority because of the [::-1] reversal of the priority list

            if self.targets[i].region.region_type == 'circle':
                poly_i = spherical_geometry_polygon.SphericalPolygon.from_cone(self.targets[i].region.coords.transform_to('icrs').ra.deg,
                self.targets[i].region.coords.transform_to('icrs').dec.deg,\
                self.targets[i].region.r.deg,
                degrees=True)

            elif self.targets[i].region.region_type == 'rectangle':
                # Create a reference to the target shapley object. This is probably uncessary, and can be sourced directly
                shapely_i = self.targets[i].region.shapely

                # Create a set of polygons using the extertiors reported by shapely to create polygons using a convex hull.
                # This is probably stupid and I should use the actual polygon methods: rectangle circle, etc.
                # Get the x-y coordinates which define the polygon of the region.
                x_i, y_i = shapely_i.exterior.coords.xy

                per_x, per_y = polygon_perimeter(x_i, y_i)
                c_poly_perimeter = astropy.coordinates.SkyCoord(per_x*astropy.units.degree, per_y*astropy.units.degree, frame=self.targets[i].frame)
                poly_i = spherical_geometry_polygon.SphericalPolygon.from_radec(c_poly_perimeter.transform_to('icrs').ra.deg, c_poly_perimeter.transform_to('icrs').deg)

            else:
                # Create a reference to the target shapley object. This is probably uncessary, and can be sourced directly
                shapely_i = self.targets[i].region.shapely

                # Create a set of polygons using the extertiors reported by shapely to create polygons using a convex hull.
                # This is probably stupid and I should use the actual polygon methods: rectangle circle, etc.
                # Get the x-y coordinates which define the polygon of the region.
                x_i, y_i = shapely_i.exterior.coords.xy

                # Convert the coordinates of the polygon into SkyCoordinates
                # This logical statemetns that check for the type of coordinate
                c_i = astropy.coordinates.SkyCoord(x_i*astropy.units.degree, y_i*astropy.units.degree, frame=self.targets[i].frame)

                # Convert the x-y coordinates, now in SkyCoordinates into polygons in icrs. 
                # This ensures that independent of what ever coordinate system i or j are in that the comparison is in the correct frame
                poly_i = spherical_geometry_polygon.SphericalPolygon.from_radec(c_i.transform_to('icrs').ra.deg, c_i.transform_to('icrs').dec.deg)


            for j in sorted_indices[i_i + 1:]:
                # j has a lower priority. So we are masking j with i
                if self.targets[j].region.region_type == 'circle':
                    poly_j = spherical_geometry_polygon.SphericalPolygon.from_cone(self.targets[j].region.coords.transform_to('icrs').ra.deg, self.targets[j].region.coords.transform_to('icrs').dec.deg, self.targets[j].region.r.deg, degrees=True)

                elif self.targets[j].region.region_type == 'rectangle':
                    # Create a reference to the target shapley object. This is probably uncessary, and can be sourced directly
                    shapely_j = self.targets[j].region.shapely

                    # Create a set of polygons using the extertiors reported by shapely to create polygons using a convex hull.
                    # This is probably stupid and I should use the actual polygon methods: rectangle circle, etc.
                    # Get the x-y coordinates which define the polygon of the region.
                    x_j, y_j = shapely_j.exterior.coords.xy

                    per_x, per_y = polygon_perimeter(x_j, y_j)
                    c_poly_perimeter = astropy.coordinates.SkyCoord(per_x*astropy.units.degree, per_y*astropy.units.degree, frame=self.targets[i].frame)
                    poly_j = spherical_geometry_polygon.SphericalPolygon.from_radec(c_poly_perimeter.transform_to('icrs').ra.deg, c_poly_perimeter.transform_to('icrs').deg)

                else:
                    # Create a reference to the target shapley object. This is probably uncessary, and can be sourced directly 
                    shapely_j = self.targets[j].region.shapely
                    
                    # Create a set of polygons using the extertiors reported by shapely to create polygons using a convex hull.
                    # This is probably stupid and I should use the actual polygon methods: rectangle circle, etc.
                    # Get the x-y coordinates which define the polygon of the region.
                    x_j, y_j = shapely_j.exterior.coords.xy

                    # Convert the coordinates of the polygon into SkyCoordinates
                    # This logical statemetns that check for the type of coordinate
                    c_j = astropy.coordinates.SkyCoord(x_j*astropy.units.degree, y_j*astropy.units.degree, frame=self.targets[j].frame)

                    # Convert the x-y coordinates, now in SkyCoordinates into polygons in icrs. 
                    # This ensures that independent of what ever coordinate system i or j are in that the comparison is in the correct frame
                    poly_j = spherical_geometry_polygon.SphericalPolygon.from_radec(c_j.transform_to('icrs').ra.deg, c_j.transform_to('icrs').dec.deg)

                # Use the spherical polygon intersection routine, replacing shapely which only works on cartesian grids.
                # short circuit the calculation on the tiles if the shapes do not overlap
                may_overlap = poly_i.intersects_poly(poly_j)

                # Note before proceeding to the next code block, if you are familiar with previous methods, we no longer have a miss match between
                # the region polygon coordinate frame and the coordinate frame of the tiles. Everything is converted to ICRS. This prevents us from having to convert everything
                # to galactic and store values, which took a lot of time for larger targets. This change seems to have off set the cost of looping over all tiles to calculate 
                # if it is contained by another target.

                if may_overlap is True:
                    # shapes overlap, so now find all pointings of j that are within i:
                    lon_j = self.pointings[j][:].ra.deg
                    lat_j = self.pointings[j][:].dec.deg

                    #Initialize array to True. This doesn't matter. We loop over all values anyway, but it's nice.
                    overlap[names[j]][names[i]] = numpy.full(len(self.pointings[j][:].ra),
                                                            False)
                    t_start = time.time()
                    # Check array to see which is false.
                    for k in range(len(lon_j)):
                        contains_True_False = poly_i.contains_radec(lon_j[k], lat_j[k], degrees=True)

                        overlap[names[j]][names[i]][k] = numpy.logical_not(contains_True_False)

                        if contains_True_False and (self.verbos_level >= 2):
                            print("%s x %s overlap at %f, %f"%(self.targets[i].name, self.targets[j].name, lon_j[k], lat_j[k]))
                    
                    if self.verbos_level >=1:
                        print("%s x %s Overlap loop exec time(s)= %f"%(self.targets[i].name, self.targets[j].name, time.time()-t_start))
                
                else:
                    overlap[names[j]][names[i]] = numpy.full(len(self.pointings[j][:].ra),
                                                            True)


                # For functional use, create a global overlap mask, to be used when scheduling
                overlap[names[j]]['global_no_overlap'] &= overlap[names[j]][names[i]]

        return overlap

    def save(self, path, overwrite=False):
        """Saves the results to a file as FITS."""

        assert isinstance(self.schedule, astropy.table.Table), \
            'cannot save empty schedule. Execute Scheduler.run() first.'

        targfile = str(self.targets.filename) if self.targets.filename is not None else 'NA'
        self.schedule.meta['targfile'] = targfile

        self.schedule.meta['tiletype'] = self.tiling_type

        self.schedule.write(path, format='fits', overwrite=overwrite)

    @classmethod
    def load(cls, path, targets=None, observing_plans=None):
        """Creates a new instance from a schedule file.

        Parameters
        ----------
        path : str or ~pathlib.Path
            The path to the schedule file.
        targets : ~lvmsurveysim.target.target.TargetList or path-like
            The `~lvmsurveysim.target.target.TargetList` object associated
            with the schedule file or a path to the target list to load. If
            `None`, the ``TARGFILE`` value stored in the schedule file will be
            used, if possible.
        observing_plans : list of `.ObservingPlan` or None
            A list with the `.ObservingPlan` to use (one for each observatory).

        """

        schedule = astropy.table.Table.read(path)

        targfile = schedule.meta.get('TARGFILE', 'NA')
        targets = targets or targfile

        if not isinstance(targets, lvmsurveysim.target.TargetList):
            assert targets is not None and targets != 'NA', \
                'invalid or unavailable target file path.'

            if not os.path.exists(targets):
                raise LVMSurveySimError(
                    f'the target file {targets!r} does not exists. '
                    'Please, call load with a targets parameter.')

            targets = lvmsurveysim.target.TargetList(target_file=targets)

        observing_plans = observing_plans or []

        tiling_type = schedule.meta.get('TILETYPE', None)
        if tiling_type is None:
            tiling_type = 'hexagonal'
            warnings.warn('No TILETYPE found in schedule file. '
                          'Assuming hexagonal tiling.', LVMSurveySimWarning)

        scheduler = cls(targets, observing_plans=observing_plans)
        scheduler.schedule = schedule

        return scheduler

    def animate_survey(self, filename='lvm_survey.mp4', step=100,
                       observatory=None, projection='mollweide'):
        """Create an animation of the survey progress and save as an mp4 file.

        Parameters
        ----------
        filename : str
            Name of the mp4 file, defaults to ``'lvm_survey.mp4'``.
        step : int
            Number of observations per frame of movie.
        observatory : str
            Either ``'LCO'`` or ``'APO'`` or `None` (plots both).
        projection : str
            Which projection of the sphere to use. Defaults to Mollweide.
        """
        data = self.schedule[self.schedule['target'] != '-']

        if observatory:
            data = data[data['observatory'] == observatory]

        ll = int(len(data) / step)

        x,y = convert_to_mollweide(data['ra'], data['dec'])
        tt = [target.name for target in self.targets]
        g = numpy.array([tt.index(i) for i in data['target']], dtype=float)
        t = data['JD']

        fig, ax = get_axes(projection=projection)
        # scat = ax.scatter(x[:1], y[:1], c=g[:1], s=1, edgecolor=None, edgecolors=None)
        scat = ax.scatter(x, y, c=g % 19, s=0.05, edgecolor=None, edgecolors=None, cmap='tab20')
        # fig.show()
        # return

        def animate(ii):
            if ii % 10 == 0:
                print('%.1f %% done\r' % (ii / ll * 100))
            scat.set_offsets(numpy.stack((x[:ii * step], y[:ii * step]), axis=0).T)
            scat.set_array(g[:ii * step])
            ax.set_title(str(t[ii]))
            return scat,

        anim = animation.FuncAnimation(fig, animate, frames=range(1, ll), interval=1,
                                       blit=True, repeat=False)
        anim.save(filename, fps=24, extra_args=['-vcodec', 'libx264'])

    def plot(self, observatory=None, projection='mollweide', fast=False, annotate=False):
        """Plots the observed pointings.

        Parameters
        ----------
        observatory : str
            Plot only the points for that observatory. Otherwise, plots all
            the pointings.
        projection : str
            The projection to use, either ``'mollweide'`` or ``'rectangular'``.
        fast : bool
            Plot IFU sized and shaped patches if `False`. This is the default.
            Allows accurate zooming and viewing. If `True`, plot scatter-plot
            dots instead of IFUs, for speed sacrificing accuracy.
            This is MUCH faster.
        annotate : bool
            Write the targets' names next to the target coordinates. Implies
            ``fast=True``.

        Returns
        -------
        figure : `matplotlib.figure.Figure`
            The figure with the plot.

        """

        if annotate is True:
            fast = True

        color_cycler = cycler.cycler(bgcolor=['b', 'r', 'g', 'y', 'm', 'c', 'k'])

        fig, ax = get_axes(projection=projection)

        data = self.schedule[self.schedule['target'] != '-']

        if observatory:
            data = data[data['observatory'] == observatory]

        if fast is True:
            if projection == 'mollweide':
                x,y = convert_to_mollweide(data['ra'], data['dec'])
            else:
                x,y = data['ra'], data['dec']
            tt = [target.name for target in self.targets]
            g = numpy.array([tt.index(i) for i in data['target']], dtype=float)
            ax.scatter(x, y, c=g % 19, s=0.05, edgecolor=None, edgecolors=None, cmap='tab20')
            if annotate is True:
                _, text_indices = numpy.unique(g, return_index=True)
                for i in range(len(tt)):
                    plt.text(x[text_indices[i]], y[text_indices[i]], tt[i], fontsize=9)
        else:
            for ii, sty in zip(range(len(self.targets)), itertools.cycle(color_cycler)):

                target = self.targets[ii]
                name = target.name

                target_data = data[data['target'] == name]

                patches = [self.ifu.get_patch(scale=target.telescope.plate_scale,
                                              centre=[pointing['ra'], pointing['dec']],
                                              edgecolor='None', linewidth=0.0,
                                              facecolor=sty['bgcolor'])[0]
                           for pointing in target_data]

                if projection == 'mollweide':
                    patches = [transform_patch_mollweide(ax, patch,
                                                         origin=__MOLLWEIDE_ORIGIN__,
                                                         patch_centre=target_data['ra'][ii])
                               for ii, patch in enumerate(patches)]

                for patch in patches:
                    ax.add_patch(patch)

        if observatory is not None:
            ax.set_title(f'Observatory: {observatory}')

        return fig

    def _create_observing_plans(self):
        """Returns a list of `.ObservingPlan` from the configuration file."""

        observing_plan = []

        for observatory in config['observing_plan']:
            obs_data = config['observing_plan'][observatory]
            start_date = obs_data['start_date']
            end_date = obs_data['end_date']
            observing_plan.append(
                ObservingPlan(start_date, end_date, observatory=observatory))

        return observing_plan

    def run(self, progress_bar=True, **kwargs):
        """Schedules the pointings.

        Parameters
        ----------
        progress_bar : bool
            If `True`, shows a progress bar.
        kwargs : dict
            Parameters to be passed to `~.Scheduler.schedule_one_night`.

        """

        # Make self.schedule a list so that we can add rows. Later we'll make
        # this an Astropy Table.
        self.schedule = []

        # Create some master arrays with all the pointings for convenience.
        s = sorted(self.pointings)

        # An array with the length of all the pointings indicating the index
        # of the target it correspond to.
        index_to_target = numpy.concatenate([numpy.repeat(idx, len(self.pointings[idx]))
                                             for idx in s])

        # All the coordinates
        coordinates = numpy.vstack(
            [numpy.array([self.pointings[idx].ra.deg, self.pointings[idx].dec.deg]).T
             for idx in s])

        # Create an array of the target's priority for each pointing
        priorities = numpy.concatenate([numpy.repeat(self.targets[idx].priority,
                                        len(self.pointings[idx]))
                                        for idx in s])

        # Array with the individual tile priorities
        tile_prio = numpy.concatenate([self.tile_priorities[idx] for idx in s])

        # Array with the total exposure time for each tile
        target_exposure_times = numpy.concatenate(
            [numpy.repeat(self.targets[idx].exptime * self.targets[idx].n_exposures,
                          len(self.pointings[idx]))
             for idx in s])

        # Array with exposure quanta (the minimum time to spend on a tile)
        exposure_quantums = numpy.concatenate(
            [numpy.repeat(self.targets[idx].exptime * self.targets[idx].min_exposures,
                          len(self.pointings[idx]))
             for idx in s])

        # Array with the airmass limit for each pointing
        max_airmass_to_target = numpy.concatenate(
            [numpy.repeat(self.targets[idx].max_airmass, len(self.pointings[idx]))
             for idx in s])

        # Array with the airmass limit for each pointing
        min_moon_to_target = numpy.concatenate(
            [numpy.repeat(self.targets[idx].min_moon_dist, len(self.pointings[idx]))
             for idx in s])

        # Array with the lunation limit for each pointing
        max_lunation = numpy.concatenate(
            [numpy.repeat(self.targets[idx].max_lunation, len(self.pointings[idx]))
             for idx in s])

        # Mask with observed exposure time for each pointing
        observed = numpy.zeros(len(index_to_target), dtype=numpy.float)

        min_date = numpy.min([numpy.min(plan['JD']) for plan in self.observing_plans])
        max_date = numpy.max([numpy.max(plan['JD']) for plan in self.observing_plans])

        dates = range(min_date, max_date + 1)

        if progress_bar:
            generator = astropy.utils.console.ProgressBar(dates)
        else:
            generator = dates

        for jd in generator:

            if progress_bar is False:
                log.info(f'scheduling JD={jd}.')

            for plan in self.observing_plans:

                # Skips JDs not found in the plan or those that don't have good weather.
                if jd not in plan['JD'] or plan[plan['JD'] == jd]['is_clear'][0] == 0:
                    continue

                observed += self.schedule_one_night(
                    jd, plan, index_to_target, max_airmass_to_target,
                    priorities, tile_prio, coordinates, target_exposure_times,
                    exposure_quantums, min_moon_to_target, max_lunation,
                    observed, **kwargs)

        # Convert schedule to Astropy Table.
        self.schedule = astropy.table.Table(
            rows=self.schedule,
            names=['JD', 'observatory', 'target', 'group', 'index', 'ra', 'dec',
                   'pixel', 'nside', 'airmass', 'lunation',
                   'lst', 'exptime', 'totaltime'],
            dtype=[float, 'S10', 'S20', 'S20', int, float, float, int, int, float,
                   float, float, float, float])

    def schedule_one_night(self, jd, plan, index_to_target, max_airmass_to_target,
                           target_priorities, tile_prio, coordinates, target_exposure_times,
                           exposure_quantums, target_min_moon_dist, max_lunation,
                           observed, zenith_avoidance=__ZENITH_AVOIDANCE__):
        """Schedules a single night at a single observatory.

        This method is not intended to be called directly. Instead, use `.run`.

        Parameters
        ----------
        jd : int
            The Julian Date to schedule. Must be included in ``plan``.
        plan : .ObservingPlan
            The observing plan to schedule for the night.
        index_to_target : ~numpy.ndarray
            An array with the length of all the pointings indicating the index
            of the target it correspond to.
        priorities : ~numpy.ndarray
            An array with the length of all the pointings indicating the
            priority of the target.
        tile_prio : ~numpy.ndarray
            An array with the length of all the pointings indicating the
            priority of the individual tiles.
        coordinates : ~astropy.coordinates.SkyCoord
            The coordinates of each one of the pointings, in the ICRS frame.
            The ordering of the coordinates is the same as in ``target_index``.
        target_exposure_times : ~numpy.ndarray
            An array with the length of all pointings with total desired
            exposure time in s for each tile.
        exposure_quantums : ~numpy.ndarray
            An array with the length of all pointings with exposure time in
            seconds to schedule for each visit.
        observed : ~numpy.ndarray
            A float array that carries the executed exposure time for each
            tile.
        max_airmass : float
            The maximum airmass to allow.
        moon_separation : float
            The minimum allowed Moon separation.
        max_lunation : float
            The maximum allowed moon illumination fraction.
        zenith_avoidance : float
            Degrees around the zenith/pole in which we should not observe.
            Defaults to the value ``scheduler.zenith_avoidance``.

        Returns
        -------
        exposure_times : `~numpy.ndarray`
            Array with the exposure times in seconds added to each tile during
            this night.

        """

        observatory = plan.observatory

        lon = plan.location.lon.deg
        lat = plan.location.lat.deg

        maxpriority = max([t.priority for t in self.targets])

        night_plan = plan[plan['JD'] == jd]
        jd0 = night_plan['evening_twilight'][0]
        jd1 = night_plan['morning_twilight'][0]

        # Start at evening twilight
        current_jd = jd0

        # Get the Moon lunation and distance to targets, assume it is constant
        # for the night for speed.
        lunation = night_plan['moon_phase'][0]

        moon_to_pointings = lvmsurveysim.utils.spherical.great_circle_distance(
            night_plan['moon_ra'], night_plan['moon_dec'],
            coordinates[:, 0], coordinates[:, 1])

        # The additional exposure time in this night
        new_observed = observed * 0.0

        # Get the coordinates in radians, this speeds up the altitude calculation
        ac = AltitudeCalculator(coordinates[:, 0], coordinates[:, 1], lon, lat)

        # convert airmass to altitude, we'll work in altitude space for efficiency
        min_alt_for_target = 90.0 - numpy.rad2deg(numpy.arccos(1.0 / max_airmass_to_target))

        # Select targets that are above the max airmass and with good
        # moon avoidance.
        moon_ok = (moon_to_pointings > target_min_moon_dist) & (lunation <= max_lunation)

        # While the current time is before morning twilight ...
        while current_jd < jd1:

            # Get current LST
            current_lst = lvmsurveysim.utils.spherical.get_lst(current_jd, lon)

            # Get the altitude at the start and end of the proposed exposure.
            alt_start = ac(lst=current_lst)
            alt_end = ac(lst=(current_lst + (exposure_quantums / 3600.)))

            # avoid the zenith!
            alt_ok = (alt_start < (90 - zenith_avoidance)) & (alt_end < (90 - zenith_avoidance))

            # Gets valid airmasses (but we're working in altitude space)
            airmass_ok = ((alt_start > min_alt_for_target) & (alt_end > min_alt_for_target))

            # Gets pointings that haven't been completely observed
            exptime_ok = (observed + new_observed) < target_exposure_times

            # Creates a mask of valid pointings with correct Moon avoidance,
            # airmass, zenith avoidance and that have not been completed.
            valid_idx = numpy.where(alt_ok & moon_ok & airmass_ok & exptime_ok)[0]

            # If there's nothing to observe, record the time slot as vacant (for record keeping)
            if len(valid_idx) == 0:
                self._record_observation(current_jd, observatory,
                                         lunation=lunation, lst=current_lst,
                                         exptime=__DEFAULT_TIME_STEP__,
                                         totaltime=__DEFAULT_TIME_STEP__)
                current_jd += __DEFAULT_TIME_STEP__ / 86400.0
                continue

            # Find observations that have nonzero exposure but are incomplete
            incomplete = ((observed + new_observed > 0) &
                          (observed + new_observed < target_exposure_times))

            # Gets the coordinates, altitudes, and priorities of possible pointings.
            valid_alt = alt_start[valid_idx]
            valid_priorities = target_priorities[valid_idx]
            valid_incomplete = incomplete[valid_idx]
            valid_tile_priorities = tile_prio[valid_idx]

            did_observe = False

            # Give incomplete observations the highest priority, imitating a high-priority target,
            # that makes sure these are completed first in all visible targets
            valid_priorities[valid_incomplete] = maxpriority + 1

            # Loops starting with targets with the highest priority (lowest numerical value).
            for priority in numpy.flip(numpy.unique(valid_priorities), axis=0):

                # Gets the indices that correspond to this priority (note that
                # these indices correspond to positions in valid_idx, not in the
                # master list).
                valid_priority_idx = numpy.where(valid_priorities == priority)[0]

                # If there's nothing to do at the current priority, try the next lower
                if len(valid_priority_idx) == 0:
                    continue

                # select all pointings with the current target priority
                valid_alt_target_priority = valid_alt[valid_priority_idx]
                valid_alt_tile_priority = valid_tile_priorities[valid_priority_idx]

                # Find the tiles with the highest tile priority
                max_tile_priority = numpy.max(valid_alt_tile_priority)
                high_priority_tiles = numpy.where(valid_alt_tile_priority == max_tile_priority)[0]

                # Gets the pointing with the highest altitude among the tiles
                # with the highest priority.
                obs_alt_idx = valid_alt_target_priority[high_priority_tiles].argmax()

                obs_tile_idx = high_priority_tiles[obs_alt_idx]
                obs_alt = valid_alt_target_priority[obs_tile_idx]

                # Gets the index of the pointing in the master list.
                observed_idx = valid_idx[valid_priority_idx[obs_tile_idx]]

                # observe it, give it one quantum of exposure
                new_observed[observed_idx] += exposure_quantums[observed_idx]

                # Gets the parameters of the pointing.
                ra = coordinates[observed_idx, 0]
                dec = coordinates[observed_idx, 1]

                target_index = index_to_target[observed_idx]
                target_name = self.targets[target_index].name
                groups = self.targets[target_index].groups
                target_group = groups[0] if groups else 'None'
                target_overhead = self.targets[target_index].overhead

                # Get the index of the first value in index_to_target that matches
                # the index of the target.
                target_index_first = numpy.nonzero(index_to_target == target_index)[0][0]

                # Get the index of the pointing within its target.
                pointing_index = observed_idx - target_index_first

                # Update the table with the schedule.
                exptime = exposure_quantums[observed_idx]
                airmass = 1.0 / numpy.cos(numpy.radians(90.0 - obs_alt))
                self._record_observation(current_jd, observatory,
                                         target_name=target_name,
                                         target_group=target_group,
                                         pointing_index=pointing_index,
                                         ra=ra, dec=dec,
                                         airmass=airmass,
                                         lunation=lunation,
                                         lst=current_lst,
                                         exptime=exptime,
                                         totaltime=exptime * target_overhead)

                did_observe = True
                current_jd += exptime * target_overhead / 86400.0

                break

            if did_observe is False:
                self._record_observation(current_jd, observatory,
                                         lst=current_lst,
                                         exptime=__DEFAULT_TIME_STEP__,
                                         totaltime=__DEFAULT_TIME_STEP__)
                current_jd += (__DEFAULT_TIME_STEP__) / 86400.0

        return new_observed

    def _record_observation(self, jd, observatory, target_name='-', target_group='-',
                            pointing_index=-1, ra=-999., dec=-999.,
                            airmass=-999., lunation=-999., lst=-999.,
                            exptime=0., totaltime=0.):
        """Adds a row to the schedule."""

        self.schedule.append((jd, observatory, target_name, target_group, pointing_index,
                              ra, dec, 0, 0, airmass, lunation, lst, exptime,
                              totaltime))

    def get_target_time(self, tname, group=False, observatory=None, lunation=None,
                        return_lst=False):
        """Returns the JDs or LSTs for a target at an observatory.

        Parameters
        ----------
        tname : str
            The name of the target or group. Use ``'-'`` for unused time.
        group : bool
            If not true, ``tname`` will be the name of a group not a single
            target.
        observatory : str
            The observatory to filter for.
        lunation : list
            Restrict the data to a range in lunation. Defaults to returning
            all lunations. Set to ``[lmin, lmax]`` to return values of
            ``lmin < lunation <= lmax``.
        return_lst : bool
            If `True`, returns an array with the LSTs of all the unobserved
            times.

        Returns
        -------
        table : `~numpy.ndarray`
            An array containing the times the target is observed at an
            observatory, as JDs. If ``return_lst=True`` returns an array of
            the corresponding LSTs.
        """

        column = 'group' if group is True else 'target'
        t = self.schedule[self.schedule[column] == tname]

        if observatory:
            t = t[t['observatory'] == observatory]

        if lunation is not None:
            t = t[(t['lunation'] > lunation[0]) * (t['lunation'] <= lunation[1])]

        if return_lst:
            return t['lst'].data
        else:
            return t['JD'].data

    def print_statistics(self, out_file=None, out_format="ascii", overwrite_out=True, observatory=None, targets=None, return_table=False):
        """Prints a summary of observations at a given observatory.

        Parameters
        ----------
        observatory : str
            The observatory to filter for.
        targets : `~lvmsurveysim.target.TargetList`
            The targets to summarize. If `None`, use ``self.targets``.
        return_table : bool
            If `True`, return a `~astropy.table.Table` with the results.
        out_file : str
            Outfile to write statistics.
        out_format : str
            Outfile format consistent with astropy.table dumps
        """

        if targets is None:
            targets = self.targets

        names = [t.name for t in targets]

        time_on_target = {}          # time spent exposing target
        exptime_on_target = {}       # total time (exp + overhead) on target
        tile_area = {}               # area of a single tile
        target_ntiles = {}           # number of tiles in a target tiling
        target_ntiles_observed = {}  # number of observed tiles
        target_nvisits = {}          # number of visits for each tile
        surveytime = 0.0             # total time of survey
        names.append('-')            # deals with unused time

        for tname, i in zip(names, range(len(names))):

            if (tname != '-'):
                target = self.targets[i]
                tile_area[tname] = target.get_pixarea(ifu=self.ifu)
                target_ntiles[tname] = len(self.pointings[i])
                target_nvisits[tname] = float(target.n_exposures / target.min_exposures)
            else:
                tile_area[tname] = -999
                target_ntiles[tname] = -999
                target_nvisits[tname] = 1

            tdata = self.schedule[self.schedule['target'] == tname]
            if observatory:
                tdata = tdata[tdata['observatory'] == observatory]

            exptime_on_target[tname] = numpy.sum(tdata['exptime'].data)
            target_ntiles_observed[tname] = len(tdata) / target_nvisits[tname]
            target_total_time = numpy.sum(tdata['totaltime'].data)
            time_on_target[tname] = target_total_time
            surveytime += target_total_time

        # targets that completely overlap with others have no tiles
        for t in self.targets:
            if target_ntiles[t.name] == 0:
                print(t.name + ' has no tiles')
                target_ntiles[t.name] = 1

        rows = [
            (t if t != '-' else 'unused',
             numpy.float(target_ntiles[t]),
             numpy.around(target_ntiles_observed[t], decimals=2),
             numpy.around(time_on_target[t] / 3600.0, decimals=2),
             numpy.around(exptime_on_target[t] / 3600.0, decimals=2),
             numpy.around(time_on_target[t] / surveytime, decimals=2),
             numpy.around(target_ntiles_observed[t] * tile_area[t],
                          decimals=2) if t != '-' else -999,
             numpy.around(float(target_ntiles_observed[t]) / float(target_ntiles[t]),
                          decimals=2) if t != '-' else -999)
            for t in names]

        stats = astropy.table.Table(rows=rows,
                                    names=['Target', 'tiles', 'tiles_obs', 'tottime/h',
                                           'exptime/h', 'timefrac', 'area', 'areafrac'],
                                    dtype=('S8', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))

        print('%s :' % (observatory if observatory is not None else 'APO+LCO'))
        stats.pprint(max_lines=-1, max_width=-1)

        if out_file is not None:
            stats.write(out_file, format=out_format, overwrite=overwrite_out)

        if return_table:
            return stats

    def plot_survey(self, observatory=None, bin_size=30., targets=None, groups=None,
                    use_groups=False, use_primary_group=True,
                    show_ungrouped=True, cumulative=False, lst=False,
                    lunation=None,
                    show_unused=True, skip_fast=False, show_mpld3=False):
        """Plot the hours spent on target.

        Parameters
        ----------
        observatory : str
            The observatory to plot. If `None`, all observatories.
        bin_size : int
            The number of days in each bin of the plot.
        targets : list
            A list with the names of the targets to plot. If empty, plots all
            targets.
        groups : list
            A list with the names of the groups to plot. If empty, plots all
            groups.
        use_groups : bool
            If set, the targets are grouped together using the
            ``Target.groups`` list.
        use_primary_group : bool
            If `True`, a target will only be added to its primary group (the
            first one in the group list). Only used when ``use_groups=True``.
        show_ungrouped : bool
            If `True`, targets that don't belong to any group are plotted
            individually. Only used when ``use_groups=True``.
        cumulative : bool or str
            If `True`, plots the cumulative sum of hours spent on each target.
            If ``'group'``, it plots the cumulative on-target hours normalised
            by the total hours needed to observe the target group. If ``'survey'``,
            plots the cumulative hours normalised by the total survey hours.
            When ``cumulative`` is not `False`, ``bin_size`` is set to 1.
        lst : bool
            Whether to bin the used time by LST instead of JD.
        show_unused : bool
            Display the unused time.
        lunation : list
            Range of lunations to include in statistics. Defaults to all lunations.
            Set to ``[lmin, lmax]`` to return values of ``lmin < lunation <= lmax``.
            Can be used to restrict lst usage plots to only bright, grey, or
            dark time.
        skip_fast : bool
            If set, do not plot targets that complete in the first 20% of the
            survey.
        show_mpld3 : bool
            If `True`, opens a browser window with an interactive version of
            the plot.

        Return
        ------
        fig : `~matplotlib.figure.Figure`
            The Matplotlib figure of the plot.
        """

        assert self.schedule is not None, 'you still have not run a simulation.'

        if not targets:
            targets = [target.name for target in self.targets]

        ncols = 2 if len(targets) > 15 else 1

        if lst:
            bin_size = 1. if bin_size == 30. else bin_size
            assert cumulative is False, 'cumulative cannot be used with lst=True.'

        if cumulative is not False:
            bin_size = 1

        fig, ax = plt.subplots(figsize=(12, 8))
        # Leaves a margin on the right to put the legend
        fig.subplots_adjust(right=0.65 if ncols == 2 else 0.8)

        ax.set_prop_cycle(color=['r', 'g', 'b', 'c', 'm', 'y', 'g', 'b', 'c', 'm', 'y', 'r', 'b',
                                 'c', 'm', 'y', 'r', 'g', 'c', 'm', 'y', 'r', 'g', 'b', ],
                          linestyle=['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.',
                                     ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--',
                                     '-.', ':'])

        min_b = (numpy.min(self.schedule['JD']) - 2451545.0) if not lst else 0.0
        max_b = (numpy.max(self.schedule['JD']) - 2451545.0) if not lst else 24.0
        b = numpy.arange(min_b, max_b + bin_size, bin_size)

        # Creates a list of groups to plot. If use_groups=False,
        # this is just the list of targets.
        if not use_groups:
            groups = [target.name for target in self.targets]
        else:
            groups = groups or self.targets.list_groups()
            # Adds the ungrouped targets.
            if show_ungrouped:
                for target in self.targets:
                    if len(target.groups) == 0:
                        groups.append(target.name)

        for group in groups:

            # Cumulated group heights
            group_heights = numpy.zeros(len(b) - 1, dtype=numpy.float)
            group_target_tot_time = 0.0

            # If we are not using groups or the "group"
            # name is that of an ungrouped target.
            if not use_groups or group in self.targets._names:
                targets = [group]
            else:
                targets = self.targets.get_group_targets(group, primary=use_primary_group)

            for tname in targets:

                t = self.targets.get_target(tname)
                tindex = [target.name for target in self.targets].index(tname)

                # plot each target
                tt = self.get_target_time(tname, observatory=observatory,
                                          lunation=lunation, return_lst=lst)

                if len(tt) == 0:
                    continue

                if not lst:
                    tt -= 2451545.0

                heights, bins = numpy.histogram(tt, bins=b)
                heights = numpy.array(heights, dtype=float)
                heights *= t.exptime * t.min_exposures / 3600.0

                target_tot_time = len(self.pointings[tindex]) * t.exptime * t.n_exposures / 3600.

                if skip_fast:
                    completion = heights.cumsum() / target_tot_time
                    if numpy.quantile(completion, 0.2) >= 1:
                        continue

                group_heights += heights
                group_target_tot_time += target_tot_time

            # Only plot the heights if they are not zero. This prevents
            # targets that are not observed at an observatory to be displayed.
            if numpy.sum(group_heights) > 0:
                if cumulative is False:
                    ax.plot(bins[:-1] + numpy.diff(bins) / 2, group_heights, label=group)
                else:
                    ax.plot(bins[:-1] + numpy.diff(bins) / 2, numpy.cumsum(group_heights)/group_target_tot_time, label=group)

        # deal with unused time
        tt = self.get_target_time('-', observatory=observatory, return_lst=lst)
        if not lst:
            tt -= 2451545.0
        heights, bins = numpy.histogram(tt, bins=b)
        heights = numpy.array(heights, dtype=float)
        heights *= __DEFAULT_TIME_STEP__ / 3600.0
        if cumulative:
            heights = heights.cumsum()

        if show_unused and cumulative is False:
            ax.plot(bins[:-1] + numpy.diff(bins) / 2, heights, ':',
                    color='k', label='Unused')

        ax.set_xlabel('JD - 2451545.0' if not lst else 'LST / h')

        if cumulative is False:
            ax.set_ylabel('Hours on target / %.f %s' % ((bin_size, 'days')
                          if not lst else (bin_size, 'h')))
        elif cumulative is True:
            ax.set_ylabel('Hours on target [cumulative]')
        elif cumulative == 'target':
            ax.set_ylabel('Fraction of target completed')
        elif cumulative == 'survey':
            ax.set_ylabel('Fraction of survey time spent on target')

        ax.set_title(observatory if observatory is not None else 'APO+LCO')

        if show_mpld3:

            if mpld3 is None:
                raise ImportError('show_mpld3 requires installing the mpld3 package.')

            handles, labels = ax.get_legend_handles_labels()

            # Resize the figure depending on the number of targets so that the
            # legend is not cut off.
            vsize = len(labels) / 8 * 2.5
            vsize = vsize if vsize > 8 else 8

            fig.set_size_inches(12, vsize)

            # Adjust the bottom of the plot so that it does not vertically
            # grow too much with figure size.
            fig.subplots_adjust(bottom=0.2 + 0.15 * vsize / 8)

            interactive_legend = mpld3.plugins.InteractiveLegendPlugin(
                handles, labels,
                start_visible=False, alpha_unsel=0.4, alpha_over=1.7)

            mpld3.plugins.connect(fig, interactive_legend)

            mpld3.show()

            # Restore figsize and margins
            fig.set_size_inches(12, 8)
            fig.subplots_adjust(bottom=0.1)

        # Move legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), ncol=ncols)

        return fig

    def plot_lunation(self, tname, group=False, observatory=None, dark_limit=0.2):
        """
        plot the lunation distribution for a target. use '-' for unused time

       Parameters
       ----------
        tname : str
            The name of the target or group. Use ``'-'`` for unused time.
        group : bool
            If not true, ``tname`` will be the name of a group not a single
            target.
        observatory : str
            The observatory to filter for.
        dark_limit : float
            Limiting lunation value to count as dark time. Defaults to 0.2.

        Return
        ------
        fig : `~matplotlib.figure.Figure`
            The Matplotlib figure of the plot.
        """
        dark = self.get_target_time(tname, group=group, lunation=[-0.01, dark_limit],
                                    observatory=observatory, return_lst=True)
        bright = self.get_target_time(tname, group=group, lunation=[dark_limit, 1.0],
                                      observatory=observatory, return_lst=True)

        bin_size = 1
        b = numpy.arange(0, 24 + bin_size, bin_size)

        heights_dark, bins = numpy.histogram(dark, bins=b)
        heights_dark = numpy.array(heights_dark, dtype=float)
        heights_bright, bins = numpy.histogram(bright, bins=b)
        heights_bright = numpy.array(heights_bright, dtype=float)

        fig, ax = plt.subplots()
        ax.plot(bins[:-1] + numpy.diff(bins) / 2, heights_dark, label='dark')
        ax.plot(bins[:-1] + numpy.diff(bins) / 2, heights_bright, label='bright')
        ax.legend()
        plt.xlabel('LST')
        plt.ylabel('# of exposures')
        plt.title('unused' if tname == '-' else tname)
        return fig
