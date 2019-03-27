#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-03-10
# @Filename: scheduler.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-27 14:34:42

import itertools

import astropy
import cycler
import healpy
import numpy

import lvmsurveysim.target
import lvmsurveysim.utils.spherical

from lvmsurveysim import IFU, config, log
from lvmsurveysim.utils.plot import __MOLLWEIDE_ORIGIN__, get_axes, plot_ellipse

import matplotlib.pyplot as plt

from .plan import ObservingPlan


__all__ = ['Scheduler', 'AltitudeCalculator']

__ZENITH_AVOIDANCE__ = config['scheduler']['zenith_avoidance']
__DEFAULT_TIME_STEP__ = config['scheduler']['timestep']


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

    Attributes
    ----------
    pointings : dict
        A dictionary with the pointings for each one of the targets in
        ``targets``. It is the direct result of calling
        `TargetList.get_healpix_tiling
        <lvmsurveysim.target.target.TargetList.get_healpix_tiling>`.
    schedule : ~astropy.table.Table
        An astropy table with the results of the scheduling. Includes
        information about the JD of each observation, the target observed,
        the index of the pointing in the target tiling, coordinates, and
        HealPix pixel.

    """

    def __init__(self, targets, observing_plans=None, ifu=None):

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

        self.pointings = targets.get_healpix_tiling(ifu=self.ifu,
                                                    return_coords=True,
                                                    to_frame='icrs')

        self.schedule = None

    def __repr__(self):

        return (f'<Scheduler (observing_plans={len(self.observing_plans)}, '
                f'n_target={len(self.pointings)})>')

    def save(self, path, overwrite=False):
        """Saves the results to a file as FITS."""

        assert isinstance(self.schedule, astropy.table.Table), \
            'cannot save empty schedule. Execute Scheduler.run() first.'

        self.schedule.meta['targets'] = ','.join(self.targets._names)
        self.schedule.write(path, format='fits', overwrite=overwrite)

    @classmethod
    def load(cls, path):
        """Creates a new instance from a schedule file."""

        schedule = astropy.table.Table.read(path)

        target_names = schedule.meta['TARGETS'].split(',')
        targets = lvmsurveysim.target.TargetList(
            [lvmsurveysim.target.Target.from_list(target_name)
             for target_name in target_names])

        scheduler = cls(targets, observing_plans=[])
        scheduler.schedule = schedule

        return scheduler

    def plot(self, observatory=None):
        """Plots the observed pointings.

        Parameters
        ----------
        observatory : str
            Plot only the points for that observatory. Otherwise, plots all
            the pointings.

        """

        color_cycler = cycler.cycler(bgcolor=['b', 'r', 'g', 'y', 'm', 'c', 'k'])

        fig, ax = get_axes(projection='mollweide')

        data = self.schedule[self.schedule['ra'] > 0.]

        if observatory:
            data = data[data['observatory'] == observatory]

        for ii, sty in zip(range(len(self.targets)), itertools.cycle(color_cycler)):

            target = self.targets[ii]
            name = target.name
            nside = target._get_nside(ifu=self.ifu)

            radius = healpy.max_pixrad(nside, degrees=True)

            target_data = data[data['target'] == name]

            plot_ellipse(ax, target_data['ra'], target_data['dec'],
                         width=radius, origin=__MOLLWEIDE_ORIGIN__, **sty)

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

    def run(self, progress_bar=False, **kwargs):
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

        # Create an array of pointing to priority.
        priorities = numpy.concatenate([numpy.repeat(self.targets[idx].priority,
                                        len(self.pointings[idx]))
                                        for idx in s])

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
                    priorities, coordinates, target_exposure_times,
                    exposure_quantums, min_moon_to_target, max_lunation,
                    observed, **kwargs)

        # Convert schedule to Astropy Table.
        self.schedule = astropy.table.Table(
            rows=self.schedule,
            names=['JD', 'observatory', 'target', 'index', 'ra', 'dec',
                   'pixel', 'nside', 'airmass', 'lunation',
                   'lst', 'exptime', 'totaltime'],
            dtype=[float, 'S10', 'S20', int, float, float, int, int, float,
                   float, float, float, float])

    def schedule_one_night(self, jd, plan, index_to_target, max_airmass_to_target,
                           target_priorities, coordinates, target_exposure_times,
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

        # Get the coordinates in radians, this speeds up then altitude calculation
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

            # Gets the coordinates and priorities of valid pointings.
            valid_alt = alt_start[valid_idx]
            valid_priorities = target_priorities[valid_idx]
            valid_incomplete = incomplete[valid_idx]

            did_observe = False

            # Give incomplete observations the highest priority
            valid_priorities[valid_incomplete] = maxpriority + 1

            # Loops starting with pointings with the highest priority.
            for priority in range(valid_priorities.max(), valid_priorities.min() - 1, -1):

                # Gets the indices that correspond to this priority (note that
                # these indices correspond to positions in valid_idx, not in the
                # master list).
                valid_priority_idx = numpy.where(valid_priorities == priority)[0]

                # If there's nothing to do at the current priority, try the next lower
                if len(valid_priority_idx) == 0:
                    continue

                valid_alt_priority = valid_alt[valid_priority_idx]

                # Gets the pointing with the highest altitude.
                obs_alt_idx = valid_alt_priority.argmax()
                obs_alt = valid_alt_priority[obs_alt_idx]

                # Gets the index of the pointing in the master list.
                observed_idx = valid_idx[valid_priority_idx[obs_alt_idx]]

                # observe it, give it one quantum of exposure
                new_observed[observed_idx] += exposure_quantums[observed_idx]

                # Gets the parameters of the pointing.
                ra = coordinates[observed_idx, 0]
                dec = coordinates[observed_idx, 1]

                target_index = index_to_target[observed_idx]
                target_name = self.targets[target_index].name
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

    def _record_observation(self, jd, observatory, target_name='-',
                            pointing_index=-1, ra=-999., dec=-999.,
                            airmass=-999., lunation=-999., lst=-999.,
                            exptime=0., totaltime=0.):
        """Adds a row to the schedule."""

        self.schedule.append((jd, observatory, target_name, pointing_index,
                              ra, dec, 0, 0, airmass, lunation, lst, exptime,
                              totaltime))

    def get_target_time(self, tname, observatory=None, return_lst=False):
        """Returns the JDs or LSTs for a target at an observatory.

        Parameters
        ----------
        tname : str
            The name of the target. Use ``'-'`` for unused time.
        observatory : str
            The observatory to filter for.
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

        t = self.schedule[self.schedule['target'] == tname]

        if observatory:
            t = t[t['observatory'] == observatory]

        if return_lst:
            return t['lst'].data
        else:
            return t['JD'].data

    def print_statistics(self, observatory=None, targets=None):
        """Prints a summary of observations at a given observatory.

        Parameters
        ----------
        observatory : str
            The observatory to filter for.
        targets : `~lvmsurveysim.target.TargetList`
            The targets to summarize. If `None`, use ``self.targets``.

        """

        if targets is None:
            targets = self.targets

        time_on_target = {}          # time spent exposing target
        exptime_on_target = {}       # total time (exp + overhead) on target
        tile_area = {}               # area of a single tile
        target_ntiles = {}           # number of tiles in a target tiling
        target_ntiles_observed = {}  # number of observed tiles
        target_nvisits = {}          # number of visits for each tile
        surveytime = 0.0             # total time of survey
        names = [t.name for t in targets]
        names.append('-')            # deals with unused time

        for tname, i in zip(names, range(len(names))):
            if (tname != '-'):
                target = self.targets[i]
                tile_area[tname] = target.get_pixarea(ifu=self.ifu)
                target_ntiles[tname] = len(self.pointings[i])
                target_nvisits[tname] = float(target.n_exposures / target.min_exposures)
            else:
                tile_area[tname] = 1.0
                target_ntiles[tname] = 1.0
                target_nvisits[tname] = 1.0
            tdata = self.schedule[self.schedule['target'] == tname]
            if observatory:
                tdata = tdata[tdata['observatory'] == observatory]
            target_exptime = numpy.sum(tdata['exptime'].data)
            target_ntiles_observed[tname] = len(tdata) / target_nvisits[tname]
            target_total_time = numpy.sum(tdata['totaltime'].data)
            exptime_on_target[tname] = target_exptime
            time_on_target[tname] = target_total_time
            surveytime += target_total_time

        print('%s :' % (observatory if observatory is not None else 'APO+LCO'))
        print('%10s\t%7s\t%8s %10s %10s %10s' % ('Target', 'tottime/h', 'exptime/h',
                                                 'timefrac', 'area', 'areafrac'))
        print('--------------------------------------------------------------------------------')
        for t in names:
            print('%10s\t%.2f\t\t%.2f\t\t%.2f\t\t%f\t\t%.2f' % (
                t if t != '-' else 'unused',
                time_on_target[t] / 3600.0,
                exptime_on_target[t] / 3600.0,
                time_on_target[t] / surveytime,
                target_ntiles[t] * tile_area[t],
                float(target_ntiles_observed[t]) / float(target_ntiles[t])))

    def plot_survey(self, observatory, bin_size=30):
        """Plot the hours spent on target.

        Parameters
        ----------
        observatory : str
            The observatory to plot.
        bin_size : int
            The number of days in each bin of the plot.

        """

        assert self.schedule is not None, 'you still have not run a simulation.'

        fig, ax = plt.subplots()

        for t in self.targets:

            tt = self.get_target_time(t.name, observatory=observatory)

            min_jd = numpy.min(self.schedule['JD'])
            max_jd = numpy.max(self.schedule['JD'])

            b = numpy.arange(min_jd, max_jd + bin_size, bin_size)

            heights, bins = numpy.histogram(tt, bins=b)
            heights = numpy.array(heights, dtype=float)
            heights *= t.exptime * t.n_exposures / 3600.0

            ax.plot(bins[:-1] + numpy.diff(bins) / 2, heights, '-', label=t.name)
            ax.set_xlabel('JD')
            ax.set_ylabel('hours on target per 30 days')
            ax.set_title(observatory)

        ax.legend()
        fig.show()
