#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-03-10
# @Filename: scheduler.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-12 19:01:11

import itertools

import astropy
import cycler
import healpy
import numpy

import lvmsurveysim.target
import lvmsurveysim.utils.spherical
from lvmsurveysim import IFU, config, log
from lvmsurveysim.utils.plot import __MOLLWEIDE_ORIGIN__, get_axes, plot_ellipse

from .plan import ObservingPlan


__all__ = ['Scheduler']

__MAX_AIRMASS__ = config['scheduler']['max_airmass']
__MOON_SEPARATION__ = config['scheduler']['moon_separation']
__EXPOSURE_TIME__ = config['scheduler']['exposure_time']
__OVERHEAD__ = config['scheduler']['overhead']
__ZENITH_AVOIDANCE__ = config['scheduler']['zenith_avoidance']


class Scheduler(object):
    """Schedules a list of targets following and observing plan.

    Parameters
    ----------
    targets : ~lvmsurveysim.target.target.TargetList
        The `~lvmsuveysim.target.target.TargetList` object with the list of
        targets to schedule.
    observing_plan : list of `.ObservingPlan` or None
        A list with the `.ObservingPlan` to use (one for each observatory).
        If `None`, the list will be created from the ``observing_plan``
        section in the configuration file.
    ifu : ~lvmsurveysim.ifu.IFU
        The `~lvmsurveysim.ifu.IFU` to use.

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

        self.schedule = astropy.table.Table(
            None, names=['JD', 'observatory', 'target', 'index', 'ra', 'dec',
                         'pixel', 'nside', 'airmass'],
            dtype=[float, 'S10', 'S20', int, float, float, int, int, float])

    def __repr__(self):

        return (f'<Scheduler (observing_plans={len(self.observing_plans)}, '
                f'n_target={len(self.pointings)})>')

    def save(self, path, overwrite=False):
        """Saves the results to a file as FITS."""

        self.schedule.meta['targets'] = ','.join(self.targets._names)
        self.schedule.write(path, format='fits', overwrite=overwrite)

    @classmethod
    def load(cls, path):

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
                         width=radius, org=__MOLLWEIDE_ORIGIN__, **sty)

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

        # Create some master arrays with all the pointings for convenience.

        # An array with the length of all the pointings indicating the index
        # of the target it correspond to.
        index_to_target = numpy.concatenate([numpy.repeat(idx, len(self.pointings[idx]))
                                             for idx in sorted(self.pointings)])

        # All the coordinates
        coordinates = numpy.vstack(
            [numpy.array([self.pointings[idx].ra.deg, self.pointings[idx].dec.deg]).T
             for idx in sorted(self.pointings)])

        # Create an array of pointing to priority.
        priorities = numpy.concatenate([numpy.repeat(self.targets[idx].priority,
                                                     len(self.pointings[idx]))
                                        for idx in sorted(self.pointings)])

        # Mask with observed pointings
        observed = numpy.zeros(len(index_to_target), dtype=numpy.bool)

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

                if jd not in plan['JD']:
                    continue

                new_observed = self.schedule_one_night(jd, plan, index_to_target,
                                                       priorities, coordinates,
                                                       observed, **kwargs)

                observed |= new_observed

    def schedule_one_night(self, jd, plan, index_to_target, priorities,
                           coordinates, observed,
                           max_airmass=__MAX_AIRMASS__,
                           moon_separation=__MOON_SEPARATION__,
                           exposure_time=__EXPOSURE_TIME__,
                           overhead=__OVERHEAD__,
                           zenith_avoidance=__ZENITH_AVOIDANCE__,
                           follow_target=False):
        """Schedules a single night in a single observatory.

        This method is not intended to be called directly. Instead, use
        `.run`.

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
        observed : ~numpy.ndarray
            A boolean array that serves as a mask for the pointings that have
            been observed.
        max_airmass : float
            The maximum airmass to allow. Defaults to the value
            ``scheduler.max_airmass`` in the configuration file.
        moon_separation : float
            The minimum allowed Moon separation. Defaults to the value
            ``scheduler.min_moon_separation`` in the configuration file.
        exposure_time : float
            Exposure time to complete each pointing, in seconds. Defaults to
            the value ``scheduler.exposure_time``.
        overhead : float
            The overhead due to operations procedures (slewing, calibrations,
            etc). Defaults to the value ``scheduler.overhead``.
        zenith_avoidance : float
            Degrees around the zenith/pole in which we should not observe.
            Defaults to the value ``scheduler.zenith_avoidance``.
        follow_target : bool
            Whether to try to stick with a target once it has started to be
            observed that night, as long as it remains within the limits of
            Moon avoidance and airmass. Otherwise the code will select the
            most optimal pointing regarding of target (but respecting
            priorities).

        """

        # Mask to mark pointings observed tonight
        new_observed = numpy.zeros(len(observed), dtype=numpy.bool)

        observatory = plan.observatory

        lon = plan.location.lon.deg
        lat = plan.location.lat.deg

        night_plan = plan[plan['JD'] == jd]
        jd0 = night_plan['evening_twilight'][0]
        jd1 = night_plan['morning_twilight'][0]

        eff_exp_time = exposure_time * overhead / 86400.

        times = numpy.arange(jd0, jd1 + eff_exp_time, eff_exp_time)
        moons = astropy.coordinates.get_moon(time=astropy.time.Time(times, format='jd'))

        # Iterates until the night is done.
        for ii in range(len(times)):

            jd = times[ii]
            moon = moons[ii]

            moon_to_pointings = lvmsurveysim.utils.spherical.great_circle_distance(
                moon.ra.deg, moon.dec.deg, coordinates[:, 0], coordinates[:, 1])

            # Select targets that are above the max airmass and with good
            # Moon avoidance.
            moon_ok = moon_to_pointings > moon_separation

            airmasses = lvmsurveysim.utils.spherical.get_altitude(
                coordinates[:, 0], coordinates[:, 1], jd=jd,
                lon=lon, lat=lat, airmass=True)

            # Gets valid airmasses
            airmass_ok = (airmasses < max_airmass) & (airmasses > 0)

            # Creates a mask of valid pointings with correct Moon avoidance,
            # airmass, and that have not been observed.
            valid_idx = numpy.where(moon_ok & airmass_ok & ~observed & ~new_observed)[0]

            if len(valid_idx) == 0:
                self._record_observation(jd, observatory)
                continue

            # Gets the coordinates and priorities of valid pointings.
            valid_airmasses = airmasses[valid_idx]
            valid_priorities = priorities[valid_idx]

            already_observed = False

            # Loops starting with pointings with the highest priority.
            for priority in range(valid_priorities.max(), valid_priorities.min() - 1, -1):

                # If we have already observed a pointing at this time, exit.
                if already_observed:
                    break

                # Gets the indices that correspond to this priority (note that
                # these indices correspond to positions in valid_idx, not in the
                # master list).
                valid_priority_idx = numpy.where(valid_priorities == priority)[0]

                if len(valid_priority_idx) == 0:
                    self._record_observation(jd, observatory)
                    continue

                valid_airmasses_priority = valid_airmasses[valid_priority_idx]

                # Gets the pointing with the smallest airmass.
                obs_airmass_idx = valid_airmasses_priority.argmin()
                obs_airmass = valid_airmasses_priority.min()

                # Gets the index of the pointing in the master list.
                observed_idx = valid_idx[valid_priority_idx[obs_airmass_idx]]

                # Mark the pointing as observed.
                new_observed[observed_idx] = True

                # Gets the parameters of the pointing.
                ra = coordinates[observed_idx, 0]
                dec = coordinates[observed_idx, 1]

                target_index = index_to_target[observed_idx]
                target_name = self.targets[target_index].name

                # Get the index of the first value in index_to_target that matches
                # the index of the target.
                target_index_first = numpy.nonzero(target_index == target_index)[0][0]

                # Get the index of the pointing within its target.
                pointing_index = observed_idx - target_index_first

                # Update the table with the schedule.
                self._record_observation(jd, observatory,
                                         target_name=target_name,
                                         pointing_index=pointing_index,
                                         ra=ra, dec=dec, airmass=obs_airmass)

                already_observed = True

        return new_observed

    def _record_observation(self, jd, observatory, target_name='-',
                            pointing_index=-1, ra=-999., dec=-999.,
                            airmass=-999.):

        self.schedule.add_row((jd, observatory, target_name, pointing_index,
                               ra, dec, 0, 0, airmass))
