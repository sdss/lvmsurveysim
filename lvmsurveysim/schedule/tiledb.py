#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Date: 2021-08-06
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


#
#
#


import astropy
import numpy

import lvmsurveysim.target
from lvmsurveysim import IFU, config, log
from lvmsurveysim.exceptions import LVMSurveySimError, LVMSurveySimWarning
from lvmsurveysim.schedule.plan import ObservingPlan

numpy.seterr(invalid='raise')


__all__ = ['TileDB']


class TileDB(object):
    """Interfaces a database holding a list of tiles to observe

    Parameters
    ----------

    Attributes
    ----------
    schedule : ~astropy.table.Table
        An astropy table with the results of the scheduling. Includes
        information about the JD of each observation, the target observed,
        the index of the pointing in the target tiling, coordinates, etc.

    """

    def __init__(self):
        self.index_to_target = []
        self.max_airmass_to_target = []
        self.min_shadowheight_to_target = []
        self.target_priorities = []
        self.tile_prio = []
        self.coordinates = []
        self.target_exposure_times = []
        self.exposure_quantums = []
        self.target_min_moon_dist = []
        self.max_lunation = []
        self.observed = []

    def init(self, index_to_target, max_airmass_to_target, min_shadowheight_to_target,
                           target_priorities, tile_prio, coordinates, target_exposure_times,
                           exposure_quantums, target_min_moon_dist, max_lunation,
                           observed):
        self.index_to_target = index_to_target
        self.max_airmass_to_target = max_airmass_to_target
        self.min_shadowheight_to_target = min_shadowheight_to_target
        self.target_priorities = target_priorities
        self.tile_prio = tile_prio
        self.coordinates = coordinates
        self.target_exposure_times = target_exposure_times
        self.exposure_quantums = exposure_quantums
        self.target_min_moon_dist = target_min_moon_dist
        self.max_lunation = max_lunation
        self.observed = observed

    def init_db(path):
        pass

    def load_from_db(path):
        pass

    def update_tile():
        pass

    def remove_tile():
        pass

    def __repr__(self):
        return (f'<TileDB (N_tiles={len(self.index_to_target)})>')


    def save(self, path, overwrite=False):
        """Saves the results to two files one FITS the other NPY.
        The FITS file contains the schedule while the npy file contains
        the overlap regions. The latter take too long to compute ..."""

        assert isinstance(self.schedule, astropy.table.Table), \
            'cannot save empty schedule. Execute Scheduler.run() first.'

        targfile = str(self.targets.filename) if self.targets.filename is not None else 'NA'
        self.schedule.meta['targfile'] = targfile

        self.schedule.meta['tiletype'] = self.tiling_type

        self.schedule.write(path+'.fits', format='fits', overwrite=overwrite)
        numpy.save(path+'.npy', self.overlap)

    @classmethod
    def load(cls, path, targets=None, observing_plans=None, verbos_level=0):
        """Creates a new instance from a schedule file.

        Parameters
        ----------
        path : str or ~pathlib.Path
            The path to the schedule file and the basename, no extension. The 
            routine expects to find path.fits and path.npy
        targets : ~lvmsurveysim.target.target.TargetList or path-like
            The `~lvmsurveysim.target.target.TargetList` object associated
            with the schedule file or a path to the target list to load. If
            `None`, the ``TARGFILE`` value stored in the schedule file will be
            used, if possible.
        observing_plans : list of `.ObservingPlan` or None
            A list with the `.ObservingPlan` to use (one for each observatory).
        verbose_level : int
            Verbosity level to pass to constructor of Schedule
        """

        schedule = astropy.table.Table.read(path+'.fits')

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

        overlap = numpy.load(path+'.npy', allow_pickle='TRUE').item()

        scheduler = cls(targets, observing_plans=observing_plans, overlap=overlap, verbos_level=verbos_level)
        scheduler.schedule = schedule

        return scheduler

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
        min_shadowheight_to_target = numpy.concatenate(
            [numpy.repeat(self.targets[idx].min_shadowheight, len(self.pointings[idx]))
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
                    jd, plan, index_to_target, max_airmass_to_target, min_shadowheight_to_target,
                    priorities, tile_prio, coordinates, target_exposure_times,
                    exposure_quantums, min_moon_to_target, max_lunation,
                    observed, **kwargs)

        # Convert schedule to Astropy Table.
        self.schedule = astropy.table.Table(
            rows=self.schedule,
            names=['JD', 'observatory', 'target', 'group', 'index', 'ra', 'dec',
                'pixel', 'nside', 'airmass', 'lunation', 'shadow_height', "moon_dist",
                'lst', 'exptime', 'totaltime'],
            dtype=[float, 'S10', 'S20', 'S20', int, float, float, int, int, float,
                float, float, float, float, float, float])

