#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import itertools

import astropy
import cycler
import matplotlib.pyplot as plt
import numpy
from matplotlib import animation

import lvmsurveysim.target
from lvmsurveysim.schedule.tiledb import TileDB
from lvmsurveysim.schedule.scheduler import Scheduler
from lvmsurveysim.schedule.plan import ObservingPlan
from lvmsurveysim import IFU, config
from lvmsurveysim.schedule.plan import ObservingPlan
from lvmsurveysim.utils.plot import __MOLLWEIDE_ORIGIN__, get_axes, transform_patch_mollweide, convert_to_mollweide


numpy.seterr(invalid='raise')


__all__ = ['Simulator']


class Simulator(object):
    """Simulates an observing schedule for a list of targets (tile database) following and observing plan.

    Parameters
    ----------
    tiledb : ~lvmsurveysim.schedule.tiledb.TileDB
        The `~lvmsurveysim.schedule.tiledb.TileDB` instance with the table of
        tiles to schedule.
    observing_plan : l`.ObservingPlan` or None
        The `.ObservingPlan` to use (one for each observatory).
        If `None`, it will be created from the ``observing_plan``
        section in the configuration file. Contains dates and sun/moon data for the 
        duration of the survey as well as Observatory data.
    ifu : ~lvmsurveysim.ifu.IFU
        The `~lvmsurveysim.ifu.IFU` to use. Defaults to the one from the
        configuration file. Used only for plotting the survey footprint.

    Attributes
    ----------
    tiledb : ~lvmsurveysim.schedule.tiledb.TileDB
        Instance of the tile database to observe.
    schedule : ~astropy.table.Table
        An astropy table with the results of the scheduling. Includes
        information about the JD of each observation, the target observed,
        the index of the pointing in the target tiling, coordinates, etc.
    """

    def __init__(self, tiledb, observing_plan=None, ifu=None):

        assert isinstance(tiledb, lvmsurveysim.schedule.tiledb.TileDB), \
            'tiledb must be a lvmsurveysim.schedule.tiledb.TileDB instances.'

        # get rid of the special tiles, we do not need them for the simulator
        tdb = tiledb.tile_table
        tiledb.tile_table = tdb[numpy.where(tdb['TileID'] >= tiledb.tileid_start)[0]]

        if observing_plan is None:
            observing_plan = self._create_observing_plan()

        assert isinstance(observing_plan, ObservingPlan), 'observing_plan is not an instance of ObservingPlan.'

        self.zenith_avoidance = config['scheduler']['zenith_avoidance']
        self.time_step = config['scheduler']['timestep']

        self.observing_plan = observing_plan
        self.tiledb = tiledb
        self.targets = tiledb.targets
        self.ifu = ifu or IFU.from_config()

        self.schedule = None

    def __repr__(self):
        return (f'<Simulator (observing_plan={self.observing_plan.observatory.name}, '
                f'tiles={len(self.tiledb.tile_table)})>')


    def save(self, path, overwrite=False):
        """
        Saves the results of the scheduling simulation to a FITS file.
        """
        assert isinstance(self.schedule, astropy.table.Table), \
            'cannot save empty schedule. Execute Scheduler.run() first.'

        targfile = str(self.targets.filename) if self.targets.filename != None else 'NA'
        self.schedule.meta['targfile'] = targfile

        self.schedule.write(path+'.fits', format='fits', overwrite=overwrite)

    @classmethod
    def load(cls, path, tiledb=None, observing_plan=None):
        """Creates a new instance from a schedule file.

        Parameters
        ----------
        path : str or ~pathlib.Path
            The path to the schedule file and the basename, no extension. The 
            routine expects to find path.fits and path.npy
        tiledb : ~lvmsurveysim.schedule.tiledb.TileDB or path-like
            Instance of the tile database to observe.
        observing_plan : `.ObservingPlan` or None
            The `.ObservingPlan` to use (one for each observatory).
        """

        schedule = astropy.table.Table.read(path+'.fits')

        if not isinstance(tiledb, lvmsurveysim.schedule.tiledb.TileDB):
            assert tiledb != None and tiledb != 'NA', \
                'invalid or unavailable tiledb file path.'

            tiledb = TileDB.load(tiledb)

        observing_plan = observing_plan or []

        sim = cls(tiledb, observing_plan=observing_plan)
        sim.schedule = schedule

        return sim


    def run(self, progress_bar=True):
        """Schedules the pointings for the whole survey defined 
        in the observing plan.

        Parameters
        ----------
        progress_bar : bool
            If `True`, shows a progress bar.

        """

        # Make self.schedule a list so that we can add rows. Later we'll make
        # this an Astropy Table.
        self.schedule = []

        plan = self.observing_plan

        # Instance of the Scheduler
        scheduler = Scheduler(plan)

        # observed exposure time for each pointing
        observed = numpy.zeros(len(self.tiledb.tile_table), dtype=numpy.float)

        # range of dates for the survey
        min_date = numpy.min(plan['JD'])
        max_date = numpy.max(plan['JD'])
        dates = range(min_date, max_date + 1)

        if progress_bar:
            generator = astropy.utils.console.ProgressBar(dates)
        else:
            generator = dates

        for jd in generator:

            if progress_bar is False:
                print(f'scheduling JD={jd}.')

            # Skips JDs not found in the plan or those that don't have good weather.
            if jd not in plan['JD'] or plan[plan['JD'] == jd]['is_clear'][0] == 0:
                continue

            # 'observed' is updated to reflect exposures taken that night
            self.schedule_one_night(jd, scheduler, observed)

        # Convert schedule to Astropy Table.
        self.schedule = astropy.table.Table(
            rows=self.schedule,
            names=['JD', 'observatory', 'target', 'group', 'tileid', 'index', 'ra', 'dec', 'pa', 
                'airmass', 'lunation', 'shadow_height', "moon_dist", 'lst', 'exptime', 'totaltime'],
            dtype=[float, 'S10', 'S20', 'S20', int, int, float, float, float, 
                   float, float, float, float, float, float, float])


    def schedule_one_night(self, jd, scheduler, observed):
        """Schedules a single night at a single observatory.

        This method is not intended to be called directly. Instead, use `.run`.

        Parameters
        ----------
        jd : int
            The Julian Date to schedule. Must be included in ``plan``.
        scheduler : .Scheduler
            The Scheduler instance that will determine the observing sequence.
        observed : ~numpy.array
            An array of the length of the tiledb that records the observing time
            accumulated on each tile thus far. This array is updated by this function
            as exposures are scheduled.

        """

        # initialize the scheduler for the night
        scheduler.prepare_for_night(jd, self.observing_plan, self.tiledb)

        # shortcut
        tdb = self.tiledb.tile_table

        # begin at twilight
        current_jd = scheduler.evening_twi

        # While the current time is before morning twilight ...
        while current_jd < scheduler.morning_twi:

            # obtain the next tile to observe
            observed_idx, current_lst, hz, alt, lunation = scheduler.get_optimal_tile(current_jd, observed)
            if observed_idx == -1:
                # nothing available
                self._record_observation(current_jd, self.observing_plan.observatory,
                                         lst=current_lst,
                                         exptime=self.time_step,
                                         totaltime=self.time_step)
                current_jd += (self.time_step) / 86400.0
                continue

            # observe it, give it one quantum of exposure
            exptime = tdb['VisitExptime'].data[observed_idx]
            observed[observed_idx] += exptime

            # collect observation data to put in table
            tileid_observed = tdb['TileID'].data[observed_idx]
            target_index = tdb['TargetIndex'].data[observed_idx]
            target_name = self.targets[target_index].name
            groups = self.targets[target_index].groups
            target_group = groups[0] if groups else 'None'
            target_overhead = self.targets[target_index].overhead

            # Get the index of the first value in index_to_target that matches
            # the index of the target.
            target_index_first = numpy.nonzero(tdb['TargetIndex'].data == target_index)[0][0]
            # Get the index of the pointing within its target.
            pointing_index = observed_idx - target_index_first
            
            # Record angular distance to moon
            dist_to_moon = scheduler.moon_to_pointings[observed_idx]

            # Update the table with the schedule.
            airmass = 1.0 / numpy.cos(numpy.radians(90.0 - alt))
            self._record_observation(current_jd, self.observing_plan.observatory,
                                        target_name=target_name,
                                        target_group=target_group,
                                        tileid = tileid_observed,
                                        pointing_index=pointing_index,
                                        ra=tdb['RA'].data[observed_idx], 
                                        dec=tdb['DEC'].data[observed_idx],
                                        pa=tdb['PA'].data[observed_idx],
                                        airmass=airmass,
                                        lunation=lunation,
                                        shadow_height= hz, #hz[valid_priority_idx[obs_tile_idx]],
                                        dist_to_moon=dist_to_moon,
                                        lst=current_lst,
                                        exptime=exptime,
                                        totaltime=exptime * target_overhead)

            current_jd += exptime * target_overhead / 86400.0


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
            ax.set_title(str(numpy.around(t[ii * step], decimals=1)))
            return scat,

        anim = animation.FuncAnimation(fig, animate, frames=range(1, ll), interval=1,
                                       blit=True, repeat=False)
        anim.save(filename, fps=24, extra_args=['-vcodec', 'libx264'])


    def plot(self, observatory=None, projection='mollweide', tname=None, fast=False, annotate=False, edge=False):
        """Plots the observed pointings.

        Parameters
        ----------
        observatory : str
            Plot only the points for that observatory. Otherwise, plots all
            the pointings.
        projection : str
            The projection to use, either ``'mollweide'`` or ``'rectangular'``.
        tname : str
            Select only a particular target name to plot
        fast : bool
            Plot IFU sized and shaped patches if `False`. This is the default.
            Allows accurate zooming and viewing. If `True`, plot scatter-plot
            dots instead of IFUs, for speed sacrificing accuracy.
            This is MUCH faster.
        annotate : bool
            Write the targets' names next to the target coordinates. Implies
            ``fast=True``.
        edge : bool
            Draw tile edges and make tiles partly transparent to better judge overlap.
            Makes zoomed-out view look odd, so use default False.

        Returns
        -------
        figure : `matplotlib.figure.Figure`
            The figure with the plot.

        """

        if annotate is True:
            fast = True

        color_cycler = cycler.cycler(bgcolor=['b', 'r', 'g', 'y', 'm', 'c', 'k'])

        fig, ax = get_axes(projection=projection)

        if tname != None:
            data = self.schedule[self.schedule['target'] == tname]
        else:
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

                if edge:
                    patches = [self.ifu.get_patch(scale=target.telescope.plate_scale, centre=[p['ra'], p['dec']], pa=p['pa'],
                                                  edgecolor='k', linewidth=1, alpha=0.5, facecolor=sty['bgcolor'])[0]
                            for p in target_data]
                else:
                    patches = [self.ifu.get_patch(scale=target.telescope.plate_scale, centre=[p['ra'], p['dec']], pa=p['pa'],
                                                  edgecolor='None', linewidth=0.0, facecolor=sty['bgcolor'])[0]
                            for p in target_data]


                if projection == 'mollweide':
                    patches = [transform_patch_mollweide(patch) for patch in patches]

                for patch in patches:
                    ax.add_patch(patch)

        if observatory != None:
            ax.set_title(f'Observatory: {observatory}')

        return fig

    def _create_observing_plan(self):
        """Returns an `.ObservingPlan` from the configuration file."""

        observatory = config['observing_plan']
        obs_data = config['observing_plan'][observatory]
        start_date = obs_data['start_date']
        end_date = obs_data['end_date']
        return ObservingPlan(start_date, end_date, observatory=observatory)


    def _record_observation(self, jd, observatory, target_name='-', target_group='-',
                            tileid=-1, pointing_index=-1, ra=-999., dec=-999., pa=-999., 
                            airmass=-999., lunation=-999., shadow_height=-999., dist_to_moon=-999.,
                            lst=-999.,
                            exptime=0., totaltime=0.):
        """Adds a row to the schedule."""

        self.schedule.append((jd, observatory, target_name, target_group, tileid, pointing_index,
                              ra, dec, pa, airmass, lunation, shadow_height, dist_to_moon, lst, exptime,
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

        if lunation != None:
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
                target_ntiles[tname] = len(numpy.where(self.tiledb.tile_table['TargetIndex'] == i)[0])
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

        print('%s :' % (observatory if observatory != None else 'APO+LCO'))
        stats.pprint(max_lines=-1, max_width=-1)

        if out_file != None:
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

        Return
        ------
        fig : `~matplotlib.figure.Figure`
            The Matplotlib figure of the plot.
        """

        assert self.schedule != None, 'you still have not run a simulation.'

        if not targets:
            targets = [target.name for target in self.targets]

        ncols = 2 if len(targets) > 15 else 1

        if lst:
            bin_size = 1. if bin_size == 30. else bin_size
            assert cumulative is False, 'cumulative cannot be used with lst=True.'

        if cumulative != False:
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
            groups = groups or self.targets.get_groups()
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
                tt = self.get_target_time(tname, observatory=observatory, lunation=lunation, return_lst=lst)

                if len(tt) == 0:
                    continue

                if not lst:
                    tt -= 2451545.0

                heights, bins = numpy.histogram(tt, bins=b)
                heights = numpy.array(heights, dtype=float)
                heights *= t.exptime * t.min_exposures / 3600.0
                ntiles = len(numpy.where(self.tiledb.tile_table['TargetIndex'].data == tindex)[0])
                target_tot_time = ntiles * t.exptime * t.n_exposures / 3600.

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
        heights *= self.time_step / 3600.0
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

        ax.set_title(observatory if observatory != None else 'APO+LCO')

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

    def plot_shadow_height(self, tname=None, group=False, observatory=None, norm=False, cumulative=0, linear_log=False):
        """
        plot the shadow height distribution for a target. use '-' for unused time

       Parameters
       ----------
        tname : str
            The name of the target or group. Use 'ALL' for all groups and group==True.
        group : bool
            If not true, ``tname`` will be the name of a group not a single
            target.
        observatory : str
            The observatory to filter for.
        norm : bool
            Normalize the histograms instead of plotting raw numbers.

        Return
        ------
        fig : `~matplotlib.figure.Figure`
            The Matplotlib figure of the plot.
        """
        if linear_log is False:
            b = numpy.logspace(numpy.log10(100.),numpy.log10(100000.),100)
        else:
            b = numpy.linspace(2, 5, 31)

        fig, ax = plt.subplots()
        self._plot_histograms(ax, 'shadow_height', b, tname=tname, group=group, observatory=observatory, 
                              norm=norm, cumulative=cumulative, linear_log=linear_log)
        if linear_log is False:
            ax.set_xscale("log")
        plt.xlabel('shadow height / km')
        plt.ylabel('# of exposures')
        plt.legend()
        #plt.show()
        return fig

    def plot_airmass(self, tname=None, group=False, observatory=None, norm=False, cumulative=0):
        """
        plot the airmass distribution for a target or group(s).

       Parameters
       ----------
        tname : str
            The name of the target or group. Use 'ALL' for all groups and group==True.
        group : bool
            If not true, ``tname`` will be the name of a group not a single
            target.
        observatory : str
            The observatory to filter for.
        norm : bool
            Normalize the histograms instead of plotting raw numbers.

        Return
        ------
        fig : `~matplotlib.figure.Figure`
            The Matplotlib figure of the plot.
        """
        b = numpy.linspace(1.0,2.0,51)
        fig, ax = plt.subplots()
        self._plot_histograms(ax, 'airmass', b, tname=tname, group=group, observatory=observatory, 
                              norm=norm, cumulative=cumulative)
        plt.xlabel('airmass')
        plt.ylabel('# of exposures' if norm==False else 'frequency')
        plt.legend()
        #plt.show()
        return fig

    def _plot_histograms(self, ax, keyword, bins, tname=None, group=False, observatory=None, 
                         norm=False, cumulative=0, linear_log=False):
        """
        plot a histogram of 'keyword' for a target or group(s).

       Parameters
       ----------
        ax : pyplot.ax
            axes object to plot into
        keyword : str
            name of the column in the schedule table to plot.
        bins : numpy.array
            the array of bins
        tname : str
            The name of the target or group. Use 'ALL' for all groups and group==True.
        group : bool
            If not true, ``tname`` will be the name of a group not a single
            target.
        observatory : str
            The observatory to filter for.
        norm : bool
            Normalize the histograms instead of plotting raw numbers.
        cumulative : int
            plot cumulative histogram (>0), reverse accumulation (<0)
        """
        column = 'group' if group is True else 'target'
        if tname != None and tname != 'ALL':
            t = self.schedule[self.schedule[column] == tname]
        else:
            t = self.schedule
        if observatory:
            t = t[t['observatory'] == observatory]

        if group==True and tname=='ALL':
            groups = self.targets.get_groups()
            for group in groups:
                tt = t[t['group'] == group]
                am = tt[keyword]
                am = am[numpy.where(am>0)]
                if linear_log is True:
                    am = numpy.log10(am)
                ax.hist(am, bins=bins, histtype='step', label=group, density=norm, cumulative=cumulative)
        else:
            am = t[keyword]
            am = am[numpy.where(am>0)]
            if linear_log is True:
                am = numpy.log10(am)
            ax.hist(am, bins=bins, histtype='step', label=tname, density=norm, cumulative=cumulative)
