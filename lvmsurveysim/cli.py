#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-03-29
# @Filename: cli.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-29 12:35:09

import click

import lvmsurveysim as lvmsurveysim_mod
import lvmsurveysim.schedule as schedule_mod
import lvmsurveysim.target as target_mod


start_date = lvmsurveysim_mod.config['observing_plan']['APO']['start_date']
end_date = lvmsurveysim_mod.config['observing_plan']['APO']['end_date']


@click.group()
def lvmsurveysim():
    """Runs commands for lvmsurveysim."""

    pass


@lvmsurveysim.command()
@click.option('--start-date', default=start_date, show_default=True, type=float,
              help='the start JD for the observing plan.')
@click.option('--end-date', default=end_date, show_default=True, type=float,
              help='the end JD for the observing plan.')
@click.option('--good-weather-fraction', type=float,
              help='the fraction of days with good weather.')
@click.option('--apo/--no-apo', default=True,
              help='whether to schedule time at APO.')
@click.option('--lco/--no-lco', default=True,
              help='whether to schedule time at LCO.')
@click.option('--target-file', type=click.Path(exists=True),
              help=('a YAML file containing the targets to observe. '
                    'Defaults to $LVMCORE/surveydesign/targets.yaml.'))
@click.option('--print-stats', is_flag=True, default=False,
              help='prints statistics of the schedule.')
@click.option('--plot', type=click.Path(),
              help='plots the results of the simulation and saves them to a file')
@click.argument('output_file', metavar='OUTPUT_FILE', type=click.Path())
def schedule(output_file, start_date=None, end_date=None,
             good_weather_fraction=None, apo=None, lco=None,
             target_file=None, print_stats=False, plot=None):
    """Runs the scheduler for a list of targets."""

    if not apo and not lco:
        raise click.UsageError('either --apo or --lco need to be set.')

    output_file = str(output_file)
    if not output_file.endswith('.fits'):
        output_file += '.fits'

    observatories = []
    if apo:
        observatories.append('APO')
    if lco:
        observatories.append('LCO')

    observing_plans = [schedule_mod.ObservingPlan(
        start=start_date, end=end_date, observatory=observatory,
        good_weather=good_weather_fraction) for observatory in observatories]

    targets = target_mod.TargetList(target_file=target_file)

    schedule = schedule_mod.Scheduler(targets, observing_plans=observing_plans)
    schedule.run(progress_bar=True)
    schedule.save(output_file, overwrite=True)

    if print_stats:
        schedule.print_statistics()

    if plot:
        fig = schedule.plot()
        fig.savefig(str(plot))
