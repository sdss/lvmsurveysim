""" main.py is the main routine calling surveysim for tests"""
# eval "$(pyenv init -)";
#%load_ext autoreload
#%autoreload 2  
#import os
#os.environ["LVMCORE_DIR"] = "/Users/droryn/prog/lvm/lvmcore/"
#https://in-the-sky.org/skymap2.php?year=2019&month=11&day=11&town=3884373
#https://lambda.gsfc.nasa.gov/product/foreground/fg_halpha_get.cfm
from lvmsurveysim.schedule import ObservingPlan, Scheduler, TileDB
from lvmsurveysim.target import TargetList
import matplotlib.pyplot as plt

#np.seterr(invalid='raise')

# Creates a list of targets/
print('Creating target list ...')
targets = TargetList(target_file='./targets.yaml')
# Create tile database
print('Creating tile database ...')
tiledb = TileDB.load('lco_tiledb')
#tiledb = TileDB(targets)
#tiledb.tile_targets()
#tiledb.save('lco_tiledb', overwrite=True)
# Creates observing plans for APO and LCO for the range sep 2021 - jun 2025.
print('Creating observing plans ...')
lco_plan = ObservingPlan(2459458, 2460856, observatory='LCO') # baseline
#lco_plan = ObservingPlan(2459458, 2460553, observatory='LCO') # 3 yr
#lco_plan = ObservingPlan(2459458, 2461283, observatory='LCO')  # 5 yr

# Creates an Scheduler instance and runs the simulation
print('Creating Scheduler ...')
#scheduler = Scheduler(targets, observing_plans=[apo_plan,lco_plan])
scheduler = Scheduler(tiledb, observing_plans=[lco_plan], verbos_level=1)
scheduler.run(progress_bar=True)

# Load/Save from as FITS table in a later session, no need to rerun:
# scheduler = Scheduler.load('lvmsurveysim_hz_1000', 'lco_tiledb')
# scheduler.save('lvmsurveysim_hz_1000', overwrite=True) # Save as FITS table

save=False

# Plot and print:
scheduler.print_statistics()
scheduler.plot_survey('LCO', use_groups=True)
if save: plt.savefig('LCO_jd_1000.pdf')
scheduler.plot_survey('LCO', use_groups=True, cumulative=True)
if save: plt.savefig('LCO_cumulative_1000.pdf')
scheduler.plot_survey('LCO', lst=True, use_groups=True)
if save: plt.savefig('LCO_lst_1000.pdf')
scheduler.plot(fast=True) # footprint
if save: plt.savefig('LCO_survey_1000.pdf')
scheduler.plot_airmass(tname='ALL', group=True, norm=True)
if save: plt.savefig('LCO_airmass_1000.pdf')
scheduler.plot_shadow_height(tname='ALL', group=True, norm=True, cumulative=True, linear_log=True)
if save: plt.savefig('LCO_hz_1000.pdf')
if save: scheduler.animate_survey(filename='lvm_survey_1000.mp4')
