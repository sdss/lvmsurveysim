""" main.py is the main routine calling surveysim for tests"""
# eval "$(pyenv init -)";
#%load_ext autoreload
#%autoreload 2  
#import os
#os.environ["LVMCORE_DIR"] = "/Users/droryn/prog/lvm/lvmcore/"
#https://in-the-sky.org/skymap2.php?year=2019&month=11&day=11&town=3884373
#https://lambda.gsfc.nasa.gov/product/foreground/fg_halpha_get.cfm
from lvmsurveysim.schedule import ObservingPlan, Simulator, TileDB, OpsDB
from lvmsurveysim.target import TargetList
import matplotlib.pyplot as plt

#np.seterr(invalid='raise')

# Creates a list of targets/
print('Creating target list ...')
targets = TargetList(target_file='./targets.yaml')
# Create tile database
print('Loading tile database ...')
#tiledb = OpsDB.load_tiledb(path='lco_tiledb', fits=True)
OpsDB.init()
tiledb = OpsDB.load_tiledb()

print('Tiling Survey ...')
# tiledb = TileDB(targets)
# tiledb.tile_targets()
# OpsDB.save_tiledb(tiledb, fits=True, path='lco_tiledb', overwrite=True)

# Creates observing plans for LCO for the range sep 2021 - jun 2025.
print('Creating observing plan ...')
lco_plan = ObservingPlan(2459458, 2460856, observatory='LCO') # baseline

# Creates an Simulator instance and runs the simulation
print('Creating Simulator ...')
sim = Simulator(tiledb, observing_plan=lco_plan)
sim.run(progress_bar=True)

# Load/Save from as FITS table in a later session, no need to rerun:
# sim = Simulator.load('lvmsurveysim_hz_1000', 'lco_tiledb')
# sim.save('lvmsurveysim_hz_1000', overwrite=True) # Save as FITS table

save=False

# Plot and print:
sim.print_statistics()
sim.plot_survey('LCO', use_groups=True)
if save: plt.savefig('LCO_jd_1000.pdf')
sim.plot_survey('LCO', use_groups=True, cumulative=True)
if save: plt.savefig('LCO_cumulative_1000.pdf')
sim.plot_survey('LCO', lst=True, use_groups=True)
if save: plt.savefig('LCO_lst_1000.pdf')
sim.plot(fast=True) # footprint
if save: plt.savefig('LCO_survey_1000.pdf')
sim.plot_airmass(tname='ALL', group=True, norm=True)
if save: plt.savefig('LCO_airmass_1000.pdf')
sim.plot_shadow_height(tname='ALL', group=True, norm=True, cumulative=True, linear_log=True)
if save: plt.savefig('LCO_hz_1000.pdf')
if save: sim.animate_survey(filename='lvm_survey_1000.mp4')



# - create db
# - tile
#  - save to FITS or DB
# - run sim
#  - save results, save plots
#  - load sim, run plots?
# - ops
# - - interactive mode, load, create objects, drop to prompt?
