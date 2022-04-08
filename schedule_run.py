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

def run(targets='./targets.yaml', tile=False, sim=False, plot=False, save=False):
   # Creates a list of targets/
   print('Creating target list ...')
   targets = TargetList(target_file='./targets.yaml')

   # Create tile database
   OpsDB.init()
   if tile:
      print('Tiling Survey ...')
      tiledb = TileDB(targets)
      tiledb.tile_targets()
      OpsDB.create_tables(drop=True)
      OpsDB.save_tiledb(tiledb) #, fits=True, path='lco_tiledb', overwrite=True)
   else:
      print('Loading tile database ...')
      #tiledb = OpsDB.load_tiledb(path='lco_tiledb', fits=True)
      tiledb = OpsDB.load_tiledb()
 
   # Creates observing plans for LCO for the range sep 2021 - jun 2025.
   print('Creating observing plan ...')
   #lco_plan = ObservingPlan(2459458, 2460856, observatory='LCO') # baseline 2021
   #lco_plan = ObservingPlan(2459945, 2459945+365*3+182, observatory='LCO') # baseline 2023, 3.5yr
   lco_plan = ObservingPlan(2459945, 2461555, observatory='LCO') # baseline 2023, until 30-May-2027

   if sim:
      # Creates an Simulator instance and runs the simulation
      print('Creating Simulator ...')
      sim = Simulator(tiledb, observing_plan=lco_plan)
      sim.run(progress_bar=True)

      # Load/Save from as FITS table in a later session, no need to rerun:
      # sim = Simulator.load('LCO_2023_4', 'lco_tiledb')
      sim.save('LCO_2023_4', overwrite=True) # Save as FITS table

   if plot:
      plot_survey(sim, basename='LCO_2023_4', save=save)

   return sim

def plot_survey(sim, basename='LCO_2023_4', save=False):
   # Plot and print:
   sim.print_statistics()
   sim.plot_survey('LCO', use_groups=True)
   if save: plt.savefig(basename+'_jd.pdf')
   sim.plot_survey('LCO', use_groups=True, cumulative=True)
   if save: plt.savefig(basename+'_cumulative.pdf')
   sim.plot_survey('LCO', lst=True, use_groups=True)
   if save: plt.savefig(basename+'_lst.pdf')
   sim.plot(fast=True) # footprint
   if save: plt.savefig(basename+'_survey.pdf')
   sim.plot_airmass(tname='ALL', group=True, norm=True)
   if save: plt.savefig(basename+'_airmass.pdf')
   sim.plot_shadow_height(tname='ALL', group=True, norm=True, cumulative=True, linear_log=True)
   if save: plt.savefig(basename+'_hz.pdf')
   #if save: sim.animate_survey(filename='lvm_survey_hz_2023.mp4')
   if not save:
      plt.show()

