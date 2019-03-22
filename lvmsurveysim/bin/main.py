""" main.py is the main routine calling surveysim for tests"""
from lvmsurveysim.schedule import ObservingPlan, Scheduler
from lvmsurveysim.target import TargetList

# Creates a list of targets/
targets = TargetList(target_file='../lvmcore/surveydesign/targets.yaml')

# Creates observing plans for APO and LCO for the range 2021-2025.
apo_plan = ObservingPlan(2459216, 2459217, observatory='APO')
lco_plan = ObservingPlan(2459216, 2461041, observatory='LCO')

# Creates an Scheduler instance and runs the simulation

scheduler = Scheduler(targets, observing_plans=[apo_plan, lco_plan])
scheduler.run(progress_bar=True)
scheduler.save('lvmsurveysim_results.fits', overwrite=True)
