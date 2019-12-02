#!/usr/bin/python3
"""Convert Survey Simulation to healpix Array of total coverage. Include secondary fit's file identifying the target with the highest priority in each spaxel"""
import astropy.io.fits as fits
from astropy_healpix import HEALPix
import healpy
import os.path
import numpy as np
import sys
import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.coordinates import Galactic
from astropy.time import Time
import astropy.table
import yaml
import astropy.units as u
from lvmsurveysim.target import TargetList
import time

def convert(coversion_params):
    print_counter = 1 # time to print status in seconds
    for coversion_params_key in coversion_params.keys():
        print(coversion_params_key,":", coversion_params[coversion_params_key])

    image_hdu_list = fits.open("Halpha_fwhm06_1024.fits")
    image_nside = image_hdu_list[0].header['NSIDE']
    image_order = image_hdu_list[0].header['ORDERING']
    image_data = np.array(image_hdu_list[1].data.tolist())[:,0]
    hp = HEALPix(nside=coversion_params['nside'], order=image_order, frame=Galactic())

    schedule = astropy.table.Table.read(coversion_params["file"])
    target_names = np.unique(schedule['target'])

    print(coversion_params["target_file"])
    targets = TargetList(target_file=coversion_params["target_file"])

    #Create the missing column in the schedule table: Priority
    schedule_priority = np.full(len(schedule['target']), -1)

    for target_i in range(len(targets)):
        target = targets[target_i].name
        if target != '-':
            target_mask = schedule['target'] == target
            schedule_priority[target_mask] = targets[target_i].priority
    
    schedule['priority'] = astropy.table.Column(schedule_priority)

    #Mask out all the unobserved values. I don't know why they are bad, but they are.
    obs_mask = schedule['target'] != "-"

    ### DEV
    # Create a mapping between the values in the table and their healpix index.
    # This allows us to directly dump information from the table onto the array.
    # healpix_indicies = hp.skycoord_to_healpix(SkyCoord(schedule['ra'][obs_mask], schedule['dec'][obs_mask], unit=u.deg))
    ### DEV

    # This is how we will store all the different healpix arrays containing different information.
    healpix_dictionary = {}

    # To later create empty arrays we need to know the number of healpix pixels.
    npix=12*coversion_params['nside']**2

    #Populate the healpix array with the highest priority of that pixel.
    healpix_dictionary['target index'] = {}

    healpix_dictionary['priorities'] =  np.full(npix, -1)
    healpix_dictionary['priority_levels'] = np.sort(np.unique(schedule['priority']))
    for priority_level in healpix_dictionary['priority_levels']:
        t0 = time.time()
        priority_mask = (schedule['priority'] == priority_level) * (schedule['target'] != "-")
        #Note because of repeat visits the need to find unique values for the healpix arrays
        tmp0= len(schedule['ra'][priority_mask])
        tmp = tmp0
        print("processing priority level: %i"%(priority_level))
        for ra, dec in zip(schedule['ra'][priority_mask], schedule['dec'][priority_mask]):
            heal_indices = hp.cone_search_skycoord(SkyCoord(ra, dec, unit=u.deg), radius=(0.25)*u.deg)
            healpix_dictionary['priorities'][heal_indices] = priority_level
            complete = 1.- float(tmp)/float(tmp0)
            tmp = tmp -1
            
            if time.time() - t0 > print_counter:
                t0 = time.time()
                print("Progress {:2.1%}".format(complete), end="\r")
    
    return(healpix_dictionary)


def run(params):
    healpix_dictionary = convert(params)

    image_hdu_list = fits.open("Halpha_fwhm06_1024.fits")
    image_nside = image_hdu_list[0].header['NSIDE']
    image_order = image_hdu_list[0].header['ORDERING']

    from lvmsurveysim.utils.healpix_plot_lib import healpix_shader

    colors =[]
    masks = []
    colors_available = ["copper", "Greens","Blues", "Purples"]
    scale = [1, 1, 1]

    priority_min = -1

    healpix_dictionary["high_res_priorities"] = healpy.pixelfunc.ud_grade(healpix_dictionary['priorities'], image_nside, power=0.0)
    for priority_i, priority in enumerate(healpix_dictionary['priority_levels']):
        if priority >= (priority_min or -1):
            masks.append(healpix_dictionary['high_res_priorities'] == priority)
            if priority_i <= len(colors_available) -1:
                colors.append(colors_available[priority_i])
            else:
                colors.append(colors_available[-1])
            if priority_i <= len(scale) -1:
                scale.append(scale[-1])
            
    
    data = np.array(image_hdu_list[1].data.tolist())[:,0]
    log_I = np.log10(data)
    log_I_max = 2.0
    log_I_min = -1.0

    healpix_shader(log_I, masks, cmaps=colors, scale=scale, title=r"MW H$\alpha$", nest=True, vmin=log_I_min, vmax=log_I_max, outfile="%s_shaded_MW.png"%(params['file'].replace(".fits","")), gui=True)

if __name__ == "__main__":
    "provide the fits file, and target file"
    params = {"file":None, "target_file":"None","nside":1024, "image_file":None}


    if len(sys.argv) > 1:
        for argument in sys.argv[1:]:
            key, value = argument.split(":")
            params[key] = value

    if params["target_file"] is "None":
        params["target_file"] = "%s/surveydesign/%s"%(os.environ['LVMCORE_DIR'], params["file"].replace(".fits",".yaml"))
        print("converting fits file name %s to target file %s"%(params["file"], params["target_file"]))

    assert(params['image_file'] != None, "No image file specified...")
    run(params)

    