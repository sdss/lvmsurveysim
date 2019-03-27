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

def convert(params):
        
    for key in params.keys():
        print(key,":", params[key])

    image_hdu_list = fits.open("Halpha_fwhm06_1024.fits")
    image_nside = image_hdu_list[0].header['NSIDE']
    image_order = image_hdu_list[0].header['ORDERING']
    image_data = np.array(image_hdu_list[1].data.tolist())[:,0]
    hp = HEALPix(nside=params['nside'], order=image_order, frame=Galactic())

    schedule = astropy.table.Table.read(params["file"])
    target_names = schedule.meta['TARGETS'].split(',')
    targets = yaml.load(open(os.environ['LVMCORE_DIR']+"/surveydesign/targets.yaml"), Loader=yaml.FullLoader)


    #Create the missing column in the schedule table: Priority
    schedule_priority = np.full(len(schedule['target']), -1)

    for target in target_names:
        target_mask = schedule['target'] == target
        schedule_priority[target_mask] = targets[target]['priority']
    
    schedule['priority'] = astropy.table.Column(schedule_priority)

    #Mask out all the bad values. I don't know why they are bad, but they are.
    obs_mask = schedule['target'] != "-"

    # Create a mapping between the values in the table and their healpix index.
    # This allows us to directly dump information from the table onto the array.
    healpix_indecies = hp.skycoord_to_healpix(SkyCoord(schedule['ra'][obs_mask], schedule['dec'][obs_mask], unit=u.deg))

    # This is how we will store all the different healpix arrays containing different information.
    healpix_dictionary = {}

    # To later create empty arrays we need to know the number of healpix pixels.
    npix=12*params['nside']**2

    #Populate the healpix array with the highest priority of that pixel.
    healpix_dictionary['target index'] = {}

    #hp.cone_search_skycoord(SkyCoord.from_name("M31"), radius=1*u.deg
    #hp.skycoord_to_healpix(SkyCoord.from_name("M31"))

    healpix_dictionary['priorities'] =  np.full(npix, -1)
    healpix_dictionary['priority_levels'] = np.sort(np.unique(schedule['priority']))
    for priority_level in healpix_dictionary['priority_levels']:
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
            print("Progress {:2.1%}".format(complete), end="\r")
            
    
    return(healpix_dictionary)

if __name__ == "__main__":
    params = {"file":"lvmsurveysim_results.fits", "nside":1024, "targets":"None"}

    if len(sys.argv) > 1:
        for argument in sys.argv[1:]:
            key, value = argument.split(":")
            params[key] = value

    healpix_dictionary = convert(params)

    image_hdu_list = fits.open("Halpha_fwhm06_1024.fits")
    image_nside = image_hdu_list[0].header['NSIDE']
    image_order = image_hdu_list[0].header['ORDERING']

    from healpix_plot_lib import healpix_shader

    colors =[]
    masks = []
    colors_available = ["copper","Blues", "Blues", "Blues", "Blues"]
    scale = [1, 1, 1, 1, 1]

    healpix_dictionary["high_res_priorities"] = healpy.pixelfunc.ud_grade(healpix_dictionary['priorities'], image_nside, power=0.0)
    for priority_i, priority in enumerate(healpix_dictionary['priority_levels']):
        masks.append(healpix_dictionary['high_res_priorities'] == priority)
        colors.append(colors_available[priority_i])

    
    image_data = np.array(image_hdu_list[1].data.tolist())[:,0]
    hp = HEALPix(nside=params['nside'], order=image_order, frame=Galactic())

    data = np.array(image_hdu_list[1].data.tolist())[:,0]
    log_I = np.log10(data)
    log_I_max = 2.0
    log_I_min = -1.0

    healpix_shader(log_I, masks, cmaps=colors, scale=scale, title=r"MW H$\alpha$", nest=True, vmin=log_I_min, vmax=log_I_max, outfile="shaded_MW.png", gui=True)
    


    


    