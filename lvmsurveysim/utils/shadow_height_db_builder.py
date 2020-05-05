import astropy
import astropy.units as u
import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.coordinates import Galactic, FK4
from astropy_healpix import HEALPix
from lvmsurveysim.utils import shadow_height_lib


class make_galaxy(object):
    def make_coords_from_healpix(self):
        self.gal_hp = HEALPix(nside=self.NSIDE, order='nested', frame=Galactic())
        self.gal_coord = self.gal_hp.healpix_to_skycoord(np.arange(self.NPIX))
        self.fk4_coord = self.gal_coord.fk4

    def __init__(self, NSIDE=4):
        self.NSIDE = NSIDE
        self.NPIX = hp.nside2npix(self.NSIDE)
        self.make_coords_from_healpix()

def jd_builder(jd_start=2459458, jd_stop=2460856, delta_jd=7, delta_hr=1.0):
    #I like a delta jd of 7 to match lunar cycles.
    
    # Create the array of hours
    hours = np.linspace(0,24-delta_hr,int(24/delta_hr))

    # Create the array of days
    days  = np.linspace(jd_start, jd_stop-delta_jd, int((jd_stop-jd_start)/delta_jd))

    # Number of jds to calculate
    jd_array = np.zeros(len(days)*len(hours))

    i = 0
    for day in days:
        for hour in hours:
            jd_array[i] = day + hour/24.0
            i+=1
    
    return(jd_array)


if __name__ == "__main__":
    import skyfield.api
    eph = skyfield.api.load('de421.bsp')
    galaxy = make_galaxy()

    shadow_calc = shadow_height_lib.shadow_calc(observatory_name='LCO', 
                            observatory_elevation=2380*u.m,
                            observatory_lat='29.0146S', observatory_lon='70.6926W',
                            eph=eph, earth=eph['earth'], sun=eph['sun'])

    jd_array = jd_builder()

    hz = {}
    N_JD = float(len(jd_array))
    for jd_i, current_jd in enumerate(jd_array):
        print("%0.6f complete\r"%((jd_i+1)/N_JD))
        hz[current_jd] = np.zeros(galaxy.NPIX)
        shadow_calc.update_t(current_jd)
        for healpix_i, coord in enumerate(galaxy.fk4_coord):
            hz[current_jd][healpix_i] = shadow_calc.height_from_radec(coord.ra, coord.dec, simple_output=True)['height']
    import pickle
    pickle.dump(hz, open("hz_lib_dict.pckl", "wb"))



# Make healpix array

# Convert healpix array to ra and dec.

# Loop over JD and calculate ra, dec, shadow_height

# Build KDE-Tree using JD, gal_l, gal_b, to get shadow height
