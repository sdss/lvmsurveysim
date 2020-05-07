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

class jd_builder(object):
    def __init__(self):
        import astral
        if float(astral.__version__.split(".")[0]) == 1:
            self._astral = astral.Astral()
        else:
            from astral.sun import sun
            self._astral = sun
        
        self.build()

    def build(self, jd_start=2459458, jd_stop=2460856, delta_jd=7, delta_hr=1.0):
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
                self.jd_array[i] = day + hour/24.0
                i+=1
        
    def _get_twilight(self, datetime_today, lon, lat, alt):
        """Returns the dusk and dawn times associated with a given JD."""

        if float(astral.__version__.split(".")[0]) == 1:
            dusk = self._astral.dusk_utc(datetime_today, lat, lon,
                                        observer_elevation=alt,
                                        depression=self.twilight_alt)

            dawn = self._astral.dawn_utc(datetime_today + _delta_dt, lat, lon,
                                        observer_elevation=alt,
                                        depression=self.twilight_alt)
        else:
            dusk = self._astral(astral.Observer(latitude=lat, longitude=lon,
                                        elevation=alt),
                                        date=datetime_today)["dusk"]

            dawn = self._astral(astral.Observer(latitude=lat, longitude=lon,
                                        elevation=alt),
                                        date=datetime_today + _delta_dt)['dawn']

        return dusk, dawn



if __name__ == "__main__":
    import skyfield.api
    eph = skyfield.api.load('de421.bsp')
    galaxy = make_galaxy()

    class location(object):
        def __init__(self, name="lco", lat=-29.0146, lon=-70.6926):
            self.lat = lat*u.deg
            self.lon = lon*u.deg
            if lat > 0:
                self.lat_cardinal="%fN"%(self.lat)
            else:
                self.lat_cardinal="%fS"%(self.lat*-1)
            if lon > 0:
                self.lat_cardinal="%fE"%(self.lon)
            else:
                self.lat_cardinal="%fW"%(self.lon*-1)

            self.name = name
    
    location = location()

    shadow_calc = shadow_height_lib.shadow_calc(observatory_name='LCO', 
                            observatory_elevation=2380*u.m,
                            observatory_lat='29.0146S', observatory_lon='70.6926W',
                            eph=eph, earth=eph['earth'], sun=eph['sun'])

    
    jd_array = jd_builder()

    hz = {}
    N_JD = float(len(jd_array))


    ap_times = astropy.time.Time(jd_array, format='jd')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        lon = self.location.lon.deg
        lat = self.location.lat.deg
        alt = self.location.height.value

        dts = ap_times.datetime

        twilights = numpy.array([self._get_twilight(dt, lon, lat, alt)
                                    for dt in dts])

    for jd_i, current_jd in enumerate(jd_array):
        print("%0.6f complete\r"%((jd_i+1)/N_JD))

        jd_dusk, jd_dawn = _get_twilight(datetime_today, lon, lat, alt)

        hz[current_jd] = np.zeros(galaxy.NPIX)
        if (current_jd > jd_dusk) and (current_jd < jd_dawn): 
            shadow_calc.update_t(current_jd)
            for healpix_i, coord in enumerate(galaxy.fk4_coord):
                hz[current_jd][healpix_i] = shadow_calc.height_from_radec(coord.ra, coord.dec, simple_output=True)['height']

    import pickle
    pickle.dump(hz, open("hz_lib_dict.pckl", "wb"))



# Make healpix array

# Convert healpix array to ra and dec.

# Loop over JD and calculate ra, dec, shadow_height

# Build KDE-Tree using JD, gal_l, gal_b, to get shadow height
