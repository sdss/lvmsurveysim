import astropy.units as u
import math
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon, Galactic
from astropy.time import Time
import numpy as np
from astropy_healpix import HEALPix
import healpy

class obs():
    def __init__(self, 
        observatory="LCO",
        obs_date="2020-9-1",
        max_sun_alt=-17.0*u.deg,
        bright_moon_sun_dist=90*u.deg,
        max_airmass=1.5,
        delta_days=1,
        nside=16,
        nside_lowres=16,
        nest=False,
        campaign_mode=False,
        dt_min=-8*u.hour,
        dt_max=8*u.hour,
        bins_per_hour=4,
        verbose=True):

        self.Flag1m = [False, True, False, False, False, False, False, True, True, True, True, True, True]
        self.verbose = verbose

        # These are are observational properties. 
        self.campaign_mode = campaign_mode
        self.date = obs_date        
        self.local_midnight = Time('%s 00:00:00'%(self.date))

        self.set_observatory(observatory)

        #This is the initialized value. These are updated each day.
        self.utc_midnight = self.local_midnight - self.utcoffset
        self.sun = get_sun(self.utc_midnight)
        self.moon = get_moon(self.utc_midnight)


        # These are survey choices
        self.max_sun_alt = max_sun_alt # I shoudl really have a unit here... and it should be degrees
        self.bright_moon_sun_dist = bright_moon_sun_dist # angular seperation between the moon and sun to be considered bright time.
        self.max_airmass = max_airmass
        self.min_continuous_window = 0.25 # hours
        self.min_exposure = 0.25 # Shortest exposure

        # These are simulation properties
        self.delta_days = delta_days # offset between simulated days.
        self.bins_per_hour = bins_per_hour

        #HPT. This should be a dictionary. In principle it can be populated from a list. Note this can be large as it is updated only upon intialization of the observation
        self.LVM_HPT = {'M31':SkyCoord.from_name("M31"),
                    "M33":SkyCoord.from_name("M33"),
                    "LMC":SkyCoord.from_name("LMC"),
                    "SMC":SkyCoord.from_name("SMC")}        

        self.max_airmass = {'M31':1.5,
                    "M33":1.5,
                    "LMC":1.75,
                    "SMC":1.75,
                    "MW":1.5}   


        try: 
            self.mkIFU(n_spaxels=parameter_list['n_spaxels'])
        except:
            print("Spaxel number not defined or unreadable. Using 1500")
            self.mkIFU(n_spaxels=1500)

        # Healpix parameters
        self.nest=nest
        if nest== True:
            self.order = "nested"
        else:
            self.order = "ring"

        self.nside = nside
        self.npix=12*self.nside**2
        self.hp = HEALPix(nside=self.nside, order=self.order, frame=Galactic())
    
        #gives you the number of pixel

        self.healpix_skycoords = self.hp.healpix_to_skycoord(range(self.npix)).transform_to(frame='fk5')

        # print the resolution of the healpix array in arcminutes
        self.angular_resolution = self.hp.pixel_resolution

        self.healpix_pixel_area = self.angular_resolution**2
        # I should calculate the minimum window based on this angular resolution. I am not.

        self.N_IFU_per_pixel = (self.angular_resolution.to(u.arcsec))**2 / self.A_IFU  

        # Simulation parameters
        if u.Unit(dt_min) == u.Unit(1):
            #No unit provided, assuming hours.
            self.dt_min = dt_min*u.hour
        else:
            self.dt_min = dt_min.to(u.hour)

        if u.Unit(dt_max) == u.Unit(1):
            #No unit provided, assuming hours.
            self.dt_max = dt_max*u.hour
        else:
            self.dt_max = dt_max.to(u.hour)

    def set_observatory(self, observatory, reset_date=True):
        if observatory == "APO":
            self.location = EarthLocation.of_site('Apache Point Observatory')
            self.utcoffset = -7.0*u.hour
            self.observatory = observatory
        elif observatory == "LCO":
            self.location = EarthLocation.of_site('Las Campanas Observatory')
            self.utcoffset = -5.0*u.hour
            self.observatory = observatory
        else:
            Exception("Observatory location not known")
        if reset_date == True:
            self.restart_date()

    def restart_date(self):
        self.local_midnight = Time('%s 00:00:00'%(self.date))
        self.utc_midnight = self.local_midnight - self.utcoffset

    def next_day(self):
        self.local_midnight = self.local_midnight + self.delta_days*u.day
        self.utc_midnight = self.local_midnight - self.utcoffset
        self.sun = get_sun(self.utc_midnight)
        self.moon = get_moon(self.utc_midnight)
        if self.verbose == True: print(self.local_midnight)

    def mkIFU(self, n_spaxels=1500, tel_diam=0.16*u.m):
        self.tel_diam = tel_diam
        self.n_spaxels = n_spaxels
        self.r_spaxel  = (37/2.) * u.arcsec * (0.168*u.m/tel_diam)
        self.A_spaxel  = 2*math.pi * self.r_spaxel**2
        self.A_IFU     = self.n_spaxels*self.A_spaxel
        self.r_IFU     = self.A_IFU**0.5
        return({""})
