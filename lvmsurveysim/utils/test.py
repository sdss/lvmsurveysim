import numpy as np
from skyfield.api import load
from skyfield.api import Topos
from skyfield.positionlib import position_from_radec, Geometric
from astropy import units as u

observatory_name="APO"
observatory_elevation=2788.0*u.m
observatory_lat='32.7802777778N'
observatory_lon = '105.8202777778W'
jd=2459458

import skyfield.api
eph = skyfield.api.load('de421.bsp')
earth = eph['earth']
sun = eph['sun']
observatory_topo = Topos(observatory_lat, observatory_lon, elevation_m=observatory_elevation.to("m").value)

ts = load.timescale()
t = ts.tt_jd(jd)
ra = np.linspace(0,23,int(1e4))
dec = np.linspace(0,-90,int(1e4))
positions = position_from_radec(ra, dec, distance=1.0, epoch=None, t=t, center=observatory_topo, target=None)
print(len(positions))