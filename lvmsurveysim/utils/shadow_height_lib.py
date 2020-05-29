#!/usr/bin/python3
import time
import numpy as np
import sys
from astropy import units as u
from skyfield.api import load
from skyfield.api import Topos
from skyfield.positionlib import position_from_radec, Geometric

class shadow_calc(object):
    def __init__(self, observatory_name="LCO",
    observatory_elevation=2380.0*u.m,
    observatory_lat='29.0146S',
    observatory_lon = '70.6926W',
    jd=2459458,
    eph=None,
    earth=None,
    sun=None,
    d=None,
    mask=None,
    pointings=None):
        """
        Initialization sets a default observatory to LCO, and the default time to the date when initialized.
        """
        super().__init__()

        # Load the ephemeral datat for the earth and sun. 
        if eph is None:
            self.eph = load('de421.bsp')
        
        # Get functions for the earth, sun and observatory
        self.earth = earth
        if self.earth is None:
            self.earth = self.eph['earth']
        
        self.sun = sun
        if self.sun is None:
            self.sun = self.eph['sun']

        self.observatory_elevation = observatory_elevation
        try:
            self.observatory_elevation.to("m")
        except:
            sys.exit("Observatory elevation does not have unit of length")

        self.observatory_name = observatory_name
        self.observatory_lat = observatory_lat
        self.observatory_lon = observatory_lon
        self.observatory_topo = Topos(observatory_lat, observatory_lon, elevation_m=self.observatory_elevation.to("m").value)
        self.ts = load.timescale()
       
        # define these at time self.t xyz_observatory_m, xyz_earth_m, xyz_sun_m

        """
        These are going to be vectors which are the x,y,z positions of the puzzle.
        Credit for this solution to the intersection goes to:  Julien Guertault @ http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/
        """

        # Those are very self explanitory 
        self.earth_radius=6.357e6*u.m
        self.sun_radius=695.700e6*u.m

        # Distance from Sun to the earth.
        self.d_se = 1*u.au

        # Distance from tip of the shadow cone to the earth
        self.d_ec = self.d_se * (self.earth_radius / self.sun_radius)

        # Opening angle of the shadow cone
        self.shadow_cone_theta  = np.arctan(self.earth_radius/self.d_ec)
        self.shadow_cone_cos_theta_sqr = np.square(np.cos(self.shadow_cone_theta))

        # Finally, updated the xyz coordinate vectors of earth, sun and observatory.
        if jd is not False:
            self.t = self.ts.tt_jd(jd)

        else:
            self.t = self.ts.now()

        self.update_time(jd)

    def cone_ra_dec(self):
        # This returns the cone RA,Dec for testing. 
        # It has a shadow height that is analyically easy to calculate
        observatory_to_c = self.xyz_c - self.xyz_observatory

        # Leave dec in steradian for just a moment
        dec = np.arctan(observatory_to_c[2]/np.sqrt(np.square(observatory_to_c[0])+np.square(observatory_to_c[1])))

        # Use dec in steradians to solve for ra.
        ra = np.degrees(np.arccos(observatory_to_c[0]/ (self.vecmag(observatory_to_c) * np.cos(dec))))
        
        #Now convert to degree
        dec = np.degrees(dec)
        
        return ra, dec

    def set_coordinates(self, ra, dec):
        # Set coordinates and calculate unit vectors.
        self.ra = ra
        self.dec = dec
        self.pointing_unit_vectors = np.zeros(shape=(len(self.ra),3))

        a = np.pi / 180.
        self.pointing_unit_vectors[:,0] = np.cos(self.ra*a)*np.cos(self.dec*a)
        self.pointing_unit_vectors[:,1] = -np.sin(self.ra*a)*np.cos(self.dec*a)
        self.pointing_unit_vectors[:,2] = np.sin(self.dec*a)

    def update_time(self, jd=None):
        """ Update the time, and all time dependent vectors """
        if jd is not None:
            self.jd = jd
        self.t = self.ts.tt_jd(self.jd)
        self.update_xyz()
        self.update_vec_c()
        self.co = self.xyz_c - self.xyz_observatory

    def update_xyz(self):
        """
        Due to the nature of the vectors, it is not possible to combine the earth and the observatory topo directly. Therefore
        this requires the calculation of an observatory position to be done using the XYZ positions defined at a particular time, and with a particular unit.
        To prevent unit issues later from arising SI units are adopted. In the case where other units are desired, like plotting the solar system, local conversions
        can be done.
        """
        self.xyz_earth = self.earth.at(self.t).position.au
        self.xyz_sun = self.sun.at(self.t).position.au
        self.xyz_observatory = self.earth.at(self.t).position.au + self.observatory_topo.at(self.t).position.au


    def update_vec_c(self):
        """
        d_se : distance sun to earth
        d_ec : distance earth to (c)one tip
        c_unit_vec: unit vector from cone to earth, or earth to sun.

        """
        # We reverse the vector sun to earth explicitly with -1.0 for clarity
        self.v = -1.0*(self.xyz_earth - self.xyz_sun)/np.sqrt(np.sum(np.square(self.xyz_earth-self.xyz_sun)))
        self.xyz_c = self.xyz_earth - self.v * self.d_ec.to("au").value

    def get_abcd(self, mask=False):
        if mask is False:
            mask = np.full(len(self.pointing_unit_vectors), True)

        N_points = len(mask)
        self.a = np.full(N_points, np.nan)
        self.b = np.full(N_points, np.nan)
        self.c = np.full(N_points, np.nan)
        self.delta = np.full(N_points, np.nan)

        self.a[mask] = np.square(np.sum(self.pointing_unit_vectors[mask]*self.v,axis=1)) -  self.shadow_cone_cos_theta_sqr
        self.b[mask] = 2*( np.sum(self.pointing_unit_vectors[mask]*self.v, axis=1) * np.sum(self.co*self.v) - np.sum(self.pointing_unit_vectors[mask]*self.co, axis=1)*self.shadow_cone_cos_theta_sqr)
        self.c[mask] = np.square(np.sum(self.co*self.v)) - np.sum(self.co * self.co) * self.shadow_cone_cos_theta_sqr
        self.delta[mask] = np.square(self.b[mask]) - 4 * self.a[mask] * self.c[mask]

    def solve_for_height(self, unit="km"):
        self.heights = np.full(len(self.a), np.nan)
        height_b1 = np.full(len(self.a), np.nan)
        height_b2 = np.full(len(self.a), np.nan)

        self.dist = np.full(len(self.a), np.nan)
        dist_b1 = np.full(len(self.a), np.nan)
        dist_b2 = np.full(len(self.a), np.nan)

        # Get P distance to point.
        positive_delta = self.delta >= 0 
        dist_b1[positive_delta] = (-self.b[positive_delta]+np.sqrt(self.delta[positive_delta])) / (2*self.a[positive_delta])
        dist_b2[positive_delta] = (-self.b[positive_delta]-np.sqrt(self.delta[positive_delta])) / (2*self.a[positive_delta])

        dist_b1[dist_b1 < 0.0] = np.nan
        dist_b2[dist_b2 < 0.0] = np.nan

        self.dist[positive_delta] = np.nanmin([dist_b1[positive_delta], dist_b2[positive_delta]], axis=0)

        # caluclate xyz of each ra, using the distance to the shadow intersection and the normal vector form the observatory
        pointing_xyz = self.pointing_unit_vectors * self.dist + self.xyz_observatory

        extra_line = ([self.xyz_observatory[0], pointing_xyz[0][0]],[self.xyz_observatory[1], pointing_xyz[0][1]])
        self.animate = orbit_animation(self)
        self.animate.snap_shot(jd=self.jd, ra=self.cone_ra_dec()[0], dec=self.cone_ra_dec()[1], show=True, extra=extra_line)

        self.heights = vecmag(self.vecmag(pointing_xyz - self.xyz_earth)*u.au - self.earth_radius).to(unit)

    def get_heights(self, jd=None, return_heights=True,unit=u.km):
        if jd is not None:
            self.jd = jd
            self.update_time()
        self.get_abcd()
        self.solve_for_height(unit="km")
        if return_heights:
            return self.heights 

    def vecmag(self, a, origin=[0,0,0]):
        """ Return the magnitude of a set of vectors around an abritrary origin """
        if len(np.shape(a)) == 1:
            return np.sqrt(np.square(a[0] - origin[0]) + np.square(a[1] - origin[1]) + np.square(a[2] - origin[2]))
        else:
            return np.sqrt(np.square(a[:, 0] - origin[:, 0]) + np.square(a[:, 1] - origin[:, 1]) + np.square(a[:, 2] - origin[:, 2]))

class orbit_animation(object):
    def __init__(self, calculator, jd0=None, djd=1/24., single=True):
        if jd0 is None:
            self.jd0 = calculator.jd
        else:
            self.jd0 = jd0
        self.djd = djd

        self.calculator = calculator
        self.calculator.observatory_zenith_topo = Topos(calculator.observatory_lat, calculator.observatory_lon, elevation_m=self.calculator.earth_radius.to("m").value*10)

        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.patches as patches

        self.plt = plt
        self.animation = animation
        self.patches = patches
        self.fig, self.ax = self.plt.subplots(figsize=(50,50))

    def init_plotting_animation(self):
        self.line1, = self.ax.plot([], [], 'ro', lw=2)
        self.line2, = self.ax.plot([], [], '.-', lw=5)
        self.line3, = self.ax.plot([], [], '.-', lw=5)
        self.path, = self.ax.plot([],[],'-', lw=1)

        self.ax.grid()
        self.xdata, self.ydata = [], []
        self.pathx, self.pathy = [], []



        self.ax.set_ylim(-150 * self.calculator.earth_radius.to("au").value, 150 * self.calculator.earth_radius.to("au").value)
        self.ax.set_xlim(-150 * self.calculator.earth_radius.to("au").value, 150 * self.calculator.earth_radius.to("au").value)
        del self.xdata[:]
        del self.ydata[:]
        self.line1.set_data(self.xdata, self.ydata)
        self.line2.set_data(self.xdata, self.ydata)

    def do_animation(self, N_days=1.0):
        ani = self.animation.FuncAnimation(self.fig, self.animation_update_positions, frames=(24*N_days), repeat_delay=0,
                                        blit=False, interval=100, init_func=self.init_plotting_animation, repeat=0)
        import os
        ani.save("/tmp/im.mp4")

    def snap_shot(self,jd,ra,dec, show=True, extra=False):
        self.init_plotting_animation()
        self.calculator.set_coordinates(np.array([ra]),np.array([dec]))

        self.calculator.update_time(jd)

        self.calculator.xyz_observatory_zenith = self.calculator.xyz_earth + self.calculator.observatory_zenith_topo.at(self.calculator.t).position.au

        height_au = 3.0 * self.calculator.earth_radius.to("au").value

        self.ax.set_xlim( ( self.calculator.xyz_earth[0] - height_au * 16, self.calculator.xyz_earth[0] + height_au * 16 ) )
        self.ax.set_ylim( ( self.calculator.xyz_earth[1] - height_au * 9, self.calculator.xyz_earth[1] + height_au * 9 ) )

        circ = self.patches.Circle((self.calculator.xyz_earth[0], self.calculator.xyz_earth[1]), self.calculator.earth_radius.to( "au" ).value, alpha=0.8, fc='yellow')
        self.ax.add_patch( circ )
        
        los_xyz =  self.calculator.xyz_observatory + self.calculator.d_ec.to("au").value * self.calculator.pointing_unit_vectors[0]

        #self.line1.set_data( (x_heights * u.m).to( "au" ).value, ( y_heights * u.m ).to("au").value )
        self.line1.set_data( [ self.calculator.xyz_c[0] ], [ self.calculator.xyz_c[1] ] )
        self.line2.set_data( [ self.calculator.xyz_observatory_zenith[0] ], [ self.calculator.xyz_observatory_zenith[1] ] )
        self.line3.set_data( [ self.calculator.xyz_observatory[0] ] , [ self.calculator.xyz_observatory[1] ] )
        self.ax.plot([self.calculator.xyz_c[0], self.calculator.xyz_sun[0]],[self.calculator.xyz_c[1], self.calculator.xyz_sun[1]], alpha=0.5)
        self.ax.plot([los_xyz[0], self.calculator.xyz_observatory[0]],[los_xyz[1], self.calculator.xyz_observatory[1]], c="k", alpha=0.5)
        if extra is not False:
            #This will plot a line using arrays provided in the extra bin.
            self.ax.plot(extra[0], extra[1])
        if show:
            self.plt.show()
        else:
            return self.plt


    def animation_update_positions(self, frame):
        if len( self.ax.patches ) > 1:
            del( self.ax.patches[-1] )

        jd = self.jd0 + frame*self.djd

        #self.calculator.t = self.calculator.ts.tt_jd( self.jd0 + frame*self.djd)
        self.calculator.update_time(jd)

        self.calculator.xyz_observatory_zenith = self.calculator.xyz_earth + self.calculator.observatory_zenith_topo.at(self.calculator.t).position.au

        height_au = 3.0 * self.calculator.earth_radius.to("au").value

        ######################################################################
        # The next set of lines calculates shadow heights. That's for later  #
        ######################################################################
        # N_ra = 24
        # N_dec = 9
        # x_heights = np.zeros(len(N_ra * N_dec)+1)
        # y_heights = np.zeros(len(N_ra * N_dec)+1)
        #
        # for ra_i, ra in enumerate(np.linspace(1,24,N_ra)):
        #     for dec_j, dec in enumerate(np.linspace(-90, 0, num=N_dec, endpoint=True)):
        #         # for height in heights:
        #             self.calculator.point_along_ray =  position_from_radec(ra, dec, distance=height_au, epoch=None, t=self.calculator.t, center=self.calculator.observatory_topo, target=None) #, observer_data=LCO_topo.at(t).observer_data) 
        #             self.calculator.xyz_along_ray = self.calculator.xyz_observatory + self.calculator.point_along_ray.position.au
        #             x_heights[ ra_i + dec_j*N_ra ] = self.calculator.xyz_along_ray[0]
        #             y_heights[ ra_i + dec_j*N_ra ] = self.calculator.xyz_along_ray[1]
        #
        # x_heights[-1] = self.calculator.xyz_earth[0]
        # y_heights[-1] = self.calculator.xyz_earth[1]
        ######################################################################
        self.ax.set_xlim( ( self.calculator.xyz_earth[0] - height_au * 150, self.calculator.xyz_earth[0] + height_au * 150 ) )
        self.ax.set_ylim( ( self.calculator.xyz_earth[1] - height_au * 150, self.calculator.xyz_earth[1] + height_au * 150 ) )

        circ = self.patches.Circle((self.calculator.xyz_earth[0], self.calculator.xyz_earth[1]), self.calculator.earth_radius.to( "au" ).value, alpha=0.8, fc='yellow')
        self.ax.add_patch( circ )

        #self.line1.set_data( (x_heights * u.m).to( "au" ).value, ( y_heights * u.m ).to("au").value )
        self.line1.set_data( [ self.calculator.xyz_c[0] ], [ self.calculator.xyz_c[1] ] )
        self.line2.set_data( [ self.calculator.xyz_observatory_zenith[0] ], [ self.calculator.xyz_observatory_zenith[1] ] )
        self.line3.set_data( [ self.calculator.xyz_observatory[0] ] , [ self.calculator.xyz_observatory[1] ] )
        self.pathx.append( self.calculator.xyz_earth[0] )
        self.pathy.append( self.calculator.xyz_earth[1] )
        self.path.set_data( self.pathx, self.pathy )
    
        #line2.set_data([sun.at(t).position.au[0]],[sun.at(t).position.au[1]])

    def defaults(self):
        self.LCO_elevation = 2380
        self.LCO_topo   = Topos('29.01597S', '70.69208W', elevation_m=LCO_elevation)

        return(LCO_topo)
        

if __name__ == "__main__":

    jd = 2459458.5+20 + 4.75/24.
    calculator = shadow_calc()
    orbit_ani = orbit_animation(calculator)
    orbit_ani.calculator.update_time(jd)
    ra, dec = orbit_ani.calculator.cone_ra_dec()
    orbit_ani.snap_shot(jd=jd, ra=ra, dec=dec)
    # orbit_ani.do_animation()

    test_results ={}
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import Angle, Latitude, Longitude
    from astropy import units as u
    # Initiate tests.
    try:
        test = "Create Class"
        calculator = shadow_calc()
        test_results[test] = "Pass"
    except:
        test_results[test] = "Fail"

    try:
        test = "Shadow RA/DEC"
        ra, dec = calculator.cone_ra_dec()
    except:
        test_results[test] = "Fail"

    try:
        test = "Set Coordinates"
        calculator.set_coordinates(np.array([ra]), np.array([dec]) )
        test_results[test] = "Pass"
    except:
        test_results[test] = "Fail"

    try:
        test = "set jd=2459458.5"
        jd = 2459458.5+20 + 4.75/24.
        calculator.update_time(jd)
        test_results[test] = "Pass"
    except:
        test_results[test] = "Fail"

    try:
        test = "Run"
        calculator.get_heights()
        test_results[test] = "Pass"
    except:
        test_results[test] = "Fail"

    try:
        test = "loop"
        for jd in np.linspace(jd, jd+1, 1/24.):
            calculator.update_time(jd)
            heights = calculator.get_heights(return_heights=True, unit="km")
            #print("height_v(jd=%f) = %f"%(jd, heights[0]))
        test_results[test] = "Pass"
    except:
        test_results[test] = "Fail"

    for test in test_results.keys():
        print ("%s -> %s"%(test, test_results[test]))

#     if speedtest is False:
#         import matplotlib.pyplot as plt
#         import matplotlib.animation as animation
#         import matplotlib.patches as patches

#     if speedtest is False:

#         calculator.fig, calculator.ax = plt.subplots(figsize=(10,10))
#         calculator.line1, = calculator.ax.plot([], [], 'ro', lw=2)
#         calculator.line2, = calculator.ax.plot([], [], '.-', lw=5)
#         calculator.line3, = calculator.ax.plot([], [], '.-', lw=5)
#         calculator.path, = calculator.ax.plot([],[],'-', lw=1)

#         calculator.ax.grid()
#         calculator.xdata, calculator.ydata = [], []
#         calculator.pathx, calculator.pathy = [], []

#     # if speedtest:
#     #     for i in range (24*365):
#     #         junk = calculator.update_positions(i, speedtest=True, goFast=True)
#     # else:
#     #     ani = animation.FuncAnimation(fig, calculator.animation_update_positions, frames=(24*365), repeat_delay=0,
#     #                                     blit=False, interval=10, init_func=calculator.init_plotting_animation, repeat=0)
#     #     import os
#     #     ani.save("%s/tmp/im.mp4"%(os.environ["HOME"]))
# #plt.show()

# def vecmag(a, origin=[0,0,0]):
#     """ Return the magnitude of a set of vectors around an abritrary origin """
#     if len(np.shape(origin)) == 1:
#         return np.sqrt(np.square(a[0] - origin[0]) + np.square(a[1] - origin[1]) + np.square(a[2] - origin[2]))
#     else:
#         return np.sqrt(np.square(a[:, 0] - origin[:, 0]) + np.square(a[:, 1] - origin[:, 1]) + np.square(a[:, 2] - origin[:, 2]))

# def ang2horizon(xyz, xyz_center, radius=6.357e6, degree=True):
#     """
#     This is the projected angle to the horizon given two vector positions and a diameter of object b.
#     Units don't matter so long as they are all the same.
#     Default radius = Earth radius in meters.
#     """

#     theta = np.pi/2.0 - np.arccos( radius / vecmag( xyz, origin=xyz_center ) )
#     if hasattr(xyz, "unit") and hasattr(xyz_center, "unit"):
#         xyz = xyz.to("m").value
#         xyz_center = xyz_center.to("m").value
    
#     if hasattr(radius, "unit"):
#         radius.to("m")

#     # Else we assume that the positions are already in meters.
#     if degree:
#         return(np.rad2deg(theta))
#     else:
#         return(theta)

# class vecmath(object):

#     def __init__(self):
#         self.function_dict = {
#             2:self.__vecdot2__,
#             3:self.__vecdot3__,
#             4:self.__vecdot4__}

#     def __vecdot2__(self, a, b, origin):
        
#         return(a[0]   * b[0]   + a[1]   * b[1]   + a[2]   * b[2])

#     def __vecdot3__(self, a, b, origin):
#         if len(np.shape(b)) > len(np.shape(a)):
#             a,b = b,a

#         return(a[:,0] * b[0]   + a[:,1] * b[1]   + a[:,2] * b[2])

#     def __vecdot4__(self, a, b, origin):
#         if (np.shape(a) == np.shape(b)):
#             return(a[:,0] * b[:,0] + a[:,1] * b[:,1] + a[:,2] * b[:,2])
#         else:
#             sys.exit("Dimensionality of arrays does not match. Arrays must be a) Two 1d arrays b) 1 1d and 2d, or c) 2 2d arrays of the same")

#     def dot(self, a,b, origin=np.array([0,0,0])):
#         a = a - origin
#         b = b - origin

#         dim_check = len(np.shape(a)) + len(np.shape(b))
#         # dim_check 2 == two 1d arrays. No slicing
#         # dim_check 3 == 1 1d array and 1 2d array. Some slicing
#         # dim_check 4 == 2 2d arrays. Full slicing

#         self.function_dict[dim_check](a,b, origin)

# class shadow_calc_old():
#     def __init__(self, observatory_name="APO",
#     observatory_elevation=2788.0*u.m,
#     observatory_lat='32.7802777778N',
#     observatory_lon = '105.8202777778W',
#     jd=False,
#     eph=None,
#     earth=None,
#     sun=None):
#         """
#         Initialization sets a default observatory to LCO, and the default time to the date when initialized.
#         """
#         super().__init__()
#         # Load the ephemeral datat for the earth and sun. 
#         if eph is None:
#             self.eph = load('de421.bsp')
        
#         self.earth = earth
#         if self.earth is None:
#             self.earth = self.eph['earth']
        
#         self.sun = sun
#         if self.sun is None:
#             self.sun = self.eph['sun']

#         # Set the radius of the earth. This is chosen to be a value with units to make later conversions easier.
#         self.r_earth = 6378*u.km

#         self.observatory_elevation = observatory_elevation
#         try:
#             self.observatory_elevation.to("m")
#         except:
#             sys.exit("Observatory elevation does not have unit of length")

#         self.observatory_name = observatory_name
#         self.observatory_lat = observatory_lat
#         self.observatory_lon = observatory_lon
#         self.observatory_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value)
#         self.ts = load.timescale()

#         if jd is not False:
#             self.t = self.ts.tt_jd(jd)

#         else:
#             self.t = self.ts.now()

#         self.update_xyz()
#         # define these at time self.t xyz_observatory_m, xyz_earth_m, xyz_sun_m

#     def set_observatory_heidelberg(self):
#         # The self object defaults to contain an observatory at LCO. This can be over written.
#         self.observatory_lat = '%fN'%(49 + 24/60. + 7.8/3600)
#         self.observatory_lon = '%fE'%(8 + 40/60. + 18.6/3600)
#         self.observatory_elevation = 1000*u.m
#         self.observatory_name = "Heidelberg"
#         self.observatory_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value)

#     def update_xyz(self):
#         """
#         Due to the nature of the vectors, it is not possible to combine the earth and the observatory topo directly. Therefore
#         this requires the calculation of an observatory position to be done using the XYZ positions defined at a particular time, and with a particular unit.
#         To prevent unit issues later from arising SI units are adopted. In the case where other units are desired, like plotting the solar system, local conversions
#         can be done.
#         """
#         self.xyz_earth_m = self.earth.at(self.t).position.m
#         self.xyz_sun_m = self.sun.at(self.t).position.m
#         self.xyz_observatory_m = self.earth.at(self.t).position.m + self.observatory_topo.at(self.t).position.m

#     def update_t(self, jd):
#         self.t = self.ts.tt_jd(jd)
#         self.update_xyz()

#     def point_in_earth_shaddow(self, xyz_point_m, xyz_sun_m, xyz_earth_m):
#         """ 
#         Determine if a point in 3d space is in the earth shadow using the projected angle of the sun from the earth compared to
#         the projected angle of the horizon from the earth.

#         Parameters
#         -----------
#         xyz_point_m: numpy.array with shape (3, N) or (3,)
#             the 3d vector position of a list of points along a ray in meters
#         xyz_sun_m: numpy.array with shape (3,)
#             3d position of the sun
#         xyz_earth_m: numpy.array with shape (3,)
#             3d position of the center of the earth

#         Returns
#         -------
#         boolean or array of booleans [True or False] for positions in the shaddow or not.        
#         """

#         if len(np.shape(xyz_point_m)) == 2 :
#             # If the number of points returned is more than one reshape the other vectors to match.
#             # The ang to sun and horizon expect the dimensions of vectors to match
#             N_points = np.shape(xyz_point_m)[0]
#             xyz_sun_m = np.repeat(np.array([xyz_sun_m]), N_points, axis=0)
#             xyz_earth_m = np.repeat(np.array([xyz_earth_m]), N_points, axis=0)

#         ang_sun_earth_from_point = vecang(xyz_sun_m, xyz_earth_m, origin=xyz_point_m, degree=True)
#         ang_horizon_from_point = ang2horizon(xyz_point_m, xyz_center=xyz_earth_m, degree=True)

#         # Return a single bol, or an array 
#         return(ang_horizon_from_point>ang_sun_earth_from_point)

#     def get_zenith_shaddow(self, hmin=1e3*u.km, hmax=1e4*u.km, dh0=1*u.km, n_subs=10, max_error=0.01, max_iteration=10, simple_output=True):
#         """ This is a zenith shaddow height calculation, so it's ok to calculate the eph or earth and sun."""

#         # initialize the iteration counter and the returned error value
#         iteration = 1
#         error = max_error + 1

#         def get_first_illuminated_m(self, heights):
#             for i_height, height in enumerate(heights):
#                 self.observatory_zenith_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=height.to("m").value)
#                 self.xyz_point_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m
            
#                 if self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m) == False:
#                     return(i_height, height.to("m").value)

#             return(-1, -1)

#         # Iterate to get an accurate shaddow height
#         while ( (iteration <= max_iteration) or (error > max_error) ):

#             heights = np.linspace(hmin, hmax, num=n_subs, endpoint=True)
#             dh_m = (hmax.to("m").value - hmin.to("m").value)/float(n_subs)

            
#             h_i, h_m = get_first_illuminated_m(self, heights)

#             # First check the first illuminated point is not the first point. If so the sun is visible.
#             if h_i > 0:
#                 # Calculate the relative error of the returned h relative to the step size
#                 error = dh_m / h_m

#                 if (error <= max_error):
#                     if simple_output:
#                         return(h_m)
#                     else:
#                         return(h_m*u.m, error, iteration)

#                 else:
#                     # Iterate the iteration counter
#                     iteration += 1
#                     hmin = heights[ h_i -1 ]
#                     hmax = heights[ h_i ]

#             elif h_i == 0:
#                 return ( np.nan*u.m, np.nan, 1 )

#             elif ( (h_i == -1) and (h_m == -1) ):
#                 return ( np.nan*u.m, np.nan, max_iteration )

#     def height_from_radec(self, ra, dec, dmin=500*u.km, dmax=1e5*u.km, n_subs=10, max_error=0.01, max_iteration=10, simple_output=True):

#         """
#         You can get the xyz coordinates with the following code:
#         xyz_earth_m = earth.at(t).position.m
#         xyz_sun_m = sun.at(t).position.m
#         xyz_observatory = xyz_earth_m + observatory_topo.at(t).position.m

#         Returns:
#             if simple_output == True:
#                 return(shadow_height)
#             else:
#                 return(shadow_height, distance, error, distance_index, iteration)
#             shadow_height == flat -> safe
#             shadow_height == +np.ing -> shadow height beyond max-calculated height
#             shadow_height == -np.inf -> in sun or below earth surface
#         """
#         # initialize the iteration counter and the returned error value
#         iteration = 1

#         # Adding one to the required error, a fraction limited to 1, ensures that the first iteration proceeds.
#         error = max_error + 1

#         def get_first_illuminated_m(distances):
#             """
#             returns the index of the first illuminated distance, that distance, and the shadow height of that poing.
#             shadow_height_m, distance_m, distance_index = get_first_illuminated_m(distances)
#             Returns
#             (-1, -1, -1): point is under the horizon
#             (altitude_point_to_earth_m, distance, -2): max distance is in shadow
#             else:
#             (altitude_point_to_earth_m, distance, distance_index): first distance illuminated
#             Note: distance_index == 0 indicates sun is up
#             """

#             # Step 1, check if the first point is above the horizon
#             distance_au = distances[0].to("au").value

#             # Calculate the 3d position vector of the first point relative to the observatory
#             self.point_along_ray =  position_from_radec(ra, dec, distance=distance_au, epoch=None, t=self.t, center=self.observatory_topo, target=None) 

#             # Calculate that position in xyz space
#             self.xyz_point_m = self.xyz_observatory_m + self.point_along_ray.position.m

#             # Calculate the altitude to see if the point faces the earth.
#             altitude_point_to_earth_m = np.sqrt(np.sum( np.array( self.xyz_point_m - self.xyz_earth_m )**2 )) - (self.r_earth.to("m").value)

#             if altitude_point_to_earth_m <= 0:
#                 # Index -1 is a flag to indicate the direction faces into the earth.

#                 return(-1, -1, -1)

#             # Step 2, check if the last distance is in the shadow, then ALL the points are in the shadow
#             distance = distances[-1]

#             # Calculate the 3d position vector of the last point relative to the observatory
#             self.point_along_ray =  position_from_radec(ra, dec, distance=distance.to("au").value, epoch=None, t=self.t, center=self.observatory_topo, target=None)

#             # Calculate that position in xyz space
#             self.xyz_point_m = self.xyz_observatory_m + self.point_along_ray.position.m

#             if self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m) == True:
#                 # Only calculate the altitude of the shadow if the point is in the shadow
#                 altitude_point_to_earth_m = np.sqrt(np.sum( np.array( self.xyz_point_m - self.xyz_earth_m )**2 )) - self.r_earth.to("m").value
#                 # Index -2 is a flag to indicate the direction at the maximum distance is still in the shadow.
#                 # This is likely still a useful shadow height however, so we return the maximum tested distance and the altitude
#                 return(altitude_point_to_earth_m, distance, -2)

#             # Step 3, check all remaining points to see which is in the shadow
#             self.points_along_ray =  position_from_radec(ra, dec, distance=distances.to("au").value, epoch=None, t=self.t, center=self.observatory_topo, target=None) 
            
#             # Transpose the returned array of positions
#             self.xyz_points_m = self.points_along_ray.position.m.T + np.repeat([self.xyz_observatory_m], len(distances), axis=0)
#             self.points_in_shaddow = self.point_in_earth_shaddow(self.xyz_points_m, self.xyz_sun_m, self.xyz_earth_m)

#             # Get the first illuminated distance, where we are NOT in the shaddow
#             i_distance = np.min(np.where(np.logical_not(self.points_in_shaddow)))

#             altitude_point_to_earth_m = np.sqrt(np.sum( np.array( self.xyz_points_m[i_distance] - self.xyz_earth_m )**2 )) - self.r_earth.to("m").value

#             return(altitude_point_to_earth_m, distances[i_distance].to("m").value, i_distance)

#             # # Modified internal routines to accept numpy arrays, the remainder is no longer necessary
#             # # start with self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m)

#             # for i_distance, distance in enumerate(distances):

#             #     # Calculate the 3d position vector relative to the observatory
#             #     self.point_along_ray =  position_from_radec(ra, dec, distance=distance.to("au").value, epoch=None, t=self.t, center=self.observatory_topo, target=None) 

#             #     # Calculate that position in xyz space
#             #     self.xyz_point_m = self.xyz_observatory_m + self.point_along_ray.position.m

#             #     # Check if point is in the shadow
#             #     if self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m) == False:
#             #         # If in the shadow calculate and return the altitude above the earth.
#             #         altitude_point_to_earth_m = np.sqrt(np.sum( np.array( self.xyz_point_m - self.xyz_earth_m )**2 )) - self.r_earth.to("m").value
                    
#             #         return(altitude_point_to_earth_m, distance.to("m").value, i_distance)

#         def final_shadow_height(self, shadow_height, distance, error, distance_index, iteration, simple_output=True):
#             """
#             This cleans the final product, to return either a shaddow hight or the full result of the calculation.
#             simple_output == True by default. Otherwise it is used for debugging purposes. 
#             usage:
#             shadow_height, distance, error, iteration = final_shadow_height(shadow_height, distance, error, i_distance, simple_output=False)
#             or
#             shadow_height = final_shadow_height(shadow_height, distance, error, i_distance, simple_output=True)
#             """
#             if simple_output:
#                 output_dict = {"height":shadow_height}
#             else:
#                 output_dict = {
#                     "height":shadow_height,
#                     "distance":distance,
#                     "error":error,
#                     "distance_index":distance_index,
#                     "iterations":iteration}
            
#             return(output_dict)


#         # Iterate to get an accurate shaddow height
#         while ( (iteration <= max_iteration) and (error > max_error) ):

#             distances = np.linspace(dmin, dmax, num=n_subs, endpoint=True)

#             try:
#                 delta_distance_m = dmax.to("m").value - dmin.to("m").value
#             except:
#                 sys.exit("distances to calculate shadow heights are not provided with astropy-lenght units. This is unsafe. Exiting...")

#             # This could fail if n_subs == to zero, or provided as not a number.
#             try:
#                 delta_distance_m /= float(n_subs)
#             except:
#                 sys.exit("Something failed in calculating the height. Check that n_subs is an integer greater than 0")

#             """
#             Get the first illuminated value. This can return 3 different outcomes
#             1) distance_index  >  0, indicating that a distance in the sun has been found, corresponding to the index d_i
#             2) distance_index ==  0, indicating that the first distance is in the sun, i.e. the observatory is illuminated
#             3) distance_index == -2, indicating that the maximum distance checked is in the shadow
            
#             """
#             shadow_height_m, shadow_distance_m, distance_index = get_first_illuminated_m(distances)

#             # First check the first illuminated point is not the first point. If so the sun is visible.
#             if distance_index > 0:
#                 # Calculate the relative error of the returned h relative to the step size
#                 error = float(delta_distance_m) / float(shadow_distance_m)

#                 # Check if the error to the distance of the shadow meets our requierments
#                 if (error > max_error):
#                     # Iterate the iteration counter
#                     iteration += 1

#                     # Revise dmin and dmax to blanket the shadow position. 
#                     # This algorithm is a modification to Newton's method, or a shooting method
#                     dmin = distances[ distance_index - 1 ]
#                     dmax = distances[ distance_index ]
#                 # If the above statement is NOT TRUE then we should exit the while loop and return the shadow height
#                 else:
#                     break

#             elif distance_index == 0:
#                 """
#                 if d_i = 0 then the observatory is in the sun.
#                 """
#                 # Height of shadow is nan, there is no shadow
#                 shadow_height_m = 0.0
#                 # Distance to shadow is nan, there is no shadow
#                 shadow_distance_m = 0.0
#                 # error is ... good?
#                 error = 0.0
#                 # d_i is already zere, which serves as a good flag.

#             elif ( distance_index == -2 ):
#                 # then the maximum calculated distance is in the shadow.
#                 # When using, a positive infinity is safe. Assume maximum shadow height
#                 shadow_height_m = 12e6
#                 shadow_distance_m = 12e6
#                 error = 0.0

#             elif( distance_index == -1 ):
#                 # Position is below the horizon, as in under the earth
#                 shadow_height_m = -1.0
#                 shadow_distance_m = -1.0
#                 error = -1
                
#         return(final_shadow_height(self, shadow_height_m, shadow_distance_m, error, distance_index, iteration, simple_output=simple_output))


#     def heidelberg_test(self, get_shaddow_only=True):
#         """ 
#         Heidelberg Test
#         Sanity test to ensure there no mistakes with angles between standard coordinate systems, which I didn't define.
#         This has been used to ensure the horizon and shadow height calculations are consistent, for example with what I can see out my window at sun set.
#         """

#         # The self object defaults to contain an observatory at LCO. This can be over written.
#         self.observatory_lat = '%fN'%(49 + 24/60. + 7.8/3600)
#         self.observatory_lon = '%fE'%(8 + 40/60. + 18.6/3600)
#         self.observatory_elevation = 1000*u.m
#         self.observatory_name = "Heidelberg"
#         self.observatory_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value)

#         # Create a new topological coordinate, using the zenith point, with an elevation 10x the elevation of Heidelberg.
#         self.observatory_zenith_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value*10.0)

#         # Animations must be done in terms of frames. We thus need to calculate the time stricktly from frame number. 
#         # The class initializes itself to the current time by default.
#         # For the purpose of animation we want to be able to add an arbitrary amount of time to the current julian date. So we save the current time.
#         jd0 = self.t.tt

#         # We calculate a time step every 15, using a linearly spaced array, from 0-23 hour, with 24*4 time steps. Prevent problems with repeat at 0 and 24, endpoint = false.
#         steps_per_hour = 4
#         for d_hr in np.linspace(0, 24, 24*steps_per_hour,endpoint=False):

#             # Must update time, which includes a call to update xyz of earth BEFORE getting xyz of zenith.
#             self.update_t(jd0 + d_hr/24); self.xyz_observatory_zenith_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m

#             horizon_alt = ang2horizon(self.xyz_observatory_m, xyz_center=self.xyz_earth_m, degree=True)

#             ang_sun_zenith = vecang(self.xyz_sun_m, self.xyz_observatory_zenith_m, origin=self.xyz_observatory_m, degree=True)

#             # Let's now get the shaddow height above heidelberg.

#             if get_shaddow_only is False:
#                 print("angle to horizon for a point %0.1fm above heidelberg = %f"%(self.observatory_elevation, horizon_alt))
#                 print("sun alt = %f"%(90.-ang_sun_zenith))
#                 print("h\ttheta1\ttheta2\tshaddow")
                

#                 for elevation in np.linspace(1e3*u.km, 10e3*u.km, 10):
#                     self.observatory_zenith_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=elevation.to("m").value)
#                     self.xyz_point_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m
                
#                     ang_sun_earth_from_point = vecang(self.xyz_sun_m, self.xyz_earth_m, origin=self.xyz_point_m, degree=True)
#                     ang_horizon_from_point = ang2horizon(self.xyz_point_m, xyz_center=self.xyz_earth_m, degree=True)
#                     in_shaddow = ang_horizon_from_point>ang_sun_earth_from_point
#                     print( "%0.1e\t%0.2f\t%0.2f\t%s"%( elevation.to("km").value, ang_sun_earth_from_point, ang_horizon_from_point, in_shaddow ) )
            
#             shaddow_h, error, iteration = self.get_zenith_shaddow(hmin=1*u.km, hmax=1e4*u.km, dh0=1*u.km, n_subs=10, max_error=0.01, max_iteration=10, simple_output=False)
#             if np.isfinite(shaddow_h):
#                 print("Found shaddow height %fkm with %f0.3 error in %i iterations"%(shaddow_h.to("km").value, error, iteration))
#             elif iteration == 1:
#                 print("the sun is up")
#             else:
#                 print("shaddow height is beyond maximum height")
                
#         return()

#     def init_plotting_animation(self,xdata,ydata,jd0):
#         self.ax.set_ylim(-1.5 * self.r_earth.to("au").value, 1.5 * self.r_earth.to("au").value)
#         self.ax.set_xlim(-1.5 * self.r_earth.to("au").value, 1.5 * self.r_earth.to("au").value)
#         del xdata[:]
#         del ydata[:]
#         self.line1.set_data(xdata, ydata)
#         self.line2.set_data(xdata, ydata)
#         global day
#         day = jd0

#     def animation_update_positions(self, frame, speedtest=False, goFast=True, dt_frame=(1/24.) ):
#         if speedtest is False:
#             if len( self.ax.patches ) > 1:
#                 del( self.ax.patches[-1] )
#         self.t = self.ts.tt_jd( jd0 + frame*dt_frame)
#         self.update_positions()

#         if goFast:
#             self.xyz_observatory_zenith_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m

#         else:
#             self.xyz_observatory_zenith_m = self.sun.at(self.t).observe(self.earth+self.observatory_zenith_topo).position.m

#         N_ra = 23
#         N_dec = 7
#         x_heights_m = np.zeros(len(N_ra * N_dec)+1)
#         y_heights_m = np.zeros(len(N_ra * N_dec)+1)

#         height_au = 3.0 * self.r_earth.to("au").value
#         for ra_i, ra in enumerate(np.linspace(1,24,N_ra)):
#             for dec_j, dec in enumerate(np.linspace(-90, 0, num=N_dec, endpoint=True)):
#                 # for height in heights:
#                     self.point_along_ray =  position_from_radec(ra, dec, distance=height_au, epoch=None, t=self.t, center=self.observatory_topo, target=None) #, observer_data=LCO_topo.at(t).observer_data) 
#                     self.xyz_along_ray_m = self.xyz_observatory_m + self.point_along_ray.position.m
#                     x_heights_m[ ra_i + dec_j*N_ra ] = self.xyz_along_ray_m[0]
#                     y_heights_m[ ra_i + dec_j*N_ra ] = self.xyz_along_ray_m[1]

#         x_heights_m[-1] = self.xyz_earth_m[0]
#         y_heights_m[-1] = self.xyz_earth_m[1]

#         if speedtest is False:
#             self.ax.set_xlim( ( self.xyz_earth_m[0] * u.m ).to( "au" ).value - height_au * 1.5, (self.xyz_earth_m[0] * u.m ).to( "au" ).value + height_au * 1.5 )
#             self.ax.set_ylim( ( self.xyz_earth_m[1] * u.m ).to( "au" ) - height_au * 1.5, (self.xyz_earth_m[1] * u.m).to( "au" ) + height_au * 1.5 )

#             circ = patches.Circle( ( self.xyz_earth_m[0] * u.m ).to( "au" ).value, ( self.xyz_earth_m[1] * u.m ).to( "au" ).value, self.r_earth.to( "au" ).value, alpha=0.8, fc='yellow') 
#             self.ax.add_patch( circ )

#             self.line1.set_data( (x_heights_m * u.m).to( "au" ).value, ( y_heights * u.m ).to("au").value )
#             self.line2.set_data( [ self.xyz_observatory_zenith[0] ], [ self.xyz_observatory_zenith[1] ] )
#             self.line3.set_data( [ self.xyz_observatory[0] ] , [ self.xyz_observatory[1] ] )
#             self.pathx.append( self.xyz_earth_m[0] )
#             self.pathy.append( self.xyz_earth_m[1] )
#             self.path.set_data( pathx, pathy )
        
#             print("frame %i"%frame)
#         else:
#             print("frame speed test %i"%frame)
#             pass

#         #line2.set_data([sun.at(t).position.au[0]],[sun.at(t).position.au[1]])
#         print("frame %i"%frame)

#     def defaults(self):
#         self.LCO_elevation = 2380
#         self.LCO_topo   = Topos('29.01597S', '70.69208W', elevation_m=LCO_elevation)

#         return(LCO_topo)
