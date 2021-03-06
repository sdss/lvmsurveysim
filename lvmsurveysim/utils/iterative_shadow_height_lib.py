#!/usr/bin/python3
import time
import numpy as np
import sys
from astropy import units as u
from skyfield.api import load
from skyfield.api import Topos
from skyfield.positionlib import position_from_radec, Geometric

def vecdot(a,b,origin=[0,0,0]):
    a = a - origin
    b = b - origin

    # To slice or not to slice
    if len(np.shape(origin)) == 1:
        return(a[0]   * b[0]   + a[1]   * b[1]   + a[2]   * b[2])    
    else:
        return(a[:,0] * b[:,0] + a[:,1] * b[:,1] + a[:,2] * b[:,2])

def vecmag(a, origin=[0,0,0]):
    """ Return the magnitude of a set of vectors around an abritrary origin """
    if len(np.shape(origin)) == 1:
        return( (( a[0]  - origin[0]   )**2 + (a[1]   - origin[1]  )**2 + (a[2]   - origin[2]  )**2)**0.5)
    else:
        return( ((a[:,0]- origin[:,0] )**2 + (a[:,1] - origin[:,1])**2 + (a[:,2] - origin[:,2])**2)**0.5)

def vecang(a, b, origin=[0,0,0],degree=False):
    """ Compute the angle between a and b with an origion, or set of origins """

    theta = np.arccos(vecdot(a,b,origin)/(vecmag(a,origin=origin) * vecmag(b,origin=origin)))

    if degree:
        return( np.rad2deg(theta) )
    else:
        return(theta)

def ang2horizon(xyz, xyz_center, radius=6.357e6, degree=True):
    """
    This is the projected angle to the horizon given two vector positions and a diameter of object b.
    Units don't matter so long as they are all the same.
    Default radius = Earth radius in meters.
    """

    theta = np.pi/2.0 - np.arccos( radius / vecmag( xyz, origin=xyz_center ) )
    if hasattr(xyz, "unit") and hasattr(xyz_center, "unit"):
        xyz = xyz.to("m").value
        xyz_center = xyz_center.to("m").value
    
    if hasattr(radius, "unit"):
        radius.to("m")

    # Else we assume that the positions are already in meters.
    if degree:
        return(np.rad2deg(theta))
    else:
        return(theta)

class shadow_calc():
    def __init__(self, observatory_name="LCO",
    observatory_elevation=2380.0*u.m,
    observatory_lat='29.0146S',
    observatory_lon = '70.6926W',
    jd=2459458,
    eph=None,
    earth=None,
    sun=None):
        """
        Initialization sets a default observatory to LCO, and the default time to the date when initialized.
        """
        super().__init__()
        # Load the ephemeral datat for the earth and sun. 
        if eph is None:
            self.eph = load('de421.bsp')
        
        self.earth = earth
        if self.earth is None:
            self.earth = self.eph['earth']
        
        self.sun = sun
        if self.sun is None:
            self.sun = self.eph['sun']

        # Set the radius of the earth. This is chosen to be a value with units to make later conversions easier.
        self.r_earth = 6378*u.km

        self.observatory_elevation = observatory_elevation
        try:
            self.observatory_elevation.to("m")
        except:
            sys.exit("Observatory elevation does not have unit of length")

        self.observatory_name = observatory_name
        self.observatory_lat = observatory_lat
        self.observatory_lon = observatory_lon
        self.observatory_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value)
        self.ts = load.timescale()

        if jd is not False:
            self.t = self.ts.tt_jd(jd)

        else:
            self.t = self.ts.now()

        self.update_xyz()
        # define these at time self.t xyz_observatory_m, xyz_earth_m, xyz_sun_m

    def set_observatory_heidelberg(self):
        # The self object defaults to contain an observatory at LCO. This can be over written.
        self.observatory_lat = '%fN'%(49 + 24/60. + 7.8/3600)
        self.observatory_lon = '%fE'%(8 + 40/60. + 18.6/3600)
        self.observatory_elevation = 1000*u.m
        self.observatory_name = "Heidelberg"
        self.observatory_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value)

    def update_xyz(self):
        """
        Due to the nature of the vectors, it is not possible to combine the earth and the observatory topo directly. Therefore
        this requires the calculation of an observatory position to be done using the XYZ positions defined at a particular time, and with a particular unit.
        To prevent unit issues later from arising SI units are adopted. In the case where other units are desired, like plotting the solar system, local conversions
        can be done.
        """
        self.xyz_earth_m = self.earth.at(self.t).position.m
        self.xyz_sun_m = self.sun.at(self.t).position.m
        self.xyz_observatory_m = self.earth.at(self.t).position.m + self.observatory_topo.at(self.t).position.m

    def update_t(self, jd):
        self.t = self.ts.tt_jd(jd)
        self.update_xyz()

    def point_in_earth_shaddow(self, xyz_point_m, xyz_sun_m, xyz_earth_m):
        """ 
        Determine if a point in 3d space is in the earth shadow using the projected angle of the sun from the earth compared to
        the projected angle of the horizon from the earth.
        Parameters
        -----------
        xyz_point_m: numpy.array with shape (3, N) or (3,)
            the 3d vector position of a list of points along a ray in meters
        xyz_sun_m: numpy.array with shape (3,)
            3d position of the sun
        xyz_earth_m: numpy.array with shape (3,)
            3d position of the center of the earth
        Returns
        -------
        boolean or array of booleans [True or False] for positions in the shaddow or not.        
        """

        if len(np.shape(xyz_point_m)) == 2 :
            # If the number of points returned is more than one reshape the other vectors to match.
            # The ang to sun and horizon expect the dimensions of vectors to match
            N_points = np.shape(xyz_point_m)[0]
            xyz_sun_m = np.repeat(np.array([xyz_sun_m]), N_points, axis=0)
            xyz_earth_m = np.repeat(np.array([xyz_earth_m]), N_points, axis=0)

        ang_sun_earth_from_point = vecang(xyz_sun_m, xyz_earth_m, origin=xyz_point_m, degree=True)
        ang_horizon_from_point = ang2horizon(xyz_point_m, xyz_center=xyz_earth_m, degree=True)

        # Return a single bol, or an array 
        return(ang_horizon_from_point>ang_sun_earth_from_point)

    def get_zenith_shaddow(self, hmin=1e3*u.km, hmax=1e4*u.km, dh0=1*u.km, n_subs=10, max_error=0.01, max_iteration=10, simple_output=True):
        """ This is a zenith shaddow height calculation, so it's ok to calculate the eph or earth and sun."""

        # initialize the iteration counter and the returned error value
        iteration = 1
        error = max_error + 1

        def get_first_illuminated_m(self, heights):
            for i_height, height in enumerate(heights):
                self.observatory_zenith_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=height.to("m").value)
                self.xyz_point_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m
            
                if self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m) == False:
                    return(i_height, height.to("m").value)

            return(-1, -1)

        # Iterate to get an accurate shaddow height
        while ( (iteration <= max_iteration) or (error > max_error) ):

            heights = np.linspace(hmin, hmax, num=n_subs, endpoint=True)
            dh_m = (hmax.to("m").value - hmin.to("m").value)/float(n_subs)

            
            h_i, h_m = get_first_illuminated_m(self, heights)

            # First check the first illuminated point is not the first point. If so the sun is visible.
            if h_i > 0:
                # Calculate the relative error of the returned h relative to the step size
                error = dh_m / h_m

                if (error <= max_error):
                    if simple_output:
                        return(h_m)
                    else:
                        return(h_m*u.m, error, iteration)

                else:
                    # Iterate the iteration counter
                    iteration += 1
                    hmin = heights[ h_i -1 ]
                    hmax = heights[ h_i ]

            elif h_i == 0:
                return ( np.nan*u.m, np.nan, 1 )

            elif ( (h_i == -1) and (h_m == -1) ):
                return ( np.nan*u.m, np.nan, max_iteration )

    def height_from_radec(self, ra, dec, dmin=500*u.km, dmax=1e5*u.km, n_subs=10, max_error=0.01, max_iteration=10, simple_output=True, ra_degree=False):

        """
        You can get the xyz coordinates with the following code:
        xyz_earth_m = earth.at(t).position.m
        xyz_sun_m = sun.at(t).position.m
        xyz_observatory = xyz_earth_m + observatory_topo.at(t).position.m
        Returns:
            if simple_output == True:
                return(shadow_height)
            else:
                return(shadow_height, distance, error, distance_index, iteration)
            shadow_height == flat -> safe
            shadow_height == +np.ing -> shadow height beyond max-calculated height
            shadow_height == -np.inf -> in sun or below earth surface
        """
        # initialize the iteration counter and the returned error value
        iteration = 1

        # Adding one to the required error, a fraction limited to 1, ensures that the first iteration proceeds.
        error = max_error + 1

        if ra_degree:
            ra = ra/15.0

        def get_first_illuminated_m(distances):
            """
            returns the index of the first illuminated distance, that distance, and the shadow height of that poing.
            shadow_height_m, distance_m, distance_index = get_first_illuminated_m(distances)
            Returns
            (-1, -1, -1): point is under the horizon
            (altitude_point_to_earth_m, distance, -2): max distance is in shadow
            else:
            (altitude_point_to_earth_m, distance, distance_index): first distance illuminated
            Note: distance_index == 0 indicates sun is up
            """

            # Step 1, check if the first point is above the horizon
            distance_au = distances[0].to("au").value

            # Calculate the 3d position vector of the first point relative to the observatory
            self.point_along_ray =  position_from_radec(ra, dec, distance=distance_au, epoch=None, t=self.t, center=self.observatory_topo, target=None) 

            # Calculate that position in xyz space
            self.xyz_point_m = self.xyz_observatory_m + self.point_along_ray.position.m

            # Calculate the altitude to see if the point faces the earth.
            altitude_point_to_earth_m = np.sum( np.array( self.xyz_point_m - self.xyz_earth_m )**2 )**0.5 - (self.r_earth.to("m").value)

            if altitude_point_to_earth_m <= 0:
                # Index -1 is a flag to indicate the direction faces into the earth.

                return(-1, -1, -1)

            # Step 2, check if the last distance is in the shadow, then ALL the points are in the shadow
            distance = distances[-1]

            # Calculate the 3d position vector of the last point relative to the observatory
            self.point_along_ray =  position_from_radec(ra, dec, distance=distance.to("au").value, epoch=None, t=self.t, center=self.observatory_topo, target=None)

            # Calculate that position in xyz space
            self.xyz_point_m = self.xyz_observatory_m + self.point_along_ray.position.m

            if self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m) == True:
                # Only calculate the altitude of the shadow if the point is in the shadow
                altitude_point_to_earth_m = np.sum( np.array( self.xyz_point_m - self.xyz_earth_m )**2 )**0.5 - self.r_earth.to("m").value
                # Index -2 is a flag to indicate the direction at the maximum distance is still in the shadow.
                # This is likely still a useful shadow height however, so we return the maximum tested distance and the altitude
                return(altitude_point_to_earth_m, distance, -2)

            # Step 3, check all remaining points to see which is in the shadow
            self.points_along_ray =  position_from_radec(ra, dec, distance=distances.to("au").value, epoch=None, t=self.t, center=self.observatory_topo, target=None) 
            
            # Transpose the returned array of positions
            self.xyz_points_m = self.points_along_ray.position.m.T + np.repeat([self.xyz_observatory_m], len(distances), axis=0)
            self.points_in_shaddow = self.point_in_earth_shaddow(self.xyz_points_m, self.xyz_sun_m, self.xyz_earth_m)

            # Get the first illuminated distance, where we are NOT in the shaddow
            i_distance = np.min(np.where(np.logical_not(self.points_in_shaddow)))

            altitude_point_to_earth_m = np.sum( np.array( self.xyz_points_m[i_distance] - self.xyz_earth_m )**2 )**0.5 - self.r_earth.to("m").value

            return(altitude_point_to_earth_m, distances[i_distance].to("m").value, i_distance)

            # # Modified internal routines to accept numpy arrays, the remainder is no longer necessary
            # # start with self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m)

            # for i_distance, distance in enumerate(distances):

            #     # Calculate the 3d position vector relative to the observatory
            #     self.point_along_ray =  position_from_radec(ra, dec, distance=distance.to("au").value, epoch=None, t=self.t, center=self.observatory_topo, target=None) 

            #     # Calculate that position in xyz space
            #     self.xyz_point_m = self.xyz_observatory_m + self.point_along_ray.position.m

            #     # Check if point is in the shadow
            #     if self.point_in_earth_shaddow(self.xyz_point_m, self.xyz_sun_m, self.xyz_earth_m) == False:
            #         # If in the shadow calculate and return the altitude above the earth.
            #         altitude_point_to_earth_m = np.sum( np.array( self.xyz_point_m - self.xyz_earth_m )**2 )**0.5 - self.r_earth.to("m").value
                    
            #         return(altitude_point_to_earth_m, distance.to("m").value, i_distance)

        def final_shadow_height(self, shadow_height, distance, error, distance_index, iteration, simple_output=True):
            """
            This cleans the final product, to return either a shaddow hight or the full result of the calculation.
            simple_output == True by default. Otherwise it is used for debugging purposes. 
            usage:
            shadow_height, distance, error, iteration = final_shadow_height(shadow_height, distance, error, i_distance, simple_output=False)
            or
            shadow_height = final_shadow_height(shadow_height, distance, error, i_distance, simple_output=True)
            """
            if simple_output:
                output_dict = {"height":shadow_height}
            else:
                output_dict = {
                    "height":shadow_height,
                    "distance":distance,
                    "error":error,
                    "distance_index":distance_index,
                    "iterations":iteration}
            
            return(output_dict)


        # Iterate to get an accurate shaddow height
        while ( (iteration <= max_iteration) and (error > max_error) ):

            distances = np.linspace(dmin, dmax, num=n_subs, endpoint=True)

            try:
                delta_distance_m = dmax.to("m").value - dmin.to("m").value
            except:
                sys.exit("distances to calculate shadow heights are not provided with astropy-lenght units. This is unsafe. Exiting...")

            # This could fail if n_subs == to zero, or provided as not a number.
            try:
                delta_distance_m /= float(n_subs)
            except:
                sys.exit("Something failed in calculating the height. Check that n_subs is an integer greater than 0")

            """
            Get the first illuminated value. This can return 3 different outcomes
            1) distance_index  >  0, indicating that a distance in the sun has been found, corresponding to the index d_i
            2) distance_index ==  0, indicating that the first distance is in the sun, i.e. the observatory is illuminated
            3) distance_index == -2, indicating that the maximum distance checked is in the shadow
            
            """
            shadow_height_m, shadow_distance_m, distance_index = get_first_illuminated_m(distances)

            # First check the first illuminated point is not the first point. If so the sun is visible.
            if distance_index > 0:
                # Calculate the relative error of the returned h relative to the step size
                error = float(delta_distance_m) / float(shadow_distance_m)

                # Check if the error to the distance of the shadow meets our requierments
                if (error > max_error):
                    # Iterate the iteration counter
                    iteration += 1

                    # Revise dmin and dmax to blanket the shadow position. 
                    # This algorithm is a modification to Newton's method, or a shooting method
                    dmin = distances[ distance_index - 1 ]
                    dmax = distances[ distance_index ]
                # If the above statement is NOT TRUE then we should exit the while loop and return the shadow height
                else:
                    break

            elif distance_index == 0:
                """
                if d_i = 0 then the observatory is in the sun.
                """
                # Height of shadow is nan, there is no shadow
                shadow_height_m = 0.0
                # Distance to shadow is nan, there is no shadow
                shadow_distance_m = 0.0
                # error is ... good?
                error = 0.0
                # d_i is already zere, which serves as a good flag.

            elif ( distance_index == -2 ):
                # then the maximum calculated distance is in the shadow.
                # When using, a positive infinity is safe. Assume maximum shadow height
                shadow_height_m = 12e6
                shadow_distance_m = 12e6
                error = 0.0

            elif( distance_index == -1 ):
                # Position is below the horizon, as in under the earth
                shadow_height_m = -1.0
                shadow_distance_m = -1.0
                error = -1
                
        return(final_shadow_height(self, shadow_height_m, shadow_distance_m, error, distance_index, iteration, simple_output=simple_output))


    def heidelberg_test(self, get_shaddow_only=True):
        """ 
        Heidelberg Test
        Sanity test to ensure there no mistakes with angles between standard coordinate systems, which I didn't define.
        This has been used to ensure the horizon and shadow height calculations are consistent, for example with what I can see out my window at sun set.
        """

        # The self object defaults to contain an observatory at LCO. This can be over written.
        self.observatory_lat = '%fN'%(49 + 24/60. + 7.8/3600)
        self.observatory_lon = '%fE'%(8 + 40/60. + 18.6/3600)
        self.observatory_elevation = 1000*u.m
        self.observatory_name = "Heidelberg"
        self.observatory_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value)

        # Create a new topological coordinate, using the zenith point, with an elevation 10x the elevation of Heidelberg.
        self.observatory_zenith_topo = Topos(self.observatory_lat, self.observatory_lon, elevation_m=self.observatory_elevation.to("m").value*10.0)

        # Animations must be done in terms of frames. We thus need to calculate the time stricktly from frame number. 
        # The class initializes itself to the current time by default.
        # For the purpose of animation we want to be able to add an arbitrary amount of time to the current julian date. So we save the current time.
        jd0 = self.t.tt

        # We calculate a time step every 15, using a linearly spaced array, from 0-23 hour, with 24*4 time steps. Prevent problems with repeat at 0 and 24, endpoint = false.
        steps_per_hour = 4
        for d_hr in np.linspace(0, 24, 24*steps_per_hour,endpoint=False):

            # Must update time, which includes a call to update xyz of earth BEFORE getting xyz of zenith.
            self.update_t(jd0 + d_hr/24); self.xyz_observatory_zenith_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m

            horizon_alt = ang2horizon(self.xyz_observatory_m, xyz_center=self.xyz_earth_m, degree=True)

            ang_sun_zenith = vecang(self.xyz_sun_m, self.xyz_observatory_zenith_m, origin=self.xyz_observatory_m, degree=True)

            # Let's now get the shaddow height above heidelberg.

            if get_shaddow_only is False:
                print("angle to horizon for a point %0.1fm above heidelberg = %f"%(elevation, horizon_alt))
                print("sun alt = %f"%(90.-ang_sun_zenith))
                print("h\ttheta1\ttheta2\tshaddow")
                

                for elevation in np.linspace(1e3*u.km, 10e3*u.km, 10):
                    self.observatory_zenith_topo = Topos('%fN'%Lat, '%fE'%Lon, elevation_m=elevation.to("m").value)
                    self.xyz_point_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m
                
                    ang_sun_earth_from_point = vecang(self.xyz_sun_m, self.xyz_earth_m, origin=self.xyz_point_m, degree=True)
                    ang_horizon_from_point = ang2horizon(self.xyz_point_m, xyz_center=self.xyz_earth_m, degree=True)
                    in_shaddow = ang_horizon_from_point>ang_sun_earth_from_point
                    print( "%0.1e\t%0.2f\t%0.2f\t%s"%( elevation.to("km").value, ang_sun_earth_from_point, ang_horizon_from_point, in_shaddow ) )
            
            shaddow_h, error, iteration = self.get_zenith_shaddow(hmin=1*u.km, hmax=1e4*u.km, dh0=1*u.km, n_subs=10, max_error=0.01, max_iteration=10, simple_output=False)
            if np.isfinite(shaddow_h):
                print("Found shaddow height %fkm with %f0.3 error in %i iterations"%(shaddow_h.to("km").value, error, iteration))
            elif iteration == 1:
                print("the sun is up")
            else:
                print("shaddow height is beyond maximum height")
                
        return()

    def init_plotting_animation(self):
        self.ax.set_ylim(-1.5 * self.r_earth.to("au").value, 1.5 * self.r_earth.to("au").value)
        self.ax.set_xlim(-1.5 * self.r_earth.to("au").value, 1.5 * self.r_earth.to("au").value)
        del xdata[:]
        del ydata[:]
        self.line1.set_data(xdata, ydata)
        self.line2.set_data(xdata, ydata)
        global day
        day = jd0

    def animation_update_positions(self, frame, speedtest=False, goFast=True, dt_frame=(1/24.) ):
        if speedtest is False:
            if len( self.ax.patches ) > 1:
                del( self.ax.patches[-1] )
        self.t = self.ts.tt_jd( jd0 + frame*dt_frame)
        self.update_positions()

        if goFast:
            self.xyz_observatory_zenith_m = self.xyz_earth_m + self.observatory_zenith_topo.at(self.t).position.m

        else:
            self.xyz_observatory_zenith_m = self.sun.at(self.t).observe(self.earth+self.observatory_zenith_topo).position.m

        N_ra = 23
        N_dec = 7
        x_heights_m = np.zeros(len(N_ra * N_dec)+1)
        y_heights_m = np.zeros(len(N_ra * N_dec)+1)

        height_au = 3.0 * self.r_earth.to("au").value
        for ra_i, ra in enumerate(np.linspace(1,24,N_ra)):
            for dec_j, dec in enumerate(np.linspace(-90, 0, num=N_dec, endpoint=True)):
                # for height in heights:
                    self.point_along_ray =  position_from_radec(ra, dec, distance=height_au, epoch=None, t=self.t, center=self.observatory_topo, target=None) #, observer_data=LCO_topo.at(t).observer_data) 
                    self.xyz_along_ray_m = self.xyz_observatory_m + self.point_along_ray.position.m
                    x_heights_m[ ra_i + dec_j*N_ra ] = self.xyz_along_ray_m[0]
                    y_heights_m[ ra_i + dec_j*N_ra ] = self.xyz_along_ray_m[1]

        x_heights_m[-1] = self.xyz_earth_m[0]
        y_heights_m[-1] = self.xyz_earth_m[1]

        if speedtest is False:
            self.ax.set_xlim( ( self.xyz_earth_m[0] * u.m ).to( "au" ).value - height_au * 1.5, (self.xyz_earth_m[0] * u.m ).to( "au" ).value + height_au * 1.5 )
            self.ax.set_ylim( ( self.xyz_earth_m[1] * u.m ).to( "au" ) - height_au * 1.5, (self.xyz_earth_m[1] * u.m).to( "au" ) + height_au * 1.5 )

            circ = patches.Circle( ( self.xyz_earth_m[0] * u.m ).to( "au" ).value, ( self.xyz_earth_m[1] * u.m ).to( "au" ).value, self.r_earth.to( "au" ).value, alpha=0.8, fc='yellow') 
            self.ax.add_patch( circ )

            self.line1.set_data( (x_heights_m * u.m).to( "au" ).value, ( y_heights * u.m ).to("au").value )
            self.line2.set_data( [ self.xyz_observatory_zenith[0] ], [ self.xyz_observatory_zenith[1] ] )
            self.line3.set_data( [ self.xyz_observatory[0] ] , [ self.xyz_observatory[1] ] )
            self.pathx.append( self.xyz_earth_m[0] )
            self.pathy.append( self.xyz_earth_m[1] )
            self.path.set_data( pathx, pathy )
        
            print("frame %i"%frame)
        else:
            print("frame speed test %i"%frame)
            pass

        #line2.set_data([sun.at(t).position.au[0]],[sun.at(t).position.au[1]])
        print("frame %i"%frame)

    def defaults():
        LCO_elevation = 2380
        LCO_topo   = Topos('29.01597S', '70.69208W', elevation_m=LCO_elevation)

        return(LCO_topo)

if __name__ == "__main__":
    speedtest = False

    #intialize the shaddow calculator
    calculator = shadow_calc()
    calculator.heidelberg_test()

    del(calculator)


    calculator = shadow_calc()

    ra_hours = 0.0
    dec_degrees = 0.0

    if speedtest is False:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.patches as patches

    if speedtest is False:

        calculator.fig, calculator.ax = plt.subplots(figsize=(10,10))
        calculator.line1, = calculator.ax.plot([], [], 'ro', lw=2)
        calculator.line2, = calculator.ax.plot([], [], '.-', lw=5)
        calculator.line3, = calculator.ax.plot([], [], '.-', lw=5)
        calculator.path, = calculator.ax.plot([],[],'-', lw=1)

        calculator.ax.grid()
        calculator.xdata, calculator.ydata = [], []
        calculator.pathx, calculator.pathy = [], []

    if speedtest:
        for i in range (24*365):
            junk = calculator.update_positions(i, speedtest=True, goFast=True)
    else:
        ani = animation.FuncAnimation(fig, calculator.animation_update_positions, frames=(24*365), repeat_delay=0,
                                        blit=False, interval=10, init_func=calculator.init_plotting_animation, repeat=0)
        import os
        ani.save("%s/tmp/im.mp4"%(os.environ["HOME"]))
#plt.show()