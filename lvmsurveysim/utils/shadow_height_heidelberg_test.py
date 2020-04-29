#!/usr/bin/python3
if __name__ == "__main__":

    import shadow_height_lib
    import astropy.units as u
    import numpy as np

    shadow_calculator = shadow_height_lib.shadow_calc()

    # Disable speed test
    shadow_height_lib.speedtest = False

    shadow_calculator.set_observatory_heidelberg()
    shadow_calculator.update_xyz()

    # Get todays JD.
    jd0 = shadow_calculator.ts.now().tt
    print("The position of the shadow along a ray to Orion, from Heidelberg, is")
    print(("%s"+"%12s"*3)%("JD", "x", "y", "z"))
    for hour in range(24):

        shadow_calculator.update_t(jd0+hour/24.)
        #shadow_calculator.heidelberg_test()
        # Get shadow height to Orion on todays date + 0.5 a day because you are probably running this during working hours.
        ra_hours = 5.5
        dec_degrees = -5.5

        # Get the index of the distance position, the distance in meters, and the altitude of the shadow height
        shadow_dict   = shadow_calculator.height_from_radec(ra_hours, dec_degrees, dmin=100*u.km, simple_output=False)
        distance_m = shadow_dict['distance']/1e3
        shadow_height_m = shadow_dict['height']/1e3
        error = shadow_dict['error'] * distance_m/1e3
        if np.isfinite(shadow_height_m):
            print("%0.2f [%0.3E, %0.3E, %0.3E]:  %0.1f km +/- %0.3f with a height of %0.1f in %i iterations"%(shadow_calculator.t.tt,
            shadow_calculator.xyz_observatory_m[0],
            shadow_calculator.xyz_observatory_m[1],
            shadow_calculator.xyz_observatory_m[2],
            distance_m,
            error,
            shadow_height_m,
            shadow_dict["iterations"]))

        else:
            print("%0.2f [%0.3E, %0.3E, %0.3E]:  in the sun"%(shadow_calculator.t.tt,
            shadow_calculator.xyz_observatory_m[0],
            shadow_calculator.xyz_observatory_m[1],
            shadow_calculator.xyz_observatory_m[2]))
