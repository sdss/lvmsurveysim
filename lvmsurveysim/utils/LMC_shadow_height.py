import numpy as np
import shadow_height_lib
calculator = shadow_height_lib.shadow_calc()


jd = 2459460.8778240727
for hr in np.linspace(0,23,24):
    calculator.update_time(jd+hr/24.0)
    ra,dec = calculator.cone_ra_dec()
    calculator.set_coordinates(np.array([ra]), np.array([dec]))
    heights = calculator.get_heights(return_heights=True, unit="km")
    print(heights)
