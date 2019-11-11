""" Create Healpix Array of the sky and convert to RA and DEC """
import healpy
from astropy_healpix import HEALPix
import math
import astropy.units as u
import lvmsurveysim.utils.obs_class as obs_class
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.coordinates import Galactic
from astropy.time import Time
import numpy as np
import gc

verbose = False
quiet = True

def delta_t(dt_min, dt_max, bins_per_hour):
    N_dt = int(((dt_max - dt_min)*bins_per_hour/u.hour).value) + 1
    return(np.linspace(dt_min, dt_max, N_dt, endpoint=True))

def init_time_dict(obs, observatories, observatory_telescopes={"APO":["0.16m", "1.0m"], "LCO":["0.16m", "1.0m"]}, nMonths=4*12):
    # A reference to the total time dictionary is passed to this routine. 
    # Empty arrays for each month and observatory are added to the dictionary.
    total_time_dict = {}

    for obj in obs.LVM_HPT.keys():
        total_time_dict[obj] = {}

    for month_i in range(nMonths+1):
        # zeroth month used to store total time
        total_time_dict[month_i, "combined", "combined"] = np.zeros(obs.npix)
        for observatory in observatories:
            for telescope in observatory_telescopes[observatory]:
                total_time_dict[month_i, observatory, telescope] = np.zeros(obs.npix)

        for obj in obs.LVM_HPT.keys():
            total_time_dict[obj][month_i] = []

    total_time_dict['nMonths'] = nMonths
    total_time_dict['campaign_mode'] = obs.campaign_mode
    return(total_time_dict)

def map_the_sky(obs,total_time_dict,
                observatories=["APO","LCO"],
                N_days=4*365,
                verbose_debug=False,
                doPlot=True,
                list_of_days=False,
                multi_year_month_flag=True):

    start_obs_year = obs.utc_midnight.datetime.year
    for observatory in observatories:

        #Change the observatory
        obs.set_observatory(observatory=observatory)

        delta_midnight = delta_t(obs.dt_min, obs.dt_max, obs.bins_per_hour)

        for day_i in range(round(N_days/obs.delta_days)):

            if (obs.campaign_mode == True) and (observatory == "APO") and obs.Flag1m[obs.utc_midnight.datetime.month]:
                """ This is not right yet. Need to keep track of campaign mode 1m time spent on target"""
                """ Could do this with a flag, and plot 1m vs 0.16m time seperately with a different color scheme"""
                telescope = "1.0m"
                if verbose == True or quiet == False:
                    print(obs.utc_midnight, "%s campaign mode[engaged]: 1.0m "%(observatory))
            else:
                telescope = "0.16m"
                if verbose == True or quiet == False:
                    print(obs.utc_midnight, "%s campaign mode[disengaged]: 0.16m"%(observatory))

            
            # if (moon_dark_flag):
            #     print("Dark: sun/moon distance : ", "%0.1f"%obs.moon.separation(obs.sun).degree, "< ", obs.bright_moon_sun_dist)
            # else:
            #     print("Bright: sun/moon distance : ", "%0.1f"%obs.moon.separation(obs.sun).degree, "> ", obs.bright_moon_sun_dist)

            secz_dict   = {}

            # Initialize the counter for today
            if obs.min_continuous_window == obs.min_exposure:
                observable_counter = np.zeros(obs.npix,dtype=int)

            else:
                observable_counter = np.zeros([obs.npix, len(delta_midnight)],dtype=int)


            # Initialize the observable window array with the first time step
            # Doing the first step means I don't need an if statement all the time in the iteration of time

            # Create blocking
            sunAltAz = obs.sun.transform_to(AltAz(obstime=obs.utc_midnight+delta_midnight, location=obs.location)).alt
            moon_AltAz = obs.moon.transform_to(AltAz(obstime=obs.utc_midnight+delta_midnight, location=obs.location)).alt
            # This is a per day check.
            moon_dark_flag = obs.moon.separation(obs.sun) <= obs.bright_moon_sun_dist

            # Replace with a obs.HPT.keys() type things
            # Should be a dictionary with secZs. Then just loop over and check for flag. No need to code individually.
            for obj in obs.LVM_HPT.keys():
                secz_dict[obj] = obs.LVM_HPT[obj].transform_to(AltAz(obstime=obs.utc_midnight+delta_midnight, location=obs.location)).secz

            #Take the first time
            dt = delta_midnight[0] 
            dt_i = 0

            if sunAltAz[dt_i] <= obs.max_sun_alt:
                # the alt_az position of each healpix position, adjusted by dt.

                secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz

                # Calculate the secz of each healpix coordinate
                # if obs.nside_lowres < obs.nside:
                #     secz_lowres = obs.healpix_skycoords_lowres.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz
                #     secz = healpy.pixelfunc.ud_grade(secz_lowres, obs.nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)

                # else:
                #     secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz


                # Mask all healpix coordinates that have an acceptable air mass
                observable_mask = (secz <= obs.max_airmass['MW']) * (secz >=1.0)

                # Create an array for this dt to store which positions are observable. 
                observable = np.zeros(obs.npix)

                # Flag the observable positiosn with 1.
                observable[observable_mask] = 1

                # Insert the observable array into the full observable array.
                if obs.min_continuous_window == obs.min_exposure: 
                    observable_counter = observable.copy()
                else:
                    observable_counter[:,dt_i] = observable.copy()

            for dt_i in range(1,len(delta_midnight)):
                #Like before, the dt from midnight, going form -6 hours to +6 hours. Change to -18 sun set to sun rise
                dt = delta_midnight[dt_i]

                # Check if the sun is down
                if verbose == True:
                    if quiet == False:
                        print("Local date/time ", obs.local_midnight+dt, "Sun(alt): ", sunAltAz[dt_i])
                if sunAltAz[dt_i] <= obs.max_sun_alt:
                    # Check if the moon phase is bright or dark.

                    #Set HP targets. If UP and dark then True. Else, False
                    HPT_flag = {'any':False}

                    for obj in obs.LVM_HPT.keys():
                        HPT_flag[obj] = (secz_dict[obj][dt_i] >=1.0) * (secz_dict[obj][dt_i] <= obs.max_airmass[obj]) * (obs.LVM_HPT[obj].separation(obs.moon) > obs.bright_moon_sun_dist)
                        #HPT_flag[obj] = (secz_dict[obj][dt_i] >=1.0) * (secz_dict[obj][dt_i] <= obs.max_airmass[obj]) * moon_dark_flag

                        #If any targets are true set the any flag to true
                        if HPT_flag[obj] == True:
                            #I can count the number of obervable 15min windows using the number of secz entries
                            total_time_dict[obj][obs.utc_midnight.datetime.month].append(secz_dict[obj][dt_i])
                            total_time_dict[obj][0].append(secz_dict[obj][dt_i])

                            HPT_flag['any'] = True
                            if verbose_debug: print("%s:"%obj, secz_dict[obj][dt_i])

                    if HPT_flag['any'] == False :

                        # Again, calcualte the alt-az
                        #obsalt_az = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)) # Use this to calcualte moon distance. Must be greater than XX degree

                        # Get sec for full healpix array
                        #secz_lowres = obs.healpix_skycoords_lowres.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz
                        #secz = healpy.pixelfunc.ud_grade(secz_lowres.value, obs.nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)

                        secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz

                        # if obs.nside_lowres < obs.nside:
                        #     secz_lowres = obs.healpix_skycoords_lowres.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz
                        #     secz = healpy.pixelfunc.ud_grade(secz_lowres, obs.nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)

                        # else:
                        #     secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz


                        #secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz



                        #which are potentially observable
                        observable_mask = (secz <= obs.max_airmass['MW']) * (secz >= 0)
                        
                        if obs.min_continuous_window == obs.min_exposure:
                            """ We removed the multiplication by observable, and turn this into a strict counter that does not reset to zero when not observable"""
                            observable_counter[observable_mask] = observable_counter[observable_mask] + 1

                        else:
                            # Create a 1d array dimensions equal to the number of healpix elements. 
                            observable = np.zeros(obs.npix)

                            # Set all healpix masks to 1 if they are observable
                            observable[observable_mask] = 1

                            # I think this is clever. If the previous 15 minute window was observable, and this time step is observable, we add 1 to the previous counter.
                            # If it was 3, now it's 4, and we have a 1-hour block of time. If the previous was 3, but not it's zero, we zero out the counter. 
                            # And if the previous was 0, and now it's 1, we are at 1.
                            # When we are done we will have the windows of where we can observe for a given amount of time.
                            observable_counter[:,dt_i] = (1 + observable_counter[:,dt_i -1 ])*observable

                            #Now, we need to decide if we want to add time to the total observability window. If the current observable window is 0, and the previous
                            # observability windows was more than 4 (1 hr), we add the previous total time to the total time observable
                            add_time_mask = (observable == 0) * (observable_counter[:,dt_i -1] >= obs.min_continuous_window *obs.bins_per_hour)

                            total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)
                            total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)

                            total_time_dict[0, observatory, telescope][add_time_mask] = total_time_dict[0, observatory, telescope][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)
                            total_time_dict[0, "combined", "combined"][add_time_mask] = total_time_dict[0, "combined", "combined"][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)

            if obs.min_continuous_window == obs.min_exposure:
                """ Add the time in the last bin to the observable window. IF obs.min_continuous_window == obs.min_exposure then we are adding the total time of each night and only do this once"""
                # We need to do a final check to ensure that we get the last observing window of the night. This does not require the current window to become zero.
                
                total_time_dict[0, observatory, telescope] = total_time_dict[0, observatory, telescope] + observable_counter/float(obs.bins_per_hour)
                total_time_dict[0, "combined", "combined"] = total_time_dict[0, "combined", "combined"] + observable_counter/float(obs.bins_per_hour)

                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"] + observable_counter/float(obs.bins_per_hour)
                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope] + observable_counter/float(obs.bins_per_hour)

            else:

                """ Add the time in the last bin to the observable window. IF obs.min_continuous_window == obs.min_exposure then we are adding the total time of each night and only do this once"""
                # We need to do a final check to ensure that we get the last observing window of the night. This does not require the current window to become zero.
                add_time_mask = (observable_counter[:,-1] >= obs.min_continuous_window *obs.bins_per_hour)
                total_time_dict[0, observatory, telescope][add_time_mask] = total_time_dict[0, observatory, telescope][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)
                total_time_dict[0, "combined", "combined"][add_time_mask] = total_time_dict[0, "combined", "combined"][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)

                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)
                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)

            obs.next_day()

            dump = False
            if dump == True:
                    "This dumps the observability as of midnight"
                    obsalt_az = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight(), location=obs.location))
                    fh = open("out.tsv","w")
                    fh.writelines("#%s\n"%(obs.date))
                    fh.writelines("#%s\n"%(obs.utc_midnight()))
                    fh.writelines("#dt_midnight\tRA_degree\tDec_degree\tgalactic_l\tgalactic_b\talt_degree\taz_degree\tsecz\tsun_alt\tobs_count\n")
                    for k in range(len(delta_midnight)):
                            for j in [0,3000]: #range(len(observable_counter)):
                                    if np.max(observable_counter[j]) > 0:
                                            fh.writelines("%0.2f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.3f\t%0.3f\t%i\n"%(delta_midnight[k].value, obs.healpix_skycoords[j].ra.value, obs.healpix_skycoords[j].dec.value, obs.healpix_skycoords[j].galactic.l.value, obs.healpix_skycoords[j].galactic.b.value, obsalt_az[j].alt.degree, obsalt_az[j].az.degree, secz_dict[k][j].value, sunAltAz.altaz[k].alt.value, observable_counter[j,k]))

                    fh.close()

def map_a_list(obs,total_time_dict,
                observatories=["APO","LCO"],
                N_days=4*365,
                verbose_debug=False,
                doPlot=True,
                list_of_days=False):

    for observatory in observatories:

        #Change the observatory
        obs.set_observatory(observatory=observatory)

        delta_midnight = delta_t(obs.dt_min, obs.dt_max, obs.bins_per_hour)

        for day_i in range(round(N_days/obs.delta_days)):

            if (obs.campaign_mode == True) and (observatory == "APO") and obs.Flag1m[obs.utc_midnight.datetime.month]:
                """ This is not right yet. Need to keep track of campaign mode 1m time spent on target"""
                """ Could do this with a flag, and plot 1m vs 0.16m time seperately with a different color scheme"""
                telescope = "1.0m"
                if verbose == True or quiet == False:
                    print(obs.utc_midnight, "%s campaign mode[engaged]: 1.0m "%(observatory))
            else:
                telescope = "0.16m"
                if verbose == True or quiet == False:
                    print(obs.utc_midnight, "%s campaign mode[disengaged]: 0.16m"%(observatory))

            
            # if (moon_dark_flag):
            #     print("Dark: sun/moon distance : ", "%0.1f"%obs.moon.separation(obs.sun).degree, "< ", obs.bright_moon_sun_dist)
            # else:
            #     print("Bright: sun/moon distance : ", "%0.1f"%obs.moon.separation(obs.sun).degree, "> ", obs.bright_moon_sun_dist)

            secz_dict   = {}

            # Initialize the counter for today
            if obs.min_continuous_window == obs.min_exposure:
                observable_counter = np.zeros(obs.npix,dtype=int)

            else:
                observable_counter = np.zeros([obs.npix, len(delta_midnight)],dtype=int)


            # Initialize the observable window array with the first time step
            # Doing the first step means I don't need an if statement all the time in the iteration of time

            # Create blocking
            sunAltAz = obs.sun.transform_to(AltAz(obstime=obs.utc_midnight+delta_midnight, location=obs.location)).alt
            moon_AltAz = obs.moon.transform_to(AltAz(obstime=obs.utc_midnight+delta_midnight, location=obs.location)).alt
            # This is a per day check.
            moon_dark_flag = obs.moon.separation(obs.sun) <= obs.bright_moon_sun_dist

            # Replace with a obs.HPT.keys() type things
            # Should be a dictionary with secZs. Then just loop over and check for flag. No need to code individually.
            for obj in obs.LVM_HPT.keys():
                secz_dict[obj] = obs.LVM_HPT[obj].transform_to(AltAz(obstime=obs.utc_midnight+delta_midnight, location=obs.location)).secz

            #Take the first time
            dt = delta_midnight[0] 
            dt_i = 0

            if sunAltAz[dt_i] <= obs.max_sun_alt:
                # the alt_az position of each healpix position, adjusted by dt.

                secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz

                # Calculate the secz of each healpix coordinate
                # if obs.nside_lowres < obs.nside:
                #     secz_lowres = obs.healpix_skycoords_lowres.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz
                #     secz = healpy.pixelfunc.ud_grade(secz_lowres, obs.nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)

                # else:
                #     secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz


                # Mask all healpix coordinates that have an acceptable air mass
                observable_mask = (secz <= obs.max_airmass['MW']) * (secz >=1.0)

                # Create an array for this dt to store which positions are observable. 
                observable = np.zeros(obs.npix)

                # Flag the observable positiosn with 1.
                observable[observable_mask] = 1

                # Insert the observable array into the full observable array.
                if obs.min_continuous_window == obs.min_exposure: 
                    observable_counter = observable.copy()
                else:
                    observable_counter[:,dt_i] = observable.copy()

            for dt_i in range(1,len(delta_midnight)):
                #Like before, the dt from midnight, going form -6 hours to +6 hours. Change to -18 sun set to sun rise
                dt = delta_midnight[dt_i]

                # Check if the sun is down
                if verbose == True:
                    if quiet == False:
                        print("Local date/time ", obs.local_midnight+dt, "Sun(alt): ", sunAltAz[dt_i])
                if sunAltAz[dt_i] <= obs.max_sun_alt:
                    # Check if the moon phase is bright or dark.

                    #Set HP targets. If UP and dark then True. Else, False
                    HPT_flag = {'any':False}

                    for obj in obs.LVM_HPT.keys():
                        HPT_flag[obj] = (secz_dict[obj][dt_i] >=1.0) * (secz_dict[obj][dt_i] <= obs.max_airmass[obj]) * (obs.LVM_HPT[obj].separation(obs.moon) > obs.bright_moon_sun_dist)
                        #HPT_flag[obj] = (secz_dict[obj][dt_i] >=1.0) * (secz_dict[obj][dt_i] <= obs.max_airmass[obj]) * moon_dark_flag

                        #If any targets are true set the any flag to true
                        if HPT_flag[obj] == True:
                            #I can count the number of obervable 15min windows using the number of secz entries
                            total_time_dict[obj][obs.utc_midnight.datetime.month].append(secz_dict[obj][dt_i])
                            total_time_dict[obj][0].append(secz_dict[obj][dt_i])

                            HPT_flag['any'] = True
                            if verbose_debug: print("%s:"%obj, secz_dict[obj][dt_i])

                    if HPT_flag['any'] == False :

                        # Again, calcualte the alt-az
                        #obsalt_az = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)) # Use this to calcualte moon distance. Must be greater than XX degree

                        # Get sec for full healpix array
                        #secz_lowres = obs.healpix_skycoords_lowres.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz
                        #secz = healpy.pixelfunc.ud_grade(secz_lowres.value, obs.nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)

                        secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz

                        # if obs.nside_lowres < obs.nside:
                        #     secz_lowres = obs.healpix_skycoords_lowres.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz
                        #     secz = healpy.pixelfunc.ud_grade(secz_lowres, obs.nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)

                        # else:
                        #     secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz


                        #secz = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight+dt, location=obs.location)).secz



                        #which are potentially observable
                        observable_mask = (secz <= obs.max_airmass['MW']) * (secz >= 0)
                        
                        if obs.min_continuous_window == obs.min_exposure:
                            """ We removed the multiplication by observable, and turn this into a strict counter that does not reset to zero when not observable"""
                            observable_counter[observable_mask] = observable_counter[observable_mask] + 1

                        else:
                            # Create a 1d array dimensions equal to the number of healpix elements. 
                            observable = np.zeros(obs.npix)

                            # Set all healpix masks to 1 if they are observable
                            observable[observable_mask] = 1

                            # I think this is clever. If the previous 15 minute window was observable, and this time step is observable, we add 1 to the previous counter.
                            # If it was 3, now it's 4, and we have a 1-hour block of time. If the previous was 3, but not it's zero, we zero out the counter. 
                            # And if the previous was 0, and now it's 1, we are at 1.
                            # When we are done we will have the windows of where we can observe for a given amount of time.
                            observable_counter[:,dt_i] = (1 + observable_counter[:,dt_i -1 ])*observable

                            #Now, we need to decide if we want to add time to the total observability window. If the current observable window is 0, and the previous
                            # observability windows was more than 4 (1 hr), we add the previous total time to the total time observable
                            add_time_mask = (observable == 0) * (observable_counter[:,dt_i -1] >= obs.min_continuous_window *obs.bins_per_hour)

                            total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)
                            total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)

                            total_time_dict[0, observatory, telescope][add_time_mask] = total_time_dict[0, observatory, telescope][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)
                            total_time_dict[0, "combined", "combined"][add_time_mask] = total_time_dict[0, "combined", "combined"][add_time_mask] + observable_counter[:,dt_i -1][add_time_mask]/float(obs.bins_per_hour)

            if obs.min_continuous_window == obs.min_exposure:
                """ Add the time in the last bin to the observable window. IF obs.min_continuous_window == obs.min_exposure then we are adding the total time of each night and only do this once"""
                # We need to do a final check to ensure that we get the last observing window of the night. This does not require the current window to become zero.
                
                total_time_dict[0, observatory, telescope] = total_time_dict[0, observatory, telescope] + observable_counter/float(obs.bins_per_hour)
                total_time_dict[0, "combined", "combined"] = total_time_dict[0, "combined", "combined"] + observable_counter/float(obs.bins_per_hour)

                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"] + observable_counter/float(obs.bins_per_hour)
                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope] + observable_counter/float(obs.bins_per_hour)

            else:

                """ Add the time in the last bin to the observable window. IF obs.min_continuous_window == obs.min_exposure then we are adding the total time of each night and only do this once"""
                # We need to do a final check to ensure that we get the last observing window of the night. This does not require the current window to become zero.
                add_time_mask = (observable_counter[:,-1] >= obs.min_continuous_window *obs.bins_per_hour)
                total_time_dict[0, observatory, telescope][add_time_mask] = total_time_dict[0, observatory, telescope][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)
                total_time_dict[0, "combined", "combined"][add_time_mask] = total_time_dict[0, "combined", "combined"][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)

                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), "combined", "combined"][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)
                total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] = total_time_dict[obs.utc_midnight.datetime.month+multi_year_month_flag*(obs.utc_midnight.datetime.year - start_obs_year), observatory, telescope][add_time_mask] + observable_counter[:,-1][add_time_mask]/float(obs.bins_per_hour)

            obs.next_day()

            dump = False
            if dump == True:
                    "This dumps the observability as of midnight"
                    obsalt_az = obs.healpix_skycoords.transform_to(AltAz(obstime=obs.utc_midnight(), location=obs.location))
                    fh = open("out.tsv","w")
                    fh.writelines("#%s\n"%(obs.date))
                    fh.writelines("#%s\n"%(obs.utc_midnight()))
                    fh.writelines("#dt_midnight\tRA_degree\tDec_degree\tgalactic_l\tgalactic_b\talt_degree\taz_degree\tsecz\tsun_alt\tobs_count\n")
                    for k in range(len(delta_midnight)):
                            for j in [0,3000]: #range(len(observable_counter)):
                                    if np.max(observable_counter[j]) > 0:
                                            fh.writelines("%0.2f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.3f\t%0.3f\t%i\n"%(delta_midnight[k].value, obs.healpix_skycoords[j].ra.value, obs.healpix_skycoords[j].dec.value, obs.healpix_skycoords[j].galactic.l.value, obs.healpix_skycoords[j].galactic.b.value, obsalt_az[j].alt.degree, obsalt_az[j].az.degree, secz_dict[k][j].value, sunAltAz.altaz[k].alt.value, observable_counter[j,k]))

                    fh.close()

def plot_the_sky(obs, total_time_dict, observatories, observatory_telescopes, gui=False, save=True, nest=False):
    print("Plotting combined now.....")
    import matplotlib.pyplot as plt

    for month in range(total_time_dict['nMonths'] + 1):
        for observatory in observatories:
            tmp = np.zeros(obs.npix)

            vmin = 0
            vmax = 2
            pitch_angle = 0

            for i, telescope in enumerate(observatory_telescopes[observatory]):
                mask = total_time_dict[month, observatory, telescope] > 0
                if telescope == "0.16m":
                    tmp[mask] = 1.0
                elif telescope == "1.0m":
                    tmp[mask] = tmp[mask] + 2.0

            cmap = plt.get_cmap('Blues', 4)
            cmap.set_under('w')

            healpy.mollview(tmp, cbar=True, nest=nest, cmap=cmap, rot=(0,0,pitch_angle), xsize=4000, min=vmin, max=vmax, title="%s: month:%i"%(observatory, month))

            # cmap = plt.get_cmap('RdBu', np.max(tmp)-np.min([max(tmp), 10])+1)
            # healpy.mollview(total_time_dict[month, observatory, telescope], cbar=True, cmap=cmap, rot=(0,0,pitch_angle), xsize=4000, min=vmin, max=vmax, title="%s: month:%i telescope:%s"%(observatory, month, telescope))

            healpy.graticule()

            if save == True:
                if total_time_dict['campaign_mode'] == True:
                    plt.savefig("plots/%s_t_pointing_fields-month%i-campaign.png"%(observatory, month))
                else:
                    plt.savefig("plots/%s_t_pointing_fields-month%i.png"%(observatory, month))
                #plt.savefig("%s_t_pointing_fields-month%i.png"%(obs.observatory,month))

            if gui==True:
                plt.show()

            plt.close()

def save_obs_times_to_file(file_name, obs, total_time_dict, observatories, observatory_telescopes={"APO":["0.16m", "1.0m"], "LCO":["0.16m"], "combined":["combined"]}, overwrite=False):
    from astropy.table import Table, Column
    import astropy.io.fits as fits

    t = Table([Column(obs.healpix_skycoords.ra.value), 
                Column(obs.healpix_skycoords.dec.value),
                Column(obs.healpix_skycoords.galactic.l.value),
                Column(obs.healpix_skycoords.galactic.b.value)],
                names=['RA', 'DEC', 'gal_l', 'gal_b'])

    for observatory in observatories:
        #t.meta["observatory"]=observatory
        #t.meta["telescope"]=telescope
        for telescope in observatory_telescopes[observatory]:
            for month in range(total_time_dict['nMonths'] + 1):
                t.add_column(Column(total_time_dict[month, observatory, telescope]/15), name='t_%s_%s_%i'%(observatory, telescope, month))

    t.write(file_name, format="fits", overwrite=overwrite)

    hdul = fits.open(file_name)

    hdul[0].header["t_bin_hr"]=obs.bins_per_hour
    hdul[0].header["r_arcmin"]= obs.angular_resolution.value
    hdul[0].header["ContWind"]= obs.min_continuous_window
    hdul[0].header["IFUs_pix"]=obs.N_IFU_per_pixel.value
    hdul[0].header["t_hpix"] = obs.N_IFU_per_pixel.value*0.25
    hdul[0].header['NSIDE'] = obs.nside
    hdul[0].header['ORDERING'] = obs.order
    for obj in obs.max_airmass.keys():
        hdul[0].header['maxsecZ_%s'%obj] = obs.max_airmass[obj]


    hdul.writeto(file_name, overwrite=overwrite)

def save_HPT_airmass(obs, total_time_dict, name="default", overwrite=False):
    from astropy.table import Table, Column
    import astropy.io.fits as fits

    t = Table()

    for obj in obs.LVM_HPT.keys():
        for month in range(total_time_dict['nMonths'] + 1):
            t = Table([Column(total_time_dict[obj][month])], names=['secz_%s_%i'%(obj, month)])

            file_name = "data/"+name+obj+"%i.fits"%month

            t.write(file_name, format="fits")
            hdul = fits.open(file_name)
            hdul[0].header['maxsecZ_%s'%obj] = obs.max_airmass[obj]

            hdul.writeto(file_name, overwrite=overwrite)



if __name__ == "__main__":

    # Read in the obs class

    # We can change the observatory location with obs.change_observatory. It will automatically reset the date by default.
    # You can disable this with reset_date=False in the call.

    verbose = True

    observatories = ["APO", "LCO"]
    observatory_telescopes = {"APO":["0.16m", "1.0m"], "LCO":["0.16m"]}

    import astropy.io.fits as fits
    hdul_ha = fits.open("Halpha_fwhm06_1024.fits")    
    nside = hdul_ha[0].header['NSIDE']
    order = hdul_ha[0].header['ORDERING']

    if order.lower() == "NESTED".lower():
        nest=True
    else:
        nest=False

    #calculate air masses and observation windwos for a lower resolution healpix array
    obs = obs_class.obs(obs_date='2020-1-1', observatory="APO", bins_per_hour=4, nest=nest, nside=16, campaign_mode=True, delta_days=1)

    total_time_dict = init_time_dict(obs, observatories, observatory_telescopes)

    print("N IFUs per pixel area:%0.2f"%(obs.N_IFU_per_pixel))
    print("Time per heal_pix (0.25h obs): %0.2f"%(obs.N_IFU_per_pixel*0.25))

    if obs.campaign_mode == False:
        file_name = "data/viewable_pointings.fits"        
    else:
        file_name = "data/viewable_campaign_pointings.fits"

    try:
        hdul_time = fits.open("data/viewable_campaign_pointings.fits")
    except:
        map_the_sky(obs, total_time_dict, observatories=["APO","LCO"])
        save_obs_times_to_file(file_name, obs, total_time_dict, observatories)
        hdul_time = fits.open("data/viewable_campaign_pointings.fits")

    #plot_the_sky(total_time_dict, observatories, observatory_telescopes, gui=False, nest=nest)


    from healpix_plot_lib import healpix_shader

    masks = []
    highres_sky_map = healpy.pixelfunc.ud_grade(hdul_time[1].data["t_APO_1.0m_7"], nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)
    masks.append(highres_sky_map > 0)

    highres_sky_map = healpy.pixelfunc.ud_grade(hdul_time[1].data["t_APO_0.16m_6"], nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)
    masks.append(highres_sky_map > 0)

    highres_sky_map = healpy.pixelfunc.ud_grade(hdul_time[1].data["t_LCO_0.16m_7"], nside, pess=False, order_in=obs.order, order_out=obs.order, power=None, dtype=None)
    masks.append(highres_sky_map > 0)
    

    data = np.array(hdul_ha[1].data.tolist())[:,0]
    log_I = np.log10(data)
    log_I_max = 2.0
    log_I_min = -1.0

    healpix_shader(log_I, masks, title=r"MW H$\alpha$", nest=nest, vmin=log_I_min, vmax=log_I_max, outfile="shaded_MW.png", gui=True)

    #plot_the_sky(total_time_dict, ["combined"], observatory_telescopes, gui=False)
    #total_time_dict, observatories, gui=False, save=True`