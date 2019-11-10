# Krisciunas K. & Schaefer B.E., 1991, PASP, 103, 1033
# https://iopscience.iop.org/article/10.1086/132921/pdf

import numpy as np
import math


class ks_moon_model(object):
   """
   Krisciunas K. & Schaefer B.E., 1991, PASP, 103, 1033
   moon sky brightness model 
   """

   def __init__(self, *args, **kwargs):
      self.dummy = 0.0

   def nLbMag(self, B):
      '''
      nanoLambert to Magnitude conversion
      '''
      return -2.5 * np.log10(1.0 / np.pi * 0.00001 * B / 108400.0)

   def MagnLb(self, M):
      '''
      Magnitude to nanoLambert conversion
      '''
      return np.pi * 100000 * 108400 * 10**(-0.4 * M)

   def X(self, Z):
      """
      Note thatn Z must be <= pi/2, i.e. above the horizon!
      """
      return 0.4 + 0.6 * (1 - 0.96 * (np.sin(Z))**2)**(-0.5)

   def f(self,rho):
      A = 5.2313
      B = 0.9351
      F = 5.9014
      return 10**A * (B + (np.cos(rho))**2) + 10**(F - np.rad2deg(rho) / 40.0)

   def I_moon(self,a):
      return 10**(-0.4 * (3.84 + 0.026 * np.fabs(a) + 4e-9 * a**4))

   def B_moon(self,zd, zd_moon, moon_dist, k, moon_alpha):
    """
    Krisciunas et. al Moon-Model, only valid for moon above horizon.

    Parameters
    ----------
    zd : float or ~numpy.ndarray
       zenith distance of the object in degrees
    zd_moon : float or ~numpy.ndarray
       zenith distance of the moon in degrees
    moon_distance : float or ~numpy.ndarray
       distance between object and the moon in  degrees
    k : float or ~numpy.ndarray
       extinction coefficient in mag
    moon_alpha: float or ~numpy.ndarray
       lunar phase in degrees

    Returns
    -------
    B_moon: float or ~numpy.ndarray
       surface brightness in nano-Lambert at location(s) of the object
    """
    zm = np.deg2rad(zd_moon)
    rho = np.deg2rad(moon_dist)
    return self.f(rho) * \
         self.I_moon(moon_alpha) * \
         (1 - 10 ** (-0.4 * k * self.X(np.deg2rad(zd)))) * \
         (10 ** (-0.4 * k * self.X(np.deg2rad(zm))) - 10 ** (-0.4 * k * self.X(np.pi / 2.0)))
