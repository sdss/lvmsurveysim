from astropy import units as u
from astropy.coordinates import SkyCoord

__all__ = ['Tile']

#
# class representing a survey tile
#


class Tile(object):
   '''
   Class representing a survey tile
   '''
   def __init__(self, coords, PA, priority):
      assert isinstance(coords, SkyCoord), "coords parameter must be astropy.coordinates.SkyCoord"
      self.coords = coords
      self.pa = PA
      self.priority = priority
      self.standard_stars = []
      self.guide_stars = []

   def __repr__(self):
      return (f'<Tile (RA={self.coords.ra.deg}, DEC={self.coords.dec.deg})>')
