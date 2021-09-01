#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


# operations database and data classes for a survey tile and a survey observation

from lvmsurveysim.exceptions import LVMSurveyOpsError
from peewee import *
import lvmsurveysim.utils.sqlite2astropy as s2a
import hashlib
import os
import astropy

from lvmsurveysim import config
import lvmsurveysim.target
import lvmsurveysim.schedule.tiledb

# we will determine the db name and properties at runtime from config
# see http://docs.peewee-orm.com/en/latest/peewee/database.html#run-time-database-configuration

__lvm_ops_database__ = SqliteDatabase(None)


# data model:

class LVMOpsBaseModel(Model):
   '''
   Base class for LVM's peewee ORM models.
   '''
   class Meta: 
      database = __lvm_ops_database__


class Tile(LVMOpsBaseModel):
   '''
   Peewee ORM class for LVM Survey Tiles
   '''
   TileID = IntegerField(primary_key=True)
   TargetIndex = IntegerField(null=True)   # TODO: not sure this needs to go into the db, maybe create on the fly?
   Target = CharField(null=False)
   Telescope = CharField(null=False)
   RA = FloatField(null=True, default=0)
   DEC = FloatField(null=True, default=0)
   PA = FloatField(null=True, default=0)
   TargetPriority = IntegerField(null=True, default=0)
   TilePriority = IntegerField(null=True, default=0)
   AirmassLimit = FloatField(null=True, default=0)
   LunationLimit = FloatField(null=True, default=0)
   HzLimit = FloatField(null=True, default=0)
   MoonDistanceLimit = FloatField(null=True, default=0)
   TotalExptime = FloatField(null=True, default=0)
   VisitExptime = FloatField(null=True, default=0)
   Status = IntegerField(null=False)       # think bit-field to keep more fine-grained status information


class Observation(LVMOpsBaseModel):
   '''
   Peewee ORM class for LVM Survey Observation records
   '''
   ObsID = IntegerField(primary_key=True)
   ObsType = CharField(null=False)         # SCI, CAL, FLAT, DARK, BIAS, TEST, ...
   TileID = ForeignKeyField(Tile, backref='observation')
   JD = FloatField(null=False)
   LST = FloatField(null=True)
   Hz = FloatField(null=True)
   Alt = FloatField(null=True)
   Lunation = FloatField(null=True)


class Metadata(LVMOpsBaseModel):
   '''
   Peewee ORM class for LVM Survey Database Metadata
   '''
   Key = CharField(unique=True)
   Value = CharField()


class OpsDB(object):
   """
   Interface the operations database for LVM. Makes the rest of the 
   LVM Operations software agnostic to peewee or any other ORM we
   might be using one day.
   """
   def __init__(self):
      pass

   @classmethod
   def get_db(cls):
      '''
      Return the database instance. Should not be called outside this class.
      '''
      return __lvm_ops_database__

   @classmethod
   def init(cls, dbpath=None):
      '''
      Intialize the database connection. Must be called exactly once upon start of the program.
      '''
      dbpath = dbpath or config['opsdb']['dbpath']
      return __lvm_ops_database__.init(dbpath, pragmas=config['opsdb']['pragmas'])

   @classmethod
   def create_tables(cls, drop=False):
      '''
      Create the database tables needed for the LVM Ops DB. Should be called only 
      once for the lifetime of the database. Optionally, drop existing tables before
      creation.
      '''
      with OpsDB.get_db().atomic():
         if drop:
            __lvm_ops_database__.drop_tables([Tile, Observation, Metadata])
         __lvm_ops_database__.create_tables([Tile, Observation, Metadata])
         # Create special, non-science tiles to allow TileIDs to be universal
         Tile.insert(TileID=0, Target='NONE', Telescope='LVM-160', Status=0).execute()
         Tile.insert(TileID=1, Target='Test', Telescope='LVM-160', Status=0).execute()
         Tile.insert(TileID=1001, Target='DomeCal', Telescope='LVM-160', Status=0).execute()

   @classmethod
   def drop_tables(cls, models):
      '''
      Delete the tables. Should not be called during Operations. Development only.
      '''
      return __lvm_ops_database__.drop_tables(models)

   @classmethod
   def close(cls):
      '''
      Close the database connection.
      '''
      return __lvm_ops_database__.close()

   @classmethod
   def get_metadata(cls, key, default_value=None):
      '''
      Get the value associated with a key from the Metadata table.
      Return the value, or `default_value` if not found.
      '''
      try:
         return Metadata.get(Metadata.Key==key).Value
      except Metadata.DoesNotExist:
         return default_value
      
   @classmethod
   def set_metadata(cls, key, value):
      '''
      Set the value associated with a key from the Metadata table. 
      Creates or replaces the key/value pair.
      '''
      return Metadata.replace(Key=key, Value=value).execute()

   @classmethod
   def del_metadata(cls, key):
      '''
      Deletes the key/value pair from the Metadata table.
      '''
      return Metadata.delete().where(Metadata.Key==key).execute()

   @classmethod
   def update_tile_status(cls, tileid, status):
      '''
      Update the tile Status column in the tile database.
      '''
      with OpsDB.get_db().atomic():
         s = Tile.update({Tile.Status:status}).where(Tile.TileID==tileid).execute()
      if s==0:
         raise LVMSurveyOpsError('Attempt to set status on unknown TildID '+str(tileid))
      return s

   @classmethod
   def record_observation(cls, TileID, obstype, jd, lst, hz, obs_alt, lunation):
      '''
      Record an LVM Observation in the database.
      '''
      return Observation.insert(TileID=TileID, ObsType=obstype, 
                                JD=jd, LST=lst, Hz=hz, Alt=obs_alt, Lunation=lunation).execute()

   @classmethod
   def save_tiledb(cls, tiledb, fits=False, path=None, overwrite=False):
      """
      Saves a tile table to the operations database, optionally into a FITS table.

      The default is to update the tile database in SQL. No parameters are needed in 
      this case.

      Parameters
      ----------
      tiledb : `~lvmsurveysim.scheduler.TileDB`
         The instance of a tile database to save
      fits : bool
         Optional, save to FITS table instead of database.
      path : str or ~pathlib.Path
         Optional, the path and basename of the fits file, no extension.
         Expects to find 'path.fits'.
      overwrite : bool
         Optional, overwrite the FITS file if it already exists. Default False
      """
      targfile = str(tiledb.targets.filename) if tiledb.targets.filename is not None else 'NA'
      targhash = cls.md5(targfile)
      tile_table = tiledb.tile_table
      if fits:
         assert path != None, "path not provided for FITS save"
         tile_table.meta['targhash'] = targhash
         tile_table.meta['targfile'] = targfile
         tile_table.meta['scitile1'] = tiledb.tileid_start
         tile_table.write(path+'.fits', format='fits', overwrite=overwrite)
         s = len(tile_table)
      else:
         # store tiles in the Ops DB
         with cls.get_db().atomic():
               # add metadata:
               cls.set_metadata('targfile', targfile)
               cls.set_metadata('targhash', targhash)
               cls.set_metadata('scitile1', tiledb.tileid_start)
               # save tile table
               s = s2a.astropy2peewee(tile_table, Tile, replace=True)
      return s

   @classmethod
   def md5(cls, fname):
      hash_md5 = hashlib.md5()
      with open(fname, "rb") as f:
         for chunk in iter(lambda: f.read(4096), b""):
               hash_md5.update(chunk)
      return hash_md5.hexdigest()

   @classmethod
   def load_tiledb(cls, targets=None, fits=False, path=None):
      """Load a tile database from the opsdb, or optionally read
      from FITS table file. Default is read from SQL operations database.

      Parameters
      ----------
      targets : ~lvmsurveysim.target.target.TargetList or path-like
         Optional, the `~lvmsurveysim.target.target.TargetList` object associated
         with the tile database or a path to the target list to load. If
         `None`, the ``TARGFILE`` value stored in the database file will be
         used to find abd load the correct target list.
      fits : boolean
         Optional, load from a FITS table rather than SQL
      path : str or ~pathlib.Path
         Optional, the path and basename of the tile fits file, no extension.
         Expects to find 'path.fits'.

      Returns
      -------
      ~lvmsurveysim.schedule.TileDB
         TileDB instance
      """

      if fits:
         assert path != None, "path not provided for FITS save"
         tile_table = astropy.table.Table.read(path+'.fits')

         targfile = tile_table.meta.get('TARGFILE', 'NA')
         targhash = tile_table.meta.get('TARGHASH', 'NA')
         scitile1 = tile_table.meta.get('SCITILE1')
         targets = targets or targfile
      else:
         with cls.get_db().atomic():
               targfile = cls.get_metadata('targfile', default_value='NA')
               targhash = cls.get_metadata('targhash', default_value='NA')
               scitile1 = cls.get_metadata('scitile1')
               targets = targets or targfile
               tile_table = s2a.peewee2astropy(Tile)

      if not isinstance(targets, lvmsurveysim.target.TargetList):
         assert targets is not None and targets != 'NA', \
               'invalid or unavailable target file path.'

         if not os.path.exists(targets):
               raise LVMSurveyOpsError(
                  f'the target file {targets!r} does not exists. '
                  'Please, call load with a targets parameter.')

         assert targhash == cls.md5(targets), 'Target file md5 hash not identical to database value'

         targets = lvmsurveysim.target.TargetList(target_file=targets)

      if (scitile1 != None):
         scitile1 = int(scitile1)
      tiledb = lvmsurveysim.schedule.TileDB(targets, tile_tab=tile_table, tileid_start=scitile1)

      return tiledb
