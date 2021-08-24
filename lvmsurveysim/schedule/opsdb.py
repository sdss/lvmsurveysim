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

from lvmsurveysim import config

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
   RA = FloatField(null=True)
   DEC = FloatField(null=True)
   PA = FloatField(null=True)
   TargetPriority = IntegerField(null=True)
   TilePriority = IntegerField(null=True)
   AirmassLimit = FloatField(null=True)
   LunationLimit = FloatField(null=True)
   HzLimit = FloatField(null=True)
   MoonDistanceLimit = FloatField(null=True)
   TotalExptime = FloatField(null=True)
   VisitExptime = FloatField(null=True)
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
   def create_tables(cls):
      '''
      Create the database tables needed for the LVM Ops DB. Should be called only 
      once for the lifetime of the database.
      '''
      with OpsDB.get_db().atomic():
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
