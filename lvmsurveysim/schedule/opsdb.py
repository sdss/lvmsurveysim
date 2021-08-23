#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Niv Drory (drory@astro.as.utexas.edu)
# @Filename: tiledb.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#


# operations database and data classes for a survey tile and a survey observation

from peewee import *

from lvmsurveysim import config

# we will determine the db name and properties at runtime from config
# see http://docs.peewee-orm.com/en/latest/peewee/database.html#run-time-database-configuration

__lvm_ops_database__ = SqliteDatabase(None)


# data model:

class LVMOpsBaseModel(Model):
   class Meta: 
      database = __lvm_ops_database__

class test(LVMOpsBaseModel):
   a = IntegerField()
   b = IntegerField()

class Tile(LVMOpsBaseModel):
   TileID = IntegerField(primary_key=True)
   TargetIndex = IntegerField()   # TODO: not sure this needs to go into the db, maybe create on the fly?
   Target = CharField()
   Telescope = CharField()
   RA = FloatField()
   DEC = FloatField()
   PA = FloatField()
   TargetPriority = IntegerField()
   TilePriority = IntegerField()
   AirmassLimit = FloatField()
   LunationLimit = FloatField()
   HzLimit = FloatField()
   MoonDistanceLimit = FloatField()
   TotalExptime = FloatField()
   VisitExptime = FloatField()
   Status = IntegerField()       # think bit-field to keep more fine-grained status information


class Observation(LVMOpsBaseModel):
   ObsID = IntegerField(primary_key=True)
   TileID = ForeignKeyField(Tile, backref='observation')
   LST = FloatField()
   Hz = FloatField()
   Alt = FloatField()
   Lunation = FloatField()


class Metadata(LVMOpsBaseModel):
   Key = CharField(unique=True)
   Value = CharField()



class OpsDB(object):
   """
   Interface the operations database for LVM
   """
   def __init__(self):
      pass

   @classmethod
   def get_db():
      return __lvm_ops_database__

   @classmethod
   def init(cls, dbpath=None):
      dbpath = dbpath or config['opsdb']['dbpath']
      return __lvm_ops_database__.init(dbpath, pragmas=config['opsdb']['pragmas'])

   @classmethod
   def create(cls, overwrite=False):
      return __lvm_ops_database__.create_tables([test, Tile, Observation, Metadata])

   @classmethod
   def drop_tables(cls, models):
      return __lvm_ops_database__.drop_tables(models)

   @classmethod
   def close(cls):
      return __lvm_ops_database__.close()

   @classmethod
   def get_metadata(cls, key, default_value=None):
      try:
         return Metadata.get(Metadata.Key==key).Value
      except Metadata.DoesNotExist:
         return default_value
      
   @classmethod
   def set_metadata(cls, key, value):
      return Metadata.replace(Key=key, Value=value).execute()

   @classmethod
   def del_metadata(cls, key):
      return Metadata.delete().where(Metadata.Key==key).execute()
