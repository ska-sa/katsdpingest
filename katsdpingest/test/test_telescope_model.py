"""Test for the sigproc module."""

import unittest
import numpy as np
from katsdpingest import telescope_model as tm
import time
import spead

class TelescopeModelTestCases(unittest.TestCase):
    """Run a number of test against the telescope model.
    First, build and test a model through static configuration.
    Then exercise a number of the import and export capabalities.
    i.e.
        Model -> SPEAD -> Model -> HDF5 -> Model
    """
    def setUp(self):
        """Basic configuration including test sensors and attributes
        to use with the model."""
        m063 = tm.AntennaPositioner(name='m063')
        m062 = tm.AntennaPositioner(name='m062')
        cbf = tm.CorrelatorBeamformer(name='cbf')
        env = tm.Enviro(name='anc_asc')
        obs = tm.Observation(name='obs')
        self.components = [m063, m062, cbf, env, obs]
        self.model = tm.TelescopeModel()

    def _build_fake_model(self, model):
         # generate some pointing sensors
        # probably should use katpoint here. But for now we hack :)
        az = [31.5,31.6,31.7,31.7,31.7,31.7]
        el = [10.2,14.2,17.3,18.5,19.1,19.1]

    def testInitialiseModel(self):
         # test adding components
        self.model.add_components(self.components)
        self.assertEqual(len(self.model.components), len(self.components))
        self.assertFalse(self.model.is_valid())

    def testModelIndex(self):
         # test building index
        self.model.add_components(self.components)
        self.model.build_index()
        sensor_total = 0
        attr_total = 0
        for c in self.components:
            sensor_total += len(c._critical_sensors + c._std_sensors)
            attr_total += len(c._critical_attributes + c._std_attributes)
        self.assertEqual(len(self.model.index), sensor_total + attr_total)

    def testItemGroupUpdate(self):
         # test updating the model from a SPEAD itemgroup
        self.model.add_components(self.components)
        self.model.build_index()
        ig = spead.ItemGroup()
         # get a sensor and attribute to use
        sensor_name = [k for k,v in self.model.index.iteritems() if type(v) == tm.Sensor][0]
        attribute_name = [k for k,v in self.model.index.iteritems() if type(v) == tm.Attribute][0]
        sensor_descr = "sensor description string"
        attribute_descr = "attribute description string"
        sensor_val = 31.5
        attribute_val = "test value"
        ig.add_item(name=sensor_name, id=0x7001, description=sensor_descr, shape=-1, fmt=spead.mkfmt(('s', 8)))
        ig.add_item(name=attribute_name, id=0x7002, description=attribute_descr, shape=-1, fmt=spead.mkfmt(('s', 8)), init_val=attribute_val)
        ig[sensor_name] = "{0} nominal {1}".format(time.time(), sensor_val)

         # update from ig
        self.model.update_from_ig(ig)

         # check values have been set
        self.assertEqual(self.model.index[sensor_name].value, sensor_val)
        self.assertEqual(self.model.index[attribute_name].value, attribute_val)

         # check spead items have been stored
        self.assertEqual(self.model.index[sensor_name].get_spead_item().description, sensor_descr)
        self.assertEqual(self.model.index[attribute_name].get_spead_item().description, attribute_descr)

    def testModelAttributes(self):
         # test manipulating attributes in the model
         # these tests use direct manipulation
        self.model.add_components(self.components)
        self.model.build_index()
         # pick a random attribute
        a = [a for a in self.model.index.itervalues() if type(a) == tm.Attribute][0]
        a_val = "test value"
         # test invalid before a value is set
        self.assertFalse(a.is_valid())
        a.set_value(a_val)
         # check value
        self.assertEqual(a.value, a_val)
         # we should now be valid
        self.assertTrue(a.is_valid())
         # make sure we cannot overwrite value
        self.assertRaises(ValueError, lambda: a.set_value(a_val))

    def testModelSensors(self):
         # test manipulating sensors in the model
         # these tests use direct updating
         # later tests exercise the SPEAD and H5 interfaces
        self.model.add_components(self.components)
        self.model.build_index()
        az_values = [31.5,31.6,31.7,31.7,31.7,31.7]
        valid_update = "{0} nominal ".format(time.time())
         # we pick an arbitrary sensor for our use here
         # we could hardcode something like pos_actual_scan_azim but
         # this is really not necessary
        s = [s for s in self.model.index.itervalues() if type(s) == tm.Sensor][0]
        base_time = time.time()
        for i,az in enumerate(az_values):
            s.set_value("{0} nominal {1}".format(base_time + i*0.1, az))
         # check sensor length
        self.assertEqual(len(s.values), len(az_values))
         # check last item
        self.assertEqual(s.value, az_values[-1])
         # check ordered item
        self.assertEqual(s.values[sorted(s.values.iterkeys())[0]], az_values[0])
         # check is valud
        self.assertTrue(s.is_valid())
         # check valid with timespec
        self.assertTrue(s.is_valid(timespec=1.0))
         # check invalid with timespec
        self.assertFalse(s.is_valid(timespec=-5.0))
         # check cleaning works
        s.clean()
        self.assertEqual(len(s.values), 0)


