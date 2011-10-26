"""Tests for the dbe7 roach x- and f-engine simulator module."""

import unittest
from katcp import Sensor

# DUT module
from  katcapture.simulator import dbe7_roach_models

class TestModelMixin(object):
    """Mixin to test that katcp.Sensor objects match specified criteria.

    Should be mixed in with unittest.TestCase subclasses.

    """

    def assert_sensors_equal(self, actual_sensors, desired_description):
        """Test that a list of Sensor objects (actual) match a description (desired)
        
        Each desired sensor is a dict with the parameters that should be
        tested
        """
        actual_sensor_dict = dict((s.name, s) for s in actual_sensors)
        desired_description_dict = dict(
            (s['name'], s) for s in desired_description)
        # Check that the sensor names match
        self.assertEqual(sorted(actual_sensor_dict.keys()),
                         sorted(desired_description_dict.keys()))

        # Build description of the actual sensors in the same format
        # as desired_description
        actual_description_dict = {}
        for name, desired_info in desired_description_dict.items():
            actual_description_dict[name] = {}
            sensor = actual_sensor_dict[name]
            for key in desired_info.keys():
                actual_description_dict[name][key] = self._get_sensor_key(
                    sensor, key)

        self.maxDiff = None     # Make unittest print differences even
                                # if they are large
        self.assertEqual(actual_description_dict, desired_description_dict)
                
    def _get_sensor_key(self, sensor, key):
        key_fns = dict(
            name=lambda s: s.name,
            status=lambda s: s._status,
            type=lambda s: s.SENSOR_TYPE_LOOKUP[s.stype],
            value=lambda s: s.value(),
            description=lambda s: s.description,
            units=lambda s: s.units,
            )
        try: key_fn = key_fns[key]
        except KeyError, e: raise KeyError('Unknown sensor key: ' + e.message)
        return key_fn(sensor)
    
class test_Roach(unittest.TestCase, TestModelMixin):
    RoachClass = dbe7_roach_models.Roach

    expected_sensors = (dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.lru.available',
            description='line replacement unit operational',
            units='',
            status=Sensor.NOMINAL,
            value=True,
            )),

    def test_sensors(self):
        roach = self.RoachClass('roachy1234')
        self.assert_sensors_equal(roach.get_sensors(), self.expected_sensors)

class test_XEngine(test_Roach):
    RoachClass = dbe7_roach_models.XEngine

class test_FEngine(unittest.TestCase, TestModelMixin):
    # TODO Update with all the sensors in doc K0000-2006V1-02
    expected_sensors = test_XEngine.expected_sensors + (dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.fpga.synchronised',
            description='signal processing clock stable',
            units='',
            value=True,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.3x.adc.overrange',
            description='adc overrange indicator',
            units='',
            value=True,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.3y.adc.overrange',
            description='adc overrange indicator',
            units='',
            value=True,
            status=Sensor.NOMINAL))

    def test_sensors(self):
        roach = dbe7_roach_models.FEngine('roachy1234', 3)
        self.assert_sensors_equal(roach.get_sensors(), self.expected_sensors)
            
        
class test_XEngines(unittest.TestCase, TestModelMixin):
    roach_names = ('roach0123', 'roach3210')
    
    def get_expected_sensors(self):
        sens_templates = (dict(
                type=Sensor.BOOLEAN,
                name='%s.lru.available',
                description='line replacement unit operational',
                units='',
                value=True,
                status=Sensor.NOMINAL),
                          )
                
        expected_sensors = []
        for rn in self.roach_names:
            for st in sens_templates:
                sens = st.copy()
                sens['name'] = st['name'] % rn
                expected_sensors.append(sens)

        return expected_sensors

    def test_sensors(self):
        x_engines = dbe7_roach_models.XEngines(self.roach_names)
        self.assert_sensors_equal(x_engines.get_sensors(),
                                  self.get_expected_sensors())


class test_FEngines(test_XEngines):
    def get_expected_sensors(self):
        sens_templates = (dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.fpga.synchronised',
                description='signal processing clock stable',
                units='',
                value=True,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.%(roachnum)dx.adc.overrange',
                description='adc overrange indicator',
                units='',
                value=True,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.%(roachnum)dy.adc.overrange',
                description='adc overrange indicator',
                units='',
                value=True,
                status=Sensor.NOMINAL))

        expected_sensors = super(test_FEngines, self).get_expected_sensors()

        for i, rn in enumerate(self.roach_names):
            for st in sens_templates:
                sens = st.copy()
                sens['name'] = st['name'] % dict(roachname=rn, roachnum=i)
                expected_sensors.append(sens)

        return expected_sensors
        
    def test_sensors(self):
        f_engines = dbe7_roach_models.FEngines(self.roach_names)
        self.assert_sensors_equal(f_engines.get_sensors(),
                                  self.get_expected_sensors())
