"""Tests for the dbe7 roach x- and f-engine simulator module."""

import unittest
from katcp import Sensor

# DUT module
from  katcapture.simulator import dbe7_roach_models
from katcore.testutils import SensorComparisonMixin

class test_Roach(unittest.TestCase, SensorComparisonMixin):
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
        self.assert_sensors_equal_description(
            roach.get_sensors(), self.expected_sensors)

class test_XEngine(test_Roach):
    RoachClass = dbe7_roach_models.XEngine

class test_FEngine(unittest.TestCase, SensorComparisonMixin):
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
            value=False,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.3y.adc.overrange',
            description='adc overrange indicator',
            units='',
            value=False,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.3x.fft.overrange',
            description='fft overrange indicator',
            units='',
            value=False,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.3y.fft.overrange',
            description='fft overrange indicator',
            units='',
            value=False,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.3x.adc.terminated',
            description='adc disabled',
            units='',
            value=False,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.BOOLEAN,
            name='roachy1234.3y.adc.terminated',
            description='adc disabled',
            units='',
            value=False,
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.FLOAT,
            name='roachy1234.3x.adc.power',
            description='approximate input signal strength',
            units='',
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.FLOAT,
            name='roachy1234.3y.adc.power',
            description='approximate input signal strength',
            units='',
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.FLOAT,
            name='roachy1234.3x.adc.amplitude',
            description='approximate input signal strength',
            units='',
            status=Sensor.NOMINAL),
                                                        dict(
            type=Sensor.FLOAT,
            name='roachy1234.3y.adc.amplitude',
            description='approximate input signal strength',
            units='',
            status=Sensor.NOMINAL),
        )

    def test_sensors(self):
        roach = dbe7_roach_models.FEngine('roachy1234', 3)
        self.assert_sensors_equal_description(
            roach.get_sensors(), self.expected_sensors)

    def test_adc_sensor_values(self):
        def test_ampl(val):
            self.assertTrue(val > 0)
        def test_power(val):
            self.assertTrue(val < 0)

        roach = dbe7_roach_models.FEngine('roachy1234', 3)

        self.assert_sensors_value_conditions(roach.get_sensors(), {
            'roachy1234.3x.adc.power': test_power,
            'roachy1234.3y.adc.power': test_power,
            'roachy1234.3x.adc.amplitude': test_ampl,
            'roachy1234.3y.adc.amplitude': test_ampl})

    def test_channels(self):
        roach = dbe7_roach_models.FEngine('roachum5', 7)
        self.assertEqual(roach.channels, ('7x', '7y'))

class test_XEngines(unittest.TestCase, SensorComparisonMixin):
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
        self.assert_sensors_equal_description(
            x_engines.get_sensors(),
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
                value=False,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.%(roachnum)dy.adc.overrange',
                description='adc overrange indicator',
                units='',
                value=False,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.%(roachnum)dx.adc.terminated',
                description='adc disabled',
                units='',
                value=False,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.%(roachnum)dy.adc.terminated',
                description='adc disabled',
                units='',
                value=False,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.%(roachnum)dx.fft.overrange',
                description='fft overrange indicator',
                units='',
                value=False,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.BOOLEAN,
                name='%(roachname)s.%(roachnum)dy.fft.overrange',
                description='fft overrange indicator',
                units='',
                value=False,
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.FLOAT,
                name='%(roachname)s.%(roachnum)dx.adc.amplitude',
                description='approximate input signal strength',
                units='',
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.FLOAT,
                name='%(roachname)s.%(roachnum)dy.adc.amplitude',
                description='approximate input signal strength',
                units='',
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.FLOAT,
                name='%(roachname)s.%(roachnum)dx.adc.power',
                description='approximate input signal strength',
                units='',
                status=Sensor.NOMINAL),
                          dict(
                type=Sensor.FLOAT,
                name='%(roachname)s.%(roachnum)dy.adc.power',
                description='approximate input signal strength',
                units='',
                status=Sensor.NOMINAL)
    )

        expected_sensors = super(test_FEngines, self).get_expected_sensors()

        for i, rn in enumerate(self.roach_names):
            for st in sens_templates:
                sens = st.copy()
                sens['name'] = st['name'] % dict(roachname=rn, roachnum=i)
                expected_sensors.append(sens)

        return expected_sensors

    def test_sensors(self):
        f_engines = dbe7_roach_models.FEngines(self.roach_names)
        self.assert_sensors_equal_description(f_engines.get_sensors(),
                                              self.get_expected_sensors())


    def test_channels(self):
        f_engines = dbe7_roach_models.FEngines(('r1', 'r2'))
        self.assertEqual(f_engines.get_channels(), ['0x', '0y', '1x', '1y'])
        self.assertTrue(f_engines.is_channel('1x'))
        self.assertFalse(f_engines.is_channel('5y'))
