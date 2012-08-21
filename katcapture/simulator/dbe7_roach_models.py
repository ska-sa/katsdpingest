"""X-engine and f-engine roach models for the DBE7 simulator

It is expected that the names of the roach engines are read from a
config file. The f-engines are numbered in sequence (i.e. if the
f-engines are roach0001, roach0020, roach0322, the f-engine channels
will be roach0001_1x, roach0001_1y, roach0020_2x, roach0020_2y,
etc...)

The *Engines classes are containers for roaches. Given a list of roach
names, they will initialise an *Engine (note singular) instance for
each roach name.

"""

import time
from katcore.dev_base import SimpleModel
from katcp import Sensor


# Sensor templates common to all DBE7 roaches
basic_roach_sensors = (dict(
        name_template='%(roachname)s.lru.available',
        type=Sensor.BOOLEAN,
        description='line replacement unit operational',
        units='',
        params=None,
        initial_value=True,
        initial_status=Sensor.NOMINAL),
                       )

# Extra sensor templates for DBE7 roach f-engines, excluding
# per-channel sensors
f_engine_extra_sensors = (dict(
        name_template='%(roachname)s.fpga.synchronised',
        type=Sensor.BOOLEAN,
        description='signal processing clock stable',
        units='',
        params=None,
        initial_value=True,
        initial_status=Sensor.NOMINAL),
                          )

f_engine_channel_sensors = (dict(
        name_template='%(roachname)s.%(channel)s.adc.overrange',
        type=Sensor.BOOLEAN,
        description='adc overrange indicator',
        units='',
        params=None,
        initial_value=False,
        initial_status=Sensor.NOMINAL),
                          dict(
        name_template='%(roachname)s.%(channel)s.adc.terminated',
        type=Sensor.BOOLEAN,
        description='adc disabled',
        units='',
        params=None,
        initial_value=False,
        initial_status=Sensor.NOMINAL),
                                  dict(
        name_template='%(roachname)s.%(channel)s.fft.overrange',
        type=Sensor.BOOLEAN,
        description='fft overrange indicator',
        units='',
        params=None,
        initial_value=False,
        initial_status=Sensor.NOMINAL),
                                  dict(
        name_template='%(roachname)s.%(channel)s.adc.power',
        type=Sensor.FLOAT,
        description='approximate input signal strength',
        units='',
        params=[-1e9, 0],
        initial_value=-23.3,
        initial_status=Sensor.NOMINAL),
                          dict(
        name_template='%(roachname)s.%(channel)s.adc.amplitude',
        type=Sensor.FLOAT,
        description='approximate input signal strength',
        units='',
        params=[0, 1],
        initial_value=0.1,
        initial_status=Sensor.NOMINAL),
        )


class Roach(SimpleModel):
    def __init__(self, name, *args, **kwargs):
        super(Roach, self).__init__()
        self.name = name
        self._init_sensors()

    def _init_sensors(self):
        for sensor_def in basic_roach_sensors:
            sname = sensor_def['name_template']  % {'roachname': self.name}
            self.add_sensor(self._setup_sensor(sname, sensor_def))

    def _setup_sensor(self, sname, sensor_def):
            sens = Sensor(sensor_def['type'], sname, sensor_def['description'],
                          sensor_def['units'], params=sensor_def['params'])
            sens.set_value(sensor_def['initial_value'], sensor_def['initial_status'],
                           time.time())
            return sens


class XEngine(Roach): pass

class FEngine(Roach):
    CHANNEL_POLARIZATIONS = ('x', 'y')

    def __init__(self, name, channel_number):
        self.channel_number = channel_number
        self.channels = tuple('%d%s' % (channel_number, pol)
                              for pol in self.CHANNEL_POLARIZATIONS)
        super(FEngine, self).__init__(name, channel_number)

    def _init_sensors(self):
        super(FEngine, self)._init_sensors()
        for sensor_def in f_engine_extra_sensors:
            sname = sensor_def['name_template'] % {
                'roachname': self.name, }
            self.add_sensor(self._setup_sensor(sname, sensor_def))

        for sensor_def in f_engine_channel_sensors:
            for chan in self.channels:
                sname = sensor_def['name_template'] % {
                    'roachname': self.name,
                    'channel': chan}
                self.add_sensor(self._setup_sensor(sname, sensor_def))


class RoachEngines(SimpleModel):
    RoachClass = Roach

    def __init__(self, roach_names):
        super(RoachEngines, self).__init__()
        self.roach_names = roach_names
        self._roaches = {}
        self._init_roaches()

    def _init_roaches(self):
        for roach_name in self.roach_names:
            roach = self.RoachClass(roach_name)
            self._roaches[roach_name] = roach
            for s in roach.get_sensors(): self.add_sensor(s)


class XEngines(RoachEngines):
    RoachClass = XEngine

class FEngines(RoachEngines):
    RoachClass = FEngine

    def __init__(self, roach_names):
        self.channels = set()
        super(FEngines, self).__init__(roach_names)

    def _init_roaches(self):
        for channel_no, roach_name in enumerate(self.roach_names):
            roach = self.RoachClass(roach_name, channel_no)
            self._roaches[roach_name] = roach
            for s in roach.get_sensors(): self.add_sensor(s)
            for c in roach.channels: self.channels.add(c)

    def get_channels(self):
        return sorted(self.channels)

    def is_channel(self, chan):
        return chan in self.channels
