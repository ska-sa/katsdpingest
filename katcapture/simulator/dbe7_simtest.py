import logging
import re
from katcore import sim_base
from katcp import Sensor
from katcp.kattypes import request, return_reply
from katcp.kattypes import Str, Int, Bool, Discrete, Timestamp
from katcp import Message
activitylogger = logging.getLogger('activity')
log_name = 'kat.k7simulator'
logger = logging.getLogger(log_name)

class AlreadyHidden(Exception): pass
class NotHidden(Exception): pass

class SimTestDevice(sim_base.SimTestDevice):
    VERSION_INFO = ("k7-simulator-test-device",0,1)
    BUILD_INFO = ("k7-simulator-test-device",0,1,"rc1")

    def __init__(self, *args, **kwargs):
        super(SimTestDevice, self).__init__(*args, **kwargs)
        self._hidden_sensors = {}

    def set_device(self, device):
        """Set device server interfaced to the model

        The device needs to implement add/remove_sensor() methods and
        an issue_device_changed() method that informs all clients that
        the device has changed.
        """
        self._device = device

    def setup_sensors(self):
        """Add the test-device sensors from the model"""
        super(SimTestDevice, self).setup_sensors()
        for sensor in self._model.test_sensors.values():
            self.add_sensor(sensor)

    def hide_sensor(self, sensor_name):
        if sensor_name in self._hidden_sensors:
            raise AlreadyHidden('Sensor %s already hidden' % sensor_name)
        sensor = self._model.get_sensor(sensor_name)
        assert(sensor is self._device.get_sensor(sensor_name))
        self._hidden_sensors[sensor.name] = sensor
        self._device.remove_sensor(sensor)
        self._device.issue_device_changed('sensor-list')
        logger.info('Hide sensor %s.' % sensor.name)

    def unhide_sensor(self, sensor_name):
        sensor = self._model.get_sensor(sensor_name)
        try: del(self._hidden_sensors[sensor.name])
        except KeyError: raise NotHidden('Sensor %s is not hidden' % sensor.name)
        self._device.add_sensor(sensor)
        self._device.issue_device_changed('sensor-list')
        logger.info('Unhide Sensor %s.' % sensor.name)

    def _get_sensor_names(self, sensor_name_re):
        sensor_name_re = re.compile(sensor_name_re)
        return [sens.name for sens in self._model.get_sensors()
                if sensor_name_re.search(sens.name)]

    def hide_sensor_re(self, sensor_name_re, already_hidden_ok=None):
        """Hide all sensors matching regex sensor_name_re

        Return a list of sensor names that were hidden.
        """
        sensor_names = self._get_sensor_names(sensor_name_re)
        not_hidden = set()
        for sensor_name in sensor_names:
            try:
                self.hide_sensor(sensor_name)
            except AlreadyHidden:
                if not already_hidden_ok: raise
                else: not_hidden.add(sensor_name)

        return list(set(sensor_names) - not_hidden)

    def unhide_sensor_re(self, sensor_name_re, not_hidden_ok=None):
        """Unhide all sensors matching regex sensor_name_re

        Return a list of sensor names that were unhidden.
        """
        sensor_names = self._get_sensor_names(sensor_name_re)
        not_unhidden = set()
        for sensor_name in sensor_names:
            try:
                self.unhide_sensor(sensor_name)
            except NotHidden:
                if not not_hidden_ok: raise
                else: not_unhidden.add(sensor_name)

        return list(set(sensor_names) - not_unhidden)

    @request(Str(), include_msg=True)
    @return_reply(Int())
    def request_hide_sensors(self, sock, req_msg, name_re):
        """Hide all sensors matching (python) regular expression name_re

        Returns the number of sensors that were hidden. Issues and
        inform for each sensor that was hiden. E.g.:

        ?hide-sensors foo.*bar
        #hide-sensors blah.foo.bar
        #hide-sensors foo-nitz.bar-bloop
        !hide-sensors ok 2
        """

        sensor_names = self.hide_sensor_re(name_re, already_hidden_ok=True)
        for sn in sensor_names:
            self.reply_inform(sock, Message.inform(req_msg.name, sn), req_msg)
        return ('ok', len(sensor_names))

    @request(Str(), include_msg=True)
    @return_reply(Int())
    def request_unhide_sensors(self, sock, req_msg, name_re):
        """Unhide all sensors matching (python) regular expression name_re

        Returns the number of sensors that were hidden. Issues and
        inform for each sensor that was hiden. E.g.:

        ?unhide-sensors foo.*bar
        #unhide-sensors blah.foo.bar
        #unhide-sensors foo-nitz.bar-bloop
        !unhide-sensors ok 2
        """

        sensor_names = self.unhide_sensor_re(name_re, not_hidden_ok=True)
        for sn in sensor_names:
            self.reply_inform(sock, Message.inform(req_msg.name, sn), req_msg)
        return ('ok', len(sensor_names))

    @return_reply(Int())
    def request_list_hidden_sensors(self, sock, req_msg):
        """List all hidden sensors names, one inform per sensor"""
        hidden_sensor_names = sorted(self._hidden_sensors.keys())
        for sn in hidden_sensor_names:
            self.reply_inform(sock, Message.inform(req_msg.name, sn), req_msg)
        return ('ok', len(hidden_sensor_names))

    @request(Bool())
    @return_reply()
    def request_hang_requests(self, sock, hang):
        """Make device hang on all requests other than watchdog"""
        self._model.get_test_sensor('hang-requests').set_value(hang)
        return ('ok',)

    @request(Str(), Str(),
             Discrete(('unknown','nominal','warn','error','failure'), optional=True),
             Timestamp(optional=True))
    @return_reply()
    def request_set_test_sensor_value(self, sock, sensorname, value, status, timestamp):
        """Set a test sensor's value, status and timestamp.

        Parameters
        ----------
        sensor : str
            Name of the sensor whose value to set.
        value : object
            The sensor's value. Type should be appropriate for named sensor.
        status : {'unknown','nominal','warn','error','failure'}
            The sensor value's status. Optional.
        timestamp : float, in milliseconds since the Unix Epoch
            The sensor value's timestamp. Optional.

        Returns
        -------
        success : {'ok', 'fail'}
            Whether setting the sensor values, status and timestamp succeeded.
        """
        try:
            sensor = self._model.get_test_sensor(sensorname)
            value = sensor.parse_value(value)
            if not status and not timestamp and hasattr(self._model, "_update_test_sensor"):
                self._model._update_test_sensor(sensor, value)
            else:
                status = sensor.STATUS_NAMES[status] if status else sensor.NOMINAL
                sensor.set_value(value, status, timestamp)
        except ValueError, e:
            return ('fail', e)

        return ['ok']
