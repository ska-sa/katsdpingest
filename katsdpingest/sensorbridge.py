"""Bridge between single sensor and corresponding SPEAD item in stream."""

import logging

from katcp import Sensor


logger = logging.getLogger(__name__)


class SensorBridge(object):
    """Bridge between single sensor and corresponding SPEAD item in stream.

    Parameters
    ----------
    name : string
        Sensor name (used to name the corresponding SPEAD item)
    description : string
        Sensor description (which becomes the SPEAD item description)

    Attributes
    ----------
    spead_id : int
        SPEAD ID associated with sensor item
    last_update : string
        Last sensor update in form of "timestamp status 'value'" string
    strategy : string
        Name of strategy used to monitor sensor
    param : string
        Parameter of selected strategy

    """

    # Pick a SPEAD id range that is not in use here
    next_available_spead_id = 0x7000

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.spead_id = SensorBridge.next_available_spead_id
        SensorBridge.next_available_spead_id += 1
        self.last_update = ''
        self.strategy = 'none'
        self.param = ''

    def start_listening(self):
        """Start listening to sensor and send updates to SPEAD stream."""
        pass

    def stop_listening(self):
        """Stop listening to sensor and stop updates to SPEAD stream."""
        pass


class KatcpSensorBridge(object):
    """Bridge between single KATCP sensor and corresponding SPEAD item in stream.

    Parameters
    ----------
    name : string
        Sensor name (used to name the corresponding SPEAD item)
    katcp_sensor : :class:`katcorelib.KATSensor` object
        Sensor object representing KATCP sensor
    server : :class:`Cam2SpeadDeviceServer` object or similar
        Device server that serves SPEAD stream (via update_sensor method)

    """
    def __init__(self, name, katcp_sensor, server):
        super(KatcpSensorBridge, self).__init__(name, katcp_sensor.description)
        self.katcp_sensor, self.server = katcp_sensor, server
        self.listening = False
        # Store katcp.Sensor which will be used to parse KATCP string in listener
        sensor_type = Sensor.parse_type(self.katcp_sensor.type)
        params = ['unknown'] if sensor_type == Sensor.DISCRETE else None
        self._sensor = Sensor(sensor_type, self.katcp_sensor.name,
                              self.katcp_sensor.description, self.katcp_sensor.units,
                              params)

    def listen(self, update_seconds, value_seconds, status, value_string):
        """Callback that pushes KATCP sensor update to SPEAD stream.

        Parameters
        ----------
        update_seconds : float
            Unix timestamp indicating when update was received by local client
        value_seconds : float
            Unix timestamp indicating when sensor value was measured
        status : string
            Status of this update ('nominal' if all is well)
        value_string : string
            Sensor value encoded as a KATCP string

        """
        # Force value to be accepted by discrete sensor
        if self._sensor.stype == 'discrete':
            self._sensor._kattype._values.append(value_string)
            self._sensor._kattype._valid_values.add(value_string)
        # First convert value string to intended type to get appropriate repr()
        value = self._sensor.parse_value(value_string)
        # Transmit event-based updates immediately, while other updates
        # are periodically resampled in a separate thread
        self.last_update = self.server.update_sensor(self.name, value,
                                                     value_seconds, status,
                                                     self.strategy == 'event')

    def start_listening(self):
        """Start listening to sensor and send updates to SPEAD stream."""
        if not self.listening:
            self.katcp_sensor.set_strategy(self.strategy, self.param)
            self.katcp_sensor.register_listener(self.listen)
            # This triggers the callback to obtain a valid last_update
            self.katcp_sensor.get_value()
            logger.debug("Start listening to sensor %r" % (self.name,))
            self.listening = True

    def stop_listening(self):
        """Stop listening to sensor and stop updates to SPEAD stream."""
        if self.listening:
            self.katcp_sensor.unregister_listener(self.listen)
            self.katcp_sensor.set_strategy('none')
            logger.debug("Stopped listening to sensor %r" % (self.name,))
            self.listening = False


class VirtualSensorBridge(SensorBridge):
    """Bridge between single virtual sensor and corresponding SPEAD item.

    Parameters
    ----------
    name : string
        Sensor name (used to name the corresponding SPEAD item)
    description : string
        Sensor description (which becomes the SPEAD item description)

    """
    def __init__(self, name, description):
        super(VirtualSensorBridge, self).__init__(name, description)
