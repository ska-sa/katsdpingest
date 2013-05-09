"""Device server that listens to KATCP sensors and passes updates to SPEAD stream."""


import logging
import threading

import spead
from katcp import DeviceServer
from katcp.kattypes import return_reply, Str
from katcapture import __version__

logger = logging.getLogger("kat.katcp2spead")


class SensorBridge(object):
    """Bridge between single KATCP sensor and corresponding SPEAD item in stream."""

    # Pick a SPEAD id range that is not in use here
    next_available_spead_id = 0x7000

    def __init__(self, name, katcp_sensor, server):
        self.name, self.katcp_sensor, self.server = name, katcp_sensor, server
        self.spead_id = SensorBridge.next_available_spead_id
        SensorBridge.next_available_spead_id += 1
        self.listening = False

    def listen(self, update_seconds, value_seconds, status, value):
        """Callback that pushes KATCP sensor update to SPEAD stream."""
        # Assemble sensor data into string for transport over SPEAD
        update = "%r %s %r" % (value_seconds, status, value)
        logger.debug("Updating sensor %r: %s" % (self.name, update))
        # A lock is needed because each KATCP device client runs in its own
        # thread while calling this callback and the main SPEAD item group
        # of the server is shared among them (blame ig.get_heap()...)
        with self.server._spead_lock:
            self.server.ig['sensor_' + self.name] = update
            self.server.tx.send_heap(self.server.ig.get_heap())

    def start_listening(self):
        """Start listening to sensor and send updates to SPEAD stream."""
        if not self.listening:
            self.katcp_sensor.register_listener(self.listen)
            logger.debug("Start listening to sensor %r" % (self.name,))
            self.listening = True

    def stop_listening(self):
        """Stop listening to sensor and stop updates to SPEAD stream."""
        if self.listening:
            self.katcp_sensor.unregister_listener(self.listen)
            logger.debug("Stopped listening to sensor %r" % (self.name,))
            self.listening = False


class Katcp2SpeadDeviceServer(DeviceServer):
    """Device server that listens to KATCP sensors and updates SPEAD stream.

    Parameters
    ----------
    kat : :class:`katcorelib.KATCoreHost` object
        Host object providing monitoring access to system
    sensors : list of tuples of 3 strings
        List of sensors to listen to, and corresponding sensor strategy to be
        set as (name, strategy, param) tuple (use full name from kat.sensors)
    spead_host : string
        Host to receive SPEAD stream
    spead_port : int
        Port on host to receive SPEAD stream

    """

    VERSION_INFO = ("katcp2spead", 0, 1)
    BUILD_INFO = ("katcp2spead", 0, 1, __version__)

    def __init__(self, kat, sensors, spead_host, spead_port, *args, **kwargs):
        super(Katcp2SpeadDeviceServer, self).__init__(*args, **kwargs)
        self.kat = kat
        self.tx = spead.Transmitter(spead.TransportUDPtx(spead_host, spead_port))
        self.ig = spead.ItemGroup()
        self._spead_lock = threading.Lock()
        self.sensor_bridges = {}
        # In future we might want to re-register sensors when starting stream
        for name, strategy, param in sensors:
            self.register_katcp_sensor(name, strategy, param)

    def setup_sensors(self):
        """Populate the dictionary of sensors (none so far)."""
        pass

    def register_katcp_sensor(self, name, strategy, param):
        """Register KATCP sensor, set sensor strategy and create bridge."""
        if name not in self.sensor_bridges:
            try:
                katcp_sensor = getattr(self.kat.sensors, name)
            except AttributeError:
                logger.warning("Could not register unavailable KATCP sensor %r" % (name,))
                return False
            self.sensor_bridges[name] = SensorBridge(name, katcp_sensor, self)
        self.sensor_bridges[name].katcp_sensor.set_strategy(strategy, param)
        logger.info("Registered KATCP sensor %r with strategy (%r, %r) and SPEAD id 0x%x" %
                    (name, strategy, param, self.sensor_bridges[name].spead_id))
        return True

    def send_initial_spead_packet(self):
        """Send initial SPEAD packet (or resend it later to re-establish SPEAD metadata)."""
        # Use a fresh item group if this is a resend of SPEAD metadata so as not to disturb main item group
        resend = len(self.ig.keys()) > 0
        ig = spead.ItemGroup() if resend else self.ig
        for name, sensor_bridge in self.sensor_bridges.iteritems():
            # Resend existing value if available, else consult the monitor store (slow)
            if resend:
                last_update = self.ig['sensor_' + name]
            else:
                # This could eventually be replaced by sensor.get_value()
                # if timestamp + status + value can be kept in sync
                history = sensor_bridge.katcp_sensor.get_stored_history(start_seconds=-1, last_known=True)
                # All KATCP events are sent as strings containing space-separated
                # value_timestamp + status + value, regardless of KATCP type
                # (consistent with the fact that KATCP data is string-based on the wire)
                last_update = "%r %s %r" % (history[0][-1], history[2][-1], history[1][-1])
            logger.debug("Adding info for sensor %r (id 0x%x) to initial packet: %s" %
                         (name, sensor_bridge.spead_id, last_update))
            ig.add_item(name='sensor_' + name, id=sensor_bridge.spead_id,
                        description=sensor_bridge.katcp_sensor.description,
                        shape=-1, fmt=spead.mkfmt(('s', 8)), init_val=last_update)
        self.tx.send_heap(ig.get_heap())
        logger.info('%s initial SPEAD packet containing %d items to %s' %
                    ('Resent' if resend else 'Sent', len(ig.ids()), self.tx.t._tx_ip_port))

    def start_listening(self):
        """Start listening to all registered sensors."""
        for sensor_bridge in self.sensor_bridges.itervalues():
            sensor_bridge.start_listening()

    def stop_listening(self):
        """Stop listening to all registered sensors."""
        for sensor_bridge in self.sensor_bridges.itervalues():
            sensor_bridge.stop_listening()

    @return_reply(Str())
    def request_start_stream(self, req, msg):
        """Start the SPEAD stream of KATCP sensor data (or resend metadata)."""
        self.send_initial_spead_packet()
        self.start_listening()
        return ("ok", "SPEAD stream started")

    @return_reply(Str())
    def request_stop_stream(self, req, msg):
        """Stop the SPEAD stream of KATCP sensor data."""
        self.stop_listening()
        return ("ok", "SPEAD stream stopped")
