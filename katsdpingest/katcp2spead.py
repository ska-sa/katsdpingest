"""Device server that listens to KATCP sensors and passes updates to SPEAD stream."""

import logging
import threading

import spead
from katcp import DeviceServer
from katcp.kattypes import request, return_reply, Str, Int
from katsdpingest import __version__
import time
import Queue
import copy

logger = logging.getLogger("kat.katcp2spead")

class SensorBridge(object):
    """Bridge between single KATCP sensor and corresponding SPEAD item in stream."""

    # Pick a SPEAD id range that is not in use here
    next_available_spead_id = 0x7000

    def __init__(self, name, katcp_sensor, server):
        self.name, self.katcp_sensor, self.server = name, katcp_sensor, server
        self.spead_id = SensorBridge.next_available_spead_id
        SensorBridge.next_available_spead_id += 1
        self.strategy = 'none'
        self.param = ''
        self.listening = False
        self.last_update = ''

    def store_strategy(self, strategy, param):
        """Store sensor strategy if it has changed."""
        if strategy == self.strategy and param == self.param:
            return
        self.strategy = strategy
        self.param = param
        logger.info("Registered KATCP sensor %r with strategy (%r, %r) and SPEAD id 0x%x" %
                    (self.name, self.strategy, self.param, self.spead_id))

    def listen(self, update_seconds, value_seconds, status, value):
        """Callback that pushes KATCP sensor update to SPEAD stream."""
        # All KATCP events are sent as strings containing space-separated
        # value_timestamp + status + value, regardless of KATCP type
        # (consistent with the fact that KATCP data is string-based on the wire)
        update = "%r %s %r" % (value_seconds, status, value)
        logger.debug("Updating sensor %r: %s" % (self.name, update))
        # A lock is needed because each KATCP device client runs in its own
        # thread while calling this callback and the main SPEAD item group
        # of the server is shared among them (blame ig.get_heap()...)
        if self.server.streaming:
            with self.server._spead_lock:
                self.server.ig['sensor_' + self.name] = update
                # if the sensor strategy is event, transmit the update
                if self.strategy == 'event':
                    self.server.transmit(self.server.ig.get_heap())
        self.last_update = update

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

class transmitThread(threading.Thread):
    """Thread which transmits a sensor heap to a particular IP and port."""

    def __init__(self, spead_host, spead_port, init_heap):
        threading.Thread.__init__(self)
        self.name = 'Thread_' + spead_host + ':' + str(spead_port)
        self.mailbox = Queue.Queue()
        self._transmit = spead.Transmitter(spead.TransportUDPtx(spead_host,
                          spead_port))
        self._thread_active = True
        self._init_heap = init_heap

    def run(self):
        # send initial spead packed to initialise
        self._transmit.send_heap(self._init_heap)

        # wait for packets to be addres to the queue, then transmit them
        while self._thread_active:
            try:
                # timeout necessary to stop waiting for queue item,
                # after the transmitter is removed
                heap = self.mailbox.get(block=True, timeout=60)
                self._transmit.send_heap(heap)
            except Queue.Empty:
                pass

    def stop(self):
        self._thread_active = False
 
class Katcp2SpeadDeviceServer(DeviceServer):
    """Device server that listens to KATCP sensors and updates SPEAD stream.

    Parameters
    ----------
    kat : :class:`katcorelib.KATCoreConn` object
        KATCoreConn object providing monitoring access to system
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

    def __init__(self, kat, sensors, tx_period, *args, **kwargs):
        super(Katcp2SpeadDeviceServer, self).__init__(*args, **kwargs)
        self.kat, self.sensor_strategies = kat, sensors
        self._spead_lock = threading.Lock()
        self.sensor_bridges = {}
        self.streaming = False
        self.transmitters = {}
        self.tx_period = float(tx_period)
        self.init_heap = None
        self.ig = None

    def setup_sensors(self):
        """Populate the dictionary of sensors (none so far)."""
        pass

    def register_sensors(self):
        """Register all requested KATCP sensors, skipping the unknown ones."""
        for name, strategy, param in self.sensor_strategies:
            if name not in self.sensor_bridges:
                try:
                    sensor = getattr(self.kat.sensors, name)
                except AttributeError:
                    logger.warning("Could not register unavailable KATCP sensor %r" % (name,))
                    continue
                self.sensor_bridges[name] = SensorBridge(name, sensor, self)
            # It is possible to change the strategy on an existing sensor bridge
            self.sensor_bridges[name].store_strategy(strategy, param)

    def register_transmitter(self,spead_host,spead_port):
        """Register transmitter."""
        transmitter_name = spead_host + '_' + str(spead_port)
        if transmitter_name not in self.transmitters:
            # create transmitter thread, add transmitter to dict, send initial heap
            self.transmitters[transmitter_name] = transmitThread(spead_host,spead_port,
                                                  copy.deepcopy(self.init_heap))
            self.transmitters[transmitter_name].start()

    def remove_transmitter(self,spead_host,spead_port):
        """Remove transmitter."""
        transmitter_name = spead_host + '_' + str(spead_port)
        # stop transmitter thread
        self.transmitters[transmitter_name].stop()
        # remove transmitter from dictionary
        del self.transmitters[transmitter_name]

    def transmit(self, heap):
        """Transmit spead packet to all recievers."""
        for tx in self.transmitters:
            self.transmitters[tx].mailbox.put(copy.deepcopy(heap))
             
    def set_initial_spead_packet(self):
        """Initialise initial SPEAD packet to establish metadata / item structure."""
        self.ig = spead.ItemGroup()
        for name, bridge in self.sensor_bridges.iteritems():
            logger.debug("Adding info for sensor %r (id 0x%x) to initial packet: %s" %
                         (name, bridge.spead_id, bridge.last_update))
            self.ig.add_item(name='sensor_' + name, id=bridge.spead_id,
                        description=bridge.katcp_sensor.description,
                        shape=-1, fmt=spead.mkfmt(('s', 8)),
                        init_val=bridge.last_update)
        self.init_heap = self.ig.get_heap()

    def start_listening(self):
        """Start listening to all registered sensors."""
        for bridge in self.sensor_bridges.itervalues():
            bridge.start_listening()

    def stop_listening(self):
        """Stop listening to all registered sensors."""
        for bridge in self.sensor_bridges.itervalues():
            bridge.stop_listening()

    def periodic_flush(self,start):
        """Periodically send heap."""
        while self.streaming:
            time.sleep(self.tx_period - (time.time()-start))
            start = time.time()
            # transmit current heap to all active transmitters
            with self._spead_lock:
                self.transmit(self.ig.get_heap())

    @return_reply(Str())
    def request_start_stream(self, req, msg):
        """Start the SPEAD stream of KATCP sensor data."""
        self.register_sensors()
        self.start_listening()
        self.set_initial_spead_packet()
        self.streaming = True
        smsg = "SPEAD stream started"
        logger.info(smsg)
        start = time.time()
        t = threading.Thread(target=self.periodic_flush,args=(time.time(),))
        t.start()
        return ("ok", smsg)

    @request(Str(), Int())
    @return_reply(Str())
    def request_add_transmitter(self, req, spead_host, spead_port):
        """Add a transmitting thread."""    
        self.register_transmitter(spead_host,spead_port)
        smsg = "Sensor SPEAD transmitter thread registered, to port %s on %s" % (spead_port, spead_host)
        logger.info(smsg)
        logger.info('Sent initial SPEAD packet containing %d items to %s' % (len(self.ig.ids()), spead_port))
        return ("ok", smsg)

    @request(Str(), Int())
    @return_reply(Str())
    def request_remove_transmitter(self, req, spead_host, spead_port):
        """Add a transmitting thread."""
        self.remove_transmitter(spead_host,spead_port)
        smsg = "Sensor SPEAD transmitter thread removed, to port %s on %s" % (spead_port, spead_host)
        logger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_stop_stream(self, req, msg):
        """Stop the SPEAD stream of KATCP sensor data."""
        self.streaming = False
        self.stop_listening()
        smsg = "SPEAD stream stopped"
        logger.info(smsg)
        return ("ok", smsg)
