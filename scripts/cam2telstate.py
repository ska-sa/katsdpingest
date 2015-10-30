#!/usr/bin/env python

from __future__ import print_function, division
import tornado
import tornado.ioloop
import tornado.gen
import katportalclient
import logging
import sys
import pprint
import katsdptelstate
import six
import signal


class Sensor(object):
    """Information about a sensor to be collected from CAM. This may later be
    replaced by a kattelmod class.
    """
    def __init__(self, cam_name, sp_name=None, sampling_strategy='event', immutable=False):
        self.cam_name = cam_name
        self.sp_name = sp_name or cam_name
        self.sampling_strategy = sampling_strategy
        self.immutable = immutable

    def prefix(self, cam_prefix, sp_prefix=None):
        """Return a copy of the sensor with a prefixed name. If `sp_prefix` is
        omitted, `cam_prefix` is used in its place.

        The prefix should omit the underscore used to join it to the name.
        """
        if sp_prefix is None:
            sp_prefix = cam_prefix
        return Sensor(cam_prefix + '_' + self.cam_name,
                      sp_prefix + '_' + self.sp_name,
                      self.sampling_strategy,
                      self.immutable)


# Per-receptor sensors, without the prefix for the receptor name
RECEPTOR_SENSORS = [
    Sensor('activity'),
    Sensor('target'),
    Sensor('pos_request_scan_azim', sampling_strategy='period 0.4'),
    Sensor('pos_request_scan_elev', sampling_strategy='period 0.4'),
    Sensor('pos_actual_scan_azim', sampling_strategy='period 0.4'),
    Sensor('pos_actual_scan_elev', sampling_strategy='period 0.4'),
    Sensor('dig_noise_diode'),
    Sensor('ap_indexer_position'),
    Sensor('rsc_rxl_serial_number'),
    Sensor('rsc_rxs_serial_number'),
    Sensor('rsc_rxu_serial_number'),
    Sensor('rsc_rxx_serial_number')
]
# Data proxy sensors without the data proxy prefix
DATA_SENSORS = [
    Sensor('target'),
    Sensor('auto_delay_enabled')
]
# Subarray sensors with the subarray name prefix
SUBARRAY_SENSORS = [
    Sensor('config_label', immutable=True),
    Sensor('band', immutable=True),
    Sensor('product', immutable=True),
    Sensor('sub_nr', immutable=True)
]
# All other sensors
OTHER_SENSORS = [
    Sensor('anc_weather_pressure'),
    Sensor('anc_weather_relative_humidity'),
    Sensor('anc_weather_temperature'),
    Sensor('anc_weather_wind_direction'),
    Sensor('anc_weather_wind_speed')
]


def configure_logging():
    if len(logging.root.handlers) > 0: logging.root.removeHandler(logging.root.handlers[0])
    formatter = logging.Formatter("%(asctime)s.%(msecs)dZ - %(filename)s:%(lineno)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logging.root.addHandler(sh)

    logger = logging.getLogger("katsdpingest.cam2telstate")
    logger.setLevel(logging.INFO)

    # configure SPEAD to display warnings about dropped packets etc...
    logging.getLogger('spead2').setLevel(logging.WARNING)
    return logger


def parse_args():
    parser = katsdptelstate.ArgumentParser()
    parser.add_argument('url', type=str, help='WebSocket URL to connect to')
    parser.add_argument('--namespace', type=str, default='sp', help='Namespace to create in katportal [%(default)s]')
    parser.add_argument('--antenna', dest='antennas', type=str, default=[], action='append', help='An antenna name in the subarray (repeat for each antenna)')
    args = parser.parse_args()
    if not args.telstate:
        print('--telstate is required', file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    return args


class Client(object):
    def __init__(self, args, logger):
        self._args = args
        self._telstate = args.telstate
        self._logger = logger
        self._loop = tornado.ioloop.IOLoop().current()
        self._portal_client = None
        self._sensors = None  #: Dictionary from CAM name to sensor object
        sensors = self.get_sensors()
        self._sensors = {x.cam_name: x for x in sensors}

    def get_sensors(self):
        """Get list of sensors to be collected from CAM. This should be
        replaced to use kattelmod.

        Returns
        -------
        sensors : list of `Sensor`
        """

        sensors = []
        for antenna_name in self._args.antennas:
            for sensor in RECEPTOR_SENSORS:
                sensors.append(sensor.prefix(antenna_name))
        # XXX Nasty hack to get SDP onto cbf name for AR1 integration
        for (cam_prefix, sp_prefix) in [('data_1', 'cbf')]:
            for sensor in DATA_SENSORS:
                sensors.append(sensor.prefix(cam_prefix, sp_prefix))
        for (cam_prefix, sp_prefix) in [('subarray_1', 'sub')]:
            for sensor in SUBARRAY_SENSORS:
                sensors.append(sensor.prefix(cam_prefix, sp_prefix))
        sensors.extend(OTHER_SENSORS)
        return sensors

    @tornado.gen.coroutine
    def start(self):
        self._portal_client = katportalclient.KATPortalClient(
            self._args.url, self.update_callback, logger=self._logger)
        yield self._portal_client.connect()
        status = yield self._portal_client.subscribe(
            self._args.namespace, self._sensors.keys())
        self._logger.info("Subscribed to %d channels", status)
        for sensor in six.itervalues(self._sensors):
            status = yield self._portal_client.set_sampling_strategy(
                self._args.namespace, sensor.cam_name, sensor.sampling_strategy)
            result = status[sensor.cam_name]
            if result[u'success']:
                self._logger.info("Set sampling strategy on %s to %s",
                    sensor.cam_name, sensor.sampling_strategy)
            else:
                self._logger.warn("Failed to set sampling strategy on %s: %s",
                    sensor.cam_name, result[u'info'])
        for signal_number in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(
                signal_number,
                lambda sig, frame: self._loop.add_callback_from_signal(self.close))
            self._logger.debug('Set signal handler for %s', signal_number)

    def process_update(self, item):
        self._logger.debug("Received update %s", pprint.pformat(item))
        data = item[u'msg_data']
        if data is None:
            return
        name = data[u'name'].encode('us-ascii')
        timestamp = data[u'timestamp']
        status = data[u'status'].encode('us-ascii')
        value = data[u'value']
        if isinstance(value, unicode):
            value = value.encode('us-ascii')
        if status == 'unknown':
            self._logger.debug("Sensor {} received update '{}' with status 'unknown' (ignored)"
                    .format(name, value))
        elif name in self._sensors:
            sensor = self._sensors[name]
            self._telstate.add(sensor.sp_name, value, timestamp, immutable=sensor.immutable)
        else:
            self._logger.debug("Sensor {} received update '{}' but we didn't subscribe (ignored)"
                    .format(name, value))

    def update_callback(self, msg):
        self._logger.info("update_callback: %s", pprint.pformat(msg))
        if isinstance(msg, list):
            for item in msg:
                self.process_update(item)
        else:
            self.process_update(msg)

    def close_handler(self, sig, frame):
        self._loop.add_callback_from_signal(self.close)

    @tornado.gen.coroutine
    def close(self):
        yield self._portal_client.unsubscribe(self._args.namespace)
        self._portal_client.disconnect()
        self._logger.info("disconnected")
        self._loop.stop()

def main():
    args = parse_args()
    logger = configure_logging()
    loop = tornado.ioloop.IOLoop().current()
    loop.install()
    client = Client(args, logger)
    loop.add_callback(client.start)
    loop.start()

if __name__ == '__main__':
    main()
