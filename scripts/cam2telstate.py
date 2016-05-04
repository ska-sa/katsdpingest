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
    def __init__(self, cam_name, sp_name=None, sampling_strategy_and_params='event', immutable=False):
        self.cam_name = cam_name
        self.sp_name = sp_name or cam_name
        self.sampling_strategy_and_params = sampling_strategy_and_params
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
                      self.sampling_strategy_and_params,
                      self.immutable)


# Per-receptor sensors, without the prefix for the receptor name
RECEPTOR_SENSORS = [
    Sensor('activity'),
    Sensor('target'),
    Sensor('pos_request_scan_azim', sampling_strategy_and_params='period 0.4'),
    Sensor('pos_request_scan_elev', sampling_strategy_and_params='period 0.4'),
    Sensor('pos_actual_scan_azim', sampling_strategy_and_params='period 0.4'),
    Sensor('pos_actual_scan_elev', sampling_strategy_and_params='period 0.4'),
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
    parser.add_argument('subarray', type=int, help='Subarray number')
    parser.add_argument('url', type=str, help='WebSocket URL to connect to')
    parser.add_argument('--namespace', type=str, help='Namespace to create in katportal [sp_subarray_N]')
    parser.add_argument('--antenna', dest='antennas', type=str, default=[], action='append', help='An antenna name in the subarray (repeat for each antenna)')
    args = parser.parse_args()
    if args.namespace is None:
        args.namespace = 'sp_subarray_{}'.format(args.subarray)
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
        self._loop = tornado.ioloop.IOLoop.current()
        self._portal_client = None
        self._sensors = None  #: Dictionary from CAM name to sensor object
        self._data_name = tornado.concurrent.Future()

    def get_sensors(self):
        """Get list of sensors to be collected from CAM. This should be
        replaced to use kattelmod. It must only be called after
        :attr:`_data_name` is resolved.

        Returns
        -------
        sensors : list of `Sensor`
        """

        sensors = []
        for antenna_name in self._args.antennas:
            for sensor in RECEPTOR_SENSORS:
                sensors.append(sensor.prefix(antenna_name))
        # Convert CAM prefixes to SDP ones
        for (cam_prefix, sp_prefix) in [(self._data_name.result(), 'cbf')]:
            for sensor in DATA_SENSORS:
                sensors.append(sensor.prefix(cam_prefix, sp_prefix))
        for (cam_prefix, sp_prefix) in [('subarray_{}'.format(self._args.subarray), 'sub')]:
            for sensor in SUBARRAY_SENSORS:
                sensors.append(sensor.prefix(cam_prefix, sp_prefix))
        sensors.extend(OTHER_SENSORS)
        return sensors

    @tornado.gen.coroutine
    def get_data_name(self):
        """Query subarray_N_pool_resources to find out which data_M resource is
        assigned to the subarray.
        """
        sensor = 'subarray_{}_pool_resources'.format(self._args.subarray)
        status = yield self._portal_client.subscribe(
            self._args.namespace, sensor)
        if status != 1:
            raise RuntimeError("Expected 1 sensor for {}, found {}".format(sensor, status))
        status = yield self._portal_client.set_sampling_strategy(
            self._args.namespace, sensor, 'event')
        result = status[sensor]
        if result[u'success']:
            self._logger.info("Set sampling strategy on %s to event", sensor)
        else:
            raise RuntimeError("Failed to set sampling strategy on {}: {}".format(
                sensor, result[u'info']))
        # Wait until we get a callback with the value
        yield self._data_name
        yield self._portal_client.unsubscribe(self._args.namespace, sensor)

    @tornado.gen.coroutine
    def start(self):
        try:
            self._portal_client = katportalclient.KATPortalClient(
                self._args.url, self.update_callback, io_loop=self._loop, logger=self._logger)
            yield self._portal_client.connect()
            # First find out which data_* resource is allocated to the subarray
            yield self.get_data_name()
            # Now we can tell which sensors to subscribe to
            self._sensors = {x.cam_name: x for x in self.get_sensors()}

            status = yield self._portal_client.subscribe(
                self._args.namespace, self._sensors.keys())
            self._logger.info("Subscribed to %d channels", status)
            for sensor in six.itervalues(self._sensors):
                status = yield self._portal_client.set_sampling_strategy(
                    self._args.namespace, sensor.cam_name, sensor.sampling_strategy_and_params)
                result = status[sensor.cam_name]
                if result[u'success']:
                    self._logger.info("Set sampling strategy on %s to %s",
                        sensor.cam_name, sensor.sampling_strategy_and_params)
                else:
                    raise RuntimeError("Failed to set sampling strategy on {}: {}".format(
                        sensor.cam_name, result[u'info']))
            for signal_number in [signal.SIGINT, signal.SIGTERM]:
                signal.signal(
                    signal_number,
                    lambda sig, frame: self._loop.add_callback_from_signal(self.close))
                self._logger.debug('Set signal handler for %s', signal_number)
        except Exception:
            self._logger.error("Exception during startup", exc_info=True)
            self._loop.stop()
        else:
            self._logger.info("Startup complete")

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
        if name == 'subarray_{}_pool_resources'.format(self._args.subarray):
            if not self._data_name.done():
                resources = value.split(',')
                for resource in resources:
                    if resource.startswith('data_'):
                        self._data_name.set_result(resource)
                        return
                self._data_name.set_exception(RuntimeError(
                    'No data_* resource found for subarray {}'.format(self._args.subarray)))
        else:
            if status == 'unknown':
                self._logger.warn("Sensor {} received update '{}' with status 'unknown' (ignored)"
                        .format(name, value))
            elif name in self._sensors:
                sensor = self._sensors[name]
                try:
                    self._telstate.add(sensor.sp_name, value, timestamp, immutable=sensor.immutable)
                except katsdptelstate.ImmutableKeyError as e:
                    self._logger.error(e)
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

    @tornado.gen.coroutine
    def close(self):
        yield self._portal_client.unsubscribe(self._args.namespace)
        self._portal_client.disconnect()
        self._logger.info("disconnected")
        self._loop.stop()

def main():
    args = parse_args()
    logger = configure_logging()
    loop = tornado.ioloop.IOLoop.instance()
    client = Client(args, logger)
    loop.add_callback(client.start)
    loop.start()

if __name__ == '__main__':
    main()
