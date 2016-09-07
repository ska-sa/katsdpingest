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
import re
import numpy as np


STATUS_KEY = 'sdp_cam2telstate_status'


def comma_split(value):
    return value.split(',')


class Sensor(object):
    """Information about a sensor to be collected from CAM. This may later be
    replaced by a kattelmod class.

    Parameters
    ----------
    cam_name : str
        Name of the sensor to pass to katportalclient
    sp_name : str, optional
        Name of the sensor within katsdptelstate (defaults to `cam_name`)
    sampling_strategy_and_params : str, optional
        Sampling method to pass to katportalclient
    immutable : bool, optional
        Passed to :meth:`katsdptelstate.TelescopeState.add`
    convert : callable, optional
        If provided, it is used to transform the sensor value before storing
        it in telescope state.
    """
    def __init__(self, cam_name, sp_name=None, sampling_strategy_and_params='event',
                 immutable=False, convert=None):
        self.cam_name = cam_name
        self.sp_name = sp_name or cam_name
        self.sampling_strategy_and_params = sampling_strategy_and_params
        self.immutable = immutable
        self.convert = convert
        self.waiting = True     #: Waiting for an initial value

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
                      self.immutable,
                      self.convert)


# Per-receptor sensors, without the prefix for the receptor name
RECEPTOR_SENSORS = [
    Sensor('observer'),
    Sensor('activity'),
    Sensor('target'),
    Sensor('pos_request_scan_azim', sampling_strategy_and_params='period 0.4'),
    Sensor('pos_request_scan_elev', sampling_strategy_and_params='period 0.4'),
    Sensor('pos_actual_scan_azim', sampling_strategy_and_params='period 0.4'),
    Sensor('pos_actual_scan_elev', sampling_strategy_and_params='period 0.4'),
    Sensor('dig_noise_diode'),
    Sensor('dig_synchronisation_epoch'),
    Sensor('ap_indexer_position'),
    Sensor('ap_point_error_tiltmeter_enabled'),
    Sensor('ap_tilt_corr_azim'),
    Sensor('ap_tilt_corr_elev'),
    Sensor('rsc_rxl_serial_number'),
    Sensor('rsc_rxs_serial_number'),
    Sensor('rsc_rxu_serial_number'),
    Sensor('rsc_rxx_serial_number'),
    Sensor('ap_version_list', immutable=True)
]
# Data proxy sensors without the data proxy prefix
DATA_SENSORS = [
    Sensor('target'),
    Sensor('auto_delay_enabled'),
    Sensor('cbf_corr_adc_sample_rate', immutable=True),
    Sensor('cbf_corr_bandwidth', immutable=True),
    Sensor('cbf_corr_baseline_ordering', immutable=True, convert=np.safe_eval),
    Sensor('cbf_corr_center_frequency', immutable=True),
    Sensor('cbf_corr_integration_time', immutable=True),
    Sensor('cbf_corr_n_accs', immutable=True),
    Sensor('cbf_corr_n_chans', immutable=True),
    Sensor('cbf_corr_n_inputs', immutable=True),
    Sensor('cbf_corr_scale_factor_timestamp', immutable=True),
    Sensor('cbf_corr_synch_epoch', immutable=True),
    Sensor('cbf_version_list', immutable=True),
    Sensor('input_labels', immutable=True, convert=comma_split),
    Sensor('loaded_delay_correction'),
    Sensor('spmc_version_list', immutable=True)
]
# Subarray sensors with the subarray name prefix
SUBARRAY_SENSORS = [
    Sensor('config_label', immutable=True),
    Sensor('band', immutable=True),
    Sensor('product', immutable=True),
    Sensor('sub_nr', immutable=True),
    Sensor('dump_rate', immutable=True),
    Sensor('pool_resources', immutable=True)
]
# All other sensors
OTHER_SENSORS = [
    Sensor('anc_air_pressure'),
    Sensor('anc_air_relative_humidity'),
    Sensor('anc_air_temperature'),
    Sensor('anc_wind_direction'),
    Sensor('anc_mean_wind_speed'),
    Sensor('mcp_dmc_version_list', immutable=True),
    Sensor('mcp_cmc_version_list', immutable=True)
]


def configure_logging():
    if len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[0])
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
    parser.add_argument('--subarray-numeric-id', type=int, help='Subarray number')
    parser.add_argument('--url', type=str, help='WebSocket URL to connect to')
    parser.add_argument('--namespace', type=str, help='Namespace to create in katportal [sp_subarray_N]')
    args = parser.parse_args()
    if args.namespace is None:
        args.namespace = 'sp_subarray_{}'.format(args.subarray_numeric_id)
    # Can't use required= on the parser, because telstate-provided arguments
    # are treated only as defaults rather than having set a value.
    if args.telstate is None:
        parser.error('argument --telstate is required')
    if args.subarray_numeric_id is None:
        parser.error('argument --subarray-numeric-id is required')
    if args.url is None:
        parser.error('argument --url is required')
    return args


class Client(object):
    def __init__(self, args, logger):
        self._args = args
        self._telstate = args.telstate
        self._logger = logger
        self._loop = tornado.ioloop.IOLoop.current()
        self._portal_client = None
        self._sensors = None  #: Dictionary from CAM name to sensor object
        self._pool_resources = tornado.concurrent.Future()
        self._active = tornado.concurrent.Future()  #: Set once subarray_N_state is active
        self._data_name = None    #: Set once _pool_resources result is set
        self._receptors = []      #: Set once _pool_resources result is set
        self._waiting = 0         #: Number of sensors whose initial value is still outstanding

    def get_sensors(self):
        """Get list of sensors to be collected from CAM. This should be
        replaced to use kattelmod. It must only be called after
        :attr:`_pool_resources` is resolved.

        Returns
        -------
        sensors : list of `Sensor`
        """

        sensors = []
        for receptor_name in self._receptors:
            for sensor in RECEPTOR_SENSORS:
                sensors.append(sensor.prefix(receptor_name))
        # Convert CAM prefixes to SDP ones
        for (cam_prefix, sp_prefix) in [(self._data_name, 'data')]:
            for sensor in DATA_SENSORS:
                sensors.append(sensor.prefix(cam_prefix, sp_prefix))
        for (cam_prefix, sp_prefix) in [('subarray_{}'.format(self._args.subarray_numeric_id), 'sub')]:
            for sensor in SUBARRAY_SENSORS:
                sensors.append(sensor.prefix(cam_prefix, sp_prefix))
        sensors.extend(OTHER_SENSORS)
        return sensors

    @tornado.gen.coroutine
    def subscribe_one(self, sensor):
        """Utility for subscribing to a single sensor. This is only used for
        "special" sensors used during startup.

        Parameters
        ----------
        sensor : str
            Name of the sensor to subscribe to
        """
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

    @tornado.gen.coroutine
    def get_pool_resources(self):
        """Query subarray_N_pool_resources to find out which data_M resource and
        which receptors are assigned to the subarray.
        """
        sensor = 'subarray_{}_pool_resources'.format(self._args.subarray_numeric_id)
        yield self.subscribe_one(sensor)
        # Wait until we get a callback with the value
        yield self._pool_resources
        yield self._portal_client.unsubscribe(self._args.namespace, sensor)

    @tornado.gen.coroutine
    def wait_active(self):
        """Subscribe to subarray_N_state and wait for its value to become
        'active'."""
        sensor = 'subarray_{}_state'.format(self._args.subarray_numeric_id)
        yield self.subscribe_one(sensor)
        # Wait until we get a callback to say that its active
        yield self._active
        yield self._portal_client.unsubscribe(self._args.namespace, sensor)

    def set_status(self, status):
        self._telstate.add(STATUS_KEY, status)

    @tornado.gen.coroutine
    def start(self):
        try:
            self.set_status('connecting')
            self._portal_client = katportalclient.KATPortalClient(
                self._args.url, self.update_callback, io_loop=self._loop, logger=self._logger)
            yield self._portal_client.connect()
            self.set_status('waiting for subarray activation')
            # Wait to be sure that the subarray is fully activated
            yield self.wait_active()
            self.set_status('initialising')
            # First find out which resources are allocated to the subarray
            yield self.get_pool_resources()
            # Now we can tell which sensors to subscribe to
            self._sensors = {x.cam_name: x for x in self.get_sensors()}

            self._waiting = len(self._sensors)
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
                    self._logger.error("Failed to set sampling strategy on %s: %s",
                                       sensor.cam_name, result[u'info'])
                    # Not going to get any values, so don't wait for it
                    self._waiting -= 1
                    sensor.waiting = False
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

    def sensor_update(self, sensor, value, status, timestamp):
        name = sensor.cam_name
        if status not in ['nominal', 'warn', 'error']:
            self._logger.warn("Sensor {} received update '{}' with status '{}' (ignored)"
                              .format(name, value, status))
            return
        try:
            if sensor.convert is not None:
                value = sensor.convert(value)
        except Exception:
            self._logger.warn('Failed to convert %s, ignoring (value was %r)',
                              name, value, exc_info=True)
            return
        try:
            self._telstate.add(sensor.sp_name, value, timestamp, immutable=sensor.immutable)
            self._logger.debug('Updated %s to %s with timestamp %s',
                               sensor.sp_name, value, timestamp)
        except katsdptelstate.ImmutableKeyError as e:
            self._logger.error('Failed to set %s to %s with timestamp %s',
                               sensor.sp_name, value, timestamp, exc_info=True)

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
        if name == 'subarray_{}_state'.format(self._args.subarray_numeric_id):
            if not self._active.done() and value == 'active' and status == 'nominal':
                self._active.set_result(None)
            return
        elif name == 'subarray_{}_pool_resources'.format(self._args.subarray_numeric_id):
            if not self._pool_resources.done() and status == 'nominal':
                resources = value.split(',')
                data_found = False
                self._receptors = []
                for resource in resources:
                    if resource.startswith('data_'):
                        self._data_name = resource
                        data_found = True
                    elif re.match(r'^m\d+$', resource):
                        self._receptors.append(resource)
                if not data_found:
                    self._pool_resources.set_exception(RuntimeError(
                        'No data_* resource found for subarray {}'.format(self._args.subarray_numeric_id)))
                else:
                    self._pool_resources.set_result(resources)
        if self._sensors is None:
            return
        if name not in self._sensors:
            self._logger.warn("Sensor {} received update '{}' but we didn't subscribe (ignored)"
                               .format(name, value))
        else:
            sensor = self._sensors[name]
            last = False
            if sensor.waiting:
                sensor.waiting = False
                self._waiting -= 1
                if self._waiting == 0:
                    last = True
            try:
                self.sensor_update(sensor, value, status, timestamp)
            finally:
                if last:
                    self._logger.info('Initial values for all sensors seen')
                    self.set_status('ready')

    def update_callback(self, msg):
        self._logger.debug("update_callback: %s", pprint.pformat(msg))
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
