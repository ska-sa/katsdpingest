#!/usr/bin/env python

from __future__ import print_function, division

import logging
import collections
import itertools
import pprint
import string
import six
import signal
import re
import json

import tornado
import tornado.ioloop
import tornado.gen
import numpy as np
import katsdptelstate
import katsdpservices
import katportalclient
import katcp

import katsdpcam2telstate


def comma_split(value):
    return value.split(',')


def convert_bitmask(value):
    """Converts a string of 1's and 0's to a numpy array of bools"""
    if not isinstance(value, six.string_types) or not re.match('^[01]*$', value):
        return None
    else:
        return np.array([c == '1' for c in value])


class Template(string.Template):
    """Template for a sensor name."""

    # Override this to allow dots in the name
    idpattern = '[A-Za-z0-9._]+'


class Sensor(object):
    """Information about a sensor to be collected from CAM. This may later be
    replaced by a kattelmod class.

    Parameters
    ----------
    cam_name : str
        Name of the sensor to pass to katportalclient
    sdp_name : str or list of str, optional
        Name of the sensor within katsdptelstate (defaults to `cam_name`). If
        a list, a copy of the sensor is placed in telstate for each name in
        the list.
    sampling_strategy_and_params : str, optional
        Sampling method to pass to katportalclient
    immutable : bool, optional
        Passed to :meth:`katsdptelstate.TelescopeState.add`
    convert : callable, optional
        If provided, it is used to transform the sensor value before storing
        it in telescope state.
    """
    def __init__(self, cam_name, sdp_name=None, sampling_strategy_and_params='event',
                 immutable=False, convert=None):
        self.cam_name = cam_name
        self.sdp_name = sdp_name or cam_name
        self.sampling_strategy_and_params = sampling_strategy_and_params
        self.immutable = immutable
        self.convert = convert
        self.waiting = True     #: Waiting for an initial value

    def expand(self, substitutions):
        """Expand a template into a list of sensors. The sensor name may
        contain keys in braces. These are looked up in `substitutions` and
        replaced with each possible value to form the new sensors, taking the
        Cartesian product if there are multiple keys.

        The `sdp_name` must not contain any keys that are not part of the CAM
        name, and must not be a list.

        Parameters
        ----------
        substitutions : dict-like
            Maps a key to a list of (cam, sdp) values to substitute. Each sdp
            name is a list of values, each of which is used.

        Raises
        ------
        KeyError
            if a key is used in `sdp_name` but not in `cam_name`
        """
        def substitute(name, params):
            """Expand a template with parameters.

            Also eliminates doubled, leading and trailing underscores from the result.
            """
            name = Template(name).substitute(params)
            parts = [part for part in name.split('_') if part]
            return '_'.join(parts)

        keys = list(set(re.findall(r'\$\{([^}]+)\}', self.cam_name)))
        iters = [substitutions[key] for key in keys]
        ans = []
        for values in itertools.product(*iters):
            cam_dict = {key: value[0] for key, value in zip(keys, values)}
            sdp_dict = {key: value[1] for key, value in zip(keys, values)}
            sdp_names = []
            # sdp_dict maps each key to a list of values to substitute.
            # Check that they are lists and not a single string.
            for value in six.itervalues(sdp_dict):
                assert isinstance(value, list)
            for sdp_values in itertools.product(*six.itervalues(sdp_dict)):
                # Recombine this specific set of values with the keys
                sdp_single_dict = dict(zip(sdp_dict.keys(), sdp_values))
                sdp_names.append(substitute(self.sdp_name, sdp_single_dict))
            ans.append(Sensor(substitute(self.cam_name, cam_dict),
                              sdp_names,
                              self.sampling_strategy_and_params,
                              self.immutable,
                              self.convert))
        return ans


#: Templates for sensors
SENSORS = [
    # Receptor sensors
    Sensor('${receptor}_observer', immutable=True),
    Sensor('${receptor}_activity'),
    Sensor('${receptor}_target'),
    Sensor('${receptor}_pos_request_scan_azim', sampling_strategy_and_params='period 0.4'),
    Sensor('${receptor}_pos_request_scan_elev', sampling_strategy_and_params='period 0.4'),
    Sensor('${receptor}_pos_actual_scan_azim', sampling_strategy_and_params='period 0.4'),
    Sensor('${receptor}_pos_actual_scan_elev', sampling_strategy_and_params='period 0.4'),
    Sensor('${receptor}_pos_adjust_pointm_azim'),
    Sensor('${receptor}_pos_adjust_pointm_elev'),
    Sensor('${receptor}_${digitiser}_noise_diode'),
    Sensor('${receptor}_ap_indexer_position'),
    Sensor('${receptor}_ap_point_error_tiltmeter_enabled'),
    Sensor('${receptor}_ap_tilt_corr_azim'),
    Sensor('${receptor}_ap_tilt_corr_elev'),
    Sensor('${receptor}_${receiver}_serial_number', immutable=True),
    Sensor('${receptor}_data_suspect'),
    Sensor('${receptor}_ap_version_list', immutable=True),
    # CBF proxy sensors
    Sensor('${cbf}_target'),
    Sensor('${cbf}_auto_delay_enabled'),
    Sensor('${cbf}_input_labels', immutable=True, convert=comma_split),
    Sensor('${cbf}_loaded_delay_correction', immutable=True),
    Sensor('${cbf}_delay_centre_frequency'),
    Sensor('${cbf}_delay_adjustments', convert=json.loads),
    Sensor('${cbf}_pos_request_offset_azim', sampling_strategy_and_params='period 0.4'),
    Sensor('${cbf}_pos_request_offset_elev', sampling_strategy_and_params='period 0.4'),
    Sensor('${cbf}_cmc_version_list', immutable=True),
    # SDP proxy sensors
    Sensor('${sdp}_spmc_version_list', immutable=True),
    # CBF sensors that are instrument specific
    Sensor('${instrument}_adc_sample_rate', immutable=True),
    Sensor('${instrument}_bandwidth', immutable=True),
    Sensor('${instrument}_n_inputs', immutable=True),
    Sensor('${instrument}_scale_factor_timestamp', immutable=True),
    Sensor('${instrument}_sync_time', immutable=True),
    # CBF sensors that are stream-specific
    Sensor('${stream.cbf.baseline_correlation_products}_bls_ordering',
           immutable=True, convert=np.safe_eval),
    Sensor('${stream.cbf.baseline_correlation_products}_int_time', immutable=True),
    Sensor('${stream.cbf.baseline_correlation_products}_n_accs', immutable=True),
    Sensor('${stream.cbf.baseline_correlation_products}_n_chans_per_substream', immutable=True),
    # Beamformer metadata are not immutable, because controlled by passband
    Sensor('${stream.cbf.tied_array_channelised_voltage}_n_chans'),
    Sensor('${stream.cbf.tied_array_channelised_voltage}_${inputn}_weight'),
    Sensor('${stream.cbf.tied_array_channelised_voltage}_n_chans_per_substream', immutable=True),
    Sensor('${stream.cbf.tied_array_channelised_voltage}_spectra_per_heap', immutable=True),
    Sensor('${stream.cbf.antenna_channelised_voltage}_n_samples_between_spectra',
           sdp_name='${stream.cbf.antenna_channelised_voltage}_ticks_between_spectra',
           immutable=True),
    Sensor('${stream.cbf.antenna_channelised_voltage}_n_chans', immutable=True),
    Sensor('${sub_stream.cbf.antenna_channelised_voltage}_centre_frequency',
           sdp_name='${sub_stream.cbf.antenna_channelised_voltage}_center_freq', immutable=True),
    # TODO: need to figure out how to deal with multi-stage FFT instruments
    Sensor('${stream.cbf.antenna_channelised_voltage}_${inputn}_fft0_shift',
           sdp_name='${stream.cbf.antenna_channelised_voltage}_fft_shift'),
    Sensor('${stream.cbf.antenna_channelised_voltage}_${inputn}_delay', convert=np.safe_eval),
    Sensor('${stream.cbf.antenna_channelised_voltage}_${inputn}_delay_ok'),
    Sensor('${stream.cbf.antenna_channelised_voltage}_${inputn}_eq', convert=np.safe_eval),
    # Subarray sensors
    Sensor('${subarray}_config_label', immutable=True),
    Sensor('${subarray}_band', immutable=True),
    Sensor('${subarray}_product', immutable=True),
    Sensor('${subarray}_sub_nr', immutable=True),
    Sensor('${subarray}_dump_rate', immutable=True),
    Sensor('${subarray}_pool_resources', immutable=True),
    Sensor('${sub_stream.cbf.antenna_channelised_voltage}_channel_mask', convert=convert_bitmask),
    Sensor('${sub_stream.cbf.tied_array_channelised_voltage}_precise_time_epoch_fraction'),
    Sensor('${sub_stream.cbf.tied_array_channelised_voltage}_precise_time_uncertainty'),
    Sensor('${subarray}_state'),
    # Misc other sensors
    Sensor('anc_air_pressure'),
    Sensor('anc_air_relative_humidity'),
    Sensor('anc_air_temperature'),
    Sensor('anc_wind_direction'),
    Sensor('anc_mean_wind_speed'),
    Sensor('anc_siggen_ku_frequency'),
    Sensor('mcp_dmc_version_list', immutable=True)
]


def parse_args():
    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--url', type=str, help='WebSocket URL to connect to')
    parser.add_argument('--namespace', type=str,
                        help='Namespace to create in katportal [sp_subarray_N]')
    parser.add_argument('-a', '--host', type=str, metavar='HOST', default='',
                        help='Hostname to bind for katcp interface')
    parser.add_argument('-p', '--port', type=int, metavar='N', default=2047,
                        help='Port to bind for katcp interface [%(default)s]')
    args = parser.parse_args()
    # Can't use required= on the parser, because telstate-provided arguments
    # are treated only as defaults rather than having set a value.
    if args.telstate is None:
        parser.error('argument --telstate is required')
    if args.url is None:
        parser.error('argument --url is required')
    return args


class DeviceServer(katcp.AsyncDeviceServer):
    VERSION_INFO = ("cam2telstate", 1, 0)
    BUILD_INFO = ("cam2telstate",) + tuple(katsdpcam2telstate.__version__.split('.', 1)) + ('',)

    def __init__(self, host, port):
        super(DeviceServer, self).__init__(host, port)

    def setup_sensors(self):
        pass


class Client(object):
    def __init__(self, args, logger):
        self._args = args
        self._telstate = args.telstate
        self._device_server = None
        self._logger = logger
        self._loop = tornado.ioloop.IOLoop.current()
        self._portal_client = None
        self._namespace = None    #: Set after connecting
        self._sensors = None  #: Dictionary from CAM name to sensor object
        self._instruments = set()  #: Set of instruments available in the current subarray
        self._streams_with_type = {}  #: Dictionary mapping stream names to stream types
        self._pool_resources = tornado.concurrent.Future()
        self._input_labels = tornado.concurrent.Future()
        self._band = tornado.concurrent.Future()
        self._sub_name = None     #: Set once connected
        self._cbf_name = None     #: Set once connected
        self._sdp_name = None     #: Set once connected
        self._receptors = []      #: Set once _pool_resources result is set
        self._waiting = 0         #: Number of sensors whose initial value is still outstanding

    def parse_streams(self):
        """Parse the stream information from telstate to populate the
        instruments and the stream_types dictionary."""
        sdp_config = self._telstate['sdp_config']
        for name, stream in six.iteritems(sdp_config.get('inputs', {})):
            if stream['type'].startswith('cbf.'):
                instrument_name = stream['instrument_dev_name']
                self._instruments.add(instrument_name)
                self._streams_with_type[name] = stream['type']

    def get_sensors(self):
        """Get list of sensors to be collected from CAM. This should be
        replaced to use kattelmod. It must only be called after
        :attr:`_pool_resources` and :attr:`_input_labels` are resolved.

        Returns
        -------
        sensors : list of `Sensor`
        """
        band = self._band.result()
        rx_name = 'rsc_rx{}'.format(band)
        dig_name = 'dig_{}_band'.format(band)
        # Build table of names for expanding sensor templates
        # Using a defaultdict removes the need to hardcode the list of stream
        # types.
        substitutions = collections.defaultdict(
            list,
            receptor=[(name, [name]) for name in self._receptors],
            receiver=[(rx_name, [rx_name])],
            digitiser=[(dig_name, [dig_name])],
            subarray=[(self._sub_name, ['sub'])],
            cbf=[(self._cbf_name, ['cbf'])],
            sdp=[(self._sdp_name, ['sdp'])]
        )
        for (number, name) in enumerate(self._input_labels.result()):
            substitutions['inputn'].append(('input{}'.format(number), [name]))
        for (cam_prefix, sdp_prefix) in substitutions['cbf']:
            # Add the per instrument specific sensors for every instrument we know about
            for instrument in self._instruments:
                cam_instrument = "{}_{}".format(cam_prefix, instrument)
                sdp_instruments = [instrument]
                substitutions['instrument'].append((cam_instrument, sdp_instruments))
            # For each stream we add type specific sensors
            for (full_stream_name, stream_type) in self._streams_with_type.iteritems():
                cam_stream = "{}_{}".format(cam_prefix, full_stream_name)
                cam_sub_stream = "{}_streams_{}".format(self._sub_name, full_stream_name)
                sdp_streams = [full_stream_name]
                substitutions['stream'].append((cam_stream, sdp_streams))
                substitutions['stream.' + stream_type].append((cam_stream, sdp_streams))
                substitutions['sub_stream'].append((cam_sub_stream, sdp_streams))
                substitutions['sub_stream.' + stream_type].append((cam_sub_stream, sdp_streams))

        sensors = []
        for template in SENSORS:
            expanded = template.expand(substitutions)
            if not expanded:
                self._logger.warning('No sensors expanded from template %s', template.cam_name)
            sensors.extend(expanded)
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
            self.namespace, sensor)
        if status != 1:
            raise RuntimeError("Expected 1 sensor for {}, found {}".format(sensor, status))
        status = yield self._portal_client.set_sampling_strategy(
            self.namespace, sensor, 'event')
        result = status[sensor]
        if result[u'success']:
            self._logger.info("Set sampling strategy on %s to event", sensor)
        else:
            raise RuntimeError("Failed to set sampling strategy on {}: {}".format(
                sensor, result[u'info']))

    @tornado.gen.coroutine
    def get_resources(self):
        """Query subarray_N_pool_resources to find out which cbf_M resource and
        which receptors are assigned to the subarray, followed by
        cbf_N_input_labels to find the input labels.
        """
        sensor = '{}_pool_resources'.format(self._sub_name)
        yield self.subscribe_one(sensor)
        # Wait until we get a callback with the value
        yield self._pool_resources
        yield self._portal_client.unsubscribe(self.namespace, sensor)
        # Now input labels
        sensor = '{}_input_labels'.format(self._cbf_name)
        yield self.subscribe_one(sensor)
        yield self._input_labels
        yield self._portal_client.unsubscribe(self.namespace, sensor)
        # Finally we need the band
        sensor = '{}_band'.format(self._sub_name)
        yield self.subscribe_one(sensor)
        yield self._band
        yield self._portal_client.unsubscribe(self.namespace, sensor)

    @tornado.gen.coroutine
    def start(self):
        try:
            self._logger.info('Connecting')
            self._portal_client = katportalclient.KATPortalClient(
                self._args.url, self.update_callback, io_loop=self._loop, logger=self._logger)
            yield self._portal_client.connect()
            if self._args.namespace is None:
                sub_nr = self._portal_client.sitemap['sub_nr']
                if sub_nr is None:
                    raise RuntimeError('subarray number not known')
                self.namespace = 'sp_subarray_{}'.format(sub_nr)
            else:
                self.namespace = self._args.namespace
            self._sub_name = yield self._portal_client.sensor_subarray_lookup('sub', '')
            self._cbf_name = yield self._portal_client.sensor_subarray_lookup('cbf', '')
            self._sdp_name = yield self._portal_client.sensor_subarray_lookup('sdp', '')
            self._logger.info('Initialising')
            # First find out which resources are allocated to the subarray
            yield self.get_resources()
            # Now we can tell which sensors to subscribe to
            self._sensors = {x.cam_name: x for x in self.get_sensors()}

            self._waiting = len(self._sensors)
            status = yield self._portal_client.subscribe(
                self.namespace, self._sensors.keys())
            self._logger.info("Subscribed to %d channels", status)
            for sensor in six.itervalues(self._sensors):
                status = yield self._portal_client.set_sampling_strategy(
                    self.namespace, sensor.cam_name, sensor.sampling_strategy_and_params)
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
            if self._device_server is not None:
                yield self._device_server.stop(timeout=None)
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
        sdp_names = sensor.sdp_name
        if not isinstance(sdp_names, list):
            sdp_names = [sdp_names]
        for name in sdp_names:
            try:
                self._telstate.add(name, value, timestamp, immutable=sensor.immutable)
                self._logger.debug('Updated %s to %s with timestamp %s',
                                   name, value, timestamp)
            except katsdptelstate.ImmutableKeyError:
                self._logger.error('Failed to set %s to %s with timestamp %s',
                                   name, value, timestamp, exc_info=True)

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
        if name == '{}_pool_resources'.format(self._sub_name):
            if not self._pool_resources.done() and status == 'nominal':
                resources = value.split(',')
                self._receptors = []
                for resource in resources:
                    if re.match(r'^m\d+$', resource):
                        self._receptors.append(resource)
                self._pool_resources.set_result(resources)
        elif self._cbf_name and name == '{}_input_labels'.format(self._cbf_name):
            if not self._input_labels.done() and status == 'nominal':
                labels = value.split(',')
                self._input_labels.set_result(labels)
        elif name == '{}_band'.format(self._sub_name):
            if not self._band.done() and status == 'nominal':
                self._band.set_result(value)

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
                    self._logger.info('Initial values for all sensors seen, starting katcp server')
                    self._device_server = DeviceServer(self._args.host, self._args.port)
                    self._device_server.start()

    def update_callback(self, msg):
        self._logger.debug("update_callback: %s", pprint.pformat(msg))
        if isinstance(msg, list):
            for item in msg:
                self.process_update(item)
        else:
            self.process_update(msg)

    @tornado.gen.coroutine
    def close(self):
        yield self._portal_client.unsubscribe(self.namespace)
        self._portal_client.disconnect()
        self._logger.info("disconnected")
        if self._device_server is not None:
            yield self._device_server.stop(timeout=None)
            self._device_server = None
        self._logger.info("device server shut down")
        self._loop.stop()


def main():
    katsdpservices.setup_logging()
    katsdpservices.setup_restart()
    args = parse_args()
    logger = logging.getLogger("katsdpcam2telstate")
    loop = tornado.ioloop.IOLoop.instance()
    client = Client(args, logger)
    client.parse_streams()
    loop.add_callback(client.start)
    loop.start()


if __name__ == '__main__':
    main()