#!/usr/bin/env python3

import argparse
import logging
import collections
import itertools
import pprint
import string
import signal
import re
import json
import asyncio
import uuid
from typing import List, Tuple, Dict, Set, Callable, Mapping, MutableMapping, Optional, Union, Any

import numpy as np
import katsdptelstate
import katsdpservices
import katportalclient
import aiokatcp

import katsdpcam2telstate


def comma_split(value: str) -> List[str]:
    return value.split(',')


def convert_bitmask(value: str) -> np.ndarray:
    """Converts a string of 1's and 0's to a numpy array of bools"""
    if not isinstance(value, str) or not re.match('^[01]*$', value):
        return None
    else:
        return np.array([c == '1' for c in value])


class Template(string.Template):
    """Template for a sensor name."""

    # Override this to allow dots in the name
    idpattern = '[A-Za-z0-9._]+'


class Sensor:
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
    ignore_missing : bool, optional
        If true, don't report an error if the sensor isn't present. This is
        used for sensors that only exist in RTS but not MeerKAT, or vice
        versa.
    """
    def __init__(self, cam_name: str, sdp_name: Union[None, str, List[str]] = None,
                 sampling_strategy_and_params: str = 'event',
                 immutable: bool = False,
                 convert: Optional[Callable[[Any], Any]] = None,
                 ignore_missing: bool = False) -> None:
        self.cam_name = cam_name
        self.sdp_name = sdp_name or cam_name
        self.sampling_strategy_and_params = sampling_strategy_and_params
        self.immutable = immutable
        self.convert = convert
        self.ignore_missing = ignore_missing
        self.waiting = True     #: Waiting for an initial value

    def expand(self, substitutions: Mapping[str, List[Tuple[str, List[str]]]]) -> List['Sensor']:
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
            for value in sdp_dict.values():
                assert isinstance(value, list)
            for sdp_values in itertools.product(*sdp_dict.values()):
                # Recombine this specific set of values with the keys
                sdp_single_dict = dict(zip(sdp_dict.keys(), sdp_values))
                sdp_names.append(substitute(self.sdp_name, sdp_single_dict))
            ans.append(Sensor(substitute(self.cam_name, cam_dict),
                              sdp_names,
                              self.sampling_strategy_and_params,
                              self.immutable,
                              self.convert,
                              self.ignore_missing))
        return ans


STREAM_TYPES = {
    'cbf.antenna_channelised_voltage',
    'cbf.baseline_correlation_products',
    'cbf.tied_array_channelised_voltage'
}
STATUS_VALID_VALUE = {'nominal', 'warn', 'error'}
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
    # CBF sensors that are instrument-specific
    Sensor('${instrument}_adc_sample_rate', immutable=True),
    Sensor('${instrument}_n_inputs', immutable=True),
    Sensor('${instrument}_scale_factor_timestamp', immutable=True),
    Sensor('${instrument}_sync_time', immutable=True),
    # CBF sensors that are stream-specific
    Sensor('${sub_stream.cbf.baseline_correlation_products}_bandwidth', immutable=True),
    Sensor('${stream.cbf.baseline_correlation_products}_bls_ordering',
           immutable=True, convert=np.safe_eval),
    Sensor('${stream.cbf.baseline_correlation_products}_int_time', immutable=True),
    Sensor('${stream.cbf.baseline_correlation_products}_n_accs', immutable=True),
    Sensor('${stream.cbf.baseline_correlation_products}_n_chans_per_substream', immutable=True),
    Sensor('${sub_stream.cbf.tied_array_channelised_voltage}_bandwidth', immutable=True),
    Sensor('${stream.cbf.tied_array_channelised_voltage}_source_indices',
           immutable=True, convert=np.safe_eval),
    Sensor('${stream.cbf.tied_array_channelised_voltage.inputn}_weight',
           ignore_missing=True),   # CBF-CAM ICD v5 - remove in future
    Sensor('${stream.cbf.tied_array_channelised_voltage}_weight',
           ignore_missing=True, convert=np.safe_eval),   # CBF-CAM ICD v6 (draft)
    Sensor('${stream.cbf.tied_array_channelised_voltage}_n_chans_per_substream', immutable=True),
    Sensor('${stream.cbf.tied_array_channelised_voltage}_spectra_per_heap', immutable=True),
    Sensor('${stream.cbf.antenna_channelised_voltage}_n_samples_between_spectra',
           sdp_name='${stream.cbf.antenna_channelised_voltage}_ticks_between_spectra',
           immutable=True),
    Sensor('${stream.cbf.antenna_channelised_voltage}_n_chans', immutable=True),
    Sensor('${sub_stream.cbf.antenna_channelised_voltage}_bandwidth', immutable=True),
    Sensor('${sub_stream.cbf.antenna_channelised_voltage}_centre_frequency',
           sdp_name='${sub_stream.cbf.antenna_channelised_voltage}_center_freq', immutable=True),
    # TODO: need to figure out how to deal with multi-stage FFT instruments
    Sensor('${stream.cbf.antenna_channelised_voltage}_${inputn}_fft0_shift',
           sdp_name='${stream.cbf.antenna_channelised_voltage}_fft_shift'),
    Sensor('${stream.cbf.antenna_channelised_voltage}_${inputn}_delay', convert=np.safe_eval),
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
    Sensor('anc_siggen_ku_frequency', ignore_missing=True),
    Sensor('anc_tfr_ktt_gnss'),
    Sensor('mcp_dmc_version_list', immutable=True)
]


def parse_args() -> argparse.Namespace:
    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--url', type=str, help='WebSocket URL to connect to')
    parser.add_argument('--namespace', type=str,
                        help='Namespace to create in katportal [UUID]')
    parser.add_argument('-a', '--host', type=str, metavar='HOST', default='',
                        help='Hostname to bind for katcp interface')
    parser.add_argument('-p', '--port', type=int, metavar='N', default=2047,
                        help='Port to bind for katcp interface [%(default)s]')
    parser.add_aiomonitor_arguments()
    args = parser.parse_args()
    # Can't use required= on the parser, because telstate-provided arguments
    # are treated only as defaults rather than having set a value.
    if args.telstate is None:
        parser.error('argument --telstate is required')
    if args.url is None:
        parser.error('argument --url is required')
    return args


class DeviceServer(aiokatcp.DeviceServer):
    VERSION = "cam2telstate-1.0"
    BUILD_STATE = "cam2telstate-" + katsdpcam2telstate.__version__


class Client:
    _device_server: Optional[DeviceServer]
    _portal_client: Optional[katportalclient.KATPortalClient]
    _namespace: Optional[str]
    _sensors: Optional[Dict[str, Sensor]]
    _instruments: Set[str]
    _streams_with_type: Dict[str, str]
    _sub_name: Optional[str]
    _sdp_name: Optional[str]

    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self._args = args
        self._telstate = args.telstate
        self._device_server = None
        self._logger = logger
        self._portal_client = None
        self._namespace = None     #: Set after connecting
        self._sensors = None       #: Dictionary from CAM name to sensor object
        self._instruments = set()  #: Set of instruments available in the current subarray
        self._streams_with_type = {}  #: Dictionary mapping stream names to stream types
        self._sub_name = None      #: Set once connected
        self._cbf_name = None      #: Set once connected
        self._sdp_name = None      #: Set once connected
        self._waiting = 0          #: Number of sensors whose initial value is still outstanding
        self._stopped = asyncio.Event()    #: Set after shutdown

    def parse_streams(self) -> None:
        """Parse the stream information from telstate to populate the
        instruments and the stream_types dictionary."""
        sdp_config = self._telstate['sdp_config']
        for name, stream in sdp_config.get('inputs', {}).items():
            if stream['type'].startswith('cbf.'):
                instrument_name = stream['instrument_dev_name']
                self._instruments.add(instrument_name)
                self._streams_with_type[name] = stream['type']

    async def get_sensor_value(self, sensor: str) -> Any:
        """Get the current value of a sensor.

        This is only used for special sensors needed to bootstrap subscriptions.
        Multiple calls can proceed in parallel, provided that they do not
        duplicate any names.
        """
        assert self._portal_client is not None
        data = await self._portal_client.sensor_value(sensor)
        if data.status not in STATUS_VALID_VALUE:
            self._logger.warning('Status %s for sensor %s is invalid, but using the value anyway',
                                 data.status, sensor)
        return data.value

    async def get_receptors(self) -> List[str]:
        """Get the list of receptors"""
        value = await self.get_sensor_value('{}_pool_resources'.format(self._sub_name))
        resources = value.split(',')
        receptors = []
        for resource in resources:
            if re.match(r'^m\d+$', resource):
                receptors.append(resource)
        return receptors

    async def get_sensors(self) -> List[Sensor]:
        """Get list of sensors to be collected from CAM.

        Returns
        -------
        sensors : list of `Sensor`
        """
        # Tell mypy that these must have been initialised
        assert self._sub_name is not None
        assert self._cbf_name is not None
        assert self._sdp_name is not None

        receptors = await self.get_receptors()
        input_labels = await self.get_sensor_value('{}_input_labels'.format(self._cbf_name))
        input_labels = input_labels.split(',')
        band = await self.get_sensor_value('{}_band'.format(self._sub_name))

        rx_name = 'rsc_rx{}'.format(band)
        dig_name = 'dig_{}_band'.format(band)
        # Build table of names for expanding sensor templates
        substitutions: Dict[str, List[Tuple[str, List[str]]]] = {
            'receptor': [(name, [name]) for name in receptors],
            'receiver': [(rx_name, [rx_name])],
            'digitiser': [(dig_name, [dig_name])],
            'subarray': [(self._sub_name, ['sub'])],
            'cbf': [(self._cbf_name, ['cbf'])],
            'sdp': [(self._sdp_name, ['sdp'])],
            'inputn': [],
            'instrument': [],
            'stream': [],
            'sub_stream': [],
            'stream.cbf.tied_array_channelised_voltage.inputn': []
        }
        for stream_type in STREAM_TYPES:
            substitutions['stream.' + stream_type] = []
            substitutions['sub_stream.' + stream_type] = []

        cam_prefix = self._cbf_name
        for (number, name) in enumerate(input_labels):
            substitutions['inputn'].append(('input{}'.format(number), [name]))
        # Add the per instrument specific sensors for every instrument we know about
        for instrument in self._instruments:
            cam_instrument = "{}_{}".format(cam_prefix, instrument)
            sdp_instruments = [instrument]
            substitutions['instrument'].append((cam_instrument, sdp_instruments))
        # For each stream we add type specific sensors
        for (full_stream_name, stream_type) in self._streams_with_type.items():
            if stream_type not in STREAM_TYPES:
                self._logger.warning('Skipping stream %s with unknown type %s',
                                     full_stream_name, stream_type)
            cam_stream = "{}_{}".format(cam_prefix, full_stream_name)
            cam_sub_stream = "{}_streams_{}".format(self._sub_name, full_stream_name)
            sdp_streams = [full_stream_name]
            substitutions['stream'].append((cam_stream, sdp_streams))
            substitutions['stream.' + stream_type].append((cam_stream, sdp_streams))
            substitutions['sub_stream'].append((cam_sub_stream, sdp_streams))
            substitutions['sub_stream.' + stream_type].append((cam_sub_stream, sdp_streams))
            # tied-array-channelised-voltage per-input sensors are special:
            # only a subset of the inputs are used and only the corresponding
            # sensors exist.
            if stream_type == 'cbf.tied_array_channelised_voltage':
                source_indices = await self.get_sensor_value(cam_stream + '_source_indices')
                source_indices = np.safe_eval(source_indices)
                sublist = substitutions['stream.{}.inputn'.format(stream_type)]
                for index in source_indices:
                    if 0 <= index < len(input_labels):
                        name = '{}_{}'.format(full_stream_name, input_labels[index])
                        sublist.append(('{}_input{}'.format(cam_stream, index), [name]))
                    else:
                        self._logger.warning('Out of range source index %d on %s',
                                             index, full_stream_name)

        sensors: List[Sensor] = []
        for template in SENSORS:
            expanded = template.expand(substitutions)
            sensors.extend(expanded)
        return sensors

    async def start(self) -> None:
        try:
            self._logger.info('Connecting')
            self._portal_client = katportalclient.KATPortalClient(
                self._args.url, self.update_callback, logger=self._logger)
            await self._portal_client.connect()
            if self._args.namespace is None:
                self.namespace = uuid.uuid4().hex
                self._logger.info('Using %s as katportalclient namespace', self.namespace)
            else:
                self.namespace = self._args.namespace
            self._sub_name = await self._portal_client.sensor_subarray_lookup('sub', '')
            self._cbf_name = await self._portal_client.sensor_subarray_lookup('cbf', '')
            self._sdp_name = await self._portal_client.sensor_subarray_lookup('sdp', '')
            self._logger.info('Initialising')
            # Now we can tell which sensors to subscribe to
            sensors = await self.get_sensors()
            self._sensors = {x.cam_name: x for x in sensors}

            self._waiting = len(self._sensors)
            status = await self._portal_client.subscribe(
                self.namespace, list(self._sensors.keys()))
            self._logger.info("Subscribed to %d channels", status)

            # Group sensors by strategy to bulk-set sampling strategies
            by_strategy: MutableMapping[str, List[Sensor]] = collections.defaultdict(list)
            for sensor in sensors:
                by_strategy[sensor.sampling_strategy_and_params].append(sensor)
            for (strategy, strategy_sensors) in by_strategy.items():
                regex = '^(?:' + '|'.join(re.escape(sensor.cam_name)
                                          for sensor in strategy_sensors) + ')$'
                status = await self._portal_client.set_sampling_strategies(
                    self.namespace, regex, strategy)
                for (sensor_name, result) in sorted(status.items()):
                    if result['success']:
                        self._logger.info("Set sampling strategy on %s to %s",
                                          sensor_name, strategy)
                    else:
                        self._logger.error("Failed to set sampling strategy on %s: %s",
                                           sensor_name, result['info'])
                        # Not going to get any values, so don't wait for it
                        self._waiting -= 1
                        self._sensors[sensor_name].waiting = False
                for sensor in strategy_sensors:
                    if sensor.cam_name not in status:
                        if not sensor.ignore_missing:
                            self._logger.error("Sensor %s not found", sensor.cam_name)
                        self._waiting -= 1
                        sensor.waiting = False

            loop = asyncio.get_event_loop()
            for signal_number in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(
                    signal_number, lambda: loop.create_task(self.close()))
                self._logger.debug('Set signal handler for %s', signal_number)
        except Exception as e:
            if isinstance(e, katportalclient.SensorLookupError):
                self._logger.error("Sensor name lookup failed. Please check the state of "
                                   "the CAM subarray to make sure it is not in error. (%s)", e)
            else:
                self._logger.error("Exception during startup", exc_info=True)
            if self._device_server is not None:
                await self._device_server.stop()
            self._stopped.set()
        else:
            self._logger.info("Startup complete")

    def sensor_update(self, sensor: Sensor, value: Any, status: str, timestamp: float) -> None:
        name = sensor.cam_name
        if status not in STATUS_VALID_VALUE:
            self._logger.info("Sensor {} received update '{}' with status '{}' (ignored)"
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

    def process_update(self, item: Mapping[str, Any]) -> None:
        self._logger.debug("Received update %s", pprint.pformat(item))
        data = item['msg_data']
        if data is None:
            return
        name = data['name']
        timestamp = data['timestamp']
        status = data['status']
        value = data['value']

        if self._sensors is None:   # We are still bootstrapping
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
                    asyncio.get_event_loop().create_task(self._device_server.start())

    def update_callback(self, msg: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        self._logger.debug("update_callback: %s", pprint.pformat(msg))
        if isinstance(msg, list):
            for item in msg:
                self.process_update(item)
        else:
            self.process_update(msg)

    async def close(self) -> None:
        if self._portal_client is not None:
            await self._portal_client.unsubscribe(self.namespace)
            self._portal_client.disconnect()
            self._logger.info("disconnected")
        if self._device_server is not None:
            await self._device_server.stop()
            self._device_server = None
        self._logger.info("device server shut down")
        self._stopped.set()

    async def join(self) -> None:
        await self._stopped.wait()

    async def run(self) -> None:
        await self.start()
        await self.join()


def main() -> None:
    katsdpservices.setup_logging()
    katsdpservices.setup_restart()
    args = parse_args()
    logger = logging.getLogger("katsdpcam2telstate")

    loop = asyncio.get_event_loop()
    client = Client(args, logger)
    client.parse_streams()
    with katsdpservices.start_aiomonitor(loop, args, locals()):
        loop.run_until_complete(client.run())


if __name__ == '__main__':
    main()
