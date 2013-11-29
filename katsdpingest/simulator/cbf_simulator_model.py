from __future__ import with_statement

import os
import time
import sys
import logging
import threading
import random
import blinker

import numpy as np

from math import floor
from collections import namedtuple

from katcore.dev_base import ThreadedModel
from katcp import Sensor

from .cbf_roach_models import XEngines, FEngines
from .cbf_spead_model import DBE7SpeadData
from .model_correlator_conf import ModelCorrConf



activitylogger = logging.getLogger('activity')
log_name = 'kat.k7simulator'
logger = logging.getLogger(log_name)


CaptureDestination = namedtuple('CaptureDestination', (
    'name', 'meta_ip', 'meta_port', 'data_ip', 'data_port'))

class AddLockSettersGetters(object):
    class add(object):
        def __init__(self, initial_value=None):
            self.initial_value = initial_value

    def __init__(self, lock_name):
        self.lock_name = lock_name

    def make_getter(self, name):
        def getter(self):
            return getattr(self, name)

        return getter

    def make_setter(self, name):
        lock_name = self.lock_name
        def setter(self, val):
            with getattr(self, lock_name):
                setattr(self, name, val)

        return setter

    def __call__(self, cls):
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, self.add):
                assert (name.startswith('_'))
                setattr(cls, name, attr.initial_value)
                setattr(cls, 'set'+name, self.make_setter(name))
                setattr(cls, 'get'+name, self.make_getter(name))
        return cls

class TestInterfaceModel(ThreadedModel):
    def __init__(self, *args, **kwargs):
        self.test_sensors = {} # Sensors meant for the test device interface
        super(TestInterfaceModel, self).__init__(*args, **kwargs)

    def add_test_sensor(self, sensor):
        """Add a test device sensor object"""
        sname = sensor.name
        if sname  in self.test_sensors:
            raise ValueError(
                'Sensor with name %s already added to device' % sname)
        self.test_sensors[sname] = sensor

    def get_test_sensor(self, sensorname):
        """Retrieve a sensor object by its name"""
        sensor = self.test_sensors.get(sensorname, None)
        if not sensor:
            raise ValueError("Unknown sensor '%s'." % sensorname)
        return sensor

    def get_test_sensors(self):
        """Get list of all sensors"""
        return self.test_sensors.values()


@AddLockSettersGetters('_data_lock')
class K7CorrelatorModel(TestInterfaceModel):
    # In standalone mode a default antenna mapping is set up. The
    # standalone variable needs to be set before start() is called
    standalone = False
    valid_modes = ('ready', 'c16n400M1k', 'c16n400M8k', 'c16n13M4k',
                   'c16n25M4k', 'c16n2M4k', 'c16n3M8k', 'c16n7M4k',
                   'bc16n400M1k', 'c8n856M32k',)

    def __init__(self, config_dir, *names, **kwargs):
        super(K7CorrelatorModel, self).__init__(*names, **kwargs)
        self.config_dir = config_dir

        # A signal indicating that a sensor has been added. The signal
        # is sent with self as the sender, and the keyword arguments:
        #
        # sensor: katcp Sensor object -- the new sensor
        self.sensor_added = blinker.Signal()

        # A signal indicating that a sensor has been changed. The signal
        # is sent with self as the sender, and the keyword arguments:
        #
        # sensor: katcp Sensor object -- the new sensor
        #
        # It is assumed that the subscriber will arrange to remove
        # listeners or sampling strategies from the sensor as required.
        self.sensor_changed = blinker.Signal()

        # A signal indicating that a sensor has been removed. The signal
        # is sent with self as the sender, and the keyword arguments:
        #
        # sensor: katcp Sensor object -- The new sensor.
        # old_sensor: katcp Sensor object -- The old sensor
        #
        # It is assumed that the subscriber will arrange to have
        # listeners or sampling strategies or whatever transferred
        # from the old to the new sensor object should that be
        # required.
        self.sensor_removed = blinker.Signal()

        # Lock that should be held whenever something that could
        # effect data generation and associated SPEAD meta data is
        # modified. For instance, while changing the dump period.
        self._data_lock = threading.Lock()
        self._init_values()
        self._init_sensors()
        self._init_test_sensors()
        self._thread_paused = True

    def _update_roaches(self):
        old_sensors = []
        try: old_sensors.extend(self._roach_x_engines.get_sensors())
        except AttributeError: pass
        try: old_sensors.extend(self._roach_f_engines.get_sensors())
        except AttributeError: pass
        old_sensor_names = set(s.name for s in old_sensors)

        self._roach_x_engines = XEngines(self.config['servers_x'])
        self.roach_x_engines = self._roach_x_engines.roach_names
        self._roach_f_engines = FEngines(self.config['servers_f'])
        self.roach_f_engines = self._roach_f_engines.roach_names

        new_sensors = (self._roach_x_engines.get_sensors() +
                       self._roach_f_engines.get_sensors())
        new_sensor_names = set(s.name for s in new_sensors)
        remove = old_sensor_names - new_sensor_names
        add = new_sensor_names - old_sensor_names
        change = new_sensor_names & old_sensor_names
        for s in new_sensors:
            if s.name in add:
                self.add_sensor(s)
            elif s.name in change:
                self.change_sensor(s)
            else:
                raise RuntimeError('An impossible error has occured')
        for s in old_sensors:
            if s.name in remove:
                self.del_sensor(s)


    def add_sensor(self, sensor, signal=True):
        super(K7CorrelatorModel, self).add_sensor(sensor)
        if signal: self.sensor_added.send(self, sensor=sensor)

    def del_sensor(self, sensor, signal=True):
        del(self.sensors[sensor.name])
        if signal: self.sensor_removed.send(self, sensor=sensor)

    def change_sensor(self, sensor, signal=True):
        old_sensor = self.get_sensor(sensor.name)
        self.del_sensor(old_sensor, signal=False)
        self.add_sensor(sensor, signal=False)
        if signal:
            self.sensor_changed.send(self, sensor=sensor, old_sensor=old_sensor)

    def _init_sensors(self):
        self.add_sensor(Sensor(Sensor.BOOLEAN, 'corr.lru.available',
                               'line replacement unit operational', ''))
        self.get_sensor('corr.lru.available').set_value(True, Sensor.NOMINAL)
        self.add_sensor(Sensor(
                Sensor.INTEGER, "sync_time", "Last sync time in epoch seconds.","seconds",
                default=0, params=[0,2**32]))
        self.add_sensor(Sensor(Sensor.INTEGER, "tone_freq",
                               "The frequency of the injected tone in Hz.","",
                               default=0, params=[0,2**32]))
        self.get_sensor('tone_freq').set_value(self.tone_freq, Sensor.NOMINAL)

        dip_sens = Sensor(
            Sensor.STRING, "destination_ip",
            "The current destination address for data and metadata.","","")
        self.add_sensor(dip_sens)

        msens = Sensor(Sensor.STRING, 'mode', 'Current DBE operating mode', '')
        msens.set_value('ready', Sensor.NOMINAL, time.time())
        self.add_sensor(msens)

        self.add_sensor(Sensor(Sensor.BOOLEAN, "ntp.synchronised", "clock good", ""))
        self.get_sensor('ntp.synchronised').set_value(True, Sensor.NOMINAL)

        f0_sens = Sensor(Sensor.INTEGER, 'centerfrequency',
                         'current selected center frequency', 'Hz',
                         [0, 400*1000000])
        f0_sens.set_value(0, Sensor.UNKNOWN)
        self.add_sensor(f0_sens)
        bw_sens = Sensor(Sensor.INTEGER, 'bandwidth',
                         'The bandwidth currently available', 'Hz',
                         [0, 400*1000000])
        bw_sens.set_value(0, Sensor.UNKNOWN)
        self.add_sensor(bw_sens)

        chan_sens = Sensor(Sensor.INTEGER, 'channels',
                           'the number of frequency channels that the '
                           'correlator provides when in the current mode.',
                           'Hz', [0, 10000])
        chan_sens.set_value(0, Sensor.UNKNOWN)
        self.add_sensor(chan_sens)
        # Make per-beam beamformer sensors. We get values from the config file
        # bc16n400M1k which, as of 2013-03-25, is the only beamformer mode. Also
        # assume that the values are valid for any other beam former modes.
        bf_conf = ModelCorrConf(os.path.join(self.config_dir, 'bc16n400M1k'))
        adc_bw = bf_conf['adc_clk'] / 2
        self.NO_BEAMS = bf_conf['bf_n_beams']
        for i in range(self.NO_BEAMS):
            bw = bf_conf['bf_bandwidth_beam%d' % i]
            f0 = bf_conf['bf_centre_frequency_beam%d' % i]
            bf_bw = Sensor.integer(
                'bf%d.bandwidth' % i, 'selected bandwidth of beam',
                'Hz', [0, adc_bw], default=bw)
            bf_f0 = Sensor.integer(
                'bf%d.centerfrequency' % i, 'selected center frequency of beam',
                'Hz', [0, adc_bw], default=f0)
            self.add_sensor(bf_bw)
            self.add_sensor(bf_f0)

        #no_beams


    def _init_test_sensors(self):
        self.add_test_sensor(Sensor(
            Sensor.BOOLEAN, 'hang-requests',
            'Requests on the real device other than watchdog hang if true',
            '', default=False))
        self.add_test_sensor(Sensor(
            Sensor.FLOAT, 'mode-change-delay',
            'Time that correlator should hang while wating for the mode to change',
            'seconds', [0, 1000], default=10))

    def _init_values(self):
        self.adc_value = 0
        self._dump_period = 1.0
        self.sample_rate = 800e6
        self.tone_freq = 302e6
        self.noise_diode = 0
        self.nd_duty_cycle = 0
        self._nd_cycles = 0
        self.set_target_az(0)
        self.set_target_el(0)
        self.set_target_flux(0)
        self.set_test_az(0)
        self.set_test_el(0)
        self.nd = 0
        self.multiplier = 100

    _target_az = AddLockSettersGetters.add()
    _target_el = AddLockSettersGetters.add()
    _target_flux = AddLockSettersGetters.add()
    _test_az = AddLockSettersGetters.add()
    _test_el = AddLockSettersGetters.add()

    def start(self, *names, **kwargs):
        self.set_mode('c8n856M32k', mode_delay=0)
        #self.spead_issue()
        super(K7CorrelatorModel, self).start(*names, **kwargs)

    def start_data_stream(self):
        if self.get_sensor('mode').value() == 'ready':
            raise RuntimeError("Cannot dump data in 'ready' mode")
        else:
            print "Unpaused in sds"
            self._thread_paused = False

    def get_adc_snap_shot(self, when, level, inputs):
        """Return fake ADC snapshot values for inputs

        Parameters
        ==========

        when -- 'pps' or 'now', see doc K0000-2006V1-04 for intended effect
        level -- ignored, see doc K0000-2006V1-04 for intended effect
        inputs -- List of DBE inputs to provide ADC snapshot for, e.g.
          ('0x', '5y')

        Return Values
        =============

        Generator of tuples

        (input, timestamp, values)

        for each input requested, with

        input -- name of the input (as in input parameter `inputs`)
        input -- name of the input (as in input parameter `inputs`
        timestamp -- timestamp in ms since Unix epoch for snapshot values
        values -- list of adc snapshop values
        """
        for inp in inputs:
            if not self._roach_f_engines.is_channel(inp):
                raise ValueError( 'Unknown input %s. Valid inputs are %s.' % (
                    inp, ','.join(self._roach_f_engines.get_channels())) )
        no_samples = 16
        if when == 'pps':
            time.sleep(random.random()) # Sleep for up to 1s
        for inp in inputs:
            yield (inp, int(time.time()*1000),
                   [random.randint(-127, 127) for i in range(no_samples)])

    def k7_frequency_select(self, center_frequency):
        """Set the nbc centre frequency to the closest nbc channel

        Returns the actual centre frequency used. Will fail if the correlator is
        not in narrow-band (nbc) mode
        Returns the actual centre frequency used
        """
        mode_type = self.config.get('mode')
        if mode_type != 'nbc':
            # Apparently this is the error returned by the real DBE :)
            raise RuntimeError('unavailable')
        min_f0 = 0
        max_f0 = self.config['adc_clk'] / 2   # Nyquist rate from ADC clock
        wideband_bw = max_f0 - min_f0

        if not (center_frequency >= min_f0 and center_frequency <= max_f0):
            raise RuntimeError('center_frequency must be between %f and %f Hz'
                               % (min_f0, max_f0))


        # Spacing of coarse channels that f0 is based on
        d_f0 = wideband_bw / self.config['coarse_chans']
        # Quantise requested center_frequency to nearest bin
        f0 = int(round(center_frequency/d_f0)*d_f0)
        self.get_sensor('centerfrequency').set_value(
            f0, Sensor.NOMINAL, time.time())
        return f0

    def set_k7_beam_passband(self, beam, bandwidth, centerfrequency):
        if not self.has_beamformer:
            raise ValueError('k7 beam passband can only be set in beamformer mode')
        if not beam.startswith('bf'):
            raise ValueError('Can only be used with beamformer (bf*) beams, '
                             'not {0}'.format(beam))
        try:
            bw_sens = self.get_sensor(beam+'.bandwidth')
            f0_sens = self.get_sensor(beam+'.centerfrequency')
        except AttributeError:
            raise ValueError('Unknown beam {0}'.format(beam))
        min_f0 = 0
        max_f0 = self.config['adc_clk'] / 2   # Nyquist rate from ADC clock
        max_bandwidth = max_f0 - min_f0
        if centerfrequency > max_f0 or centerfrequency < min_f0:
            raise ValueError('{0} <= centerfrequency <={1} must be true'.format(
                min_f0, max_f0))
        if bandwidth > max_bandwidth or bandwidth < 0:
            raise ValueError('{0} <= bandwidth <={1} must be true'.format(
                0, max_bandwidth))

        bw_sens.set_value(bandwidth)
        f0_sens.set_value(centerfrequency)

    def set_k7_beam_weights(self, beam, input, channel_weights):
        """
        Set the k7 beam former weights.

        Arguments
        ---------

        beam -- beam name, e.g. bf0
        input -- correlator input channel, e.g. 1x or (if mapped) ant1H
        channel_weights -- Array of channel weights, one per frequency channel

        Behaviour
        ---------

        Currently a dummy implementation that checks if the right number of
        channel_weights were passed, and hangs around for 5.5s
        """
        no_chans = self.get_sensor('channels').value()
        if len(channel_weights) != no_chans:
            raise ValueError('Channel weights should be an array of length %d, '
                             'not %d.' % (no_chans, len(channel_weights)))
        time.sleep(5.5)

    def run(self):
        while not self._stopEvent.isSet():
            st = time.time()
            if not self._thread_paused:
                st = time.time()
                with self._data_lock:
                    self.data = self.generate_data()
                self.send_dump()
                tt = time.time() - st
                status = ("Sending correlator dump at %.3f (%.2f) "
                          "(dump period: %f s, multiplier: %i, noise_diode: %s)\n" %
                          (time.time(), tt, self._dump_period, self.multiplier,
                           (self.nd > 0 and 'On' or 'Off')))
                sys.stdout.write(status)
                sys.stdout.flush()
                time.sleep(max(self._dump_period - (time.time() - st), 0.))
        self.send_stop()
        print "Correlator tx halted."

    def gaussian(self,x,y):
         # for now a gaussian of height 1 and width 1
         # beam width is 0.8816 degrees (sigma of 0.374) at 1.53 GHZ
         #equates to coefficient of
        return np.exp(-(0.5/(0.374*0.374)) * (x*x + y*y))

    def generate_data(self):
        source_value = self.get_target_flux() * self.gaussian(
            self.get_target_az() - self.get_test_az(),
            self.get_target_el() - self.get_test_el())
         # generate a flux contribution from the synthetic source (if any)
        tsys_elev_value = 25 - np.log(self.get_test_el() + 1) * 5
        self.nd = 0
        if self.noise_diode > 0:
            self.noise_diode -= 1
            self.nd = 100

        if self.nd_duty_cycle > 0:
            self._nd_cycles += 1
            if self._nd_cycles >= (1.0/self.nd_duty_cycle):
                self.nd = 100
                self._nd_cycles = 0

        self.multiplier = 50 + source_value + tsys_elev_value
        n_chans = self.config['n_chans']
        samples_per_dump = n_chans * 8
         # not related to actual value. just for calculation purposes
        n = np.arange(samples_per_dump)
        x = np.cos(2 * np.pi * self.tone_freq / self.sample_rate * n)
        data = np.fft.fft(x, n_chans)[:n_chans]
        data = (data.view(np.float64)*self.multiplier).astype(np.int32).reshape((n_chans,2))
        data = np.tile(data, self.config['n_bls'])
        data = data.reshape((n_chans,self.config['n_bls'],2), order='C')
        for ib in range (self.config['n_bls']):#for different baselines
            (a1,a2)= self._spead_model.bls_ordering[ib]
            if a1[:-1] == a2[:-1]:
                auto_d=np.abs(data[:,ib,:]+((ib*32131+48272)%1432)/1432.0*20.0 +
                              np.random.randn(n_chans*2).reshape(
                        [n_chans,2])*10.0) + 200
                auto_d[:,1] = 0
                data[:,ib,:]=auto_d
            else:
                data[:,ib,:]=data[:,ib,:]+(((ib*32131+48272)%1432)/1432.0*20.0 +
                np.random.randn(n_chans*2).reshape(
                    [n_chans,2])*10.0)
        data = data.astype(np.int32) + self.nd
        return data

    def get_crosspol_order(self):
        "Returns the order of the cross-pol terms out the X engines"
        pol1=self.config['rev_pol_map'][0]
        pol2=self.config['rev_pol_map'][1]
        return (pol1+pol1,pol2+pol2,pol1+pol2,pol2+pol1)

    def spead_issue(self):
        with self._data_lock:
            print "Issuing SPEAD meta data to %s\n" % self.config['rx_meta_ip_str']
            self._spead_model.spead_issue()

    def send_dump(self):
        """Send a single correlator dump..."""
        with self._data_lock:
            data_ig = self._spead_model.data_ig
            tx = self._spead_model.tx
            data_ig['timestamp'] = int(
                (time.time() - self._spead_model.sync_time) *
                self.config['spead_timestamp_scale_factor'])
            data_ig['xeng_raw'] = self.data
            tx.send_heap(data_ig.get_heap())

    def send_stop(self):
        self._spead_model.send_stop()

    def set_dump_period(self, dump_period):
        with self._data_lock:
            self._dump_period = dump_period
            self._spead_model.set_acc_time(dump_period)

    def get_dump_period(self):
        return self._dump_period

    def set_mode(self, mode, progress_callback=None, mode_delay=None):
        """Set DBE mode

        Parameters
        ==========

        mode -- the desired mode; any mode listed in attribute valid_modes may be used
        progress_callback -- optional callback that is called with a string
            describing the 'progress' of mode changing.
        mode_delay -- optional number of seconds to delay while doing
            mode change. Defaults to the value of the test_sensor
            'mode-change-delay'

        """
        if mode_delay is None:
            mode_delay = self.get_test_sensor('mode-change-delay').value()

        if progress_callback is None: progress_callback = lambda x: x
        if mode not in self.valid_modes:
            raise ValueError('Bad mode %s. Mode should be one of %s.' % (
                mode, ','.join( "'%s'" %m for m in self.valid_modes)))
        if mode == self.get_sensor('mode').value():
            progress_callback('Correlator already in mode %s' % mode)
            return

        self._thread_paused = True
        logger.info('Sleeping %f s for cbf mode change' % mode_delay)

        for i in range(int(floor(mode_delay))):
            time.sleep(1)
            smsg = 'Doing some mode changing stuff on DBE %d' % (i+1)
            progress_callback(smsg)
        if mode_delay % 1 > 0.000001:
            time.sleep(mode_delay % 1)
            smsg = 'Doing some mode changing stuff on DBE.'
            progress_callback(smsg)


        if mode == 'ready':
            # Fake ready mode using the c8n856M32k config as basis
            config_file = os.path.join(self.config_dir, 'c8n856M32k')
        else:
            config_file = os.path.join(self.config_dir, mode)
        self.config = ModelCorrConf(config_file)

        # can be wbc for wideband modes (i.e. uses the whole ADC range) or nbc
        # for narrow band modes that select sub-windows of the ADC range
        try: mode_type = self.config['mode']
        except KeyError: mode_type = None
        f0_sens = self.get_sensor('centerfrequency')
        max_bw = self.config['adc_clk'] / 2   # Nyquist rate from ADC clock

        if  mode_type == 'wbc':
            # For wbc modes the centre frequency cannot be selected and is 'hard
            # coded' to half the available bandwidth
            f0_sens.set_value(max_bw / 2, Sensor.NOMINAL)
            self.get_sensor('bandwidth').set_value(max_bw, Sensor.NOMINAL)
        elif mode_type == 'nbc':
            f0_sens.set_value(f0_sens.value(), Sensor.NOMINAL)
            self.get_sensor('bandwidth').set_value(
                max_bw/self.config['coarse_chans'], Sensor.NOMINAL)

        if self.has_beamformer:
            # Seems we have a beamformer mode
            assert self.NO_BEAMS == self.config['bf_n_beams']
            for b in range(self.NO_BEAMS):
                self.get_sensor('bf%d.bandwidth' % b).set_value(
                    self.config['bf_bandwidth_beam%d' % b])
                self.get_sensor('bf%d.centerfrequency' % b).set_value(
                    self.config['bf_centre_frequency_beam%d' % b])
        self._spead_model = DBE7SpeadData(self.config)
        self._spead_model.set_acc_time(self.get_dump_period())
        self._update_roaches()

        self.get_sensor('sync_time').set_value(self._spead_model.sync_time, Sensor.NOMINAL)
        self.get_sensor('destination_ip').set_value(self.config['rx_meta_ip_str'])
        self.get_sensor('mode').set_value(mode, Sensor.NOMINAL, time.time())
        self.get_sensor('channels').set_value(self.config['n_chans'])
        #if mode != 'ready':
        #    print "unpaused in mode ready"
        #    self._thread_paused = False

    def pause_data(self):
        self._thread_paused = True

    def unpause_data(self):
        print "unpaused"
        self._thread_paused = False

    def set_capture_destination(
            self, stream, meta_ip, meta_port, data_ip=None, data_port=None):
        """
        Set data capture destination for correlator

        If data_ip or data_port are not specified, they default to meta_ip/port

        The k7 stream does not (currently?) support different meta and data ports
        """
        meta_port = int(meta_port)
        data_ip = data_ip or meta_ip
        data_port = data_port or meta_port
        data_port = int(data_port)
        streams = set(s[0] for s in self.capture_list)
        if stream not in streams:
            raise ValueError('Unknown correlator stream {0}'.format(stream))

        c = self.config
        paused = self._thread_paused

        try:
            self._thread_paused = True
            if stream == 'k7':
                if data_port != meta_port:
                    raise ValueError('k7 stream must have meta_port = data_port')
                c['rx_udp_ip_str'] = data_ip
                c['rx_udp_port'] = data_port
                c['rx_meta_ip_str'] = meta_ip
            else:
                stream_type = stream[:-1]
                stream_no = int(stream[-1])
                (c['%s_rx_udp_ip_str_beam%i' % (stream_type, stream_no)],
                 c['%s_rx_udp_port_beam%i' % (stream_type, stream_no)],
                 c['%s_rx_meta_ip_str_beam%i' % (stream_type, stream_no)],
                 c['%s_rx_meta_port_beam%i' % (stream_type, stream_no)]) = (
                    data_ip, data_port, meta_ip, meta_port)
            # Close any other stream that was listening
            self._spead_model.send_stop()
            # Re-init spead model with new IPs
            self._spead_model.init_spead()
        finally:
            self._thread_paused = paused

    @property
    def has_beamformer(self):
        """Whether or not the current mode has a beamformer"""
        return bool(self.config.get('bf_n_beams'))

    @property
    def labels(self):
        """Build a dict of input channel names -> antenna labels"""
        if self.get_sensor('mode').value() == 'ready':
            raise RuntimeError("Can't access antenna mapping in 'ready' mode")

        return dict(zip(self.config.get_unmapped_channel_names(),
                        self.config['antenna_mapping']))

    @property
    def capture_list(self):
        c = self.config
        # It seems that the config does not specify a separate port
        capture_list = [CaptureDestination('k7',
                                           c['rx_meta_ip_str'], c['rx_udp_port'],
                                           c['rx_udp_ip_str'], c['rx_udp_port'],)]

        if self.has_beamformer:
            for i in range(c['bf_n_beams']):
                bf = 'bf%i' % i
                capture_list.append(CaptureDestination(
                    bf,
                    c['bf_rx_meta_ip_str_beam%i' % i],
                    c['bf_rx_meta_port_beam%i' % i],
                    c['bf_rx_udp_ip_str_beam%i' % i],
                    c['bf_rx_udp_port_beam%i' % i],))

        return capture_list

    def set_antenna_mapping(self, channel, ant_name):
        """Set the antenna name of `channel` (e.g. 0x, 3y) to `ant_name`"""
        if self.get_sensor('mode').value() == 'ready':
            raise RuntimeError("Can't set antenna mapping in 'ready' mode")
        self.config.set_antenna_mapping(channel, ant_name)
