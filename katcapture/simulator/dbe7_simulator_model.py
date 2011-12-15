from __future__ import with_statement

import time
import sys
import logging
import numpy as np
from .dbe7_roach_models import XEngines, FEngines
from .dbe7_spead_model import DBE7SpeadData
from katcore.dev.base import ThreadedModel
from katcp import Sensor
import random

activitylogger = logging.getLogger('activity')
log_name = 'kat.k7simulator'
logger = logging.getLogger(log_name)

class K7CorrelatorModel(ThreadedModel):
    def __init__(self, config_file, *names, **kwargs):
        self.config = self.read_config(config_file)
        self._spead_model = DBE7SpeadData(self.config)
        super(K7CorrelatorModel, self).__init__(*names, **kwargs)
        self._init_values()
        self._init_sensors()
        self._init_roaches()
        self._init_spead()
        self._thread_paused = False
        self.data = self.generate_data()


    def _init_roaches(self):
        roach_x_names = [s.strip() for s in self.config['servers_x'].split(',')]
        roach_f_names = [s.strip() for s in self.config['servers_f'].split(',')]
        self._roach_x_engines = XEngines(roach_x_names)
        self._roach_f_engines = FEngines(roach_f_names)
        for s in self._roach_x_engines.get_sensors(): self.add_sensor(s)
        for s in self._roach_f_engines.get_sensors(): self.add_sensor(s)

    def _init_sensors(self):
        self.add_sensor(Sensor(
                Sensor.INTEGER, "sync_time", "Last sync time in epoch seconds.","seconds",
                default=0, params=[0,2**32]))
        self.add_sensor(Sensor(Sensor.INTEGER, "tone_freq",
                               "The frequency of the injected tone in Hz.","",
                               default=0, params=[0,2**32]))
        dip_sens = Sensor(
            Sensor.STRING, "destination_ip",
            "The current destination address for data and metadata.","","")
        dip_sens.set_value(self.config['rx_meta_ip'])
        self.add_sensor(dip_sens)
        msens = Sensor(Sensor.DISCRETE, 'mode', 'Current DBE operating mode', '',
                       ['basic', 'ready', 'wbc', 'nbc'])
        msens.set_value('ready', Sensor.NOMINAL, time.time())
        self.add_sensor(msens)
        nbc_f_sens = Sensor(Sensor.INTEGER, 'nbc.frequency.current',
                            'current selected center frequency', 'Hz',
                            [0, 387.5*1000000])
        nbc_f_sens.set_value(0, Sensor.UNKNOWN)
        self.add_sensor(nbc_f_sens)

        self.add_sensor(Sensor(Sensor.BOOLEAN, "ntp_synchronised", "clock good", ""))
        self.get_sensor('sync_time').set_value(self._spead_model.sync_time, Sensor.NOMINAL)
        self.get_sensor('tone_freq').set_value(self.tone_freq, Sensor.NOMINAL)
        self.get_sensor('ntp_synchronised').set_value(True, Sensor.NOMINAL)

    def _init_values(self):
        self.adc_value = 0
        self.dump_period = 1.0
        self.sample_rate = 800e6
        self.tone_freq = 302e6
        self.noise_diode = 0
        self.nd_duty_cycle = 0
        self._nd_cycles = 0
        self.target_az = 0
        self.target_el = 0
        self.target_flux = 0
        self.test_az = 0
        self.test_el = 0
        self.nd = 0
        self.multiplier = 100

    def _init_spead(self):
        self._spead_model.init_spead()

    def read_config(self, config_file):
        config = {}
        try:
            f = open(config_file)
        except IOError, e:
            raise IOError("Specified config file (%s) could not be read. "
                          "Unable to start simulator. Read error: %s" % (
                              config_file, e))
        for s in f.readlines():
            try:
                if s.index(' = ') > 0:
                    (k,v) = s[:-1].split(' = ')
                    try:
                        config[k] = int(v)
                    except ValueError:
                        config[k] = v
            except ValueError:
                pass
        f.close()
        config['xeng_sample_bits']=32
        config['n_xfpgas']=len(config['servers_x'])
        config['n_xeng']=config['x_per_fpga']*config['n_xfpgas']
        config['n_bls']=config['n_ants']*(config['n_ants']+1)/2 * config['n_stokes']
        config['n_chans_per_x']=config['n_chans']/config['n_xeng']
        config['bandwidth']=config['adc_clk']/2.
        config['center_freq']=config['adc_clk']/4.
        config['pcnt_scale_factor']=config['bandwidth']/config['xeng_acc_len']
        config['spead_timestamp_scale_factor']=(config['pcnt_scale_factor'] /
                                                config['n_chans'])
        config['10gbe_ip']=12
        config['n_accs']=config['acc_len']*config['xeng_acc_len']
        config['int_time']= float(config['n_chans'])*(
            config['n_accs']/config['bandwidth'])
        config['pols']=['x','y']
        config['n_pols'] = 2
        return config

    def start(self, *names, **kwargs):
        self.set_mode('wbc', mode_delay=0)
        self.spead_issue()
        super(K7CorrelatorModel, self).start(*names, **kwargs)


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

    def k7_frequency_select(self, centre_frequency):
        """Set the nbc centre frequency to the closest nbc channel

        Returns the actual centre frequency used. Will fail if the correlator is
        not in narrow-band (nbc) mode
        """
        # TODO Should see if we can get this from actual DBE code/config
        min_f0 = 0 ; max_f0 = 400e6 ; d_f0 = 12.5e6
        assert(centre_frequency >= min_f0)
        assert(centre_frequency <= max_f0)
        mode = self.get_sensor('mode').value()
        if mode != 'nbc':
            raise RuntimeError('Correlator must be in nbc mode to set nbc '
                               'centre frequency. Correleator is currently in '
                               '%s mode' % mode)

        f0 = int(round(centre_frequency/d_f0)*d_f0)
        self.get_sensor('nbc.frequency.current').set_value(
            f0, Sensor.NOMINAL, time.time())
        return f0

    def run(self):
        while not self._stopEvent.isSet():
            if not self._thread_paused:
                self.send_dump()
                status = ("\rSending correlator dump at %s "
                          "(dump period: %f s, multiplier: %i, noise_diode: %s)" %
                          (time.ctime(), self.dump_period, self.multiplier,
                           (self.nd > 0 and 'On' or 'Off')))
                sys.stdout.write(status)
                sys.stdout.flush()
            st = time.time()
            self.data = self.generate_data()
            time.sleep(max(self.dump_period - (time.time() - st), 0.))
        self.send_stop()
        print "Correlator tx halted."

    def gaussian(self,x,y):
         # for now a gaussian of height 1 and width 1
         # beam width is 0.8816 degrees (sigma of 0.374) at 1.53 GHZ
         #equates to coefficient of
        return np.exp(-(0.5/(0.374*0.374)) * (x*x + y*y))

    def generate_data(self):
        source_value = self.target_flux * self.gaussian(
            self.target_az - self.test_az, self.target_el - self.test_el)
         # generate a flux contribution from the synthetic source (if any)
        tsys_elev_value = 25 - np.log(self.test_el + 1) * 5
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
        print "Issuing SPEAD meta data to %s\n" % self.config['rx_meta_ip']
        self._spead_model.spead_issue()



    def send_dump(self):
        """Send a single correlator dump..."""
        data_ig = self._spead_model.data_ig
        tx = self._spead_model.tx
        data_ig['timestamp'] = int(
            (time.time() - self._spead_model.sync_time) *
            self.config['spead_timestamp_scale_factor'])
        data_ig['xeng_raw'] = self.data
        tx.send_heap(data_ig.get_heap())

    def send_stop(self):
        self._spead_model.send_stop()

    def set_mode(self, mode, progress_callback=None, mode_delay=10):
        """Set DBE mode, can be one of 'wbc' or 'nbc'

        Parameters
        ==========

        mode -- 'nbc' or 'wbc', the desired mode
        progress_callback -- optional callback that is called with a string
            describing the 'progress' of mode changing.
        mode_delay -- optional integer number of seconds to delay while doing
            mode change. Defaults to 10

        """
        valid_modes = ('nbc', 'wbc')

        if progress_callback is None: progress_callback = lambda x: x
        if mode not in valid_modes:
            raise ValueError('Mode should be one of ' + ','.join(
                "'%s'" %m for m in valid_modes))
        if mode == self.get_sensor('mode').value():
            progress_callback('Correlator already in mode %s' % mode)
            return

        logger.info('Sleeping %f s for dbe7 mode change' % mode_delay)

        for i in range(mode_delay):
            time.sleep(1)
            smsg = 'Doing some mode changing stuff on DBE %d' % (i+1)
            progress_callback(smsg)

        if mode == 'wbc':
            nbc_f_sens = self.get_sensor('nbc.frequency.current')
            nbc_f_sens.set_value(nbc_f_sens.value(), Sensor.UNKNOWN)

        self.get_sensor('mode').set_value(mode, Sensor.NOMINAL, time.time())

