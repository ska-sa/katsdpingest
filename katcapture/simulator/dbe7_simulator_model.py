from __future__ import with_statement

import time
import sys
import copy
import numpy as np
from katcore.dev.base import ThreadedModel
from katcp import Sensor
import spead

class K7CorrelatorModel(ThreadedModel):
    def __init__(self, config_file, *names, **kwargs):
        self.config = self.read_config(config_file)
        super(K7CorrelatorModel, self).__init__(*names, **kwargs)
        self._init_values()
        self._init_sensors()
        self._init_spead()
        self._thread_paused = False
        self.data = self.generate_data()

    def _init_sensors(self):
        self.sensors["sync-time"] = Sensor(
            Sensor.INTEGER, "sync_time", "Last sync time in epoch seconds.","",
            default=0, params=[0,2**32])
        self.sensors["tone_freq"] = Sensor(
            Sensor.INTEGER, "tone_freq",
            "The frequency of the injected tone in Hz.","",
            default=0, params=[0,2**32])
        self.sensors["destination_ip"] = Sensor(
            Sensor.STRING, "destination_ip",
            "The current destination address for data and metadata.","","")
        self.sensors["destination_ip"].set_value(self.config['rx_meta_ip'])
        msens = Sensor( Sensor.DISCRETE, 'mode', 'Current DBE operating mode', '',
                        ['basic', 'ready', 'wbc', 'nbc'])
        msens.set_value('ready', Sensor.NOMINAL, time.time())
        self.sensors[msens.name] = msens

    def _init_values(self):
        self.labels = dict([[str(x)+y,'ant' + str(x+1)+{'x':'H','y':'V'}[y]]
                            for x in range(8) for y in ['x','y']])
        # in np form so it will work as a spead item descriptor
        self.sync_time = int(time.time())
        self.adc_value = 0
        self.update_bls_ordering()
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
        self.tx=spead.Transmitter(spead.TransportUDPtx(
                self.config['rx_meta_ip'],self.config['rx_udp_port']))
        self.data_ig=spead.ItemGroup()
        self._data_meta_descriptor = None
        self.init_data_descriptor()

    def read_config(self, config_file):
        config = {}
        try:
            f = open(config_file)
        except IOError:
            print "Specified config file (%s) not found. Unable to start simulator." % config_file
            sys.exit(0)
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
        super(K7CorrelatorModel, self).start(*names, **kwargs)
        self.spead_issue()

    def update_bls_ordering(self):
        """Update the mapping based on the specified input labelling."""
        self.bls_ordering = np.array(self.get_default_bl_map())

    def get_default_bl_map(self):
        """Return a default baseline mapping by replacing inputs with proper antenna names."""
        bls = []
        for b in self.get_bl_order():
            for p in ['xx','yy','xy','yx']:
                bls.append([self.labels[str(b[0])+p[0]], self.labels[str(b[1])+p[1]]])
        return bls

    def get_bl_order(self):
        """Return the order of baseline data output by a CASPER correlator X engine."""
        n_ants=self.config['n_ants']
        order1, order2 = [], []
        for i in range(n_ants):
            for j in range(int(n_ants/2),-1,-1):
                k = (i-j) % n_ants
                if i >= k: order1.append((k, i))
                else: order2.append((i, k))
        order2 = [o for o in order2 if o not in order1]
        return tuple([o for o in order1 + order2])

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
        samples_per_dump = self.config['n_chans'] * 8
         # not related to actual value. just for calculation purposes
        n = np.arange(samples_per_dump)
        x = np.cos(2 * np.pi * self.tone_freq / self.sample_rate * n)
        data = np.fft.fft(x, self.config['n_chans'])[:self.config['n_chans']]
        data = (data.view(np.float64)*self.multiplier).astype(np.int32).reshape((512,2))
        data = np.tile(data, self.config['n_bls'])
        data = data.reshape((self.config['n_chans'],self.config['n_bls'],2), order='C')
        for ib in range (self.config['n_bls']):#for different baselines
            (a1,a2)= self.bls_ordering[ib]
            if a1[:-1] == a2[:-1]:
                auto_d=np.abs(data[:,ib,:]+((ib*32131+48272)%1432)/1432.0*20.0 +
                              np.random.randn(self.config['n_chans']*2).reshape(
                        [self.config['n_chans'],2])*10.0) + 200
                auto_d[:,1] = 0
                data[:,ib,:]=auto_d
            else:
                data[:,ib,:]=data[:,ib,:]+(((ib*32131+48272)%1432)/1432.0*20.0 +
                np.random.randn(self.config['n_chans']*2).reshape(
                    [self.config['n_chans'],2])*10.0)
        data = data.astype(np.int32) + self.nd
        return data

    def get_crosspol_order(self):
        "Returns the order of the cross-pol terms out the X engines"
        pol1=self.config['rev_pol_map'][0]
        pol2=self.config['rev_pol_map'][1]
        return (pol1+pol1,pol2+pol2,pol1+pol2,pol2+pol1)

    def spead_issue(self):
        print "Issuing SPEAD meta data to %s\n" % self.config['rx_meta_ip']
        self.spead_static_meta_issue()
        self.spead_time_meta_issue()
        self.spead_data_descriptor_issue()
        self.spead_eq_meta_issue()

    def spead_static_meta_issue(self):
        """Issues the SPEAD metadata packets containing the payload and options descriptors and unpack sequences."""
        ig=spead.ItemGroup()

        ig.add_item(name="adc_clk",id=0x1007,
            description="Clock rate of ADC (samples per second).",
            shape=[],fmt=spead.mkfmt(('u',64)),
            init_val=self.config['adc_clk'])

        ig.add_item(name="n_bls",id=0x1008,
            description="The total number of baselines in the data product.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['n_bls'])

        ig.add_item(name="n_chans",id=0x1009,
            description="The total number of frequency channels present in any integration.",
            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['n_chans'])

        ig.add_item(name="n_ants",id=0x100A,
            description="The total number of dual-pol antennas in the system.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['n_ants'])

        ig.add_item(name="n_xengs",id=0x100B,
            description="The total number of X engines in the system.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['n_xeng'])

        ig.add_item(name="bls_ordering",id=0x100C,
            description=("The output ordering of the baselines from each X engine. "
                         "Packed as a pair of unsigned integers, "
                         "ant1,ant2 where ant1 < ant2."),
            init_val=self.bls_ordering)

        ig.add_item(name="input_labelling",id=0x100E,
           description="The physical location of each antenna connection.",
           init_val=np.array([[label, str(inp), 'roachXXXXXX', str(inp%2)] for (inp, label) in enumerate(sorted(self.labels.itervalues()))]))

        ig.add_item(name="center_freq",id=0x1011,
            description="The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['center_freq'])

        ig.add_item(name="bandwidth",id=0x1013,
            description="The analogue bandwidth of the digitally processed signal in Hz.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['bandwidth'])

        #1015/1016 are taken (see time_metadata_issue below)

        ig.add_item(name="fft_shift",id=0x101E,
            description="The FFT bitshift pattern. F-engine correlator internals.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['fft_shift'])

        ig.add_item(name="xeng_acc_len",id=0x101F,
            description="Number of spectra accumulated inside X engine. Determines minimum integration time and user-configurable integration time stepsize. X-engine correlator internals.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['xeng_acc_len'])

        ig.add_item(name="requant_bits",id=0x1020,
            description="Number of bits after requantisation in the F engines (post FFT and any phasing stages).",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['feng_bits'])

        ig.add_item(name="feng_pkt_len",id=0x1021,
            description="Payload size of 10GbE packet exchange between F and X engines in 64 bit words. Usually equal to the number of spectra accumulated inside X engine. F-engine correlator internals.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['10gbe_pkt_len'])

        ig.add_item(name="rx_udp_port",id=0x1022,
            description="Destination UDP port for X engine output.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['rx_udp_port'])

        ig.add_item(name="feng_udp_port",id=0x1023,
            description="Destination UDP port for F engine data exchange.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['10gbe_port'])

        ig.add_item(name="rx_udp_ip",id=0x1024,
            description="Destination IP address for X engine output UDP packets.",
            shape=[-1],fmt=spead.STR_FMT,
            init_val=self.config['rx_udp_ip'])

        ig.add_item(name="feng_start_ip",id=0x1025,
            description="F engine starting IP address.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['10gbe_ip'])

        ig.add_item(name="xeng_rate",id=0x1026,
            description="Target clock rate of processing engines (xeng).",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['xeng_clk'])

        ig.add_item(name="x_per_fpga",id=0x1041,
            description="Number of X engines per FPGA.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['x_per_fpga'])

        ig.add_item(name="n_ants_per_xaui",id=0x1042,
            description="Number of antennas' data per XAUI link.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['n_ants_per_xaui'])

        ig.add_item(name="ddc_mix_freq",id=0x1043,
            description="Digital downconverter mixing freqency as a fraction of the ADC sampling frequency. eg: 0.25. Set to zero if no DDC is present.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['ddc_mix_freq'])

        ig.add_item(name="ddc_decimation",id=0x1044,
            description="Frequency decimation of the digital downconverter (determines how much bandwidth is processed) eg: 4",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['ddc_decimation'])

        ig.add_item(name="adc_bits",id=0x1045,
            description="ADC quantisation (bits).",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['adc_bits'])

        ig.add_item(name="xeng_out_bits_per_sample",id=0x1048,
            description="The number of bits per value of the xeng accumulator output. Note this is for a single value, not the combined complex size.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['xeng_sample_bits'])

        self.tx.send_heap(ig.get_heap())

    def spead_time_meta_issue(self):
        """Issues a SPEAD packet to notify the receiver that we've resync'd the system, acc len has changed etc."""
        ig=spead.ItemGroup()

        ig.add_item(name="n_accs",id=0x1015,
            description="The number of spectra that are accumulated per integration.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=8)

        ig.add_item(name="int_time",id=0x1016,
            description="Approximate (it's a float!) integration time per accumulation in seconds.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['int_time'])

        ig.add_item(name='sync_time',id=0x1027,
            description="Time at which the system was last synchronised (armed and triggered by a 1PPS) in seconds since the Unix Epoch.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.sync_time)

        ig.add_item(name="scale_factor_timestamp",id=0x1046,
            description="Timestamp scaling factor. Divide the SPEAD data packet timestamp by this number to get back to seconds since last sync.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['spead_timestamp_scale_factor'])

        self.tx.send_heap(ig.get_heap())

    def spead_eq_meta_issue(self):
        """Issues a SPEAD heap for the RF gain and EQ settings."""
        ig=spead.ItemGroup()

        if self.config['adc_type'] == 'katadc':
            for ant in range(self.config['n_ants']):
                for pn,pol in enumerate(self.config['pols']):
                    ig.add_item(name="rf_gain_%i%c"%(ant,pol),id=0x1200+ant*self.config['n_pols']+pn,
                        description="The analogue RF gain applied at the ADC for input %i%c in dB."%(ant,pol),
                        shape=[],fmt=spead.mkfmt(('f',64)),
                        init_val=self.config['rf_gain_%i%c'%(ant,pol)])

        for ant in range(self.config['n_ants']):
            for pn,pol in enumerate(self.config['pols']):
                ig.add_item(name="eq_coef_%i%c"%(ant,pol),id=0x1400+ant*self.config['n_pols']+pn,
                    description="The unitless per-channel digital scaling factors implemented prior to requantisation, post-FFT, for input %i%c. Complex number real,imag 32 bit integers."%(ant,pol),
                    shape=[self.config['n_chans'],2],fmt=spead.mkfmt(('u',32)),
                    init_val=[[np.real(coeff),np.imag(coeff)] for coeff in np.zeros(512, dtype=np.complex64)])


        self.tx.send_heap(ig.get_heap())


    def send_dump(self):
        """Send a single correlator dump..."""
        self.data_ig['timestamp'] = int(
            (time.time() - self.sync_time) * self.config['spead_timestamp_scale_factor'])
        self.data_ig['xeng_raw'] = self.data
        self.tx.send_heap(self.data_ig.get_heap())

    def send_stop(self):
        self.tx.send_halt()

    def init_data_descriptor(self):
        """Issues the SPEAD data descriptors for the HW 10GbE output,

           Enables receivers to decode the data.
        """

        if self.config['xeng_sample_bits'] != 32:
            raise RuntimeError("Invalid bitwidth of X engine output. You "
                               "specified %i, but I'm hardcoded for 32." % (
                    self.config['xeng_sample_bits']) )

        self.data_ig.add_item(
            name=('timestamp'), id=0x1600,
            description='Timestamp of start of this integration. uint counting '
            'multiples of ADC samples since last sync (sync_time, id=0x1027). '
            'Divide this number by timestamp_scale (id=0x1046) to get back to '
            'seconds since last sync when this integration was actually '
            'started. Note that the receiver will need to figure out the centre '
            'timestamp of the accumulation (eg, by adding half of int_time, '
            'id 0x1016).',
            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)), init_val=0)

        self.data_ig.add_item(
            name=("xeng_raw"),id=0x1800,
            description="Raw data for %i xengines in the system. This item "
            "represents a full spectrum (all frequency channels) assembled from "
            "lowest frequency to highest frequency. Each frequency channel "
            "contains the data for all baselines (n_bls given by SPEAD ID "
            "0x100B). For a given baseline, -SPEAD ID 0x1040- stokes parameters "
            "are calculated (nominally 4 since xengines are natively "
            "dual-polarisation; software remapping is required for "
            "single-baseline designs). Each stokes parameter consists of a "
            "complex number (two real and imaginary unsigned integers)." % (
                self.config['n_xeng']),
            ndarray=(np.dtype(np.int32),
                     (self.config['n_chans'],
                      self.config['n_bls'],2)))

        self._data_meta_descriptor = self.data_ig.get_heap()

    def spead_data_descriptor_issue(self):
        mdata = copy.deepcopy(self._data_meta_descriptor)
        self.tx.send_heap(mdata)
 
