import logging
import copy
import spead
import time
import numpy

class DBE7SpeadData(object):
    """
    DBE7 spead data configuration stealing some code from
    corr.corr_functions.Correlator (spead_*issue, get_bl_order,
    adc_lru_mapping_get, get_input_location, is_wideband, is_narrowband,
    map_input_to_ant
    """

    def __init__(self, config, logger=None):
        self.MODE_WB = 0
        self.MODE_NB = 1
        self.MODE_DDC = 2
        if logger is None:
            self.syslogger = logging.getLogger('kat.k7simulator')
        else:
            self.syslogger = logger
        self.config = config

        self.xsrvs = self.config['servers_x']
        self.fsrvs = self.config['servers_f']
        self.allsrvs = self.fsrvs + self.xsrvs
        # Bogus initial accumulation time. Should be set by user with
        # set_acc_time(). This _acc_time is used to short-circuit the
        # "real" DBE calculation using the number of integrations,
        # etc.
        self.set_acc_time(0)
        self.init_spead()

    def init_spead(self):
        self.tx=spead.Transmitter(spead.TransportUDPtx(
                self.config['rx_meta_ip_str'],self.config['rx_udp_port']))
        self.data_ig=spead.ItemGroup()
        self._data_meta_descriptor = None
        self.init_data_descriptor()

    @property
    def sync_time(self):
        return self.config['sync_time']

    @property
    def bls_ordering(self):
        return self.get_bl_order()

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
        dp_bls = tuple([o for o in order1 + order2])
        rv=[]
        for bl in dp_bls:
            rv.append(tuple((self.map_input_to_ant(bl[0]*2),self.map_input_to_ant(bl[1]*2))))
            rv.append(tuple((self.map_input_to_ant(bl[0]*2+1),self.map_input_to_ant(bl[1]*2+1))))
            rv.append(tuple((self.map_input_to_ant(bl[0]*2),self.map_input_to_ant(bl[1]*2+1))))
            rv.append(tuple((self.map_input_to_ant(bl[0]*2+1),self.map_input_to_ant(bl[1]*2))))
        return rv

    def adc_lru_mapping_get(self):
        """Map all the antennas to lru and physical inputs"""
        rv=[]
        for input_n,ant_str in enumerate(self.config._get_ant_mapping_list()):
            ffpga_n,xfpga_n,fxaui_n,xxaui_n,feng_input = self.get_input_location(input_n)
            rv.append((ant_str,input_n,self.fsrvs[ffpga_n],feng_input))
        return rv

    def get_input_location(self, input_n):
        " Returns the (ffpga_n,xfpga_n,fxaui_n,xxaui_n,feng_input) location for a given system-wide input number."
        if input_n > self.config['n_inputs'] or input_n < 0:
            raise RuntimeError("There is no input %i in this design (total %i inputs)."%(input_n,self.config['n_inputs']))
        ant = input_n / 2 #dual-pol ant, as transmitted across XAUI links
        ffpga_n  = ant/self.config['f_per_fpga']
        fxaui_n  = ant/self.config['n_ants_per_xaui']%self.config['n_xaui_ports_per_ffpga']
        xfpga_n  = ant/self.config['n_ants_per_xaui']/self.config['n_xaui_ports_per_xfpga']
        xxaui_n  = ant/self.config['n_ants_per_xaui']%self.config['n_xaui_ports_per_xfpga']
        feng_input = input_n%self.config['f_inputs_per_fpga']
        return (ffpga_n,xfpga_n,fxaui_n,xxaui_n,feng_input)

    def spead_labelling_issue(self):
        tx=self.tx
        ig=spead.ItemGroup()

#        ig.add_item(name="bls_ordering",id=0x100C,
#            description="The output ordering of the baselines from each X engine. Packed as a pair of unsigned integers, ant1,ant2 where ant1 < ant2.",
#            shape=[self.config['n_bls'],2],fmt=spead.mkfmt(('u',16)),
#            init_val=[[bl[0],bl[1]] for bl in self.get_bl_order()])

        ig.add_item(name="bls_ordering",id=0x100C,
            description="The output ordering of the baselines from each X engine.",
            #shape=[self.config['n_bls']],fmt=spead.STR_FMT,
            init_val=numpy.array([bl for bl in self.get_bl_order()]))

        ig.add_item(name="input_labelling",id=0x100E,
            description="The physical location of each antenna connection.",
            #shape=[self.config['n_inputs']],fmt=spead.STR_FMT,
            init_val=numpy.array([(ant_str,input_n,lru,feng_input) for (ant_str,input_n,lru,feng_input) in self.adc_lru_mapping_get()]))

#        ig.add_item(name="crosspol_ordering",id=0x100D,
#            description="The output ordering of the cross-pol terms. Packed as a pair of characters, pol1,pol2.",
#            shape=[self.config['n_stokes'],self.config['n_pols']],fmt=spead.mkfmt(('c',8)),
#            init_val=[[bl[0],bl[1]] for bl in self.get_crosspol_order()])
        tx.send_heap(ig.get_heap())
        self.syslogger.info("Issued SPEAD metadata describing baseline labelling and input mapping to %s:%i."%(self.config['rx_meta_ip_str'],self.config['rx_udp_port']))


    def spead_static_meta_issue(self):
        """ Issues the SPEAD metadata packets containing the payload and options descriptors and unpack sequences."""
        #tested ok corr-0.5.0 2010-08-07
        tx=self.tx
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

        ig.add_item(name="center_freq",id=0x1011,
            description="The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['center_freq'])

        ig.add_item(name="bandwidth",id=0x1013,
            description="The analogue bandwidth of the digitally processed signal in Hz.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['bandwidth'])

        #1015/1016 are taken (see time_metadata_issue below)

        if self.is_wideband():
            ig.add_item(name="fft_shift",id=0x101E,
                description="The FFT bitshift pattern. F-engine correlator internals.",
                shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
                init_val=self.config['fft_shift'])
        elif self.is_narrowband():
            ig.add_item(name="fft_shift_fine",id=0x101C,
                description="The FFT bitshift pattern for the fine channelisation FFT. F-engine correlator internals.",
                shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
                init_val=self.config['fft_shift_fine'])
            ig.add_item(name="fft_shift_coarse",id=0x101D,
                description="The FFT bitshift pattern for the coarse channelisation FFT. F-engine correlator internals.",
                shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
                init_val=self.config['fft_shift_coarse'])

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

        ig.add_item(name="rx_udp_ip_str",id=0x1024,
            description="Destination IP address for X engine output UDP packets.",
            shape=[-1],fmt=spead.STR_FMT,
            init_val=self.config['rx_udp_ip_str'])

        ig.add_item(name="feng_start_ip",id=0x1025,
            description="F engine starting IP address.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['10gbe_ip'])

        ig.add_item(name="xeng_rate",id=0x1026,
            description="Target clock rate of processing engines (xeng).",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['xeng_clk'])

#        ig.add_item(name="n_stokes",id=0x1040,
#            description="Number of Stokes parameters in output.",
#            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
#            init_val=self.config['n_stokes'])

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

        tx.send_heap(ig.get_heap())
        self.syslogger.info("Issued misc SPEAD metadata to %s:%i."%(self.config['rx_meta_ip_str'],self.config['rx_udp_port']))
        self.spead_labelling_issue()

    def spead_time_meta_issue(self):
        """Issues a SPEAD packet to notify the receiver that we've resync'd the system, acc len has changed etc."""
        #tested ok corr-0.5.0 2010-08-07
        tx=self.tx
        ig=spead.ItemGroup()

        ig.add_item(name="n_accs",id=0x1015,
            description="The number of spectra that are accumulated per integration.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.acc_n_get())

        ig.add_item(name="int_time",id=0x1016,
            description="Approximate (it's a float!) integration time per accumulation in seconds.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.acc_time_get())

        ig.add_item(name='sync_time',id=0x1027,
            description="Time at which the system was last synchronised (armed and triggered by a 1PPS) in seconds since the Unix Epoch.",
            shape=[],fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
            init_val=self.config['sync_time'])

        ig.add_item(name="scale_factor_timestamp",id=0x1046,
            description="Timestamp scaling factor. Divide the SPEAD data packet timestamp by this number to get back to seconds since last sync.",
            shape=[],fmt=spead.mkfmt(('f',64)),
            init_val=self.config['spead_timestamp_scale_factor'])


        tx.send_heap(ig.get_heap())
        self.syslogger.info("Issued SPEAD timing metadata to %s:%i."%(self.config['rx_meta_ip_str'],self.config['rx_udp_port']))

    def spead_eq_meta_issue(self):
        """Issues a SPEAD heap for the RF gain and EQ settings."""
        tx=self.tx
        ig=spead.ItemGroup()

        if self.config['adc_type'] == 'katadc':
            for input_n,ant_str in enumerate(self.config._get_ant_mapping_list()):
                ig.add_item(name="rf_gain_%i"%(input_n),id=0x1200+input_n,
                    description="The analogue RF gain applied at the ADC for input %i (ant %s) in dB."%(input_n,ant_str),
                    shape=[],fmt=spead.mkfmt(('f',64)),
                    init_val=self.config['rf_gain_%i'%(input_n)])

        if self.config['eq_type']=='scalar':
            for in_n,ant_str in enumerate(self.config._get_ant_mapping_list()):
                ig.add_item(name="eq_coef_%s"%(ant_str),id=0x1400+in_n,
                    description="The unitless per-channel digital amplitude scaling factors implemented prior to requantisation, post-FFT, for input %s."%(ant_str),
                    init_val=self.eq_spectrum_get(ant_str))

        elif self.config['eq_type']=='complex':
            for in_n,ant_str in enumerate(self.config._get_ant_mapping_list()):
                ig.add_item(name="eq_coef_%s"%(ant_str),id=0x1400+in_n,
                    description="The unitless per-channel digital scaling factors implemented prior to requantisation, post-FFT, for input %s. Complex number real,imag 32 bit integers."%(ant_str),
                    shape=[self.config['n_chans'],2],fmt=spead.mkfmt(('u',32)),
                    init_val=[[numpy.real(coeff),numpy.imag(coeff)] for coeff in self.eq_spectrum_get(ant_str)])

        else: raise RuntimeError("I don't know how to deal with your EQ type.")

        tx.send_heap(ig.get_heap())
        self.syslogger.info("Issued SPEAD EQ metadata to %s:%i."%(self.config['rx_meta_ip_str'],self.config['rx_udp_port']))


    def init_data_descriptor(self):
        """ Initialises the SPEAD data descriptors for the HW 10GbE output.
        To  issue the descriptor to enable receivers to decode the
        data, calls spead_data_descriptor_issue
        """
        #tested ok corr-0.5.0 2010-08-07
        ig=self.data_ig

        if self.config['xeng_sample_bits'] != 32: raise RuntimeError("Invalid bitwidth of X engine output. You specified %i, but I'm hardcoded for 32."%self.config['xeng_sample_bits'])


        if self.config['xeng_format'] == 'cont':
            ig.add_item(name=('timestamp'), id=0x1600,
                description='Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
                shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)),
                init_val=0)

            ig.add_item(name=("xeng_raw"),id=0x1800,
                description="Raw data for %i xengines in the system. This item represents a full spectrum (all frequency channels) assembled from lowest frequency to highest frequency. Each frequency channel contains the data for all baselines (n_bls given by SPEAD ID 0x100B). Each value is a complex number -- two (real and imaginary) unsigned integers."%(self.config['n_xeng']),
            ndarray=(numpy.dtype(numpy.int32),(self.config['n_chans'],self.config['n_bls'],2)))


        elif self.config['xeng_format'] =='inter':
            for x in range(self.config['n_xeng']):

                ig.add_item(name=('timestamp%i'%x), id=0x1600+x,
                    description='Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
                    shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)),init_val=0)

                ig.add_item(name=("xeng_raw%i"%x),id=(0x1800+x),
                    description="Raw data for xengine %i out of %i. Frequency channels are split amongst xengines. Frequencies are distributed to xengines in a round-robin fashion, starting with engine 0. Data from all X engines must thus be combed or interleaved together to get continuous frequencies. Each xengine calculates all baselines (n_bls given by SPEAD ID 0x100B) for a given frequency channel. For a given baseline, -SPEAD ID 0x1040- stokes parameters are calculated (nominally 4 since xengines are natively dual-polarisation; software remapping is required for single-baseline designs). Each stokes parameter consists of a complex number (two real and imaginary unsigned integers)."%(x,self.config['n_xeng']),
                    ndarray=(numpy.dtype(numpy.int32),(self.config['n_chans']/self.config['n_xeng'],self.config['n_bls'],2)))

        self._data_meta_descriptor = self.data_ig.get_heap()

    def spead_data_descriptor_issue(self):
        mdata = copy.deepcopy(self._data_meta_descriptor)
        self.tx.send_heap(mdata)
        self.syslogger.info("Issued SPEAD data descriptor to %s:%i."%(self.config['rx_meta_ip_str'],self.config['rx_udp_port']))


    def spead_issue_all(self):
        """Issues all SPEAD metadata."""
        st = time.time()
        self.spead_data_descriptor_issue()
        self.spead_static_meta_issue()
        self.spead_time_meta_issue()
        self.spead_eq_meta_issue()
        self.syslogger.info("Took %.2f seconds to issue metadata" % (time.time() - st))

    spead_issue = spead_issue_all         # For compatibility with other simulator code

    def is_wideband(self):
        return self.config['mode'] == self.MODE_WB

    def is_narrowband(self):
        return self.config['mode'] == self.MODE_NB


    def eq_spectrum_get(self,ant_str):
        """Dummy function returning fake  (unity) equaliser settings

        Docstring of original function:

        Retrieves the equaliser settings currently programmed in an F
        engine for the given antenna. Assumes equaliser of 16
        bits. Returns an array of length n_chans."""

        return numpy.ones(self.config['n_chans'])

    def map_input_to_ant(self,input_n):
        """Maps an input number to an antenna string."""
        return self.config._get_ant_mapping_list()[input_n]

    def acc_n_get(self):
        """get dummy value of number of accumulations"""
        return 8

    def acc_time_get(self):
        ## Two lines below is how the DBE really does it. We will hack it
        # n_accs = self.acc_n_get()
        # return float(self.config['n_chans'] * n_accs) / self.config['bandwidth']
        ##
        return self._acc_time

    def set_acc_time(self, acc_time):
        """Set the accumulation time in seconds

        This accumulation time is used to short-circuit the 'real' DBE
        calculation using the number of integrations, etc.
        """
        self._acc_time = acc_time

    def send_stop(self):
        try:
            self.tx.send_halt()
        except AttributeError:
            # Don't error out if the transmitter has not been initialized yet.
            pass
