from __future__ import print_function, division, absolute_import
import spead2
import spead2.recv
import spead2.recv.trollius
import katcp
from katcp.kattypes import request, return_reply, Str
import os
import os.path
import trollius
from trollius import From
import tornado
import logging
import katsdpingest
import subprocess
import time
import sys
import netifaces
import signal
import time

import argparse
import pyfits
from astropy.time import Time


_logger = logging.getLogger(__name__)

def _get_interface_address(interface):
    return netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']

class _CaptureSession(object):
    """Object encapsulating a co-routine that runs for a single capture session
    (from ``capture-init`` to end of stream or ``capture-done``.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments. See :class:`CaptureServer`.
    loop : :class:`trollius.BaseEventLoop`
        IO Loop for the coroutine

    Attributes
    ----------
    _args : :class:`argparse.Namespace`
        Command-line arguments passed to the constructor
    _loop : :class:`trollius.BaseEventLoop`
        Event loop passed to the constructor
    _file : :class:`h5py.File`
        Handle to HDF5 file
    _bf_raw_dataset : :class:`h5py.Dataset`
        Handle to the ``bf_raw`` dataset
    _timestamps : :class:`list`
        Timestamp for the start of each heap

    _manual_stop : :class:`bool`
        Whether :meth:`stop` has been called
    _ig : :class:`spead2.ItemGroup`
        Item group updated with each heap
    _stream : :class:`spead2.recv.trollius.Stream`
        SPEAD stream for incoming data
    _run_future : :class:`trollius.Task`
        Task for the coroutine that does the work
    _timestep : int
        Time interval (in ADC clocks) between spectra
    """
    def __init__(self, args, loop):
        _logger.info("Capture-init received")
        _logger.info("ARGS")
        _logger.info(args)
        self.args=args
        self._loop = loop
        self._args = args
        self._dada_dbdisk_process = None
        self._capture_process = None
        self.backend = None
        self.n_pol = None
        self._dada_dbdisk_process = None
        self._capture_process = None
        self._speadmeta_process = None
        self._digifits_process = None
        self._dspsr_process = None
        self._speadmeta_process = None
        self._dada_header_process = None

        _logger.info("Grabbing script_args")

        config = args.telstate.get('config')
        ant_mask = config['antenna_mask']
        ants = config['antenna_mask'].split(',')

        self.script_args = args.telstate.get('obs_script_arguments')
        self.driftscan = self.script_args['drift_scan']
        
        #script_args is only populated for beamformer observations.
        if self.script_args and self.script_args['backend'] != '':
            _logger.info ("Running with script args of %s"%str(self.script_args))
            pubsub_thread = PubSubThread(1,eval(config['stream_sources'])['cam.http']['camdata'],_logger,self,ants)
            pubsub_thread.start()
            backend = self.script_args['backend']
            bandwidth = self.script_args['beam_bandwidth']
            self.backend = self.script_args['backend']
            backend_args=self.script_args['backend_args']
            _logger.info("Bandwidth = %d"%bandwidth)
            if 'digifits' in backend and backend_args and '-p 4' in backend_args and bandwidth >= 428:
                _logger.info("Bandwidth set to 428 as this is the max bandwidth digifits can handle with p=4") 
                bandwidth = 428
            elif 'digifits' in backend and bandwidth >= 642:
                _logger.info("Bandwidth set to 642 as this is the max bandwidth digifits can handle with p=1")
                bandwidth = 642

            _logger.info("ben_y")
            beam_y_multicast = eval(config['stream_sources'])["cbf.tied_array_channelised_voltage"]['i0.tied-array-channelised-voltage.0y'].split(":")[1][2:]
            _logger.info("after")
            total_parts=int(beam_y_multicast.split('+')[-1])+1
            part_bandwidth=856.0/total_parts
            n_parts = bandwidth/part_bandwidth
            resolution = 4194304 * n_parts / total_parts
            _logger.info("BEFORE BUFFER")
            if self.backend in "dada_dbdisk": #Make massive RAM buffer, but first we need to increase RAM provided by Mesos
                self._create_dada_buffer(resolution*bandwidth/856*16*4, nBuffers=64)
            else:
                _logger.info("CREATING BUFFER")
                self._create_dada_buffer(resolution*bandwidth/856*16*4, nBuffers=64)
            _logger.info("Created dada_buffer")
            self._create_dada_header()
            if ("digifits" in self.backend):
                if (self.script_args['backend_args']):
                    self._create_digifits(args.affinity[0], backend_args=backend_args)
                else:
                    self._create_digifits(args.affinity[0])
            elif ("dspsr" in self.backend):
                if (backend_args):
                    self._create_dspsr(args.affinity[0], backend_args=backend_args)
                else:
                    self._create_dspsr(args.affinity[0])
            elif ("dada_dbdisk" in self.backend):
                self._create_dada_dbdisk()
            time.sleep(1)
            beam_x_multicast = eval(config['stream_sources'])["cbf.tied_array_channelised_voltage"]['i0.tied-array-channelised-voltage.0x'].split(":")[1][2:]
            beam_y_multicast = eval(config['stream_sources'])["cbf.tied_array_channelised_voltage"]['i0.tied-array-channelised-voltage.0y'].split(":")[1][2:]
            _logger.info("Subscribing to beam_x on %s"%beam_x_multicast)
            _logger.info("Subscribing to beam_y on %s"%beam_y_multicast)
            data_port = int(eval(config['stream_sources'])["cbf.tied_array_channelised_voltage"]['i0.tied-array-channelised-voltage.0x'].split(":")[-1])
            self._run_future = trollius.async(self._run(bandwidth=bandwidth, obs_length = self.script_args['target_duration'], centre_freq=self.script_args["beam_centre_freq"], targets=self.script_args["targets"], cores=args.affinity[1:], interface=args.interface, halfband=True, beam_x_multicast=beam_x_multicast, beam_y_multicast=beam_y_multicast, data_port=data_port), loop=self._loop)

    def _create_dada_buffer(self, bufsz, dadaId = 'dada', numaCore = 1, nBuffers =32):
        """Create the dada buffer. Must be run before capture and dbdisk.'

        Parameters
        ----------
        dadaId :
            Dada buffer ID.
        numaCore :
            NUMA node to attach dada buffer to
        """
        cmd = ['dada_db', '-k', dadaId, '-b', str(bufsz),'-c', '%d'%self.args.affinity[0],'-p','-l','-n', '%d'%nBuffers]
        dada_buffer_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )
        dada_buffer_process.wait()
        _logger.info("dada buffer creation output :")
        _logger.info(dada_buffer_process.communicate())

    def _create_dada_header(self):
        cmd = ['dada_header']
        self._dada_header_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def _create_dada_dbdisk (self, dadaId = 'dada', cpuCore = 1, outputDir = '/data'):
        """Create the dada_dbdisk process which writes data from dada buffer to disk.
        Must be run after creating the dada_buffer and before the capture process.
        Must have the same NUMA node as the dada_buffer

         Parameters
            ----------
            dadaId :
                Dada buffer ID.
            cpuCore :
                CPU core to bind processing to
            outputDir :
                Location to write captured data
        """
        _logger.info("dada_dbdisk")
        self.save_dir = "/data/%.0fdada"%time.time()
        os.mkdir("%s.writing"%self.save_dir)
        cmd = ['dada_dbdisk', '-k', dadaId, '-b','%d'%self.args.affinity[1], '-D', "%s.writing"%self.save_dir, '-z', '-s']
        self._dada_dbdisk_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )

    def get_digifits_args(self, backend_args):
        _logger.info("backend_args = %s"%backend_args)
        parser = argparse.ArgumentParser(description='Grab arguments')
        parser.add_argument('-t', dest='-t', type=float, help='integration time (s) per output sample (default=64mus)')
        parser.add_argument('-overlap', dest='-overlap', action='store_true', help='disable input buffering')
        parser.add_argument('-header', dest='-header', type=str, help='command line arguments are header values (not filenames)')
        parser.add_argument('-S', dest='-S', type=int, help='start processing at t=seek seconds')
        parser.add_argument('-T', dest='-T', type=int, help='process only t=total seconds')
        parser.add_argument('-set', dest='-set', type=str, help='key=value     set observation attributes')
        parser.add_argument('-r', dest='-r', action='store_true', help='report time spent performing each operation')
        parser.add_argument('-dump', dest='-dump', action='store_true', help='dump time series before performing operation')
        parser.add_argument('-D', dest='-D', type=float, help='set the dispersion measure')
        parser.add_argument('-do_dedisp', dest='-do_dedisp', action='store_true', help='enable coherent dedispersion (default: false)')
        parser.add_argument('-c', dest='-c', action='store_true', help='keep offset and scale constant')
        parser.add_argument('-I', dest='-I', type=int, help='rescale interval in seconds')
        parser.add_argument('-p', dest='-p', type=str, choices = ['1','4'], help='output 1 (Intensity), or 4 (Coherence) products')
        parser.add_argument('-b', dest='-b', type=str, choices = ['1','2','4','8'], help='number of bits per sample output to file [1,2,4,8]')
        parser.add_argument('-F', dest='-F', type=int, help='nchan[:D]     * create a filterbank (voltages only)')
        parser.add_argument('-nsblk', dest='-nsblk', type=int, help='output block size in samples (default=2048)')
        parser.add_argument('-k', dest='-K', action='store_true', help='remove inter-channel dispersion delays')
        opts = parser.parse_args (backend_args.split(" "))
        opts_list = [list(i) for i in  zip(vars(opts).keys(),vars(opts).values())]
        self.n_pol = "eish"
        return [str(item) for sublist in opts_list for item in sublist if (sublist[1] != None and sublist[1] != False and item != True)] 
                

    def _create_digifits (self, core, backend_args = '-t 0.000153121770088 -p 1'):
        passed_args = self.get_digifits_args(backend_args)
        self.save_dir = "/data/%.0fsf"%time.time()
        if '-p' in passed_args:
            self.n_pol= passed_args[passed_args.index('-p')+1]
        #if "profile" in self.backend:
        #    cmd =["nvprof","--analysis-metrics","--export-profile","%s.writing/digifits.nvprof"%self.save_dir,"numactl", "-C", "%i"%core, "digifits"] + passed_args + ["-v","-D","0","-c","-b","8","-v","-nsblk","256","-cuda","0","/home/kat/dada.info"]
        #else:
        cmd =["numactl", "-C", "%i"%core, "digifits"] + passed_args + ["-v","-D","0","-c","-b","8","-v","-nsblk","256","-cuda","0","/home/kat/dada.info"]
        os.mkdir("%s.writing"%self.save_dir)
        _logger.info("Starting digifits with args:")
        _logger.info(cmd)
        with open("%s.writing/digifits.log"%self.save_dir,"a") as logfile:
            _logger.info(cmd)
            self._digifits_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=logfile, stderr=logfile, cwd="%s.writing"%self.save_dir
            )

    def get_dspsr_args(self, backend_args):
        _logger.info(backend_args)
        if backend_args:
            parser = argparse.ArgumentParser(description='Grab arguments') 
            parser.add_argument('-overlap', dest='-overlap', action='store_true', help='disable input buffering')
            parser.add_argument('-header', dest='-header', type=str, help='command line arguments are header values (not filenames)')
            parser.add_argument('-S', dest='-S', type=int, help='start processing at t=seek seconds')
            parser.add_argument('-T', dest='-T', type=int, help='process only t=total seconds')
            parser.add_argument('-set', dest='-set', type=str, help='key=value     set observation attributes')
            parser.add_argument('-W', dest='-W', action='store_true', help='disable weights (allow bad data)')
            parser.add_argument('-r', dest='-r', action='store_true', help='report time spent performing each operation')
            parser.add_argument('-B', dest='-B', type=float, help='set the bandwidth in MHz')
            parser.add_argument('-f', dest='-f', type=float, help='set the centre frequency in MHz')
            parser.add_argument('-k', dest='-k', type=str, help='set the telescope name')
            parser.add_argument('-N', dest='-N', type=str, help='set the source name')
            parser.add_argument('-C', dest='-C', type=float, help='adjust clock byset the source name')
            parser.add_argument('-m', dest='-m', type=str, help='set the start MJD of the observation')
            parser.add_argument('-2', dest='-2', action='store_true', help='unpacker options ("2-bit" excision)')
            parser.add_argument('-skz', dest='-skz', action='store_true', help='apply spectral kurtosis filterbank RFI zapping')
            parser.add_argument('-noskz_too', dest='-noskz_too', action='store_true', help='also produce un-zapped version of output')
            parser.add_argument('-skzm', dest='-skzm', type=int, help='samples to integrate for spectral kurtosis statistics')
            parser.add_argument('-skzs', dest='-skzs', type=int, help='number of std deviations to use for spectral kurtosis excisions')
            parser.add_argument('-skz_start', dest='-skz_start', type=int, help='first channel where signal is expected')
            parser.add_argument('-skz_end', dest='-skz_end', type=int, help='last channel where signal is expected')
            parser.add_argument('-skz_no_fscr', dest='-skz_no_fscr', action='store_true', help=' do not use SKDetector Fscrunch feature')
            parser.add_argument('-skz_no_tscr', dest='-skz_no_tscr', action='store_true', help='do not use SKDetector Tscrunch feature')
            parser.add_argument('-skz_no_ft', dest='-skz_no_ft', action='store_true', help='do not use SKDetector despeckeler')
            parser.add_argument('-sk_fold', dest='-sk_fold', action='store_true', help='fold the SKFilterbank output')
            parser.add_argument('-F', dest='-F', type=str, help=' <N>[:D]          * create an N-channel filterbank')
            parser.add_argument('-G', dest='-G', type=int, help='nbin          create phase-locked filterbank')
            parser.add_argument('-cyclic', dest='-cyclic', type=int, help='form cyclic spectra with N channels (per input channel)')
            parser.add_argument('-cyclicoversample', dest='-cyclicoversample', type=int, help='use M times as many lags to improve cyclic channel isolation (4 is recommended)')
            parser.add_argument('-D', dest='-D', type=float, help='over-ride dispersion measure')
            parser.add_argument('-K', dest='-K', type=float, help='remove inter-channel dispersion delays')
            parser.add_argument('-d', dest='-d', type=int, choices=[1,2,3,4], help='1=PP+QQ, 2=PP,QQ, 3=(PP+QQ)^2 4=PP,QQ,PQ,QP')
            parser.add_argument('-n', dest='-n', action='store_true', help='[experimental] ndim of output when npol=4')
            parser.add_argument('-4', dest='-4', action='store_true', help='compute fourth-order moments')
            parser.add_argument('-b', dest='-b', type=int, help='number of phase bins in folded profile')
            parser.add_argument('-c', dest='-c', type=float, help='folding period (in seconds)')
            parser.add_argument('-cepoch', dest='-cepoch', type=str, help='MJD           reference epoch for phase=0 (when -c is used)')
            parser.add_argument('-p', dest='-p', type=float, help='reference phase of rising edge of bin zero')
            parser.add_argument('-E', dest='-E', type=str, help='pulsar ephemeris used to generate predictor')
            parser.add_argument('-P', dest='-P', type=str, help='phase predictor used for folding')
            parser.add_argument('-X', dest='-X', type=str, help='additional pulsar to be folded')
            parser.add_argument('-asynch-fold', dest='-asynch-fold', action='store_true', help='fold on CPU while processing on GPU')
            parser.add_argument('-A', dest='-A', action='store_true', help='output single archive with multiple integrations')
            parser.add_argument('-nsub', dest='-nsub', type=int, help='output archives with N integrations each')
            parser.add_argument('-s', dest='-s', action='store_true', help='create single pulse sub-integrations')
            parser.add_argument('-turns', dest='-turns', type=int, help='create integrations of specified number of spin periods')
            parser.add_argument('-L', dest='-L', type=float, help='create integrations of specified duration')
            parser.add_argument('-Lepoch', dest='-Lepoch', type=str, help='start time of first sub-integration (when -L is used)')
            parser.add_argument('-Lmin', dest='-Lmin', type=float, help='minimum integration length output')
            parser.add_argument('-y', dest='-y', action='store_true', help='output partially completed integrations')
            opts = parser.parse_args (backend_args.split(" "))
            opts_list = [list(i) for i in  zip(vars(opts).keys(),vars(opts).values())]
            return [str(item) for sublist in opts_list for item in sublist if (sublist[1] != None and sublist[1] != False and item != True)]
        else:
            return []

    def _create_dspsr(self, core, backend_args="-D 0 -L 10"):
        
        passed_args = self.get_dspsr_args(backend_args)
        passed_args.append(["-D","0","-Q","-cuda","0","/home/kat/dada.info"])

        self.save_dir = "/data/%.0far"%time.time()
        os.mkdir("%s.writing"%self.save_dir)
        #if "profile" in self.backend:
        #    cmd = ["nvprof","--analysis-metrics","--export-profile","%s.writing/dspsr.nvprof"%self.save_dir,"numactl","-C",str(core),"/usr/local/kat/pulsar/linux_64/bin/dspsr", "-D", "0", "-Q", "-minram", "512", "-L", "10", "-b", "1024", "-cuda", "0", "/home/kat/dada.info"]
        #else:
        cmd = ["numactl","-C",str(core),"/usr/local/kat/pulsar/linux_64/bin/dspsr", "-D", "0", "-Q", "-minram", "512", "-L", "10", "-b", "1024", "-cuda", "0", "/home/kat/dada.info"]
        with open("%s.writing/dspsr.log"%self.save_dir,"a") as logfile:
            _logger.info("Starting dspsr with args:")
            _logger.info(cmd)
            self._dspsr_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=logfile, stderr=logfile, cwd="%s.writing"%self.save_dir
            )

    @trollius.coroutine
    def _run(self, bandwidth, obs_length, centre_freq, targets, cores,interface, backend_args=None, halfband=False, interface_ip="10.100.205.11", beam_x_multicast="239.9.3.30", beam_y_multicast="239.9.3.31", data_port="7148"):
        """Does the work of capturing a stream. This is a coroutine."""
        import re
        content = ""

        total_parts=int(beam_y_multicast.split('+')[-1])+1
        part_bandwidth=856.0/total_parts
        part_chan=4096/total_parts
        part_bytes_p_s=3424000000.0/total_parts
        n_parts = bandwidth/part_bandwidth
        part_centre = (centre_freq)/part_bandwidth*part_bandwidth            
        resolution = 4194304 * n_parts / total_parts    
 
        c_file = '/home/kat/hardware_cbf_4096chan_2pol.cfg'

        import struct, socket
        ip2int = lambda ipstr: struct.unpack('!I', socket.inet_aton(ipstr))[0]
        int2ip = lambda n: socket.inet_ntoa(struct.pack('!I', n))

        bottom_chan_ip = (part_centre - 856 - bandwidth / 2) / part_bandwidth
        x_ip = int2ip(ip2int(beam_x_multicast.split('+')[0]) + bottom_chan_ip)
        y_ip = int2ip(ip2int(beam_y_multicast.split('+')[0]) + bottom_chan_ip)
        start_chan = bottom_chan_ip*part_chan
        end_chan = start_chan + n_parts * part_chan - 1
        replace = "ADC_SYNC_TIME       %s"%self.args.telstate.get("cbf_sync_time") 
        _logger.info ("Config file is %s"%c_file)

        with open('%s.template'%(c_file), 'r+') as content_file:
            content = content_file.read()
            insert = "BW                  %f"%(n_parts*part_bandwidth)
            content = re.sub("BW                  856", "BW                  %f"%(n_parts*part_bandwidth),content)
            content = re.sub("BYTES_PER_SECOND    3424000000.0","BYTES_PER_SECOND    %f"%(n_parts*part_bytes_p_s),content)
            content = re.sub("END_CHANNEL         4095","END_CHANNEL         %i"%end_chan, content)
            content = re.sub("START_CHANNEL       0","START_CHANNEL       %i"%start_chan,content)
            content = re.sub("NCHAN               4096","NCHAN               %i"%(n_parts*part_chan),content)
            content = re.sub("RESOLUTION          4194304","RESOLUTION          %i"%resolution, content)
            content = re.sub("ADC_SYNC_TIME",replace,content)
            content = re.sub("FREQ                1283.8955078125", "FREQ                %f"%part_centre, content)
            content = re.sub("DATA_HOST_0         10.100.205.10", "DATA_HOST_0         %s"%_get_interface_address(interface), content)
            content = re.sub("DATA_HOST_1         10.100.205.10", "DATA_HOST_1         %s"%_get_interface_address(interface), content)
            content = re.sub("DATA_MCAST_0        239.9.3.30", "DATA_MCAST_0        %s+%i"%(x_ip,n_parts-1), content)
            content = re.sub("DATA_MCAST_1        239.9.3.31", "DATA_MCAST_1        %s+%i"%(y_ip,n_parts-1), content)
            content = re.sub("SOURCE              J0835-4510", "SOURCE              %s"%targets[0], content)
            content = re.sub("DATA_PORT_0         7148", "DATA_PORT_0         %i"%data_port, content)
            content = re.sub("DATA_PORT_1         7148", "DATA_PORT_1         %i"%data_port, content)
            _logger.info ("Running with : \n%s"%content)
        with open (c_file, 'r+') as content_file:
            content_file.seek(0)
            content_file.write(content)
            content_file.truncate() 
       
        self.startTime=time.strftime("%Y-%m%dT%H:%M:%S")
        cap_env = os.environ.copy()

        cap_env["LD_PRELOAD"] = "libvma.so"
        cap_env["VMA_MTU"] = "9200"
        cap_env["VMA_RX_POLL_YIELD"] = "1"
        cap_env["VMA_RX_UDP_POLL_OS_RATIO"] = "0"
        self.capture_log = open("%s.writing/capture.log"%self.save_dir,"a")
        _logger.info("Capturing in /scratch/data/%s.writing"%self.save_dir[6:])
        
    def capture_start (self):

        cores=self.args.affinity[1:]
        config_file = "/home/kat/hardware_cbf_4096chan_2pol.cfg"
        cap_env = os.environ.copy()

        cap_env["LD_PRELOAD"] = "libvma.so"
        cap_env["VMA_MTU"] = "9200"
        cap_env["VMA_RX_POLL_YIELD"] = "1"
        cap_env["VMA_RX_UDP_POLL_OS_RATIO"] = "0"
        cmd = ["numactl","-C","%i,%i,%i"%(cores[0],cores[1],cores[2]),"meerkat_udpmergedb", "-b","%i,%i"%(cores[1],cores[2]),config_file,"-f", "spead"]
        self.capture_log = open("%s.writing/capture.log"%self.save_dir,"a")

        #Sleep to ensure data is flowing
        self._capture_process = subprocess.Popen(
        cmd, subprocess.PIPE, stdout=self.capture_log, stderr=self.capture_log, env=cap_env
        )

        _logger.info("Capture started with args:")
        _logger.info(cmd)
        time.sleep(int(self.script_args['target_duration']))
        _logger.info("Killing capture process")
        self._capture_process.send_signal(signal.SIGINT)
        self._capture_process.send_signal(signal.SIGINT)
         

    #@trollius.coroutine
    def stop(self):
        """Shut down the stream and wait for the coroutine to complete. This
        is a coroutine.
        """
        if self._capture_process is not None and self._capture_process.poll() is None:
            self._capture_process.send_signal(signal.SIGINT)
            _logger.info("Capture process still running, sending SIGINT")
        if self._capture_process is not None and self._capture_process.poll() is None:
            self._capture_process.send_signal(signal.SIGINT)
            _logger.info("Capture process still running, sending SIGINT")
        if  self._capture_process is not None and self._capture_process.poll() is None:
            self._capture_process.send_signal(signal.SIGINT)
            self._capture_process.send_signal(signal.SIGINT)
            self.capture_log.close()
        if self._dada_dbdisk_process is not None and self._dada_dbdisk_process.poll() is None:
            self._dada_dbdisk_process.send_signal(signal.SIGINT)
        count = 0
        while self._digifits_process is not None and self._digifits_process.poll() is None and count < 100:
            _logger.info("Digifits did not stop %i"%count)
            self._digifits_process.send_signal(signal.SIGINT)
            time.sleep(1)
            count+=1 
        if self._dspsr_process is not None and self._dspsr_process.poll() is None:
            self._dspsr_process.send_signal(signal.SIGTERM)
            _logger.info("dspsr did not stop")
        if self._dada_header_process: 
            comm = self._dada_header_process.communicate()
            try:
                dada_header = dict([d.split() for d in comm[0].split('\n')][:-1])
            except:
                _logger.info("Could not parse dada header:\n%s"%str(comm))

        if self.backend and (self.backend == 'digifits' or self.backend == 'dspsr'):
            try: 
                obs_info = open ("%s.writing/obs_info.dat"%self.save_dir, "w+")
                obs_info.write ("observer;%s\n"%self.script_args['observer'])
                obs_info.write ("program_block_id;%s\n"%self.script_args["program_block_id"])
                obs_info.write ("targets;%s\n"%self.script_args["targets"])
                obs_info.write ("sb_id_code;%s\n"%self.script_args["sb_id_code"])
                obs_info.write ("target_duration;%s\n"%self.script_args["target_duration"])
                obs_info.write ("proposal_id;%s\n"%self.script_args["proposal_id"])
                obs_info.write ("description;%s\n"%self.script_args["description"])
                obs_info.write ("backend_args;%s\n"%self.script_args["backend_args"])
                obs_info.write ("experiment_id;%s\n"%self.script_args["experiment_id"])
                obs_info.write ("PICOSECONDS;%s\n"%dada_header["PICOSECONDS"])
                obs_info.write ("UTC_START;%s\n"%dada_header["UTC_START"])
                obs_info.close() 
            except:

                _logger.info("Obs info file not correctly filled")

        if self.backend and self.backend == 'digifits':
            
            data_files = os.listdir("%s.writing"%self.save_dir)
            index = 0
            while index < len(data_files) and data_files[index][-2:] != 'sf':
                index+=1

            try:
                data = pyfits.open("%s.writing/%s"%(self.save_dir, data_files[index]), mode="update", memmap=True, save_backup=False)
                target = [s.strip() for s in self.args.telstate.get("data_target").split(",")]
                hduPrimary=data[0].header
                start_time = Time([hduPrimary['STT_IMJD'] + hduPrimary['STT_SMJD'] / 3600.0 / 24.0],format='mjd')
                start_time.format = 'isot'
                self.startTime=start_time.value[0][:-3] + str(hduPrimary['STT_OFFS'])[2:]
                 

                _logger.info(target)
                _logger.info(self.startTime.split('.')[0])
                hduPrimary["PROJID"]=self.script_args['sb_id_code']
                hduPrimary["OBSERVER"]=self.script_args['observer']
                hduPrimary["ANT_X"]=5109318.8410
                hduPrimary["ANT_Y"]=2006836.3673
                hduPrimary["ANT_Z"]=-3238921.7749
                hduPrimary["BACKEND"]="BFI1"
                hduPrimary["TELESCOP"]="MEERKAT"
                hduPrimary["FRONTEND"]="L-BAND"
                hduPrimary["RA"]=target[-2]
                hduPrimary["DEC"]=target[-1]
                hduPrimary["STT_CRD1"]=target[-2]
                hduPrimary["STT_CRD2"]=target[-1]
                hduPrimary["STP_CRD1"]=target[-2]
                hduPrimary["STP_CRD2"]=target[-1]
                hduPrimary["TRK_MODE"]="TRACK"
                hduPrimary["OBS_MODE"]="SEARCH"
                hduPrimary["TCYCLE"]=0
                hduPrimary["ANT_X"]=5109318.8410
                hduPrimary["ANT_Y"]=2006836.3673
                hduPrimary["ANT_Z"]=-3238921.7749
                hduPrimary["NRCVR"]=2
                hduPrimary["CAL_MODE"]="OFF"
                hduPrimary["CAL_FREQ"]=0.0
                hduPrimary["CAL_DCYC"]=0.0
                hduPrimary["CAL_PHS"]=0.0
                hduPrimary["CAL_NPHS"]=0.0
                hduPrimary["CHAN_DM"] = 0.0
                hduPrimary["DATE-OBS"]=self.startTime.split('.')[0]
                hduPrimary["DATE"]=self.startTime
                hduPrimary["SCANLEN"]=self.script_args['target_duration']
                
                hduSubint = data[2].header
                hduSubint["NPOL"]=self.n_pol
                _logger.info("n_pol = %s"%self.n_pol)
                if int(self.n_pol) == 1:
                    hduSubint["POL_TYPE"]="AA+BB"
                if int(self.n_pol) == 4:
                    hduSubint["POL_TYPE"]="AABBCRCI"
                hduSubint["NCHNOFFS"] = 0 
                hduSubint["NSUBOFFS"] = 0
                _logger.info("Updated fits headers")
            except:
                _logger.info("Oh no failed header update")
        try:
            os.rename("%s.writing"%self.save_dir,self.save_dir)
        except:
            _logger.info ("No save_dir, never started capturing data")
        cmd = ['dada_db', '-d']
        dada_buffer_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )
        capturing = False

        _logger.info("KILLED ALL CAPTURE PROCESSES")


class CaptureServer(object):
    """Beamformer capture. This contains all the core functionality of the
    katcp device server, without depending on katcp. It is split like this

                              Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments. The following arguments are required:

        cbf_channels
          Total number of PFB channels, or ``None`` for unknown
        cbf_spead
          ist of :class:`katsdptelstate.endpoint.Endpoint` for receiving SPEAD data
        file_base
          Directory in which to write files
    loop : :class:`trollius.Baseos-3c52c442-e322-41af-8b6c-0381856777f5-S0.702cdbcb-05dc-4084-a2ea-b9b87489c386ventLoop`
        IO Loop for running coroutines

    Attributes
    ----------
    capturing : :class:`bool`
        Whether a capture session is in progress. Note that a session is
        considered to be in progress until explicitly stopped with
        :class:`stop_capture`, even if the stream has terminated.
    _args : :class:`argparse.Namespace`
        Command-line arguments passed to constructor
    _loop : :class:`trollius.BaseEventLoop`
        IO Loop passed to constructor
    _capture : :class:`_CaptureSession`
        Current capture session, or ``None`` if not capturing
    """
    def __init__(self, args, loop):
        self._args = args
        self._loop = loop
        self._capture = None

    @property
    def capturing(self):
        return self._capture is not None

    def start_capture(self):
        """Start capture to file, if not already in progress."""
        if self._capture is None:
            try:
                self._capture = _CaptureSession(self._args, self._loop)
            except Exception as e:
                print ("Exception caught " + str(e))
                #self.stop_capture()

    def capture_start(self):
        if self._capture is not None:
            try:
                self._capture.capture_start()
            except Exception as e:
                print ("Exception caught " + str(e))
                                                        
    #@trollius.coroutine
    def stop_capture(self):
        """Stop capture to file, if currently running. This is a co-routine."""
        if self._capture is not None:
            self._capture.stop()
            self._capture = None


class KatcpCaptureServer(CaptureServer, katcp.DeviceServer):
    """katcp device server for beamformer capture.

    Parameters
    ----------4194304 * n_parts / total_parts
    args : :class:`argparse.Namespace`
        Command-line arguments (see :class:`CaptureServer`).
        The following additional arguments are required:

        host
          Hostname to bind to ('' for none)
        port
          Port number to bind to
    loop : :class:`trollius.BaseEventLoop`
        IO Loop for running coroutines
    """

    VERSION_INFO = ('ptuse-ingest', 1, 0)
    BUILD_INFO = ('katsdpingest',) + tuple(katsdpingest.__version__.split('.', 1)) + ('',)

    def __init__(self,
 args, loop):
        CaptureServer.__init__(self, args, loop)
        katcp.DeviceServer.__init__(self, args.host, args.port)

    def setup_sensors(self):
        pass

    @request(Str(optional=True))
    @return_reply()
    def request_capture_init(self, sock, program_block_id=None):
        """Start capture to file."""
        if self.capturing:
            return ('fail', 'already capturing')
        try:
            self.start_capture()
        except Exception as e:
            return ('fail',e.message)
        return ('ok',)

    #@tornado.gen.coroutine
    def _stop_capture(self):
        """Tornado variant of :meth:`stop_capture`"""
        #stop_future = trollius.async(self.stop_capture(), loop=self._loop)
        #yield tornado.platform.asyncio.to_tornado_future(stop_future)
        self.stop_capture()

    @request()
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_done(self, sock):
        """Stop a capture that is in progress."""
        self._stop_capture()
        raise tornado.gen.Return(('ok',))

    @request()
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_start(self, sock):
        """Start a capture process."""
        self.capture_start()
        raise tornado.gen.Return(('ok',))


    @tornado.gen.coroutine
    def stop(self):
        yield self._stop_capture()
        yield katcp.DeviceServer.stop(self)

    stop.__doc__ = katcp.DeviceServer.stop.__doc__

import tornado.gen
from katportalclient import KATPortalClient
import logging
import threading
import time

class PubSubThread (threading.Thread):

  def __init__ (self, id, server, _logger, ingest,ants):

    _logger.info("CAM server = %s"%server)
    threading.Thread.__init__(self)
    self.ingest = ingest
    self.ants = dict(zip(ants,[0]*len(ants)))

    self.metadata_server = server
    self.logger = _logger
    self.logger.info(server)
    self.io_loop = []
    self.policy = "event-rate 1.0 300.0"
    self.title  = "bf_unconfigured"
    self.running = False
    self.restart_io_loop = True
    self.tracks = 0

  def run (self):

    while self.restart_io_loop:

      # open connection to CAM
      self.io_loop = tornado.ioloop.IOLoop()
      self.io_loop.add_callback (self.connect, self.logger)
      self.running = True
      self.restart_io_loop = False
      self.logger.info("starting io loop")
      self.io_loop.start()
      self.running = False
      self.io_loop = []
      self.ws_client.unsubscribe(self.title)
      self.ws_client.disconnect()

  def join (self):
    self.stop()
    time.sleep(0.1)

  def stop (self):
    self.logger.info("Stopping pubsub")
    if self.running:
      self.running = False
      self.io_loop.stop()
    return

  def restart (self):
    self.restart_io_loop = True
    if self.running:
      self.stop()
    return

  @tornado.gen.coroutine
  def connect (self, logger):
    self.ws_client = KATPortalClient(self.metadata_server, self.on_update_callback, logger=logger)
    yield self.ws_client.connect()
    result = yield self.ws_client.subscribe(self.title)

    list = ['%s.activity'%ant for ant in self.ants.keys()]
    self.logger.info(list)
    results = yield self.ws_client.set_sampling_strategies(
        self.title, list, self.policy)

  def on_update_callback (self, msg):
    self.logger.info(msg)
    self.logger.info(self.ants)
    if msg['msg_data']['value'] == 'track':
      self.logger.info(msg['msg_data']['name'][:4])
      self.logger.info(self.ants[msg['msg_data']['name'][:4]])
      self.ants[msg['msg_data']['name'][:4]] +=1
      self.logger.info(self.ants[msg['msg_data']['name'][:4]])
      self.tracks=sum(self.ants.values())/len(self.ants)

    if msg['msg_data']['value'] == 'track' and self.ingest._capture_process is None and self.tracks == 2 and hasattr(self.ingest, 'driftscan') and self.ingest.driftscan:
      self.logger.info("STARTING CAPTURE")
      self.ingest.capture_start()
      self.join() 
    elif msg['msg_data']['value'] == 'track' and self.ingest._capture_process is None and self.tracks == 1 and hasattr(self.ingest, 'driftscan') and self.ingest.driftscan == False:
      self.logger.info("STARTING CAPTURE")
      self.ingest.capture_start()
      self.join()
    elif msg['msg_data']['value'] == 'track' and self.tracks > 2:
      self.logger.info("Terminate")
      self.join()

__all__ = ['CaptureServer', 'KatcpCaptureServer']
