from __future__ import print_function, division, absolute_import
import spead2
import spead2.recv
import spead2.recv.trollius
import katcp
from katcp.kattypes import request, return_reply
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
        script_args = args.telstate.get('obs_script_arguments')
        self.run = False
        self.args=args
        self._loop = loop
        self._args = args
        self._dada_dbdisk_process = None
        self._capture_process = None
        self._timestamps = []
        self._manual_stop = False
        self._endpoints = []
        self.cpture_log = None
        self._timestep = None
        self._ig = spead2.ItemGroup()
        self._stream = spead2.recv.trollius.Stream(spead2.ThreadPool(), 0, loop=self._loop)
        for endpoint in args.cbf_spead:
            _logger.info('Listening on endpoint {}'.format(endpoint))
            _logger.info(endpoint.host)

        self._dada_dbdisk_process = None
        self._capture_process = None
        self._speadmeta_process = None
        self._digifits_process = None
        self._dspsr_process = None
        self._speadmeta_process = None
        self._dada_header_process = None

        self._create_dada_buffer()
        self._create_dada_header()
        _logger.info("Created dada_buffer\n")
        if script_args:
            backend = script_args['backend']
            self.backend = script_args['backend']
            _logger.info(backend)
            if (backend in "digifits"):
                if (script_args['backend_args']):
                    self._create_digifits(backend_args=script_args['backend_args'])
                else:
                    self._create_digifits()
            elif (backend in "dspsr"):
                if (script_args['backend_args']):
                    self._create_dspsr(backend_args=script_args['backend_args'])
                else:
                    self._create_dspsr()
            elif (backend in "dada_dbdisk"):
                self._create_dada_dbdisk()
            time.sleep(1)
            #_logger.info(args.telstate.get(script_args))
            #_logger.info(args.telstate.get('config'))
            #_logger.info(args)
            beam_x_multicast = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speadx'].split(":")[0]
            beam_y_multicast = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speady'].split(":")[0]
            _logger.info(beam_x_multicast)
            _logger.info(beam_y_multicast)
            data_port = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speady'].split(":")[1]
            if (args.halfband or backend in "digifits"):
                self._run_future = trollius.async(self._run(obs_length = script_args['target_duration'], centre_freq=script_args["beam_centre_freq"], targets=script_args["targets"], halfband=True, beam_x_multicast=beam_x_multicast, beam_y_multicast=beam_y_multicast, data_port=data_port), loop=self._loop)
            else:
                self._run_future = trollius.async(self._run(obs_length = script_args['target_duration'], centre_freq=script_args["beam_centre_freq"], targets=script_args["targets"], beam_x_multicast=beam_x_multicast, beam_y_multicast=beam_y_multicast, data_port=data_port), loop=self._loop)

    def _create_dada_buffer(self, dadaId = 'dada', numaCore = 1, nBuffers =64):
        """Create the dada buffer. Must be run before capture and dbdisk.'

        Parameters
        ----------
        dadaId :
            Dada buffer ID.
        numaCore :
            NUMA node to attach dada buffer to
        """
        cmd = ['dada_db', '-k', dadaId, '-b', '268435456','-c', '%d'%numaCore, '-p', '-l', '-n', '%d'%nBuffers]
        dada_buffer_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )
        print ('creating buffers')
        dada_buffer_process.wait()
        print ('complete buffer creation')
        print (dada_buffer_process.communicate())

    def _create_dada_header(self):
        cmd = ['dada_header']
        self._dada_header_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def _create_dada_dbdisk (self, dadaId = 'dada', cpuCore = 0, outputDir = '/data'):
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
        cmd = ['dada_dbdisk', '-k', dadaId, '-c','%d'%cpuCore, '-D', outputDir, '-z', '-s']
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
        parser.add_argument('-p', dest='-p', type=str, choices = ['1','2','4'], help='output 1 (Intensity), 2 (AABB), or 4 (Coherence) products')
        parser.add_argument('-b', dest='-b', type=str, choices = ['1','2','4','8'], help='number of bits per sample output to file [1,2,4,8]')
        parser.add_argument('-F', dest='-F', type=int, help='nchan[:D]     * create a filterbank (voltages only)')
        parser.add_argument('-nsblk', dest='-nsblk', type=int, help='output block size in samples (default=2048)')
        parser.add_argument('-k', dest='-K', action='store_true', help='remove inter-channel dispersion delays')
        opts = parser.parse_args (backend_args.split(" "))
        opts_list = [list(i) for i in  zip(vars(opts).keys(),vars(opts).values())]
        _logger.info(opts_list)
        #for i in opts_list:
        #  if i[1] and i[0] != '-p':
        #      _logger.info(i)
        #      i[1] = '' 
        return [str(item) for sublist in opts_list for item in sublist if (sublist[1] != None and sublist[1] != False and item != True)] 
                

    def _create_digifits (self, backend_args = '-t 0.000153121770088 -p 1 -c'):
        _logger.info("digifits")
        passed_args = self.get_digifits_args(backend_args)
        cmd =["taskset", "7", "digifits"] + passed_args + ["-D","0","-b","8","-v","-nsblk","128","-cuda","0","/home/kat/dada.info"]
        self.save_dir = "/data/%.0fsf"%time.time()
        os.mkdir(self.save_dir)
        #_logger.info(passed_args)
        #_logger.info(cmd)
        with open("/tmp/digifits.log","a") as logfile:
            _logger.info(cmd)
            self._digifits_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=logfile, stderr=logfile, cwd=self.save_dir
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
            _logger.info(opts_list)
            #for i in opts_list:
            #   if i[1] == True and i[1] != 1:
            #       _logger.info(i)
            #       i[1] = ''
            return [str(item) for sublist in opts_list for item in sublist if (sublist[1] != None and sublist[1] != False and item != True)]
        else:
            return []

    def _create_dspsr(self, backend_args="-D 0 -L 10"):
        _logger.info("dspsr")
        
        passed_args = self.get_dspsr_args(backend_args)
        _logger.info(passed_args)
        passed_args.append(["-D","0","-Q","-cuda","0","/home/kat/dada.info"])
        _logger.info (passed_args)
        _logger.info(passed_args)

        self.save_dir = "/data/%.0far"%time.time()
        os.mkdir(self.save_dir)
 
        with open("/tmp/dspsr.log","a") as logfile:
            cmd = ["taskset","7","dspsr"] + passed_args + ["-cuda","0","/home/kat/dada.info"]
            cmd = ["dspsr","-t","2","-D","0","-Q","-L","10","-cuda","0","/home/kat/dada.info"]
            _logger.info(cmd)
            self._dspsr_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=logfile, stderr=logfile, cwd=self.save_dir
            )

    def _create_metaspead (self, pol_h_host = "10.100.205.11", pol_h_mcast = "239.9.3.30", pol_h_port = 7148):
        print ("IN METADATA")
        cmd = ["meerkat_speadmeta", pol_h_host, pol_h_mcast, "-p", "%d"%pol_h_port]

        speadmeta_process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


        speadmeta_process.wait()
        # we need to capture the output from this process
        adc_sync_time, error = speadmeta_process.communicate()
        print ("meta spead run")
        # next we need to put this ADC_SYNC_TIME into the spipConfig file provided
        runtimeConfig = spipConfig + ".live"
        f_read = open (spipConfig, 'r')


        f_write = open (spipConfig, 'w')
        for line in f_read:
            if line.find("ADC_SYNC_TIME"):
                f_write.write("ADC_SYNC_TIME %s\n"%(adc_sync_time))
            else:
                f_write.write(line + "\n")
        f_read.close()
        f_write.close()
        print ("meta complete with ts = %d"%adc_sync_time)



    def _create_capture_process(self, spipConfig="/home/kat/hardware_cbf_4096chan_2pol.cfg", core1=3, core2=5):
        """Create the beamformer capture process which writes data from the NIC to dada.
        Must be run after creating the dada_buffer and dada_dbdisk.
        Must have the same NUMA node as the dada_buffer

         Parameters
            ----------
            spipConfig :
                Location of the SPIP config file.
            core1 :
                core to run capture on
            core2 :
                core to run capture on
        """
        print ("capture_process+++++++______!!")
        self._create_metaspead()
        cap_env = os.environ.copy()
        _logger.info(cap_env)

        cap_env["LD_PRELOAD"] = "libvma.so"
        cap_env["VMA_MTU"] = "9200"
        cap_env["VMA_RX_POLL_YIELD"] = "1"
        cap_env["VMA_RX_UDP_POLL_OS_RATIO"] = "0"
        cap_env["VMA_RING_ALLOCATION_LOGIC_RX"]="1"                                                   
        cmd = ["meerkat_udpmergedb", "runtimeConfig", "-f", "spead", "-b", "%d"%core1, "-c" "%d"%core2]

        self._capture_process = subprocess.Popen(
        cmd, subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=cap_env
        )

    @trollius.coroutine
    def _run(self, obs_length, centre_freq, targets, backend_args=None, halfband=False, interface_ip="10.100.205.11", beam_x_multicast="239.9.3.30", beam_y_multicast="239.9.3.31", data_port="7148"):
        """Does the work of capturing a stream. This is a coroutine."""
        _logger.info("running")
        print ("capture_process+++++++______!!")
        _logger.info("IN METADATA")
        cmd = ["meerkat_speadmeta", _get_interface_address("p4p1"), beam_y_multicast, "-p", data_port]
        self.run = True
        #keys = [kv[0].split(' ', 1)[0] for kv in args.telstate.get_range('obs_params', st=0)]
        #values = [eval(kv[0].split(' ', 1)[1]) for kv in t.get_range('obs_params', st=0)]
        #obs_params = dict(zip(keys, values))
        #_logger.info(obs_params)

        with open("/tmp/metadata.log","w") as logfile: 
            self._speadmeta_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=logfile, stderr=logfile
            )

        #error, adc_sync_time = self._speadmeta_process.communicate()
        #self._speadmeta_process = None
        #_logger.info((error,adc_sync_time))
        content = ""
        with open("/tmp/metadata.log","r") as metafile:
            content = metafile.read()
            _logger.info("content is %s"%content)
            count = 0
            while count < 1000 and content == "":
                count+=1
                time.sleep(0.1)
                content = metafile.read()
                _logger.info("content is %s"%content)
            #print(self._speadmeta_process.poll())

        if content == "":
            self._speadmeta_process.kill()
            self._speadmeta_process = None
            _logger.info("metadata packet did not arrive by timeout")
            return
        _logger.info("passed exit")
        #_logger.info(dir(self._metadata_process))

        #error, adc_sync_time = self._speadmeta_process.communicate()
        #self._speadmeta_process = None
        split_content = content.splitlines()
        _logger.info(content)
        adc_sync_time = split_content[-1]
        _logger.info(adc_sync_time)
        #adc_sync_time = content

        #keys = [kv[0].split(' ', 1)[0] for kv in t.get_range('obs_params', st=0)]
        #values = [eval(kv[0].split(' ', 1)[1]) for kv in t.get_range('obs_params', st=0)]
        #obs_params = dict(zip(keys, values))


        import re
        replace = "ADC_SYNC_TIME %s"%adc_sync_time.strip()
        content = ""
        if halfband:
            c_file = '/home/kat/hardware_cbf_2048chan_2pol.cfg'
        else:
            c_file = '/home/kat/hardware_cbf_4096chan_2pol.cfg'
        
        _logger.info ("c_file = %s"%c_file)
        with open('%s.template'%(c_file), 'r+') as content_file:
            content = content_file.read()
            print (content)
            content = re.sub("ADC_SYNC_TIME",replace,content)
            content = re.sub("FREQ                1283.8955078125", "FREQ                %f"%centre_freq, content)
            content = re.sub("DATA_HOST_0         10.100.205.11", "DATA_HOST_0         %s"%_get_interface_address("p4p1"), content)
            content = re.sub("DATA_HOST_1         10.100.205.11", "DATA_HOST_1         %s"%_get_interface_address("p4p1"), content)
            content = re.sub("DATA_MCAST_0        10.100.205.10", "DATA_MCAST_0        %s"%beam_x_multicast, content)
            content = re.sub("DATA_MCAST_1        10.100.205.10", "DATA_MCAST_1        %s"%beam_y_multicast, content)
            content = re.sub("SOURCE              J0835-4510", "SOURCE              %s"%targets[0], content)
            _logger.info (content)
        _logger.info('yo')
        with open (c_file, 'r+') as content_file:
            print(0)
            content_file.seek(0)
            print (2)
            content_file.write(content)
            print (3)
            content_file.truncate() 
            _logger.info("written meta")
       

        #count = 0
        #while count < 1000 and not self._speadmeta_process.poll():
        #    count+=1
        #    time.sleep(0.1)
            
        #if not self._metadata_process.poll():
        #    self._metadata_process.kill()
        #    self._metadata_process = None
        #    print ("metadata packet did not arrive by timeout")
        #    return
 
        _logger.info("meta complete with ts = %s"%adc_sync_time)
        self._speadmeta_process = None
        self.startTime=time.strftime("%Y-%m%dT%H:%M:%S")
        cap_env = os.environ.copy()

        cap_env["LD_PRELOAD"] = "libvma.so"
        cap_env["VMA_MTU"] = "9200"
        cap_env["VMA_RX_POLL_YIELD"] = "1"
        cap_env["VMA_RX_UDP_POLL_OS_RATIO"] = "0"
        cmd = ["meerkat_udpmergedb", c_file, "-f", "spead", "-b", "%d,%d"%(3,5)]
        self.capture_log = open("/tmp/capture.log","a")
        self._capture_process = subprocess.Popen(
        cmd, subprocess.PIPE, stdout=self.capture_log, stderr=self.capture_log, env=cap_env
        )

        _logger.info("capture started_________________------12-----!!")
        #import signal
        _logger.info ("obs_length")
        _logger.info(obs_length - 5)
        #time.sleep(int(obs_length-2))
        #_logger.info("kill cap--------------11!!--------------------")
        #time.sleep(1)
        #if self._capture_process.poll() is None:
        #    self._capture_process.send_signal(signal.SIGINT)
        #_logger.info("kill cap--------------11--------------------")
        #time.sleep(1)
        #if self._capture_processi.poll() is None:
        #    self._capture_process.send_signal(signal.SIGINT)
 
        #if self._capture_process.poll() is None:
        #    self._capture_process.kill()
        #    self._capture_process.kill()

    #@trollius.coroutine
    def stop(self):
        """Shut down the stream and wait for the coroutine to complete. This
        is a coroutine.
        """
        #import signal
        _logger.info("--------------------------------------------")
        #_logger.info(dir(self.args.telstate))
        #for k in self.args.telstate.keys():
        #    _logger.info('%s, %s'%(str(k),str(self.args.telstate.get(k))))
        #_logger.info(self.args.telstate.get('config'))
        #_logger.info(self.args)
        print ("STOPPING")
        self._manual_stop = True
        if  self._capture_process is not None and self._capture_process.poll() is None:
            self._capture_process.send_signal(signal.SIGINT)
            _logger.info("kill cap----------------------------------")
            self._capture_process.send_signal(signal.SIGINT)
            _logger.info(self._capture_process.communicate())
            self.capture_log.close()
        time.sleep(5)
        if self._dada_dbdisk_process is not None and self._dada_dbdisk_process.poll() is None:
            self._dada_dbdisk_process.send_signal(signal.SIGINT)
        if self._digifits_process is not None and self._digifits_process.poll() is None:
            #self._digifits_process.send_signal(signal.SIGINT)
            #_logger.info(self._digifits_process.communicate())
            _logger.info("digifits still running")
        if self._dspsr_process is not None and self._dspsr_process.poll() is None:
            self._dspsr_process.send_signal(signal.SIGINT)
            _logger.info(self._dspsr_process.communicate())
        if self._speadmeta_process is not None and self._speadmeta_process.poll() is None:
            self._speadmeta_process.send_signal(signal.SIGINT)
        cmd = ['dada_db', '-d']
        dada_buffer_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )
        _logger.info(dada_buffer_process.communicate())
        _logger.info(self._dada_header_process.communicate())
        if self.run and self.backend == 'digifits':
            data_files = os.listdir(self.save_dir)
            _logger.info(data_files)
            data = pyfits.open("%s/%s"%(self.save_dir, data_files[0]), mode="update", memmap=True, save_backup=False)
            target = [s.strip() for s in self.args.telstate.get("data_target").split(",")]
            _logger.info(target)
            hduPrimary=data[0].header
            hduPrimary["TELESCOP"]="MeerKAT"
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
            hduPrimary["NRCVR"]=0
            hduPrimary["CAL_MODE"]="OFF"
            hduPrimary["CAL_FREQ"]=0.
            hduPrimary["CAL_DCYC"]=0.
            hduPrimary["CAL_PHS"]=0.
            hduPrimary["CAL_NPHS"]=0
            hduPrimary["CHAN_DM"] = 0.0
            hduPrimary["DATE-OBS"]=self.startTime
            hduPrimary["DATE"]=self.startTime
            
            hduSubint = data[2].header
            hduSubint["NPOL"]=1
            hduSubint["POL_TYPE"]="AA+BB"
            hduSubint["NCHNOFFS"] = 0 

        _logger.info("KILLED ALL CAPTURE PROCESSES")
        #yield From(self._run_future)


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
    loop : :class:`trollius.BaseEventLoop`
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
                                                        
    #@trollius.coroutine
    def stop_capture(self):
        """Stop capture to file, if currently running. This is a co-routine."""
        if self._capture is not None:
            self._capture.stop()
            self._capture = None


class KatcpCaptureServer(CaptureServer, katcp.DeviceServer):
    """katcp device server for beamformer capture.

    Parameters
    ----------
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

    def __init__(self, args, loop):
        CaptureServer.__init__(self, args, loop)
        katcp.DeviceServer.__init__(self, args.host, args.port)

    def setup_sensors(self):
        pass

    @request()
    @return_reply()
    def request_capture_init(self, sock):
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
        if not self.capturing:
            raise tornado.gen.Return(('fail', 'not capturing'))
        self._stop_capture()
        raise tornado.gen.Return(('ok',))

    @tornado.gen.coroutine
    def stop(self):
        yield self._stop_capture()
        yield katcp.DeviceServer.stop(self)

    stop.__doc__ = katcp.DeviceServer.stop.__doc__


__all__ = ['CaptureServer', 'KatcpCaptureServer']
                                                          
                                      
