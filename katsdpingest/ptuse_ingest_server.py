from __future__ import print_function, division, absolute_import
import spead2
import spead2.recv
import spead2.recv.trollius
import katcp
from katcp.kattypes import request, return_reply
#import h5py
#import numpy as np
#import time
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

        #beam_x_multicast = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speadx'].split(":")[0]
        #_logger.info(beam_x_multicast)
        #beam_y_multicast = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speady'].split(":")[0]
        #_logger.info(beam_y_multicast)
        #data_port = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speady'].split(":")[1]
        #_logger.info(data_port)

        #_logger.info("IN METADATA")
        #cmd = ["meerkat_speadmeta", "10.100.205.11", beam_x_multicast, "-p", data_port]

        #self._speadmeta_process = subprocess.Popen(
        #cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        #)


        #speadmeta_process.wait(timeout=10)
        # we need to capture the output from this process
        #error, adc_sync_time = self._speadmeta_process.communicate()
        #_logger.info((error,adc_sync_time))
        #_logger.info("meta spead run")


        _logger.info(_get_interface_address("p4p1"))

        _logger.info(args)
        _logger.info(args.telstate.keys())
#        antenna_mask', 'cal_bp_solint', 'cal_g_solint', 'cal_k_chan_sample', 'cal_k_solint', 'cal_refant', 'config', 'sdp_cbf_channels', 'sdp_node_detail', 'subarray_product_id
        #_logger.info(args.telstate.get('antenna_mask'))
        _logger.info(args.telstate.get('obs_script_arguments'))
        script_args = args.telstate.get('obs_script_arguments')
        _logger.info(type(script_args)) 
        _logger.info("YOLO-----------------------")
        _logger.info(script_args['backend'])
        #_logger.info(args.telstate.get('config'))
        #_logger.info(args.telstate.get('sdp_cbf_channels'))
        #_logger.info(args.telstate.get('sdp_ndoe_detail'))
        #_logger.info(args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speady'])
        #_logger.info(args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speadx'])
        #_logger.info('YOLO')
        #_logger.info(args.telstate.get('config')['sdp']['filewriter']['1']['telstate'])
        #t = katsdptelstate.TelescopeState(args.telstate.get('config')['sdp']['filewriter']['1']['telstate'])
        #_logger.info('YOHO')
        #_logger.info(t.get_range('obs_params', st=0))
        #_logger.info(args.telstate.getRange('obs_params',st=0))
        #print (sys.version_info)
        self._loop = loop
        self._args = args
        self._dada_dbdisk_process = None
        self._capture_process = None
        self._timestamps = []
        self._manual_stop = False
        self._endpoints = []
        #if args.cbf_channels:
        #    self._timestep = 2 * args.cbf_channels
        #else:
        self._timestep = None
        self._ig = spead2.ItemGroup()
        self._stream = spead2.recv.trollius.Stream(spead2.ThreadPool(), 0, loop=self._loop)
        for endpoint in args.cbf_spead:
            _logger.info('Listening on endpoint {}'.format(endpoint))
            _logger.info(endpoint.host)
        #    self._endpoints.append(endpoint)

        self._dada_dbdisk_process = None
        self._capture_process = None
        self._speadmeta_process = None
        self._digifits_process = None
        self._dspsr_process = None
        self._speadmeta_process = None

        self._create_dada_buffer()
        _logger.info("Created dada_buffer\n")
        backend = script_args['backend']
        _logger.info(backend)
        #print (args.backend)
        #print (type(args.backend))
        if (backend in "digifits"):
            self._create_digifits()
        elif (backend in "dspsr"):
            self._create_dspsr()
        elif (backend in "dada_dbdisk"):
            self._create_dada_dbdisk()
        _logger.info ("target_duration")
        _logger.info (script_args['target_duration'])
        time.sleep(1)
        #_run(self, halfband=False, interface_ip="10.100.205.11", beam_x_multicast="239.9.3.30", beam_y_multicast="239.9.3.31", data_port="7148"):
        beam_x_multicast = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speadx'].split(":")[0]
        _logger.info(beam_x_multicast)
        beam_y_multicast = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speady'].split(":")[0]
        _logger.info(beam_y_multicast)
        data_port = args.telstate.get('config')['cbf']['bf_output']['1']['cbf_speady'].split(":")[1]
        _logger.info(data_port)
        #_logger.info(args.telstate.getRange('obs_params',st=0))
        if (args.halfband or backend == "digifits"):
            self._run_future = trollius.async(self._run(obs_length = script_args['target_duration'], centre_freq=script_args["beam_centre_freq"], targets=script_args["targets"], halfband=True, beam_x_multicast=beam_x_multicast, beam_y_multicast=beam_y_multicast, data_port=data_port), loop=self._loop)
        else:
            self._run_future = trollius.async(self._run(obs_length = script_args['target_duration'], centre_freq=script_args["beam_centre_freq"], targets=script_args["targets"], beam_x_multicast=beam_x_multicast, beam_y_multicast=beam_y_multicast, data_port=data_port), loop=self._loop)

        #if self._dada_dbdisk_process == None:
        #    raise Exception("data_db process failed to start after seconds")
        #if self._speadmeta_process == None:
            #raise Exception("metadata_process failed to start after seconds")

    def _create_dada_buffer(self, dadaId = 'dada', numaCore = 4, nBuffers =16):
        """Create the dada buffer. Must be run before capture and dbdisk.'

        Parameters
        ----------
        dadaId :
            Dada buffer ID.
        numaCore :
            NUMA node to attach dada buffer to
        """
        #cmd = ['dada_db', '-k', dadaId, '-d']
        #dada_buffer_process = subprocess.Popen(
        #cmd, stdout=subprocess.PIPE
        #)
        #print (dada_buffer_process.communicate())
        
        #dada_buffer_process.wait()
        cmd = ['dada_db', '-k', dadaId, '-b', '268435456', '-p', '-l', '-n', '%d'%nBuffers, '-c', '%d'%numaCore]
        dada_buffer_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )
        print ('creating buffers')
        dada_buffer_process.wait()
        print ('complete buffer creation')
        print (dada_buffer_process.communicate())

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
        cmd = ['numactl', '-C', '%d'%cpuCore, 'dada_dbdisk', '-k', dadaId, '-D', outputDir, '-z', '-s']
        self._dada_dbdisk_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )

    def _create_digifits (self):
        _logger.info("digifits")
        #cmd=["printenv"]
        #temp_proc = subprocess.Popen(
        #cmd, stdin=subprocess.PIPE, stdout= subprocess.PIPE, stderr=subprocess.PIPE, cwd="/data"
        #)
        #_logger.info(temp_proc.communicate())
        with open("/tmp/digifits.log","w+") as logfile:
            cmd = ["digifits","-v","-b","8","-t",".0001531217700876","-c","-p","1","-nsblk","128", "-cuda", "0","/home/kat/dada.info"]
            _logging.info(cmd)
            self._digifits_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=logfile, stderr=logfile, cwd="/data"
            )

    def _create_dspsr(self):
        _logger.info("dspsr")
        #numactl -C 7 dspsr -cuda 0 /home/kat/dada.info 
        with open("/tmp/dspsr.log","w+") as logfile:
            cmd = ["taskset","7","dspsr","-D","0","-Q","-L","10","-cuda","0","/home/kat/dada.info"]
            _logging.info(cmd)
            self._dspsr_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/data"
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
        print ("meta complete with ts = %d"%ads_sync_time)



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

        #result, error = yield self._capture_process.communicate()

        #raise Return((result, error))

    @trollius.coroutine
    def _run(self, obs_length, centre_freq, targets, halfband=False, interface_ip="10.100.205.11", beam_x_multicast="239.9.3.30", beam_y_multicast="239.9.3.31", data_port="7148"):
        """Does the work of capturing a stream. This is a coroutine."""
        _logger.info("running")
        #_logger.info(args.telstate.getRange('obs_params',st=0))
        #_logger.info(args.telstate.get('obs_params'))
        #self._create_dada_buffer()
        #_logger.info("Created dada_buffer\n" + result)
        #print ("Created dada_buffer\n")
        #self._create_dada_dbdisk()
        #_logger.info ("dada_dbdisk output :\n %s\n error\n %s"%(result,error))
        #print ("dada_dbdisk complete") 
        #self._create_capture_process()
        print ("capture_process+++++++______!!")
        #self._create_metaspead()
        #def _create_metaspead (self, pol_h_host = "10.100.21.5", pol_h_mcast = "239.9.3.30", pol_h_port = 7148):
        _logger.info("IN METADATA")
        #_logger.info(
        cmd = ["meerkat_speadmeta", _get_interface_address("p4p1"), beam_y_multicast, "-p", data_port]

        self._speadmeta_process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


        #speadmeta_process.wait(timeout=10)
        # we need to capture the output from this process
        error, adc_sync_time = self._speadmeta_process.communicate()
        self._speadmeta_process = None
        _logger.info((error,adc_sync_time))
        _logger.info("meta spead run")

#        print (adc_sync_time)
 #       print (type(adc_sync_time))
        #print (error)
        # next we need to put this ADC_SYNC_TIME into the spipConfig file provided
        #runtimeConfig = "/home/kat/hardware_cbf_4096chan_2pol.cfg" + ".live"
        #f_read = open ("/home/kat/hardware_cbf_4096chan_2pol.cfg", 'r')

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
        #for line in fileinput.input("/home/kat/hardware_cbf_4096chan_2pol.cfg", inplace=1):
            #print(line)
            #if "ADC_SYNC_TIME" in line:
            #    line = "ADC_SYNC_TIME %s/n"%(adc_sync_time)
            #sys.stdout.write(line)
        #for line in f_read:
        #    if line.find("ADC_SYNC_TIME"):
        #        f_write.write("ADC_SYNC_TIME %s\n"%(adc_sync_time))
        #    else:
        #        f_write.write(line + "\n")
        #f_read.close()
        #f_write.close()
        _logger.info("meta complete with ts = %s"%adc_sync_time)
        self._speadmeta_process = None
        cap_env = os.environ.copy()

        cap_env["LD_PRELOAD"] = "libvma.so"
        cap_env["VMA_MTU"] = "9200"
        cap_env["VMA_RX_POLL_YIELD"] = "1"
        cap_env["VMA_RX_UDP_POLL_OS_RATIO"] = "0"
        #_create_capture_process(self, spipConfig="/home/kat/hardware_cbf_4096chan_2pol.cfg", core1=3, core2=5):
        cmd = ["meerkat_udpmergedb", c_file, "-f", "spead", "-b", "%d,%d"%(3,5)]

        self._capture_process = subprocess.Popen(
        cmd, subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=cap_env
        )

        #result, error = yield self._capture_process.communicate()

        #raise Return((result, error))
        #_logger.info ("capture output :\n %s\n error\n %s"%(result,error))
        _logger.info("capture started_________________------12-----!!")
        #while (self._capture_process == None):
         #   _logger.info ("No capture")
          #  time.sleep(1)
        import signal
        _logger.info ("obs_length")
        _logger.info(obs_length - 5)
        time.sleep(int(obs_length))
        _logger.info("kill cap--------------11!!--------------------")
        self._capture_process.send_signal(signal.SIGINT)
        #time.sleep(5)
        #self._capture_process.send_signal(signal.SIGINT)
        _logger.info("kill cap--------------11--------------------")
            #time.sleep(2) 
        #if self._capture_process != None:
        #self._capture_process.send_signal(signal.SIGINT)
        #self._capture_process.send_signal(signal.SIGINT)
        _logger.info(self._capture_process.communicate())
        self._capture_process = None
        #self._capture_process.wait()
        #result, error = yield self._capture_process.communicate()

        #raise Return((result, error))

        #print ("capture_process_output = %s"%(result,error))
        #ret.append (self._dada_dbdisk_process.wait_for_exit())

        #ret = yield self._capture_process.communicate()

    #@trollius.coroutine
    def stop(self):
        """Shut down the stream and wait for the coroutine to complete. This
        is a coroutine.
        """
        import signal
        print ("STOPPING")
        self._manual_stop = True
        if self._capture_process != None:
            self._capture_process.send_signal(signal.SIGINT)
            _logger.info("kill cap----------------------------------")
            #time.sleep(2) 
        #if self._capture_process != None:
            self._capture_process.send_signal(signal.SIGINT)
            _logger.info(self._capture_process.communicate())
            #time.sleep(4)
        if self._dada_dbdisk_process != None:
            self._dada_dbdisk_process.send_signal(signal.SIGINT)
        if self._digifits_process != None:
            self._digifits_process.send_signal(signal.SIGINT)
            #_logger.info("comm digifits")
            #for line in self._digifits_process.stdout.readlines():
            #    print (line)
            _logger.info(self._digifits_process.communicate())
        if self._dspsr_process != None:
            self._dspsr_process.send_signal(signal.SIGINT)
            _logger.info(self._dspsr_process.communicate())
        if self._speadmeta_process != None:
            self._speadmeta_process.send_signal(signal.SIGINT)
        time.sleep(5)
        cmd = ['dada_db', '-d']
        dada_buffer_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE
        )
        _logger.info(dada_buffer_process.communicate())
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
                                                          
                                      
