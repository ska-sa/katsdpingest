from __future__ import print_function, division, absolute_import
#import spead2
#import spead2.recv
#import spead2.recv.trollius
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


_logger = logging.getLogger(__name__)


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
        self._loop = loop
        self._args = args
        self._dada_dbdisk_process = None
        self._capture_process = None
        self._timestamps = []
        self._manual_stop = False
        self._endpoints = []
        if args.cbf_channels:
            self._timestep = 2 * args.cbf_channels
        else:
            self._timestep = None
        self._ig = spead2.ItemGroup()
        # self._stream = spead2.recv.trollius.Stream(spead2.ThreadPool(), 0, loop=self._loop)
        for endpoint in args.cbf_spead:
            _logger.info('Listening on endpoint {}'.format(endpoint))
            self._endpoints.append(endpoint)
        self._run_future = trollius.async(self._run(), loop=self._loop)

    def _create_dada_buffer(self, dadaId = 'dada', numaNode = 1):
        """Create the dada buffer. Must be run before capture and dbdisk.'

        Parameters
        ----------
        dadaId :
            Dada buffer ID.
        numaNode :
            NUMA node to attach dada buffer to
        """
        cmd = 'dada_db -k %s -b 268435456 -p -l -n 4 -c %d'%(dadaId, numaNode)
        dada_buffer_process = tornado.process.Subprocess(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        return sub_process.wait_for_exit()

    def _create_dada_dbdisk (self, dadaId = 'dada', numaNode = 1, outputDir = '/data'):
        """Create the dada_dbdisk process which writes data from dada buffer to disk.
        Must be run after creating the dada_buffer and before the capture process.
        Must have the same NUMA node as the dada_buffer

         Parameters
            ----------
            dadaId :
                Dada buffer ID.
            numaNode :
                NUMA node to attach dada buffer to
            outputDir :
                Location to write captured data
        """

        STREAM = tornado.process.Subprocess.STREAM
        cmd = 'numactl -C 1 dada_dbdisk -k %s -D %s -z -s -d;\
               numactl -C 1 dada_dbdisk -k %s -D %s -z -s'%(numaNode, dadaId, outputDir)
        self._dada_dbdisk_process = tornado.process.Subprocess(
        cmd, stdin=STREAM, stdout=STREAM, stderr=STREAM
        )

        # sub_process.stdin.close()

        result, error = yield [
        Task(self._dada_dbdisk_process.stdout.read_until_close),
        Task(self._dada_dbdisk_process.stderr.read_until_close)
        ]

        raise Return((result, error))


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
        STREAM = tornado.process.Subprocess.STREAM
        cap_env = os.environ.copy()
        cap_env["LD_PRELOAD"] = "libvma.so"
        cap_env["VMA_MTU"] = "9200"
        cap_env["VMA_RX_POLL_YIELD"] = "1"
        cap_env["VMA_RX_UDP_POLL_OS_RATIO"] = "0"
        cap_env["VMA_RING_ALLOCATION_LOGIC_RX"] = "10"

        cmd = "meerkat_udpmergedb %s -f spead -b %d -c %d"%(spipConfig,core1,core2)

        self._capture_process = tornado.process.Subprocess(
        cmd, stdin=STREAM, stdout=STREAM, stderr=STREAM, env=cap_env
        )

        # sub_process.stdin.close()

        result, error = yield [
        Task(self._capture_process.stdout.read_until_close),
        Task(self._capture_process.stderr.read_until_close)
        ]

        raise Return((result, error))

    @trollius.coroutine
    def _run(self):
        """Does the work of capturing a stream. This is a coroutine."""
        result = self._create_dada_buffer()
        _logger.info("Created dada_buffer\n" + result)

        result, error = yield self._create_dada_dbdisk()
        _logger.info ("dada_dbdisk output :\n %s\n error\n %s"%(result,error))

        result, error = yield self._create_capture_process()
        _logger.info ("capture output :\n %s\n error\n %s"%(result,error))

        ret = [self._capture_process.wait_for_exit(),]

        ret.append (self._dada_dbdisk_process.wait_for_exit())

        # return yield ret

        
    @trollius.coroutine
    def stop(self):
        """Shut down the stream and wait for the coroutine to complete. This
        is a coroutine.
        """
        self._manual_stop = True
        self._capture_process.kill()
        self._capture_process.kill()
        self._dada_dbdisk_process.kill()
        yield From(self._run_future)


class CaptureServer(object):
    """Beamformer capture. This contains all the core functionality of the
    katcp device server, without depending on katcp. It is split like this
    to facilitate unit testing.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Command-line arguments. The following arguments are required:

        cbf_channels
          Total number of PFB channels, or ``None`` for unknown
        cbf_spead
          List of :class:`katsdptelstate.endpoint.Endpoint` for receiving SPEAD data
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
            self._capture = _CaptureSession(self._args, self._loop)

    @trollius.coroutine
    def stop_capture(self):
        """Stop capture to file, if currently running. This is a co-routine."""
        if self._capture is not None:
            yield From(self._capture.stop())
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
        self.start_capture()
        return ('ok',)

    @tornado.gen.coroutine
    def _stop_capture(self):
        """Tornado variant of :meth:`stop_capture`"""
        stop_future = trollius.async(self.stop_capture(), loop=self._loop)
        yield tornado.platform.asyncio.to_tornado_future(stop_future)

    @request()
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_done(self, sock):
        """Stop a capture that is in progress."""
        if not self.capturing:
            raise tornado.gen.Return(('fail', 'not capturing'))
        yield self._stop_capture()
        raise tornado.gen.Return(('ok',))

    @tornado.gen.coroutine
    def stop(self):
        yield self._stop_capture()
        yield katcp.DeviceServer.stop(self)

    stop.__doc__ = katcp.DeviceServer.stop.__doc__


__all__ = ['CaptureServer', 'KatcpCaptureServer']
