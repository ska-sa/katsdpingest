from __future__ import print_function, division, absolute_import
import spead2
import spead2.recv
import spead2.recv.trollius
import katcp
from katcp.kattypes import request, return_reply
import h5py
import numpy as np
import time
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
        self._file = None
        self._bf_raw_dataset = None
        self._timestamps = []
        self._manual_stop = False
        if args.cbf_channels:
            self._timestep = 2 * args.cbf_channels
        else:
            self._timestep = None
        self._ig = spead2.ItemGroup()
        self._stream = spead2.recv.trollius.Stream(spead2.ThreadPool(), 0, loop=self._loop)
        for endpoint in args.cbf_spead:
            _logger.info('Listening on endpoint {}'.format(endpoint))
            self._stream.add_udp_reader(endpoint.port, bind_hostname=endpoint.host)
        self._run_future = trollius.async(self._run(), loop=self._loop)

    def _create_file_and_memory_pool(self):
        """Create the HDF5 file and dataset. This is called once the data shape
        is known.
        """
        filename = os.path.join(self._args.file_base, '{}.h5'.format(int(time.time())))
        # Allocate enough cache space for 3 chunks - should be enough for one active, one
        # being written back, and a bit extra for timestamps.
        n_chans, n_time = self._ig['bf_raw'].shape[0:2]
        dtype = self._ig['bf_raw'].dtype
        self._file = h5py.File(filename, mode='w')
        self._file.attrs['version'] = 1
        data_group = self._file.create_group('Data')
        shape = (n_chans, 0, 2)
        maxshape = (n_chans, None, 2)
        self._bf_raw_dataset = data_group.create_dataset(
            'bf_raw', shape, maxshape=maxshape, dtype=dtype, chunks=(n_chans, n_time, 2))
        if self._timestep is None:
            _logger.info('Assuming %d PFB channels; if not, pass --cbf-channels', n_chans)
            self._timestep = 2 * n_chans
        chunk_size = n_time * n_chans * 2 * dtype.itemsize
        memory_pool = spead2.MemoryPool(chunk_size, chunk_size + 4096, 8, 2)
        self._stream.set_memory_pool(memory_pool)

    @trollius.coroutine
    def _run(self):
        """Does the work of capturing a stream. This is a coroutine."""
        data_heaps = 0
        try:
            try:
                while True:
                    heap = yield From(self._stream.get())
                    updated = self._ig.update(heap)
                    is_data_heap = 'timestamp' in updated and 'bf_raw' in updated
                    # Check whether we need to, and can, create the file. We need
                    # the descriptor for bf_raw, but CBF is also known to send
                    # multiple versions of some metadata, so we wait for an
                    # indication that the metadata is complete.
                    if (not self._file and
                            (heap.is_start_of_stream() or is_data_heap) and
                            'bf_raw' in self._ig):
                        self._create_file_and_memory_pool()
                    if not is_data_heap:
                        _logger.info('Received non-data heap %d', heap.cnt)
                        continue
                    timestamp = self._ig['timestamp'].value
                    bf_raw = self._ig['bf_raw'].value
                    _logger.debug('Received heap with timestamp %d', timestamp)

                    n_chans, n_time = bf_raw.shape[0:2]
                    if n_chans != self._bf_raw_dataset.shape[0]:
                        _logger.warning('Dropping heap because number of channels does not match')
                        continue
                    idx = self._bf_raw_dataset.shape[1]   # Number of spectra already received
                    self._bf_raw_dataset.resize(idx + n_time, axis=1)
                    self._bf_raw_dataset[:, idx : idx + n_time, :] = bf_raw
                    self._timestamps.append(timestamp)
                    data_heaps += 1
                    # Check free space periodically, but every heap is excessive
                    if data_heaps % 100 == 0:
                        _logger.info('Received %d data heaps', data_heaps)
                        stat = os.statvfs(self._file.filename)
                        free_bytes = stat.f_frsize * stat.f_bavail
                        # We check only every 100 dumps, so this actually only
                        # guarantees 200 dumps free. That's a reasonable gap to
                        # allow for buffering, HDF5 overheads etc.
                        if free_bytes < 300 * bf_raw.nbytes + 8 * n_time * data_heaps:
                            _logger.warn('Capture stopped due to lack of disk space')
                            break
            except spead2.Stopped:
                if self._manual_stop:
                    _logger.info('Capture terminated by request')
                else:
                    _logger.info('Capture terminated by stop heap')
            except Exception:
                _logger.error('Capture coroutine threw uncaught exception', exc_info=True)
                raise
        finally:
            self._stream.stop()
            if self._file:
                # Write the timestamps to file
                n_time = self._ig['bf_raw'].shape[1]
                ds = self._file['Data'].create_dataset(
                    'timestamp',
                    shape=(n_time * len(self._timestamps),),
                    dtype=np.uint64)
                idx = 0
                for timestamp in self._timestamps:
                    ds[idx : idx + n_time] = np.arange(
                        timestamp, timestamp + n_time * self._timestep, self._timestep,
                        dtype=np.uint64)
                    idx += n_time
                self._file.close()
                self._file = None

            if self._timestamps:
                elapsed = max(self._timestamps) - min(self._timestamps)
                expected_heaps = elapsed // (n_time * self._timestep) + 1
            else:
                expected_heaps = 0
            _logger.info('Received %d heaps, expected %d based on min/max timestamps',
                         data_heaps, expected_heaps)
            if data_heaps < expected_heaps:
                _logger.warn('%d heaps missing', expected_heaps - data_heaps)
            elif data_heaps > expected_heaps:
                _logger.warn('%d more heaps than expected (timestamp errors?)',
                             data_heaps - expected_heaps)

    @trollius.coroutine
    def stop(self):
        """Shut down the stream and wait for the coroutine to complete. This
        is a coroutine.
        """
        self._manual_stop = True
        self._stream.stop()
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

    VERSION_INFO = ('bf-ingest', 1, 0)
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