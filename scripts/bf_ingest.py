#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import spead2
import spead2.recv
import spead2.recv.trollius
import katcp
from katcp.kattypes import request, return_reply
import h5py
import signal
import trollius
import tornado
from tornado.platform.asyncio import AsyncIOMainLoop, to_asyncio_future
import logging
from trollius import From, Return
import argparse
import katsdptelstate.endpoint
import katversion


_logger = logging.getLogger('')


class CaptureSession(object):
    def __init__(self, endpoint, loop):
        self._loop = loop
        self._file = None
        self._manual_stop = False
        self._ig = spead2.ItemGroup()
        _logger.info('Listening on endpoint {}'.format(endpoint))
        self._stream = spead2.recv.trollius.Stream(spead2.ThreadPool(), 0, loop=self._loop)
        self._stream.add_udp_reader(endpoint.port, bind_hostname=endpoint.host)
        self._run_future = trollius.async(self._run(), loop=self._loop)

    def _process_heap(self, heap):
        updated = self._ig.update(heap)
        if 'timestamp' in updated:
            timestamp = self._ig['timestamp'].value
            _logger.info('Received heap with timestamp %d', timestamp)
            if not self._file:
                self._file = h5py.File(str(timestamp) + '.h5')

    @trollius.coroutine
    def _run(self):
        try:
            while True:
                heap = yield From(self._stream.get())
                self._process_heap(heap)
        except spead2.Stopped:
            if self._manual_stop:
                _logger.info('Capture terminated by request')
            else:
                _logger.info('Capture terminated by stop heap')
        except Exception:
            _logger.error('Capture coroutine threw uncaught exception', exc_info=True)
        finally:
            if self._file:
                self._file.close()
                self._file = None

    @trollius.coroutine
    def stop(self):
        self._manual_stop = True
        self._stream.stop()
        yield From(self._run_future)


class CaptureServer(katcp.DeviceServer):
    """katcp device server for beamformer simulation"""
    VERSION_INFO = ('bf-ingest', 1, 0)
    BUILD_INFO = ('katsdpingest',) + tuple(katversion.get_version(__file__).split('.', 1)) + ('',)

    def __init__(self, args, loop):
        super(CaptureServer, self).__init__(args.host, args.port)
        self._endpoint = args.endpoint
        self._loop = loop
        self._capture = None

    def setup_sensors(self):
        pass

    @request()
    @return_reply()
    def request_capture_init(self, sock):
        """Start capture to file"""
        if self._capture is not None:
            return ('fail', 'already capturing')
        self._capture = CaptureSession(self._endpoint, self._loop)
        return ('ok',)

    @tornado.gen.coroutine
    def _stop_capture(self):
        if self._capture is not None:
            stop_future = trollius.async(self._capture.stop(), loop=self._loop)
            yield tornado.platform.asyncio.to_tornado_future(stop_future)
            self._capture = None

    @request()
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_done(self, sock):
        """Stop a capture that is in progress"""
        if self._capture is None:
            raise tornado.gen.Return(('fail', 'not capturing'))
        yield self._stop_capture()
        raise tornado.gen.Return(('ok',))

    @tornado.gen.coroutine
    def stop(self):
        yield self._stop_capture()
        yield super(CaptureServer, self).stop()


@trollius.coroutine
def on_shutdown(server):
    _logger.info('Shutting down')
    yield From(to_asyncio_future(server.stop()))
    trollius.get_event_loop().stop()


def main():
    parser = katsdptelstate.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', '-p', type=int, default=2050, help='katcp host port')
    parser.add_argument('--host', '-a', type=str, default='', help='katcp host address')
    parser.add_argument('--logging', '-l', type=str, metavar='LEVEL', default='INFO', help='log level')
    parser.add_argument('endpoint', type=katsdptelstate.endpoint.endpoint_parser(7148), help='Multicast group and port')
    args = parser.parse_args()
    logging.basicConfig(level=args.logging, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    ioloop = AsyncIOMainLoop()
    ioloop.install()
    server = CaptureServer(args, trollius.get_event_loop())
    server.set_concurrency_options(thread_safe=False, handler_thread=False)
    server.set_ioloop(ioloop)
    trollius.get_event_loop().add_signal_handler(signal.SIGINT,
        lambda: trollius.async(on_shutdown(server)))
    trollius.get_event_loop().add_signal_handler(signal.SIGTERM,
        lambda: trollius.async(on_shutdown(server)))
    ioloop.add_callback(server.start)
    trollius.get_event_loop().run_forever()


if __name__ == '__main__':
    main()
