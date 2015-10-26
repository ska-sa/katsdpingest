#!/usr/bin/env python

from __future__ import print_function, division
import tornado
import tornado.ioloop
import tornado.gen
import katportalclient
import logging
import sys
import katsdptelstate


def configure_logging():
    if len(logging.root.handlers) > 0: logging.root.removeHandler(logging.root.handlers[0])
    formatter = logging.Formatter("%(asctime)s.%(msecs)dZ - %(filename)s:%(lineno)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logging.root.addHandler(sh)

    logger = logging.getLogger("katsdpingest.cam2telstate")
    logger.setLevel(logging.INFO)

    # configure SPEAD to display warnings about dropped packets etc...
    logging.getLogger('spead2').setLevel(logging.WARNING)
    return logger


def parse_args():
    parser = katsdptelstate.ArgumentParser()
    parser.add_argument('url', type=str, help='WebSocket URL to connect to')
    args = parser.parse_args()
    if not args.telstate:
        print('--telstate is required', file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    return args


class Client(object):
    def __init__(self, args, logger):
        self._args = args
        self._telstate = args.telstate
        self._logger = logger
        self._portal_client = None

    @tornado.gen.coroutine
    def start(self):
        self._portal_client = katportalclient.KATPortalClient(
            self._args.url, self.update_callback, logger=self._logger)
        yield self._portal_client.connect()
        yield self._portal_client.subscribe('', ['*'])

    def process_update(self, item):
        data = item[u'msg_data']
        name = data[u'name'].encode('us-ascii')
        timestamp = data[u'timestamp']
        status = data[u'status'].encode('us-ascii')
        value = data[u'value']
        if isinstance(value, unicode):
            value = value.encode('us-ascii')
        if status == 'unknown':
            self._logger.debug("Sensor {} received update '{}' with status 'unknown' (ignored)"
                    .format(name, value))
        else:
            # XXX Nasty hack to get SDP onto cbf name for AR1 integration
            name = name.replace('data_1_', 'cbf_')
            self._telstate.add(name, value, timestamp)

    def update_callback(self, msg):
        if isinstance(msg, list):
            for item in msg:
                self.process_update(item)
        else:
            self.process_update(msg)


def main():
    args = parse_args()
    logger = configure_logging()
    loop = tornado.ioloop.IOLoop().current()
    client = Client(args, logger)
    loop.add_callback(client.start)
    loop.start()

if __name__ == '__main__':
    main()
