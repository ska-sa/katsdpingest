#! /usr/bin/env python
"""
Some of this hacked from Jason's correlator package.
"""

import Queue
import threading
import sys
import optparse
import logging
from pkg_resources import resource_filename

import katconf
from katsdpingest.simulator.cbf_simulator import DBE7DeviceServer
from katsdpingest.simulator.cbf_simtest import SimTestDevice, NotHidden
from katsdpingest.simulator.cbf_simulator_model import K7CorrelatorModel
from katcore.dev_base import SimpleModel
from katcore.utils import address

def parse_opts(argv):
    parser = optparse.OptionParser()
    default_conf = resource_filename("katsdpingest","") + "/conf"
    parser.add_option('-c', '--config',
                      dest='config',
                      type="string",
                      default=default_conf,
                      help='k7 correlator config directory to use.')
    parser.add_option('-p', '--port',
                      dest='port',
                      type=long,
                      default=2041,
                      metavar='N',
                      help='attach master controller katcp interface to port N '
                      '(default=%default)')
    parser.add_option('--test-addr', default=':2042',
                      help='HOST:PORT for the test interface to listen on '
                      '(default=%default)')
    parser.add_option('--startup-sensor-hide-re',
                      dest='startup_sensor_hide_re',
                      type='string',
                      default=None,       # Should not match any valide sensor name
                      help='Hide all sensors matching (python) regex')
    parser.add_option('--unhide-sensor-delay',
                      dest='unhide_sensor_delay',
                      type=float,
                      default=None,
                      help='Unhide sensors hidden by --startup-sensor-hide-re after '
                      'specified number of seconds')
    parser.add_option('-a', '--host',
                      dest='host',
                      type="string",
                      default="",
                      metavar='HOST',
                      help='listen to HOST (default="%default" - all hosts)')
    parser.add_option('-s', '--system',
                      default='systems/local.conf',
                      help='system configuration file to use. [default=%default]')
    parser.add_option('-f', '--sysconfig',
                      dest='sysconfig',
                      default='/var/kat/katconfig',
                      help='look for configuration files in folder CONF '
                      '[default is KATCONF environment variable'
                      'or %default]')
    parser.add_option('-l', '--logging',
                      dest='logging',
                      type='string',
                      default=None,
                      metavar='LOGGING',
                      help='level to use for basic logging or name of logging '
                      'configuration file; default is /log/log.<SITENAME>.conf')
    parser.add_option('--standalone',
                      dest='standalone',
                      default=False,
                      action='store_true',
                      help='Standalone mode. Sets default antenna channel '
                      'mappings and does a spead issue at '
                      'startup. Also sets logging to local with "warn"'
                      'level; can be overidden with the -l option')

    return parser.parse_args(argv)

def setup_standalone(server, model):
    model.set_mode('c8n856M32k',mode_delay=0)
    for i in range(1,8):
        for ch,pol in zip(['x', 'y'], ['h', 'v']):
            model.set_antenna_mapping('%d%s' %(i, ch), 'ant%d%s' %(i, pol))

if __name__ == '__main__':
    opts, args = parse_opts(sys.argv)

    # Setup configuration source
    katconf.set_config(katconf.environ(opts.sysconfig))

    # set up Python logging
    if opts.standalone and opts.logging is None:
        opts.logging = 'warn'
    katconf.configure_logging(opts.logging)
    log_name = 'kat.k7simulator'
    logger = logging.getLogger(log_name)
    logger.info("Logging started")
    activitylogger = logging.getLogger('activity')
    activitylogger.setLevel(logging.INFO)
    activitylogger.info("Activity logging started")

    restart_queue = Queue.Queue()
    model = K7CorrelatorModel(opts.config)
    server = DBE7DeviceServer(model, opts.host, opts.port)
    test_host, test_port = address(opts.test_addr)
    testserver = SimTestDevice(model, test_host, test_port)
    testserver.set_device(server)

    # Hide sensors as specified by command line regex
    if not opts.startup_sensor_hide_re is None:
        hidden_sensors = testserver.hide_sensor_re(opts.startup_sensor_hide_re)
    else:
        hidden_sensors = []

    if opts.standalone:
        activitylogger.info('Doing standalone mode setup')
        setup_standalone(server, model)

    # Start device server
    server.set_restart_queue(restart_queue)
    server.start()

    # Unhide sensors after specified delay
    def unhide_sensors():
        for s in hidden_sensors:
            try: testserver.unhide_sensor(s)
            except NotHidden: logger.warn(
                    'Attempted to unhide sensor %s that is not hidden' % s)

    if not opts.unhide_sensor_delay is None:
        threading.Timer(
            opts.unhide_sensor_delay, unhide_sensors).start()

    # Start 'back-door' test-device server
    testserver.set_restart_queue(restart_queue)
    testserver.start()
    smsg = "Started k7-simulator server."
    activitylogger.info(smsg)
    try:
        while True:
            try:
                device = restart_queue.get(timeout=0.5)
            except Queue.Empty:
                device = None
            if device is not None:
                smsg = "Stopping ..."
                activitylogger.info(smsg)
                device.stop()
                device.join()
                smsg = "Restarting ..."
                activitylogger.info(smsg)
                device.start()
                smsg = "Started."
                activitylogger.info(smsg)
    except KeyboardInterrupt:
        smsg = "Shutting down ..."
        print smsg
        activitylogger.info(smsg)
        server.stop()
        testserver.stop()
        server.join()
        testserver.join()
