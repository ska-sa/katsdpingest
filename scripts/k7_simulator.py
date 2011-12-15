#! /usr/bin/env python
"""
Some of this hacked from Jason's correlator package.
"""

import Queue
import sys
import optparse
import logging
from pkg_resources import resource_filename

import katconf
from katcapture.simulator.dbe7_simulator import SimulatorDeviceServer
from katcapture.simulator.dbe7_simulator_model import K7CorrelatorModel
from katcore.dev.base import SimpleModel

def parse_opts(argv):
    parser = optparse.OptionParser()
    default_conf = resource_filename("katcapture","") + "/conf/k7-local.conf"
    parser.add_option('-c', '--config',
                      dest='config',
                      type="string",
                      default=default_conf,
                      help='k7 correlator config file to use.')
    parser.add_option('-p', '--port',
                      dest='port',
                      type=long,
                      default=2041,
                      metavar='N',
                      help='attach katcp interface to port N (default=2041)')
    parser.add_option('-a', '--host',
                      dest='host',
                      type="string",
                      default="",
                      metavar='HOST',
                      help='listen to HOST (default="" - all hosts)')
    parser.add_option('-s', '--system',
                      default='systems/local.conf',
                      help='system configuration file to use. [default=%default]')
    parser.add_option('-f', '--sysconfig',
                      dest='sysconfig',
                      default='/var/kat/katconfig',
                      help='look for configuration files in folder CONF '
                      '[default is KATCONF environment variable'
                      'or /var/kat/katconfig]')
    parser.add_option('-l', '--logging',
                      dest='logging',
                      type='string',
                      default=None,
                      metavar='LOGGING',
                      help='level to use for basic logging or name of logging '
                      'configuration file; default is /log/log.<SITENAME>.conf')
    return parser.parse_args(argv)


if __name__ == '__main__':
    opts, args = parse_opts(sys.argv)

    # Setup configuration source
    katconf.set_config(katconf.environ(opts.sysconfig))

    # set up Python logging
    katconf.configure_logging(opts.logging)
    log_name = 'kat.k7simulator'
    logger = logging.getLogger(log_name)
    logger.info("Logging started")
    activitylogger = logging.getLogger('activity')
    activitylogger.setLevel(logging.INFO)
    activitylogger.info("Activity logging started")

    restart_queue = Queue.Queue()
    model = K7CorrelatorModel(opts.config)
    server = SimulatorDeviceServer(model, opts.host, opts.port)
    server.set_restart_queue(restart_queue)
    server.start()
    smsg = "Started k7-capture server."
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
        server.join()

