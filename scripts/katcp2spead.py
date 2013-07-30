#!/usr/bin/env python
#
# Server that pushes KATCP sensor data to a SPEAD stream.
#

import optparse
import Queue
import logging

import spead
import katconf
import katcorelib
from katcapture.katcp2spead import Katcp2SpeadDeviceServer

sensors = [
    ('dbe7_target', 'event', ''),
    ('ant1_activity', 'event', ''),
    ('ant2_activity', 'event', ''),
    ('ant3_activity', 'event', ''),
    ('ant4_activity', 'event', ''),
    ('ant5_activity', 'event', ''),
    ('ant6_activity', 'event', ''),
    ('ant7_activity', 'event', ''),
#    ('ant2_pos_actual_scan_azim', 'period', '0.4'),
#    ('ant2_pos_actual_scan_elev', 'period', '0.4'),
]

# Parse command-line options
parser = optparse.OptionParser(usage='%prog [options]',
                               description="Stream KATCP sensors via SPEAD.")
parser.add_option('-s', '--system',
                  help='System configuration file (default = site default)')
parser.add_option('-c', '--sysconfig', default='/var/kat/katconfig',
                  help='Configuration directory, can be overrided by KATCONF '
                       'environment variable (default=%default)')
parser.add_option('--spead-host',
                  help='Address of host where SPEAD sensor data is sent')
parser.add_option('--spead-port', type=int, default=7148,
                  help='Port on spead-host where SPEAD sensor data is sent '
                       '(default=%default)')
parser.add_option('--ctl-host', default='',
                  help='Address of host that will receive KATCP commands '
                       '(default all hosts)')
parser.add_option('--ctl-port', type=int, default=2045,
                  help='Port on which to receive KATCP commands '
                       '(default=%default)')
parser.add_option('-l', '--logging',
                  help='Level to use for basic logging or name of logging '
                       'configuration file (default log/log.<SITENAME>.conf)')
opts, args = parser.parse_args()

# Setup configuration source and configure logging (via conf file or directly)
katconf.set_config(katconf.environ(opts.sysconfig))
katconf.configure_logging(opts.logging)
# Suppress SPEAD info messages
spead.logger.setLevel(logging.WARNING)
logger = logging.getLogger("kat.katcp2spead")

# Get host object through which to access system sensors
kat = katcorelib.tbuild(system=opts.system)
# Create device server that is main bridge between KATCP and SPEAD
server = Katcp2SpeadDeviceServer(kat, sensors, opts.spead_host, opts.spead_port,
                                 host=opts.ctl_host, port=opts.ctl_port)
server.set_restart_queue(Queue.Queue())
# Spawn new thread to handle KATCP requests to device server
server.start()
# Use the rest of this thread to manage restarts of the device server
try:
    while True:
        try:
            device = server._restart_queue.get(timeout=0.5)
        except Queue.Empty:
            device = None
        if device is not None:
            logger.info("Stopping")
            device.stop()
            device.join()
            logger.info("Restarting")
            device.start()
            logger.info("Started")
finally:
    server.stop()
    server.join()