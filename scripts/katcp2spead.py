#!/usr/bin/env python
#
# Server that pushes KATCP sensor data to a SPEAD stream.
#

import optparse
import Queue
import logging
import os.path

import numpy as np
from pkg_resources import resource_filename

import spead
import katconf
import katcorelib
from katsdpingest.katcp2spead import Katcp2SpeadDeviceServer


# Obtain default sensor list
conf_dir = resource_filename('katsdpingest', 'conf')
default_sensors = os.path.join(conf_dir, 'rts_sensors.csv')

# Parse command-line options
parser = optparse.OptionParser(usage='%prog [options]',
                               description="Stream KATCP sensors via SPEAD.")
parser.add_option('-s', '--system',
                  help='System configuration file (default = site default)')
parser.add_option('-c', '--sysconfig', default='/var/kat/katconfig',
                  help='Configuration directory, can be overrided by KATCONF '
                       'environment variable (default=%default)')
parser.add_option('--sensor-list', default=default_sensors,
                  help='Name of text file containing list of sensors to serve '
                       '(default=%default)')
parser.add_option('--ctl-host', default='',
                  help='Address of host that will receive KATCP commands '
                       '(default all hosts)')
parser.add_option('--ctl-port', type=int, default=2045,
                  help='Port on which to receive KATCP commands '
                       '(default=%default)')
parser.add_option('--fake-cam', action='store_true', default=False,
                  help="Connect to fake CAM sensors for simulation")
parser.add_option('--fake-cam-host', default='localhost',
                  help='Host address of fake CAM sensor server (default=%default)')
parser.add_option('--fake-cam-port', type=int, default=2047,
                  help='Port of fake CAM sensor KATCP interface (default=%default)')
parser.add_option('-l', '--logging',
                  help='Level to use for basic logging or name of logging '
                       'configuration file (default log/log.<SITENAME>.conf)')
opts, args = parser.parse_args()

# Setup configuration source and configure logging (via conf file or directly)
katconf.set_config(katconf.environ(opts.sysconfig))
katconf.configure_logging(opts.logging)
# Suppress SPEAD info messages
spead.logger.setLevel(logging.WARNING)
logger = logging.getLogger("kat.ingest.katcp2spead")

# Get group of sensor objects, either via KAT connection or a fake CAM device
if opts.fake_cam:
    fake_cam = katcorelib.build_client('fake_cam', opts.fake_cam_host,
                                       opts.fake_cam_port, required=True)
    all_sensors = fake_cam.sensor
else:
    kat = katcorelib.tbuild(system=opts.system)
    all_sensors = kat.sensors

# Load names of sensors to be streamed
sensors = np.loadtxt(opts.sensor_list, delimiter=',', skiprows=1, dtype=np.str)
sensor_names = [line[0].strip() for line in sensors]
# Antenna position sensors are currently the only high-frequency sensors that
# update too regularly to be treated as event sensors
sensor_list = [(name, 'period', '0.4') if name.find('_pos_') > 0 else
               (name, 'event', '') for name in sensor_names]
logger.info('Listening to %d sensors selected from %d %s ones' %
            (len(sensor_list), len([k for k in vars(all_sensors) if k.find('_') > 0]),
             'fake' if opts.fake_cam else 'real'))

# Create device server that is main bridge between KATCP and SPEAD
server = Katcp2SpeadDeviceServer(all_sensors, sensor_list, tx_period=0.5,
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
    if opts.fake_cam:
        fake_cam.disconnect()
