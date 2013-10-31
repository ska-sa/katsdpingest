#!/usr/bin/env python
#
# Script that simulates a basic observation, starting and stopping various
# components (CBF simulator, ingest and katcp2spead) and playing out the
# relevant sensors from the CAM system to katcp2spead.
#

import optparse
import logging
import gzip
import time
import re
import os.path

import numpy as np
from pkg_resources import resource_filename

from katcp import DeviceServer, Sensor
import katconf
from katcorelib import build_client
from katsdpingest import __version__


class FakeCamSensorServer(DeviceServer):
    """Device server that serves fake CAM sensors for a simulated observation.

    Parameters
    ----------
    sensors : file object, string, list of strings or generator
        File-like object or filename of CSV file listing sensors to serve, with
        header row followed by rows with format 'name, description, unit, type'

    """

    VERSION_INFO = ("fake_cam_sensor", 0, 1)
    BUILD_INFO = ("fake_cam_sensor", 0, 1, __version__)

    def __init__(self, sensors, *args, **kwargs):
        self.sensors = np.loadtxt(sensors, delimiter=',', skiprows=1, dtype=np.str)
        super(FakeCamSensorServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        """Populate the dictionary of sensors."""
        for fields in self.sensors:
            name, description, units, sensor_type = [f.strip() for f in fields]
            sensor_type = Sensor.parse_type(sensor_type)
            params = ['unknown'] if sensor_type == Sensor.DISCRETE else None
            self.add_sensor(Sensor(sensor_type, name, description, units, params))


# Obtain default sensor list and observation event files
conf_dir = resource_filename('katsdpingest', 'conf')
default_sensors = os.path.join(conf_dir, 'rts_sensors.csv')
default_events = os.path.join(conf_dir, 'rts_scp_events.txt.gz')

# Parse command-line options
parser = optparse.OptionParser(usage='%prog [options]',
                               description="Perform simulated observation for ingest.")
parser.add_option('--sensor-list', default=default_sensors,
                  help='Name of text file containing list of sensors to serve '
                       '(default=%default)')
parser.add_option('--sensor-events', default=default_events,
                  help='Name of text file containing sequence of sensor events '
                       '(default=%default)')
parser.add_option('--fake-cam-host', default='',
                  help='Address of host that will receive KATCP commands for '
                       'fake CAM sensor server (default all hosts)')
parser.add_option('--fake-cam-port', type=int, default=2047,
                  help='Port on which to receive KATCP commands for fake CAM '
                       'sensor server (default=%default)')
parser.add_option('--cbf-host', default='localhost',
                  help='Host address of CBF simulator (default=%default)')
parser.add_option('--cbf-port', type=int, default=2041,
                  help='Port of CBF simulator KATCP interface (default=%default)')
parser.add_option('--ingest-host', default='localhost',
                  help='Host address of ingest system (default=%default)')
parser.add_option('--ingest-kcp-port', type=int, default=2040,
                  help='Port of ingest KATCP interface (default=%default)')
parser.add_option('--ingest-spead-port', type=int, default=7147,
                  help='Port of ingest SPEAD metadata interface (default=%default)')
parser.add_option('--katcp2spead-host', default='localhost',
                  help='Host address of katcp2spead server (default=%default)')
parser.add_option('--katcp2spead-port', type=int, default=2045,
                  help='Port of katcp2spead KATCP interface (default=%default)')
parser.add_option('-c', '--sysconfig', default='/var/kat/katconfig',
                  help='Configuration directory, can be overrided by KATCONF '
                       'environment variable (default=%default)')
parser.add_option('-l', '--logging',
                  help='Level to use for basic logging or name of logging '
                       'configuration file (default log/log.<SITENAME>.conf)')
opts, args = parser.parse_args()

# Setup configuration source and configure logging (via conf file or directly)
katconf.set_config(katconf.environ(opts.sysconfig))
katconf.configure_logging(opts.logging)
logger = logging.getLogger("kat.ingest.simobserve")

# Load sensor events file that simulates an observation
event_pattern = re.compile(r'^(\S+)\s+(\S+)\s+(\S+)\s+(.+)')
events_file = gzip.open(opts.sensor_events) if opts.sensor_events.endswith('.gz') \
              else open(opts.sensor_events)
# Extract mini-header containing dataset, CBF instrument, start time and end time
dataset = events_file.readline().split()[1]
cbf_instrument = events_file.readline().split()[1]
file_start_time = float(events_file.readline().split()[1])
file_end_time = float(events_file.readline().split()[1])
events = [event_pattern.match(line.strip()).groups() for line in events_file]
events_file.close()
logger.info('Loaded %d sensor events associated with %g seconds of dataset %s' %
            (len(events), file_end_time - file_start_time, dataset))

# Create device server that is main bridge between KATCP and SPEAD
server = FakeCamSensorServer(opts.sensor_list, host=opts.fake_cam_host,
                             port=opts.fake_cam_port)
# Spawn new thread to handle KATCP requests to device server
server.start()

# Connect to components of ingest system (ingest, katcp2spead and CBF simulator)
cbf = build_client('cbf', opts.cbf_host, opts.cbf_port,
                   required=True, controlled=True)
ingest = build_client('ingest', opts.ingest_host, opts.ingest_kcp_port,
                      required=True, controlled=True)
katcp2spead = build_client('katcp2spead', opts.katcp2spead_host, opts.katcp2spead_port,
                           required=True, controlled=True)
logger.info('Connected to ingest system components')

# Use the rest of this thread to play through observation sequence
try:
    # Initialise mini capture session
    ingest.req.capture_init()
    katcp2spead.req.start_stream()
    katcp2spead.req.add_destination(opts.ingest_host, opts.ingest_spead_port)
    cbf.req.capture_start(cbf_instrument)

    start_time = time.time()
    shift_time = lambda t: t - file_start_time + start_time
    progress = 0
    cbf_running = True
    for n, (timestamp, name, status, value) in enumerate(events):
        now = time.time()
        event_time = shift_time(float(timestamp))
        if (100 * n) // len(events) == progress:
            logger.info('Progress: %d%% complete, event happened %.1f seconds from start' %
                        (progress, event_time - start_time))
            progress += 5
        # Stop correlator if end_time reached but keep going with last bit of sensor data
        if now >= shift_time(file_end_time) and cbf_running:
            cbf.req.capture_stop(cbf_instrument)
            cbf_running = False
        # Wait until next event
        time_to_event = event_time - now
        if time_to_event > 0:
            time.sleep(time_to_event)
        sensor = server.get_sensor(name)
        # Force value to be accepted by discrete sensor
        if sensor.stype == 'discrete':
            sensor._kattype._values.append(value)
            sensor._kattype._valid_values.add(value)
        # Update sensor with new event
        sensor.set(event_time, sensor.STATUS_NAMES[status], sensor.parse_value(value))
    logger.info('Progress: 100%% complete, last event happened %.1f seconds from start' %
                (event_time - start_time,))

    katcp2spead.req.remove_destination(opts.ingest_host, opts.ingest_spead_port)
    katcp2spead.req.stop_stream()
    ingest.req.capture_done()
finally:
    server.stop()
    server.join()
    ingest.disconnect()
    cbf.disconnect()
    katcp2spead.disconnect()
