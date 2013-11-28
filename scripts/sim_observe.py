#!/usr/bin/env python
#
# Script that simulates a basic observation, starting and stopping various
# components (CBF simulator, ingest and cam2spead) and playing out the
# relevant attribute and sensor events from the CAM system to cam2spead.
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
from katcp.kattypes import return_reply, Str, Address
import katconf
from katcorelib import build_client
from katsdpingest import __version__


class FakeCamEventServer(DeviceServer):
    """Device server that serves fake CAM events for a simulated observation.

    Parameters
    ----------
    attributes : dict mapping string to string
        Attributes as key-value string pairs which are streamed once upfront
    sensors : file object, string, list of strings or generator
        File-like object or filename of CSV file listing sensors to serve, with
        header row followed by rows with format 'name, description, unit, type'

    """

    VERSION_INFO = ("fake_cam_event", 0, 1)
    BUILD_INFO = ("fake_cam_event", 0, 1, __version__)

    def __init__(self, attributes, sensors, *args, **kwargs):
        self.attributes = attributes
        self.sensors = np.loadtxt(sensors, delimiter=',', skiprows=1, dtype=np.str)
        super(FakeCamEventServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        """Populate sensor objects on server."""
        for fields in self.sensors:
            name, description, units, sensor_type = [f.strip() for f in fields]
            sensor_type = Sensor.parse_type(sensor_type)
            params = ['unknown'] if sensor_type == Sensor.DISCRETE else None
            self.add_sensor(Sensor(sensor_type, name, description, units, params))

    @return_reply(Str())
    def request_get_attributes(self, req, msg):
        """Return dictionary of attributes."""
        logger.info('Returning %d attributes' % (len(attributes,)))
        return ("ok", repr(self.attributes))


def load_events(filename):
    """Load events file that simulates an observation."""
    attr_pattern = re.compile(r'^([^#]\w+)\s*[:=]\s*(\S.*)\s*$')
    sensor_pattern = re.compile(r'^([^#][\d.]+)\s+(\w+)\s+(\w+)\s+(\S.*)\s*$')
    events_file = gzip.open(filename) if filename.endswith('.gz') else open(filename)
    lines = events_file.readlines()
    # Extract sensor events wherever sensor pattern matches
    sensor_matches = [sensor_pattern.match(line) for line in lines]
    sensors = [m.groups() for m in sensor_matches if m]
    # Extract attributes (including simulation-specific ones) from remaining lines
    attr_matches = [attr_pattern.match(line) for (line, m) in
                    zip(lines, sensor_matches) if m is None]
    attributes = dict([m.groups() for m in attr_matches if m])
    return attributes, sensors


# Obtain default sensor list and observation event files
conf_dir = resource_filename('katsdpingest', 'conf')
default_sensors = os.path.join(conf_dir, 'rts_sensors.csv')
default_events = os.path.join(conf_dir, 'rts_scp_events.txt')

# Parse command-line options
parser = optparse.OptionParser(usage='%prog [options]',
                               description="Perform simulated observation for ingest.")
parser.add_option('--sensor-list', default=default_sensors,
                  help='Name of text file containing list of sensors to serve '
                       '(default=%default)')
parser.add_option('--cam-events', default=default_events,
                  help='Name of text file containing sequence of CAM events '
                       '(default=%default)')
parser.add_option('--fake-cam-host', default='',
                  help='Address of host that will receive KATCP commands for '
                       'fake CAM event server (default all hosts)')
parser.add_option('--fake-cam-port', type=int, default=2047,
                  help='Port on which to receive KATCP commands for fake CAM '
                       'event server (default=%default)')
parser.add_option('--cbf-host', default='localhost',
                  help='Host address of CBF simulator (default=%default)')
parser.add_option('--cbf-port', type=int, default=2041,
                  help='Port of CBF simulator KATCP interface (default=%default)')
parser.add_option('--ingest-host', default='localhost',
                  help='Host address of ingest system (default=%default)')
parser.add_option('--ingest-kcp-port', type=int, default=2040,
                  help='Port of ingest KATCP interface (default=%default)')
parser.add_option('--ingest-cam-spead-port', type=int, default=7147,
                  help='Port of ingest CAM SPEAD interface (default=%default)')
parser.add_option('--ingest-cbf-spead-port', type=int, default=7148,
                  help='Port of ingest CBF SPEAD interface (default=%default)')
parser.add_option('--cam2spead-host', default='localhost',
                  help='Host address of cam2spead server (default=%default)')
parser.add_option('--cam2spead-port', type=int, default=2045,
                  help='Port of cam2spead KATCP interface (default=%default)')
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

attributes, events = load_events(opts.cam_events)
dataset = attributes.pop('sim_dataset')
subarray = attributes.pop('sim_subarray')
cbf_instrument = attributes.pop('sim_cbf_instrument')
data_product = '_'.join((subarray, cbf_instrument))
file_start_time = float(attributes.pop('sim_start_time'))
file_end_time = float(attributes.pop('sim_end_time'))
logger.info('Loaded %d attributes and %d sensor events associated with %g seconds of dataset %s' %
            (len(attributes), len(events), file_end_time - file_start_time, dataset))

# Create device server that is main bridge between KATCP and SPEAD
server = FakeCamEventServer(attributes, opts.sensor_list,
                            host=opts.fake_cam_host, port=opts.fake_cam_port)
# Spawn new thread to handle KATCP requests to device server
server.start()

# Connect to components of ingest system (ingest, cam2spead and CBF simulator)
cbf = build_client('cbf', opts.cbf_host, opts.cbf_port,
                   required=True, controlled=True)
ingest = build_client('ingest', opts.ingest_host, opts.ingest_kcp_port,
                      required=True, controlled=True)
cam2spead = build_client('cam2spead', opts.cam2spead_host, opts.cam2spead_port,
                           required=True, controlled=True)
logger.info('Connected to ingest system components')

# Use the rest of this thread to play through observation sequence
try:
    # Initialise mini capture session
    cbf.req.capture_destination(cbf_instrument, opts.ingest_host,
                                opts.ingest_cbf_spead_port)
    ingest_dest = Address().pack((opts.ingest_host, opts.ingest_cam_spead_port))
    cam2spead.req.stream_configure(data_product, ingest_dest)
    ingest.req.capture_init()
    cam2spead.req.stream_start(data_product)
    cbf.req.capture_start(cbf_instrument)
    cam2spead.req.set_label(data_product, '')
    cam2spead.req.set_obs_params(data_product, 'observer', 'Simmer')

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

    cbf.req.capture_stop(cbf_instrument)
    cam2spead.req.stream_stop(data_product)
    ingest.req.capture_done()
finally:
    server.stop()
    server.join()
    ingest.disconnect()
    cbf.disconnect()
    cam2spead.disconnect()
