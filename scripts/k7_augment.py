#!/usr/bin/python

# This script is for augmenting Kat-7 HDF5v2 files
#
# The output file should conform to the specification as described in the hdf5 v2 design record.
#
# Briefly this means it will contain Data and MetaData main groups.
# MetaData is further split into:
#  Configuration (static config values at the start of obs)
#  Markup (flags and other markup produced during processing)
#  History (log of changes to the file and observer observations)
#  Sensor (mandatory sensors as required by the minor version and optional sensors as supplied by the observer)
#

import sys
import re
import time
import os
import signal
import logging
import traceback
from optparse import OptionParser

import numpy as np
from h5py import File

import katcorelib
import katcorelib.targets
import katconf

major_version = 2
 # only augment files of this major version
augment_version = 1
 # the minor version number created by this version of augment
 # will override any existing version

errors = 0
section_reports = {}

def get_input_info(array_config):
    """Get correlator input mapping and delays from config system.

    Parameters
    ----------
    array_config : :class:`katcorelib.targets.ArrayConfig` object
        ArrayConfig object from which to extract correlator info

    Returns
    -------
    config_antennas : set of ints
        The numbers of antennas that are connected to the correlator
    dbe_delay : dict
        Mapping from DBE input string to delay in seconds (also a string)
    real_to_dbe : dict
        Mapping from antenna+pol string to DBE input string

    """
    config_antennas, dbe_delay, real_to_dbe = set(), {}, {}
    for k, v in array_config.correlator.inputs.iteritems():
        # Connection on DBE side is labelled '0x', '1y', etc.
        dbe = '%d%s' % k
        # Connection on antenna side is labelled '1h', '2h', etc.
        # This assumes the antenna name is 'antX', where X is the antenna number
        ant_num, pol, delay = int(v[0][3:]), v[1], v[2]
        real = '%d%s' % (ant_num, pol)
        config_antennas.add(ant_num)
        dbe_delay[dbe] = '%.16e' % delay
        real_to_dbe[real] = dbe
    return config_antennas, dbe_delay, real_to_dbe

def get_antenna_info(array_config):
    """Get antenna objects, positions and diameter from config system.

    Parameters
    ----------
    array_config : :class:`katcorelib.targets.ArrayConfig` object
        ArrayConfig object from which to extract antenna info

    Returns
    -------
    antennas : dict
        Mapping from antenna name to :class:`katpoint.Antenna` object
    antenna_positions : array, shape (N, 3)
        Antenna positions in ECEF coordinates, in metres
    antenna_diameter : float
        Antenna dish diameter (taken from first dish in array)

    """
    antenna_positions, antennas = [], {}
    for ant_name, ant_cfg in array_config.antennas.iteritems():
        antenna_positions.append(ant_cfg.observer.position_ecef)
        antennas[ant_name] = ant_cfg.observer
    antenna_positions = np.array(antenna_positions)
    antenna_diameter = antennas.values()[0].diameter
    return antennas, antenna_positions, antenna_diameter

def load_csv_with_header(csv_file):
    """Load CSV file containing commented-out header with key-value pairs.

    This is used to load the noise diode model CSV files, which contain extra
    metadata in its headers.

    Parameters
    ----------
    csv_file : file object or string
        File object of opened CSV file, or string containing the loaded csv lines

    Returns
    -------
    csv_array : array, shape (N, M)
        CSV data as a 2-dimensional array with N rows and M columns
    attrs : dict
        Key-value pairs extracted from header

    """
    try:
        if isinstance(csv_file, basestring):
            csv_lines = csv_file.splitlines()
            #Each line has comma-delimited string with "Frequency, Temperature" both as floats
            csv_array = np.array([ [float(l[0]),float(l[1])] for l in [line.split(",") for line in csv_file.splitlines() if line[0] != '#']])
        else:
            open(csv_file)
            start = csv_file.tell()
            csv_array = np.loadtxt(csv_file, comments='#', delimiter=',')
            csv_file.seek(start) #Go back to start of file
            csv_lines = csv_file.readlines()
            close(csv_file)
    except Exception, e:
        print "Failed to load csv_file (%s). %s\n" % (csv_file, e)
        raise
    header = [line[1:].strip() for line in csv_lines if line[0] == '#']
    keyvalue = re.compile('\A([a-z]\w*)\s*[:=]\s*(.+)')
    attrs = dict([keyvalue.match(line).groups() for line in header if keyvalue.match(line)])
    return csv_array, attrs


def get_sensor_data(sensor, start_time, end_time, dither=1, initial_value=False):
    """Returns a recarray containing timestamp and value columns for the specified sensor.

    Parameters
    ----------
    sensor : KATSensor
        The sensor from which to retrieve the data
    start_time : integer
        The start time of the period for which to retrieve data
    end_time : integer
        The end time of the period for which to retrieve data
    dither : integer
        The number of seconds either side of the specified start and end to retrieve data for
        default: 1
    initial_value : boolean
        If true then an initial value for the sensor (i.e. the last known good value before the requested range)
        is retrieve and inserted as the first entry in the returned array. Note this is an expensive
        operation and should be reserved for those cases known to require it.
        default: False
    Returns
    -------
    array : dtype=[('timestamp','float'),('value', '<f4')]
    """
    print "Pulling data for sensor %s from %i to %i\n" % (sensor.name, start_time, end_time)
    start_time = start_time - dither
    end_time = end_time + dither
    initial_data = [[], [], []]
    if initial_value:
        initial_data = sensor.get_stored_history(select=False,start_seconds=start_time,end_seconds=start_time, last_known=True)
        print "Initial value fetch:",initial_data
    stime = time.time()
    data = sensor.get_stored_history(select=False,start_seconds=start_time,end_seconds=end_time)
    print "Retrieved data of length",len(data[1]),"in",time.time()-stime,"s"
    return np.rec.fromarrays([initial_data[0] + data[0], initial_data[1] + data[1], initial_data[2] + data[2]], names='timestamp, value, status')

def insert_sensor(device, name, group, obs_start, obs_end, int_time, iv=False, default=None):
    global errors
    pstime = time.time()
    sensor_len = 0
    full_name = device + "_" + name
    try:
        sensor_i = getattr(kat.sensors, full_name)
        if sensor_i.name in group:
            sensor_len = group[sensor_i.name].len()
            section_reports[full_name] = "Success (Note: Existing data for this sensor was not changed.)"
            return sensor_len
        data = get_sensor_data(sensor_i, obs_start, obs_end, int_time, initial_value=iv)
        sensor_len = np.multiply.reduce(data.shape)
        if sensor_len == 0:
            if default is not None:
                section_reports[full_name] = "Warning: Sensor %s has no data for the specified time period. Inserting default value of" % (default,)
                s_dset = group.create_dataset(sensor_i.name, data=np.rec.fromarrays([[time.time()],[default],['failure']], names='timestamp, value, status'))
            else:
                section_reports[full_name] = "Warning: Sensor %s has no data for the specified time period. Inserting empty dataset."
                s_dset = group.create_dataset(sensor_i.name, [], maxshape=None)
        else:
            s_dset = group.create_dataset(sensor_i.name, data=data)
            section_reports[full_name] = "Success"
        s_dset.attrs['name'] = sensor_i.name
        s_dset.attrs['description'] = sensor_i.description
        s_dset.attrs['units'] = sensor_i.units
        s_dset.attrs['type'] = sensor_i.type
    except AttributeError:
         # sensor does not exist
        section_reports[full_name] = "Error: Cannot find sensor %s. This is most likely a configuration issue." % (full_name,)
        errors += 1
    except Exception, err:
        if not str(err).startswith('Name already exists'):
            section_reports[full_name] = "Error: Failed to create dataset for "+ full_name + " (" + str(err) + ")"
            errors += 1
        else:
            section_reports[full_name] = "Success (Note: Existing data for this sensor was not changed.)"
    smsg = "Creation of dataset for sensor " + full_name + " took " + str(time.time() - pstime) + "s"
    if options.verbose: print smsg
    return sensor_len

def create_group(f, name):
    try:
        ng = f.create_group(name)
    except Exception:
        ng = f[name]
    return ng

def terminate(_signum, _frame):
    print "augment - User requested terminate..."
    print "augment stopped"
    sys.exit(0)


def get_files_in_dir(directory):
    files = []
    p = os.listdir(directory+"/")
    p.sort()
    while p:
        x = p.pop()
        if x.endswith("unaugmented.h5"):
            files.append(directory+"/" + x)
    return files

def get_single_value(group, name):
    """Return a single value from an attribute or dataset of the given name.

       If data is retrieved from a dataset, this functions raises an error
       if the values in the dataset are not all the same. Otherwise it
       returns the first value."""
    value = group.attrs.get(name, None)
    if value is not None:
        return value
    dataset = group.get(name, None)
    if dataset is None:
        raise ValueError("Could not find attribute or dataset named %r/%r" % (group.name, name))
    if not dataset.len():
        raise ValueError("Found dataset named %r/%r but it was empty" % (group.name, name))
    if not all(dataset.value == dataset.value[0]):
        raise ValueError("Not all values in %r/%r are equal. Values found: %r" % (group.name, name, dataset.value))
    return dataset.value[0]

def print_tb():
    """Print a traceback if options.verbose is True."""
    if options.verbose:
        traceback.print_exc()

######### Start of augment script #########

parser = OptionParser()
parser.add_option("-b", "--batch", action="store_true", default=False, help="If set augment will process all unaugmented files in the directory specified by -d, and then continue to monitor this directory. Any new files that get created will be augmented in sequence.")
parser.add_option("-c", "--config", dest='config', default='/var/kat/katconfig', help='look for configuration files in folder CONF [default is KATCONF environment variable or /var/kat/katconfig]')
parser.add_option("-d", "--dir", default='/var/kat/data', help="Process all unaugmented files in the specified directory. [default=%default]")
parser.add_option("-f", "--file", default="", help="Fully qualified path to a specific file to augment. [default=%default]")
parser.add_option("-s", "--system", default="systems/local.conf", help="System configuration file to use. [default=%default]")
parser.add_option("-o", "--override", dest="force", action="store_true", default=False, help="If set, previously augmented files will be re-augmented. Only useful in conjunction with a single specified file.")
parser.add_option("--dbe", dest="dbe_name", default="dbe7", help="Name of kat client to use as the correlator proxy. [default=%default]")
parser.add_option("-v", "--verbose", action="store_true", default=False, help="Verbose output.")
parser.add_option('-l', '--logging', dest='logging', type='string', default=None, metavar='LOGGING',
            help='level to use for basic logging or name of logging configuration file; ' \
            'default is /log/log.<SITENAME>.conf')

options, args = parser.parse_args()

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)

# Setup configuration source
katconf.set_config(katconf.environ(options.config))
    
# set up Python logging
katconf.configure_logging(options.logging)

log_name = 'kat.k7aug'
logger = logging.getLogger(log_name)
logger.info("Logging started")
activitylogger = logging.getLogger('activity')
activitylogger.setLevel(logging.INFO)
activitylogger.info("Activity logging started")

state = ["|","/","-","\\"]
batch_count = 0

antenna_sensors = ["activity","target",
                   "pos_actual_scan_azim","pos_actual_scan_elev","pos_actual_refrac_azim","pos_actual_refrac_elev",
                   "pos_actual_pointm_azim","pos_actual_pointm_elev","pos_request_scan_azim","pos_request_scan_elev",
                   "pos_request_refrac_azim","pos_request_refrac_elev","pos_request_pointm_azim","pos_request_pointm_elev",
                   "rfe3_rfe15_noise_pin_on","rfe3_rfe15_noise_coupler_on",
                   "rfe5_attenuator_horizontal", "rfe5_attenuator_vertical"]
 # a list of antenna sensors to insert
enviro_sensors = ["asc_air_temperature","asc_air_pressure","asc_air_relative_humidity","asc_wind_speed","asc_wind_direction"]
 # a list of enviro sensors to insert
rfe_sensors = ["rfe7_lo1_frequency"]
 # a list of RFE sensors to insert
beam_sensors = ["target"]
 # a list of sensors for beam 0
dbe_sensors = ["target", "auto_delay", "dbe_mode"]
 # a (partial) list of DBE sensors to insert
dbe_proxy = options.dbe_name

sensors_iv = ["activity", "target", "rfe3_rfe15_noise_pin_on", "rfe3_rfe15_noise_coupler_on",
              "rfe5_attenuator_horizontal", "rfe5_attenuator_vertical",
              "rfe7_lo1_frequency", "auto_delay", "dbe_mode"]
 # indicate which sensors will require an initial value fetch

######### Start of augment code #########

files = []

if options.file == "":
    files = get_files_in_dir(options.dir)
else:
    files.append(options.file)

if len(files) == 0 and not options.batch:
    print "No files matching the specified criteria where found..."
    sys.exit(0)

smsg = "Found %d files to process" % len(files)
print smsg
activitylogger.info(smsg)

# build an kat object for history gathering purposes
print "Creating KAT connections..."
kat = katcorelib.tbuild(system=options.system, log_file="kat.k7aug.log", log_level=logging.ERROR)

# check that we have basic connectivity (i.e. two antennas)
time.sleep(2)
while not kat.rfe7.is_connected():
     # wait for at least rfe7 to become stable as we query it straight away.
     # also serves as a basic connectivity check.
    status = "\r%s Connection to RFE7 not yet established. Waiting for connection (possibly futile)..." % str(state[batch_count % 4])
    sys.stdout.write(status)
    sys.stdout.flush()
    time.sleep(30)
    batch_count += 1

#kat.disconnect()   # we dont need live connection anymore
section_reports['configuration'] = str(options.system)

katconfig = katcorelib.conf.KatuilibConfig(str(options.system))
arrpath = katconfig.conf.get("array","array_kat7")
arrconf = katconf.ArrayConfig(arrpath)
array_config = katcorelib.targets.ArrayConfig(arrconf)
 # retrieve array configuration object for correlator
config_antennas, dbe_delay, real_to_dbe = get_input_info(array_config)
 # return dicts showing the current mapping between dbe inputs and real antennas
antennas, antenna_positions, antenna_diameter = get_antenna_info(array_config)
 # build the description and position (in ecef coords) arrays for the antenna in the selected configuration
 # map of noise diode model names to filenames
print "Antennas", antennas, antenna_positions, antenna_diameter
for antenna in config_antennas:
    ant_name = 'ant' + str(antenna)
    print ant_name, sorted(array_config.antennas[ant_name].ant_config_dict.keys())
input_map = sorted(('ant' + real, dbe) for real, dbe in real_to_dbe.items())
 # map of antenna inputs (e.g. ant1h) to dbe inputs (e.g. 0x)

while(len(files) > 0 or options.batch):
    for fname in files:
        errors = 0
        fst = time.time()
        smsg = "Starting augment of file %s" % fname
        print "\n%s" % smsg
        logger.info(smsg)
        activitylogger.info(smsg)
        new_extension = "h5"
        new_path = ""
        try:
            f = File(fname, 'r+')
            current_version = f['/'].attrs.get('version', "0.0").split('.', 1)
            if current_version[0] != str(major_version):
                smsg = "This version of augment required HDF5 files of version %i to augment. Your file has major version %s\n" % (major_version, current_version[0])
                print smsg
                logger.info(smsg)
                continue
            last_run = f['/'].attrs.get('augment_ts',None)
            f['/'].attrs['augment_errors'] = 0
            if last_run:
                smsg = "Warning: This file has already been augmented: " + str(last_run)
                print smsg
                logger.warn(smsg)
                activitylogger.warn(smsg)
                if not options.force:
                    smsg = "To force reprocessing, please use the -o option."
                    print smsg
                    logger.info(smsg)
                    continue
                else:
                    section_reports['reaugment'] = "Augment was previously done in this file on " + str(last_run)
            f['/'].attrs['version'] = "%i.%i" % (major_version, augment_version)

            obs_start = f['/Data/timestamps'].value[1]
             # first timestamp is currently suspect
            f['/Data'].attrs['ts_of_first_timeslot'] = obs_start
            obs_end = f['/Data/timestamps'].value[-1]
            smsg = "Observation session runs from %s to %s\n" % (time.ctime(obs_start), time.ctime(obs_end))
            print smsg
            logger.info(smsg)
            int_time = get_single_value(f['/MetaData/Configuration/Correlator'], 'int_time')
            f['/MetaData/Configuration/Correlator'].attrs['input_map'] = input_map

            hist = create_group(f,"/History")
            sg = create_group(f, "/MetaData/Sensors")
            ag = create_group(sg, "Antennas")
            acg = create_group(f, "/MetaData/Configuration/Antennas")
            bg = create_group(sg, "Beams")
            dbeg = create_group(sg, "DBE")
            rfeg = create_group(sg, "RFE")
            eg = create_group(sg, "Enviro")

            for antenna in range(1,8):
                antenna = str(antenna)
                ant_name = 'ant' + antenna
                a = create_group(ag, ant_name)
                ac = create_group(acg, ant_name)
                stime = time.time()
                for sensor in antenna_sensors:
                    insert_sensor(ant_name, sensor, a, obs_start, obs_end, int_time, iv=sensor in sensors_iv)
                for pol in ('h', 'v'):
                    sensor = "rfe7_downconverter_%s_%s_attenuation" % (ant_name, pol)
                    insert_sensor("rfe7", sensor, rfeg, obs_start, obs_end, int_time, iv=True)
                 # add RFE7 attenuators too
                if options.verbose:
                    smsg = "Overall creation of sensor table for antenna " + antenna + " took " + str(time.time()-stime) + "s"
                    print smsg
                    logger.debug(smsg)
                # noise diode models
                try:
                    ac.attrs['description'] = antennas[ant_name].description
                except Exception:
                    print_tb()
                    section_reports[ant_name + ' description'] = "Error: Cannot find description for antenna %r" % (ant_name,)
                for pol in ['h','v']:
                    for nd in ['coupler','pin']:
                        nd_name = "noise_diode_model_%s_%s" % (nd, pol)
                        nd_fname = "unknown"
                        model = np.zeros((1,2), dtype=np.float32)
                        attrs = {}
                        try:
                            model, attrs = load_csv_with_header(array_config.antennas[ant_name].ant_config_dict[nd_name])
                        except Exception, e:
                            print_tb()
                            smsg = "Failed to open noise diode model (for %s). Inserting null noise diode model. (%s)" % (nd_name, e)
                            print smsg
                            logger.error(smsg)
                        try:
                            nd = ac.create_dataset("%s_%s_noise_diode_model" % (pol, nd), data=model)
                            for key,val in attrs.iteritems(): nd.attrs[key] = val
                        except Exception:
                            print_tb()
                            smsg = "Dataset %s.%s_%s_noise_diode_model already exists. Not replacing existing model." % (ac.name, pol, nd)
                            print smsg
                            logger.info(smsg)
             # end of antenna loop

            b0 = create_group(bg, "Beam0")
            for sensor in beam_sensors:
                insert_sensor(dbe_proxy, sensor, b0, obs_start, obs_end, int_time, iv=sensor in sensors_iv)

            for sensor in dbe_sensors:
                insert_sensor(dbe_proxy, sensor, dbeg, obs_start, obs_end, int_time, iv=sensor in sensors_iv)
            try:
                if hasattr(kat.sensors, dbe_proxy + "_dbe_centerfrequency"):
                    insert_sensor(dbe_proxy, "dbe_centerfrequency", dbeg, obs_start, obs_end, int_time, iv=True)
                    dbe_center_freq = dbeg[getattr(kat.sensors, dbe_proxy + "_dbe_centerfrequency").name].value
                else:
                    dbe_mode = dbeg[getattr(kat.sensors, dbe_proxy + "_dbe_mode").name].value[-1][1]
                     # get the value of the last known good DBE mode
                    sensor = 'dbe_' + dbe_mode + '_centerfrequency'
                    insert_sensor(dbe_proxy, sensor, dbeg, obs_start, obs_end, int_time, iv=True)
                    try:
                        dbe_center_freq = dbeg[getattr(kat.sensors, dbe_proxy + "_" + sensor).name].value
                        dbeg.create_dataset('dbe.centerfrequency', data=dbe_center_freq)
                    except Exception:
                        print_tb()
                        smsg = "DBE center frequency already saved. Not replacing existing dbe.centerfrequency."
                        print smsg
                        logger.info(smsg)
            except Exception:
                print_tb()
                smsg = "Failure obtaining DBE center frequency - using default of 200 MHz."
                print smsg
                logger.info(smsg)
                dbe_center_freq = np.rec.fromarrays([[obs_start - 3600], [200e6], ['nominal']], names='timestamp, value, status')
                # make it appear as if DBE center freq was set to the default value an hour before the observation

            for sensor in rfe_sensors:
                insert_sensor("rfe7", sensor, rfeg, obs_start, obs_end, int_time, iv=sensor in sensors_iv, default=np.nan)
                if sensor == "rfe7_lo1_frequency":
                    try:
                        lo1_freq = np.sort(rfeg[getattr(kat.sensors, "rfe7_" + sensor).name].value, order=['timestamp'])
                        dbe_center_freq = np.sort(dbe_center_freq, order=['timestamp'])
                         # collect rfe7 lo1 frequencies and dbe center frequencies and ensure that they are sorted
                        on_or_preceding = dbe_center_freq['timestamp'].searchsorted(lo1_freq['timestamp'] + int_time, side='right') - 1
                         # find DBE center freq event preceding or within 1 dump of each lo1 freq change
                         # this assumes that DBE center freq is only set once at the start or in lock-step with lo1 freq changes
                        on_or_preceding[on_or_preceding < 0] = 0
                         # since the first DBE center freq event still precedes the dataset, it is safe to revert to it
                        center_freq = lo1_freq.copy()
                        center_freq['value'] += 200e6 - dbe_center_freq['value'][on_or_preceding] - 4.2e9
                         # calculate sky center frequency in Hz (see Mantis #1531)
                        rfeg.create_dataset('center-frequency-hz', data=center_freq)
                    except Exception:
                        print_tb()
                        smsg = "Center frequency already saved. Not replacing existing center-frequency-hz."
                        print smsg
                        logger.info(smsg)

            stime = time.time()
            for sensor in enviro_sensors:
                insert_sensor("anc", sensor, eg, obs_start, obs_end, int_time)
            if options.verbose:
                smsg = "Overall creation of enviro sensor table took " + str(time.time()-stime) + "s"
                print smsg
                logger.debug(smsg)
            f['/'].attrs['augment_ts'] = time.time()

        except Exception, err:
            print_tb()
            section_reports["general"] = "Exception: " + str(err)
            errors += 1
            smsg = "Failed to run augment. File will be  marked as 'failed' and ignored:  (" + str(err) + ")"
            print smsg
            logger.error(smsg)
            activitylogger.error(smsg)
            new_extension = "augment_failed.h5"
            new_path = "./failed/"

        try:
            log = np.rec.fromarrays([np.array(section_reports.keys()), np.array(section_reports.values())], names='section, message')
            f['/'].attrs['augment_errors'] = errors
            if options.force:
                try:
                    del hist['augment_log']
                except KeyError:
                    pass # no worries if the augment log does not exist, a new one is written...
            try:
                hist.create_dataset("augment_log", data=log)
            except ValueError:
                hist['augment_log'].write_direct(log)
        except Exception, err:
            print_tb()
            smsg = "Warning: Unable to create augment_log dataset. (" + str(err) + ")"
            print smsg
            logger.error(smsg)
            activitylogger.error(smsg)
        f.close()

        if options.verbose:
            print "\n\nReport"
            print "======"
            logger.debug("=====Report")
            keys = section_reports.keys()
            keys.sort()
            for k in keys:
                print k.ljust(50),section_reports[k]
                logger.debug("%s %s" % (k.ljust(50), section_reports[k]))
        try:
            #Drop the last two extensions of the file 123456789.xxxxx.h5 becomes 123456789.
            #And then add the new extension in its place thus 123456789.unaugmented.h5 becomes 123456789.h5 or 123456789.failed.h5
            lst = fname.split(".")
            renfile = new_path + lst[0] + "." + new_extension
            os.rename(fname, renfile)
            smsg = "File has been renamed to " + str(renfile) + "\n"
            print smsg
            logger.info(smsg)
        except Exception:
            print_tb()
            smsg = "Failed to rename " + str(fname) + " to " + str(renfile)
            print smsg + ". This is most likely a permissions issue. Please resolve these and either manually rename the file or rerun augment with the -o option."
            logger.error(smsg)
            activitylogger.error(smsg)
            continue
        smsg = (errors == 0 and "No errors found." or str(errors) + " potential errors found. Please inspect the augment log by running 'h5dump -d /History/augment_log " + str(renfile) + "'.")
        print smsg
        if errors == 0:
            logger.info(smsg)
            activitylogger.info(smsg)
        else:
            logger.error(smsg)
            activitylogger.error(smsg)

    # if in batch mode check for more files...
    files = []
    if options.batch:
        time.sleep(2)
        status = "\rChecking for new files in %s: %s" % (options.dir,str(state[batch_count % 4]))
        sys.stdout.write(status)
        sys.stdout.flush()
        files = get_files_in_dir(options.dir)
        batch_count += 1

