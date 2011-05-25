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

import katuilib
import katcore.targets
import katconf

major_version = 2
 # only augment files of this major version
augment_version = 0
 # the minor version number created by this version of augment
 # will override any existing version

errors = 0
section_reports = {}

def get_input_info(array_cfg):
    """Get correlator input mapping and delays from config system.

    Parameters
    ----------
    array_cfg : :class:`katcore.targets.ArrayConfig` object
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
    for k, v in array_cfg.correlator.inputs.iteritems():
        # Connection on DBE side is labelled '0x', '1y', etc.
        dbe = '%d%s' % k
        # Connection on antenna side is labelled '1H', '2V', etc.
        # This assumes the antenna name is 'antX', where X is the antenna number
        ant_num, pol, delay = int(v[0][3:]), v[1].capitalize(), v[2]
        real = '%d%s' % (ant_num, pol)
        config_antennas.add(ant_num)
        dbe_delay[dbe] = '%.16e' % delay
        real_to_dbe[real] = dbe
    return config_antennas, dbe_delay, real_to_dbe

def get_antenna_info(array_cfg):
    """Get antenna objects, positions and diameter from config system.

    Parameters
    ----------
    array_cfg : :class:`katcore.targets.ArrayConfig` object
        ArrayConfig object from which to extract antenna info

    Returns
    -------
    antennas : dict
        Mapping from antenna name to :class:`katpoint.Antenna` object
    antenna_positions : array, shape (N, 3)
        Antenna positions in ECEF coordinates, in metres
    antenna_diameter : float
        Antenna dish diameter (taken from first dish in array)
    noise_diode_models : dict
        Mapping from '%(antenna)_%(diode)_%(pol)' to noise diode model CSV file

    """
    antenna_positions, antennas, noise_diode_models = [], {}, {}
    for ant_name, ant_cfg in array_cfg.antennas.iteritems():
        antenna_positions.append(ant_cfg.observer.position_ecef)
        antennas[ant_name] = ant_cfg.observer
        for nd_name, nd_file in ant_cfg.noise_diode_models.iteritems():
            noise_diode_models[ant_name + "_" + nd_name] = nd_file
    antenna_positions = np.array(antenna_positions)
    antenna_diameter = antennas.values()[0].diameter
    return antennas, antenna_positions, antenna_diameter, noise_diode_models

def load_csv_with_header(csv_file):
    """Load CSV file containing commented-out header with key-value pairs.

    This is used to load the noise diode model CSV files, which contain extra
    metadata in its headers.

    Parameters
    ----------
    csv_file : file object or string
        File object of opened CSV file, or string containing the file name

    Returns
    -------
    csv : array, shape (N, M)
        CSV data as a 2-dimensional array with N rows and M columns
    attrs : dict
        Key-value pairs extracted from header

    """
    try:
        csv_file = open(csv_file) if isinstance(csv_file, basestring) else csv_file
    except Exception, e:
        print "Failed to load csv_file (%s). %s\n" % (csv_file, e)
        raise
    start = csv_file.tell()
    csv = np.loadtxt(csv_file, comments='#', delimiter=',')
    csv_file.seek(start)
    header = [line[1:].strip() for line in csv_file.readlines() if line[0] == '#']
    keyvalue = re.compile('\A([a-z]\w*)\s*[:=]\s*(.+)')
    attrs = dict([keyvalue.match(line).groups() for line in header if keyvalue.match(line)])
    return csv, attrs


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
        initial_data = sensor.get_stored_history(select=False,start_time=start_time,end_time=start_time, last_known=True)
        print "Initial value fetch:",initial_data
    stime = time.time()
    data = sensor.get_stored_history(select=False,start_time=start_time,end_time=end_time)
    print "Retrieved data of length",len(data[1]),"in",time.time()-stime,"s"
    return np.rec.fromarrays([initial_data[0] + data[0], initial_data[1] + data[1], initial_data[2] + data[2]], names='timestamp, value, status')

def insert_sensor(name, dataset, obs_start, obs_end, int_time, iv=False, default=None):
    global errors
    pstime = time.time()
    sensor_len = 0
    try:
        sensor_i = kat.sensors.__dict__[name]
        if sensor_i.name in dataset:
            sensor_len = dataset[sensor_i.name].len()
            section_reports[name] = "Success (Note: Existing data for this sensor was not changed.)"
            return sensor_len
        data = get_sensor_data(sensor_i, obs_start, obs_end, int_time, initial_value=iv)
        sensor_len = np.multiply.reduce(data.shape)
        if sensor_len == 0:
            if default is not None:
                section_reports[name] = "Warning: Sensor %s has no data for the specified time period. Inserting default value of" % (default,)
                s_dset = dataset.create_dataset(sensor_i.name, data=np.rec.fromarrays([[time.time()],[default],[0]], names='timestamp, value, status'))
            else:
                section_reports[name] = "Warning: Sensor %s has no data for the specified time period. Inserting empty dataset."
                s_dset = dataset.create_dataset(sensor_i.name, [], maxshape=None)
        else:
            s_dset = dataset.create_dataset(sensor_i.name, data=data)
            section_reports[name] = "Success"
        s_dset.attrs['name'] = sensor_i.name
        s_dset.attrs['description'] = sensor_i.description
        s_dset.attrs['units'] = sensor_i.units
        s_dset.attrs['type'] = sensor_i.type
    except KeyError:
         # sensor does not exist
        section_reports[name] = "Error: Cannot find sensor %s. This is most likely a configuration issue." % (name,)
        errors += 1
    except Exception, err:
        if not str(err).startswith('Name already exists'):
            section_reports[name] = "Error: Failed to create dataset for "+ name + " (" + str(err) + ")"
            errors += 1
        else:
            section_reports[name] = "Success (Note: Existing data for this sensor was not changed.)"
    if options.verbose: print "Creation of dataset for sensor " + name + " took " + str(time.time() - pstime) + "s"
    return sensor_len

def create_group(f, name):
    try:
        ng = f.create_group(name)
    except Exception:
        ng = f[name]
    return ng

def get_lo1_frequency(start_time):
    try:
        return kat.sensors.rfe7_rfe7_lo1_frequency.get_stored_history(select=False, include_central=True, start_time=start_time, end_time=start_time, last_known=True)[1][0]
    except Exception:
        section_reports['lo1_frequency'] = "Warning: Failed to get a stored value for lo1 frequency. Defaulting to 6022000000.0"
        return 6022000000.0

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
parser.add_option("-c", "--config", dest='config', default='/var/kat/conf', help='look for configuration files in folder CONF [default is KATCONF environment variable or /var/kat/conf]')
parser.add_option("-u", "--central_monitor_url", default=None, help="Override the central monitor url in the configuration with the one specified.")
parser.add_option("-d", "--dir", default=katuilib.defaults.kat_directories["data"], help="Process all unaugmented files in the specified directory. [default=%default]")
parser.add_option("-f", "--file", default="", help="Fully qualified path to a specific file to augment. [default=%default]")
parser.add_option("-s", "--system", default="systems/local.conf", help="System configuration file to use. [default=%default]")
parser.add_option("-o", "--override", dest="force", action="store_true", default=False, help="If set, previously augmented files will be re-augmented. Only useful in conjunction with a single specified file.")
parser.add_option("--dbe", dest="dbe_name", default="dbe", help="Name of kat device to use as the correlator proxy.")
parser.add_option("-v", "--verbose", action="store_true", default=False, help="Verbose output.")

options, args = parser.parse_args()

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)

# Setup configuration source
katconf.set_config(katconf.environ(options.config))

state = ["|","/","-","\\"]
batch_count = 0

pointing_sensors = ["activity","target","pos_actual_scan_azim","pos_actual_scan_elev","pos_actual_refrac_azim","pos_actual_refrac_elev","pos_actual_pointm_azim","pos_actual_pointm_elev","pos_request_scan_azim","pos_request_scan_elev","pos_request_refrac_azim","pos_request_refrac_elev","pos_request_pointm_azim","pos_request_pointm_elev"]
 # a list of pointing sensors to insert
enviro_sensors = ["asc_air_temperature","asc_air_pressure","asc_air_relative_humidity","asc_wind_speed","asc_wind_direction"]
 # a list of enviro sensors to insert
pedestal_sensors = ["rfe3_rfe15_noise_pin_on", "rfe3_rfe15_noise_coupler_on"]
 # a list of pedestal sensors to insert
rfe_sensors = ["rfe7_lo1_frequency"]
 # a list of RFE sensors to insert
beam_sensors = ["%s_target" % (options.dbe_name,)]
 # a list of sensor for beam 0

sensors = {'ant':pointing_sensors, 'ped':pedestal_sensors, 'ped1':enviro_sensors, 'rfe7':rfe_sensors}
 # mapping from sensors to proxy

sensors_iv = {"rfe3_rfe15_noise_pin_on":True, "rfe3_rfe15_noise_coupler_on":True, "activity":True, "target":True,"observer":True,"lock":True}
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

print "Found",len(files),"files to process"

 # build an kat object for history gathering purposes
print "Creating KAT connections..."
kat = katuilib.tbuild(options.system, log_level=logging.ERROR, central_monitor_url=options.central_monitor_url)
 # check that we have basic connectivity (i.e. two antennas and pedestals)
time.sleep(1)
while not kat.ant1.katcpobj.is_connected() or not kat.ant2.katcpobj.is_connected() or not kat.ped1.katcpobj.is_connected() or not kat.ped2.katcpobj.is_connected():
    status = "\r%s Connections to basic system of two antennas and two pedestals not available. Waiting for these to appear (possibly futile)..." % str(state[batch_count % 4])
    sys.stdout.write(status)
    sys.stdout.flush()
    time.sleep(30)
    batch_count += 1
initial_lo1 = kat.rfe7.sensor.rfe7_lo1_frequency.get_value()
 # get initial lo1 frequency value (mostly due to simulated system not providing it...)
dbe_device = getattr(kat, options.dbe_name)
array_config = dbe_device.sensor.array_config.get_value()
 # get array config from the correlator we're using
#kat.disconnect()
 # we dont need live connection anymore
section_reports['configuration'] = str(options.system)


array_cfg = katcore.targets.ArrayConfig(array_config)
 # retrieve array configuration object for correlator
config_antennas, dbe_delay, real_to_dbe = get_input_info(array_cfg)
 # return dicts showing the current mapping between dbe inputs and real antennas
antennas, antenna_positions, antenna_diameter, noise_diode_models = get_antenna_info(array_cfg)
 # build the description and position (in ecef coords) arrays for the antenna in the selected configuration
diodes = set([name.split('_')[1] for name in noise_diode_models])
 # map of noise diode model names to filenames
input_map = sorted(('ant' + real, dbe) for real, dbe in real_to_dbe.items())
 # map of antenna inputs (e.g. ant1H) to dbe inputs (e.g. 0x)

while(len(files) > 0 or options.batch):
    for fname in files:
        errors = 0
        fst = time.time()
        print "\nStarting augment of file",fname
        new_extension = "h5"
        try:
            f = File(fname, 'r+')
            current_version = f['/'].attrs.get('version', "0.0").split('.', 1)
            if current_version[0] != str(major_version):
                print "This version of augment required HDF5 files of version %i to augment. Your file has major version %s\n" % (major_version, current_version[0])
                continue
            last_run = f['/'].attrs.get('augment_ts',None)
            f['/'].attrs['augment_errors'] = 0
            if last_run:
                print "Warning: This file has already been augmented: " + str(last_run)
                if not options.force:
                    print "To force reprocessing, please use the -o option."
                    continue
                else:
                    section_reports['reaugment'] = "Augment was previously done in this file on " + str(last_run)
            f['/'].attrs['version'] = "%i.%i" % (major_version, augment_version)

            obs_start = f['/Data/timestamps'].value[1]
             # first timestamp is currently suspect
            f['/Data'].attrs['ts_of_first_timeslot'] = obs_start
            obs_end = f['/Data/timestamps'].value[-1]
            print "Observation session runs from %s to %s\n" % (time.ctime(obs_start), time.ctime(obs_end))
            int_time = get_single_value(f['/MetaData/Configuration/Correlator'], 'int_time')
            f['/MetaData/Configuration/Correlator'].attrs['input_map'] = input_map

            hist = create_group(f,"/History")
            sg = create_group(f, "/MetaData/Sensors")
            ag = create_group(sg, "Antennas")
            acg = create_group(f, "/MetaData/Configuration/Antennas")
            pg = create_group(sg, "Pedestals")
            rfeg = create_group(sg, "RFE")
            eg = create_group(sg, "Enviro")
            bg = create_group(sg, "Beams")

            for antenna in range(1,8):
                antenna = str(antenna)
                ant_name = 'ant' + antenna
                a = create_group(ag, ant_name)
                ac = create_group(acg, ant_name)
                stime = time.time()
                for sensor in pointing_sensors:
                    insert_sensor(ant_name + "_" + sensor, a, obs_start, obs_end, int_time, iv=(sensors_iv.has_key(sensor) and True or False))
                if options.verbose: print "Overall creation of sensor table for antenna " + antenna + " took " + str(time.time()-stime) + "s"
                # noise diode models
                try:
                    ac.attrs['description'] = antennas[ant_name].description
                except Exception:
                    print_tb()
                    section_reports[ant_name + ' description'] = "Error: Cannot find description for antenna %r" % (ant_name,)
                for pol in ['h','v']:
                    for nd in ['coupler','pin']:
                        nd_name = "%s_%s_%s" % (ant_name, nd, pol)
                        nd_fname = "unknown"
                        model = np.zeros((1,2), dtype=np.float32)
                        attrs = {}
                        try:
                            nd_fname = noise_diode_models[nd_name]
                            model, attrs = load_csv_with_header(katconf.resource_stream(nd_fname))
                        except Exception, e:
                            print_tb()
                            print "Failed to open noise diode model file %s (for %s). Inserting null noise diode model. (%s)" % (nd_fname, nd_name, e)
                        try:
                            nd = ac.create_dataset("%s_%s_noise_diode_model" % (pol, nd), data=model)
                            for key,val in attrs.iteritems(): nd.attrs[key] = val
                        except Exception:
                            print_tb()
                            print "Dataset %s.%s_%s_noise_diode_model already exists. Not replacing existing model." % (ac.name, pol, nd)

            for ped in range(1,8):
                ped = str(ped)
                ped_name = 'ped' + ped
                p = create_group(pg, ped_name)
                stime = time.time()
                for sensor in pedestal_sensors:
                    insert_sensor(ped_name + "_" + sensor, p, obs_start, obs_end, int_time, iv=(sensors_iv.has_key(sensor) and True or False))
                if options.verbose: print "Overall creation of sensor table for pedestal " + ped + " took " + str(time.time()-stime) + "s"

            b0 = create_group(bg, "Beam0")
            for sensor in beam_sensors:
                insert_sensor(sensor, b0, obs_start, obs_end, int_time, iv=(sensors_iv.has_key(sensor) and True or False))

            for sensor in rfe_sensors:
                sensor_len = insert_sensor("rfe7_" + sensor, rfeg, obs_start, obs_end, int_time, default=initial_lo1)
                try:
                    rfeg.create_dataset('center-frequency-hz', data=np.rec.fromarrays([[obs_start], [initial_lo1 - 4.2e9], [0]], names='timestamp, value, status'))
                except Exception:
                    print_tb()
                    print "Centre frequency already saved. Not replacing existing center-frequency."

            stime = time.time()
            for sensor in enviro_sensors:
                insert_sensor("anc_" + sensor, eg, obs_start, obs_end, int_time)
            if options.verbose: print "Overall creation of enviro sensor table took " + str(time.time()-stime) + "s"
             # end of antenna loop
            f['/'].attrs['augment_ts'] = time.time()

        except Exception, err:
            print_tb()
            section_reports["general"] = "Exception: " + str(err)
            errors += 1
            print "Failed to run augment. File will be  marked as 'failed' and ignored:  (" + str(err) + ")"
            new_extension = "failed.h5"

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
            print "Warning: Unable to create augment_log dataset. (" + str(err) + ")"
        f.close()

        if options.verbose:
            print "\n\nReport"
            print "======"
            keys = section_reports.keys()
            keys.sort()
            for k in keys:
                print k.ljust(50),section_reports[k]
        try:
            #Drop the last two extensions of the file 123456789.xxxxx.h5 becomes 123456789.
            #And then add the new extension in its place thus 123456789.unaugmented.h5 becomes 123456789.h5 or 123456789.failed.h5
            lst = fname.split(".")
            y = ".".join(l for l in lst[:-2]) + "."
            renfile = y + new_extension
            os.rename(fname, renfile)
            print "File has been renamed to " + str(renfile) + "\n"
        except Exception:
            print_tb()
            print "Failed to rename " + str(fname) + " to " + str(renfile) + ". This is most likely a permissions issue. Please resolve these and either manually rename the file or rerun augment with the -o option."
            continue
        print (errors == 0 and "No errors found." or str(errors) + " potential errors found. Please inspect the augment log by running 'h5dump -d /History/augment_log " + str(renfile) + "'.")

    # if in batch mode check for more files...
    files = []
    if options.batch:
        time.sleep(2)
        status = "\rChecking for new files in %s: %s" % (options.dir,str(state[batch_count % 4]))
        sys.stdout.write(status)
        sys.stdout.flush()
        files = get_files_in_dir(options.dir)
        batch_count += 1
