import numpy as np
import time
import h5py
import os
import spead

### Model definitions

debug = True
hdf5_version = 3.0
 # the version number is intrinsically linked to the telescope model, as this
 # is the arbiter of file structure and format

class SPEADItemUpdate(object):
    def __init__(self, key, ts, value):
        self.key = key
        self.ts = ts
        self.value = value

class Attribute(object):
    def __init__(self, name, critical=False, init_val=None):
        self.name = name
        self.value = init_val
        self.critical = critical
        self.update_ts = None
        self.spead_item = None
         # when known, this is set to a spead item that describes this attribute

    def set_value(self, value):
        if self.value is None:
            self.value = value
            if debug: print "Set attribute {0} to {1}".format(self.name,value)
            self.update_ts = time.time()
        else: raise ValueError("Attribute has already been set.")

    def get_spead_item(self):
        """Return a spead item describing this attribute.
        A generic item is create if an extant one is not available."""
        if self.spead_item is None:
            self.spead_item = spead.Item(name=self.name, description='', shape=-1, fmt=spead.mkfmt(('s', 8)))
        return self.spead_item

    def is_valid(self):
        return self.value is not None

class Sensor(object):
    def __init__(self, name, critical=False):
        self.name = name
        self.critical = critical
        self.values = {}
        self.statii = {}
        self.update_tss = {}
        self.value_time = 0
        self.value = None
        self.status = "unknown"
        self.update_ts = None
        self.spead_item = None
         # when known, this is set to a spead item that describes this sensor

    def set_value(self, value):
        (value_ts, status, value) = value[0].split(" ",2)
        if debug: print "Updated sensors {0} with ({1},{2},{3})".format(self.name,value_ts,status,value)
        self.values[value_time] = value
        self.statii[value_time] = status
        self.value_time = value_time
        self.value = value
        self.status = status
        self.update_ts = time.time()
        self.update_tss[value_time] = self.update_ts

    def is_valid(self,timespec=None):
        if self.critical:
            if len(self.values) == 0: return False
            if timespec is not None:
                # check to see if most recent update is within timespec
                return True if float(max(self.values)) + timespec >= time.time() else False
        return True

    def get_spead_item(self):
        """Return a spead item describing this attribute.
        A generic item is create if an extant one is not available."""
        if self.spead_item is None:
            self.spead_item = spead.Item(name=self.name, description='', shape=-1, fmt=spead.mkfmt(('s', 8)))
        return self.spead_item

    def get_dataset(self):
        return np.rec.fromarrays([self.values.keys(), self.values.values(), self.statii.values()],names='timestamp, value, status')

class TelescopeComponent(object):
    def __init__(self,name,proxy_path=None):
        self.name = name
        self._h5_path = ""
        self.proxy_path = proxy_path if proxy_path is not None else name
        self._critical_sensors = []
        self._std_sensors = []
        self._critical_attributes = []
        self._std_attributes = []
        self.sensors = {}
        self.attributes = {}
        self._build()

    def set_attribute(self, key, value):
        """Directly set attribute on component. Used specifically
        in cases in which the attributes are not know at build time."""
        if self.attributes.has_key(key) and self.attributes[key].value is not None: raise ValueError("This attribute already exists and is set to {0}. Not overwriting.".format(self.attributes[key].value))
        else:
            self.attributes[key] = Attribute(key, critical=False, init_val=value)
             # by definition it is not critical
            self.attributes[key].update_ts = time.time()

    def _build(self):
        for s in self._critical_sensors + self._std_sensors:
            self.sensors[s] = Sensor(s,critical=s in self._critical_sensors)
        for a in self._critical_attributes + self._std_attributes:
            self.attributes[a] = Attribute(a,critical=a in self._critical_attributes)

    def is_valid(self, timespec=None, check_sensors=True, check_attributes=True):
        retval = True
        if check_sensors:
            for s in [v for (k,v) in self.sensors.iteritems() if k in self._critical_sensors]:
                if not s.is_valid(timespec=timespec):
                    print "Sensor {0} is invalid ({1})".format("{0}/{1}".format(self.proxy_path,s.name),"no sensor value in timespec" if timespec is not None else "no values")
                    retval = False
        if check_attributes:
            for a in [v for (k,v) in self.attributes.iteritems() if k in self._critical_attributes]:
                if not a.is_valid(): print "Attribute {0} has not been set".format(a.name)
                retval = retval and a.is_valid()
        return retval

class TelescopeModel(object):
    def __init__(self):
        self.components = {}
        self.model_version = hdf5_version
        self.index = {}

    def add_components(self, components):
        for component in components:
            if self.components.has_key(component):
                print "Component name {0} is not unique".format(component_name)
                continue
            self.components[component.name] = component

    def is_valid(self, timespec=None):
        retval = True
        for c in self.components.itervalues():
            if not c.is_valid(timespec=timespec): retval = False
        return retval

    def build_index(self):
        for c in self.components.itervalues():
            for k,v in c.sensors.iteritems():
                self.index["{0}_{1}".format(c.proxy_path,k)] = v
            for k,v in c.attributes.iteritems():
                self.index["{0}_{1}".format(c.proxy_path,k)] = v

    def init_from_h5(self, filename):
        """Initialise the telescope model from the contents
        of an HDF5 format.
        File needs to have a version that matches the current model.

        Currently has very little error checking....
        """
        f = h5py.File(filename, mode="r")
        version = str(f['/'].attrs['version'])
        if version != str(hdf5_version):
            print "Attempt to load version {0} into a version {1} model.".format(version, hdf5_version)
            return
        cls_count = sensor_count = attr_count = 0
        for h5_component in f['/TelescopeModel'].itervalues():
            cls = globals()[h5_component.attrs['class']]
            c = cls(name=h5_component.name)
            self.add_components([c])
            cls_count += 1
            for k,v in h5_component.attrs.iteritems():
                c.set_attribute(k,v)
                attr_count += 1
            for h5_dataset in h5_component.itervalues():
                for row in h5_dataset.value:
                    c.sensors[h5_dataset.name].set_value("{0} {1} {2}".format(row[0],row[1],row[2]))
                    sensor_count += 1
        self.build_index()
        print "Completed initialisation. Added {0} components with {1} attributes and {2} sensor rows".format(cls_count, attr_count, sensor_count)

    def create_h5_file(self, filename):
        """Initialises an HDF5 output file as appropriate
        for this version of the telescope model."""
        f = h5py.File(filename, mode="w")
        f['/'].create_group('Data')
        f['/'].create_group('MetaData')
        f['/'].create_group('MetaData/Configuration')
        f['/'].create_group('MetaData/Configuration/Observation')
        f['/'].create_group('MetaData/Configuration/Correlator')
        f['/'].create_group('Markup')
        f['/Markup'].create_dataset('labels', [1], maxshape=[None], dtype=np.dtype([('timestamp', np.float64), ('label', h5py.new_vlen(str))]))
         # create a label storage of variable length strings
        f['/'].create_group('History')
        f['/History'].create_dataset('script_log', [1], maxshape=[None], dtype=np.dtype([('timestamp', np.float64), ('log', h5py.new_vlen(str))]))
        f['/History'].create_dataset('process_log',[1], maxshape=[None], dtype=np.dtype([('process', h5py.new_vlen(str)), ('arguments', h5py.new_vlen(str)), ('revision', np.int32)]))
        f['/'].attrs['version'] = hdf5_version
        return f

    def finalise_h5_file(self,f,base_path="/TelescopeModel"):
        h5py._errors.silence_errors()
         # needed to supress h5py error printing in child threads. 
         # exception handling and logging are used to print
         # more informative messages.

        for c in self.components.values():
            comp_base = "{0}/{1}/{2}/".format(base_path,c._h5_path,c.name)
            try:
                c_group = f.create_group(comp_base)
                c_group.attrs['class'] = str(c.__class__.__name__)
            except ValueError:
                c_group = f[comp_base]
                print "Failed to create group {0} (likely to already exist)".format(comp_base)
            for (k,v) in c.attributes.iteritems():
                if v.value is not None: c_group.attrs[k] = v.value
            for sensor in sorted(c.sensors.keys()):
                s = c.sensors[sensor]
                try:
                    c_group.create_dataset(s.name,data=s.get_dataset())
                except ValueError:
                    print "Failed to create dataset {0}/{1} as the model has no values".format(comp_base, s.name)
                except RuntimeError:
                    print "Failed to insert dataset {0}/{1} as it already exists".format(comp_base, s.name)

    def close_h5_file(self, f):
        filename = f.filename
         # grab filename before we close
        f.flush()
        f.close()
         # make sure to close and flush gracefully
        output_file = filename[:filename.find(".writing.h5")] + ".unaugmented.h5"
        try:
            os.rename(filename, output_file)
        except Exception, e:
            print "Failed to rename output file {0} to {1} ({2})".format(filename, output_file, e)
            return ("fail","Failed to rename output file from {0} to {1}.".format(filename, output_file))
        return "File renamed to {0}".format(output_file)

    def update(self, update_dict):
        """Expects a dict of sensor names with each value a space
        seperated string containing value_timestamp status and value"""
        for sensor_name,value_string in update_dict.iteritems():
            if sensor_name in self.index:
                sensor = self.index[sensor_name]
                (value_ts, status, value) = value_string.split(" ",2)
                sensor.add_value(value_ts, status, value)

    def _add_spead_item(self, ig, item):
        if item.id is None:
            item.id = 2**12 + len(ig._items)
            while ig._items.has_key(item.id): item.id += 1
        ig._items[item.id] = item
        ig._new_names.append(item.name)
        ig._update_keys()

    def generate_heaps(self, time_spacing=1.0):
        """Used to create a generator that will yield SPEAD heaps ready to transmit
        that describe the telescope model. A basic timeline is constructed to order the events
        and the heap transmitter can use the timestamping to attempt a realtime simulation 
        if this is desired."""

         # first step is to build metadata
         # we traverse the entire model and pick up all the SPEAD items
         # as we go (for missing items these are synthesised as best we can)
        ig = spead.ItemGroup()
        all_item_updates = []
         # the array of Attributes and Sensors to actually send

        for c in self.components.values():
            for a in c.attributes.itervalues():
                self._add_spead_item(ig, a.get_spead_item())
                print "Attribute {0} has update ts {1}".format(a.name,a.update_ts)
                if a.update_ts is not None:
                    print "Adding attribute {0} to spead update".format(a.name)
                    all_item_updates.append(SPEADItemUpdate(a.name,a.update_ts,a.value))
            for s in c.sensors.itervalues():
                self._add_spead_item(ig, s.get_spead_item())
                for (sensor_time, sensor_value) in s.values.iteritems():
                    sensor_state = s.statii[sensor_time]
                    update_time = s.update_tss[sensor_time]
                    print "Adding sensor value for sensor {0} to spead update".format(s.name)
                    all_item_updates.append(SPEADItemUpdate(s.name, update_time, "{0} {1} {2}".format(sensor_time, sensor_state, sensor_value)))
        all_item_updates.sort(key=lambda x: x.ts)
         # sort items by their update time

        yield ig.get_heap()
         # send metadata

        while len(all_item_updates) > 0:
            update = [all_item_updates.pop()]
            while len(all_item_updates) > 0 and all_item_updates[0].ts < (update[0].ts + time_spacing):
                update.append(all_item_updates.pop())
             # ok, we now have a group of updates to send
            for u in update:
                ig[u.key] = u.value
            yield ig.get_heap()

    def update_from_ig(self, ig, proxy_path=None, debug=False):
        """Traverses a SPEAD itemgroup looking for any changed items that match items expected in the model
        index and then updating these as appropriate. Attributes are non-volatile and will not 
        be overwritten, whilst conforming sensor are inserted into Sensor objects."""
        for item_name in ig.keys():
            # ig has no iteritems or equivalent
            item = ig.get_item(item_name)
            if item._changed:
                try:
                    if proxy_path is not None: item_name = "{0}_{1}".format(proxy_path,item_name)
                     # some sensors and attributes lack proper identification
                    dest = self.index[item_name]
                    item_value = item.get_value()
                    try:
                        dest.set_value(item_value)
                        if dest.spead_item is None: dest.spead_item = item
                        item._changed = False
                    except ValueError as e:
                        print "Failed to set value: {0}".format(e.message)
                        continue
                except KeyError:
                    pass #print "Item {0} not in index.".format(item_name)

#### Component Definitions

class AntennaPositioner(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(AntennaPositioner, self).__init__(*args, **kwargs)
        self._critical_sensors = ['activity','target','pos_actual_scan_elev','pos_request_scan_elev','pos_actual_scan_azim','pos_request_scan_azim']
        self._critical_attributes = ['description']
        self._build()

class CorrelatorBeamformer(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(CorrelatorBeamformer, self).__init__(*args, **kwargs)
        self._critical_sensors = ['dbe_mode','target','center_frequency_hz']
        self._std_sensors = ['auto_delay']
        self._critical_attributes = ['n_chans','n_accs','n_bls','bls_ordering','bandwidth', 'sync_time', 'int_time','scale_factor_timestamp']
        self._std_attributes = ['int_time','center_freq']
        self._build()

class Enviro(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(Enviro, self).__init__(*args, **kwargs)
        self._std_sensors = ['air_pressure','air_relative_humidity','temperature','wind_speed','wind_direction']
        self._build()

class Digitiser(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(Digitiser, self).__init__(*args, **kwargs)
        self._std_sensors = ['overflow']
        self._build()

class Observation(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(Observation, self).__init__(*args, **kwargs)
        self._build()

