import numpy as np
import time
import h5py

### Model definitions

class Sensor(object):
    def __init__(self, name, critical=False):
        self.name = name
        self.critical = critical
        self.values = {}
        self.statii = {}
        self.value_time = 0
        self.value = None
        self.status = "unknown"

    def add_value(self, value_time, status, value):
        self.values[value_time] = value
        self.statii[value_time] = status
        self.value_time = value_time
        self.value = value
        self.status = status

    def is_valid(self,timespec=None):
        if self.critical:
            if len(self.values) == 0: return False
            if timespec is not None:
                # check to see if most recent update is within timespec
                return True if float(max(self.values)) + timespec >= time.time() else False
        return True

    def get_dataset(self):
        return np.rec.fromarrays([self.values.keys(), self.values.values(), self.statii.values()],names='timestamp, value, status')

class TelescopeComponent(object):
    def __init__(self,name,proxy_path=None):
        self.name = name
        self._h5_path = ""
        self._proxy_path = proxy_path if proxy_path is not None else name
        self._critical_sensors = []
        self._std_sensors = []
        self._critical_attributes = {}
        self._std_attributes = []
        self.sensors = {}
        self.attributes = {}
        self._build()

    def _build(self):
        for s in self._critical_sensors + self._std_sensors:
            self.sensors["{0}_{1}".format(self._proxy_path,s)] = Sensor(s,critical=s in self._critical_sensors)

    def is_valid(self, timespec=None, check_sensors=True, check_attributes=True):
        retval = True
        if check_sensors:
            [l for (l,k) in u.iteritems() if l in q]
            for s in [s for (k,v) in self.sensors.iteritems() if k in self._critical_sensors]:
                if not s.is_valid(timespec=timespec):
                    print "Sensor {0} is invalid ({1})".format("{0}/{1}".format(self._proxy_path,s.name),"no sensor value in timespec" if timespec is not None else "no values")
                    retval = False
        if check_attributes:
            for a in self._critical_attributes:
                retval = retval and self.attributes.has_key(a)
        return retval

class TelescopeModel(object):
    def __init__(self):
        self.components = {}
        self.index = {}

    def add_components(self, components):
        for component in components:
            self.components[component.name] = component

    def is_valid(self, timespec=None):
        retval = True
        for c in self.components.itervalues():
            if not c.is_valid(timespec=timespec): retval = False
        return retval

    def build_index(self):
        tmp_index = []
        for c in self.components.itervalues():
            tmp_index += c.sensors.items()
        self.index = dict(tmp_index)

    def write_h5(self,f,base_path="/MetaData/Sensors"):
        h5py._errors.silence_errors()
         # needed to supress h5py error printing in child threads. 
         # exception handling and logging are used to print
         # more informative messages.
        for c in self.components.values():
            comp_base = "{0}/{1}/{2}/".format(base_path,c._h5_path,c.name)
            try:
                c_group = f.create_group(comp_base)
            except ValueError:
                c_group = f[comp_base]
                print "Failed to create group {0} (likely to already exist)".format(comp_base)
            for sensor in sorted(c.sensors.keys()):
                s = c.sensors[sensor]
                try:
                    c_group.create_dataset(s.name,data=s.get_dataset())
                except ValueError:
                    print "Failed to create dataset {0}/{1} as the model has no values".format(comp_base, s.name)
                except RuntimeError:
                    print "Failed to insert dataset {0}/{1} as it already exists".format(comp_base, s.name)

    def update(self, update_dict):
        """Expects a dict of sensor names with each value a space
        seperated string containing value_timestamp status and value"""
        for sensor_name,value_string in update_dict.iteritems():
            if sensor_name in self.index:
                sensor = self.index[sensor_name]
                (value_ts, status, value) = value_string.split(" ",2)
                sensor.add_value(value_ts, status, value)

    def update_attributes_from_ig(self, ig, component_name, debug=False):
        """Traverses an item group looking for possible attributes.
        These are inserted into the named component if appropriate."""
        try:
            comp = self.components[component_name]
        except KeyError:
            print "Invalid component name specified"
            return False
        for item_name in ig.keys():
            item = ig.get_item(item_name)
            if not item._changed: continue
            if item_name in (comp._critical_attributes + comp._std_attributes):
                value = item.get_value()
                comp.attributes[item_name] = value
                if debug: print "Update attribute {0} on {1} to {2}".format(item_name, comp.name, value)

    def update_sensors_from_ig(self, ig, debug=False):
        """Expects a SPEAD itemgroup containing sensor information."""
        for item_name in ig.keys():
            # ig has no iteritems or equivalent
            item = ig.get_item(item_name)
            if item.has_changed():
                try:
                    sensor = self.index[item_name]
                    (value_ts, status, value) = item.get_value()[0].split(" ",2)
                    sensor.add_value(value_ts, status, value)
                    if debug: print "Updated sensors {0} with ({1},{2},{3})".format(item_name,value_ts,status,value)
                except KeyError:
                    print "Sensor {0} not in index.".format(item_name)

#### Component Definitions

class AntennaPositioner(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(AntennaPositioner, self).__init__(*args, **kwargs)
        self._h5_path = "Antennas"
        self._critical_sensors = ['activity','target','pos_actual_scan_elev','pos_request_scan_elev','pos_actual_scan_azim','pos_request_scan_azim']
        self._build()

class CorrelatorBeamformer(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(CorrelatorBeamformer, self).__init__(*args, **kwargs)
        self._critical_sensors = ['mode','target','center_frequency_hz']
        self._std_sensors = ['auto_delay']
        self._critical_attributes = ['n_chans','n_accs','n_bls','bls_ordering','bandwidth', 'sync_time', 'int_time']
        self._std_attributes = ['int_time','center_freq']
        self._build()

class Enviro(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(Enviro, self).__init__(*args, **kwargs)
        self._h5_path = "Enviro"
        self._std_sensors = ['air_pressure','air_relative_humidity','temperature','wind_speed','wind_direction']
        self._build()

class Digitiser(TelescopeComponent):
    def __init__(self, *args, **kwargs):
        super(Digitiser, self).__init__(*args, **kwargs)
        self._h5_path = "Digitiser"
        self._std_sensors = ['overflow']
        self._build()

