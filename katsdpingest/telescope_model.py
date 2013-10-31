import numpy as np
import time
import h5py

### Model definitions

debug = True

class Attribute(object):
    def __init__(self, name, critical=False):
        self.name = name
        self.value = None
        self.critical = critical

    def set_value(self, value):
        if self.value is None:
            self.value = value
            if debug: print "Set attribute {0} to {1}".format(self.name,value)
        else: raise ValueError("Attribute has already been set.")

    def is_valid(self):
        return self.value is not None

class Sensor(object):
    def __init__(self, name, critical=False):
        self.name = name
        self.critical = critical
        self.values = {}
        self.statii = {}
        self.value_time = 0
        self.value = None
        self.status = "unknown"

    def set_value(self, value):
        (value_ts, status, value) = value[0].split(" ",2)
        if debug: print "Updated sensors {0} with ({1},{2},{3})".format(self.name,value_ts,status,value)
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
        self.proxy_path = proxy_path if proxy_path is not None else name
        self._critical_sensors = []
        self._std_sensors = []
        self._critical_attributes = []
        self._std_attributes = []
        self.sensors = {}
        self.attributes = {}
        self._build()

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
    def __init__(self, minor_version=2):
        self.components = {}
        self.minor_version = minor_version
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

    def write_h5(self,f,base_path="/TelescopeModel"):
        h5py._errors.silence_errors()
         # needed to supress h5py error printing in child threads. 
         # exception handling and logging are used to print
         # more informative messages.
        current_version = f['/'].attrs.get('version', "2.0").split(".",1)
        f['/'].attrs['version'] = "{0}.{1}".format(current_version[0],self.minor_version)

        for c in self.components.values():
            comp_base = "{0}/{1}/{2}/".format(base_path,c._h5_path,c.name)
            try:
                c_group = f.create_group(comp_base)
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
                item._changed = False
                if debug: print "Update attribute {0} on {1} to {2}".format(item_name, comp.name, value)

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

