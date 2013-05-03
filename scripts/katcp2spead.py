#!/usr/bin/python
#
# Ludwig Schwardt
# 4 April 2013
#

import numpy as np
import spead
import katcorelib

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
listeners = []
start_id = 0x7000

site, system = katcorelib.conf.get_system_configuration()
try:
    kat = katcorelib.tbuild(system=system)
except ValueError:
    raise ValueError("Could not build KAT connection for %s" % (system,))

tx = spead.Transmitter(spead.TransportUDPtx('192.168.56.101', 7148))
ig = spead.ItemGroup()

class Listener(object):
    def __init__(self, name):
        self.name = name
    def listen(self, update_seconds, value_seconds, status, value):
        """Push sensor update to SPEAD stream."""
        update = "%r %s %r" % (value_seconds, status, value)
        print "Updating sensor %r: %s" % (self.name, update)
        ig['sensor_' + self.name] = update
        tx.send_heap(ig.get_heap())

for n, (name, strategy, param) in enumerate(sensors):
    try:
        sensor = getattr(kat.sensors, name)
    except AttributeError:
        continue
    sensor.set_strategy(strategy, param)
    history = sensor.get_stored_history(start_seconds=-1, last_known=True)
    last_update = "%r %s %r" % (history[0][-1], history[2][-1], history[1][-1])
    print "Adding sensor %r: %s" % (name, last_update)
    ig.add_item(name='sensor_' + name, id=start_id + n, description=sensor.description,
                shape=-1, fmt=spead.mkfmt(('s', 8)), init_val=last_update)
    listeners.append(Listener(name))

tx.send_heap(ig.get_heap())

for listener in listeners:
    print "Registering sensor %r" % (listener.name,)
    sensor = getattr(kat.sensors, listener.name)
    sensor.register_listener(listener.listen)

try:
    while(True): pass
finally:
    for listener in listeners:
        sensor = getattr(kat.sensors, listener.name)
        sensor.unregister_listener(listener.listen)
    kat.disconnect()
