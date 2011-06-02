#!/usr/bin/env python

import katconf
from optparse import OptionParser

try:
    if kat.dbe7.is_connected():
        print "Using connection to DBE7 proxy to set input mappings..."
except NameError:
    print "This script must be run from within an iPython session that has a live kat connection and a dbe7 object within. (run -i <script_name>)"

parser = OptionParser()
parser.add_option("-c", "--config", dest='config', default='/var/kat/conf', help='look for configuration files in folder CONF [default is KATCONF environment variable or /var/kat/conf]')
options, args = parser.parse_args()

katconf.set_config(katconf.environ(options.config))
conf = katconf.get_config()
kc = conf.resource_config("correlators/karoo.katcorrelator.conf")

for inp in kc.options('inputs'):
    descr = kc.get('inputs',inp).split(",")
    label = descr[0] + descr[1][1:].upper()
    dbe_input = inp[5:]
    kat.dbe7.req.dbe_label_input(dbe_input,label)
    print "Labelling dbe input %s with label %s" % (dbe_input,label)

print "Done.\n\nDBE reports following labelling:"
ret = kat.dbe7.req.dbe_label_input()
print ret
