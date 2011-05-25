#!/usr/bin/env python

# Silly script to demonstrate basic functionality of the sigproc classes in the katsdisp package.
# This really needs to move to it's own package soon :)
#
# Shows the basic flow, which is to have a numpy array history container that is linked
# to an attribute in the base sigproc class ProcBlock.
# Classes inheriting from this will have a reference to this history array.
#
# Some modules such as the thresholder produce products (in this case flags),
# whilst others such as the scale simply modify the data in situ.
#
# In a live system the blocks will all be instantiated and then sit in a loop
# which updates history and then calls the proc method on each block in turn.
#

import katcapture.sigproc as sp
import numpy as np

n_channels = 1024
n_baselines = 144
 # current kat-7 correlator numbers
n_timestamps = 5
 # keep a history of 5 data points
scale_factor = 7.5
rfi_sigma = 2
 # number of std deviations from mean at which to flag data

history = []
for t in range(n_timestamps):
    history.append(np.random.random((1024,144)).astype(np.complex64) * 1000)
history = np.array(history)
 # create some random data

sp.ProcBlock.current = history[0]
sp.ProcBlock.history = history
 # setup the data pointers in the base class

scale = sp.Scale(scale_factor)
 # create a scaler
rfi = sp.RFIThreshold(rfi_sigma)
 # create a simple thresholder

print "Pre scale value",history[0][0][0]
scale.proc()
print "Post scale value",history[0][0][0],"\n"

history[0][0][5] = 20000+0j
 # add a single rfi spike
print "RFI should be at 0,0,5"
mask = rfi.proc()
print "First 8 flags in mask are:",np.unpackbits(mask)[:8]

print "\nOverall stats:"
rfi.stats()
 # note this shows stats for all operations to date
