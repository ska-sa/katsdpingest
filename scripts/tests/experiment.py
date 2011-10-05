from __future__ import with_statement

from katcp import BlockingClient, Message, KatcpClientError
import string
import time
import subprocess
import os.path
import contextlib
from pkg_resources import resource_filename
import ConfigParser
import logging
import spead
import numpy as np
import katcp

logging.root.setLevel(logging.INFO)


# k7_simulator_file = os.path.join(
#     os.path.dirname(__file__), '..', 'k7_simulator.py')

k7_simulator_file = '/usr/local/bin/k7_simulator.py'

device_host = "127.0.0.1"
device_port = 2041
default_conf = resource_filename("katcapture","") + "/conf/k7-local.conf"

cfg = ConfigParser.SafeConfigParser()
cfg.read(default_conf)
spead_udp_port = cfg.getint('receiver', 'rx_udp_port')
k7_simulator_logfile = open('k7_simulator.log', 'w')

@contextlib.contextmanager
def closing_process(process, closing_files=[]):
    try:
        yield process
    finally:
        process.terminate()
        time.sleep(0.1)
        process.kill()
        for f in closing_files:
            f.close()
        
@contextlib.contextmanager
def stopjoining_client(client):
    try:
        yield client
    finally:
        client.stop()
        client.join()

@contextlib.contextmanager
def stopping(thing):
    try:
        yield thing
    finally:
        thing.stop()
        
def retry(fn, exception, timeout=1, sleep_interval=0.05):
    trytime = 0
    while trytime < timeout:
        try:
            retval = fn()
        except exception:
            time.sleep(sleep_interval)
            trytime += sleep_interval
            if trytime >= timeout:
                raise
            continue
        break
    return retval

contexts = (
    closing_process(subprocess.Popen([k7_simulator_file, '-p', '%d' % device_port],
                                     stdout=k7_simulator_logfile,
                                     stderr=k7_simulator_logfile),
                    closing_files=[k7_simulator_logfile]
                    ),
    stopjoining_client(BlockingClient(device_host, device_port)),
    contextlib.closing(k7_simulator_logfile),
    stopping(spead.TransportUDPrx(
            spead_udp_port, pkt_count=1024, buffer_size=51200000)),
    )

spead_ig_before = spead.ItemGroup()
spead_ig_after = spead.ItemGroup()

req_retry = lambda client, *req: retry(
    lambda : client.blocking_request(Message.request(*req)),
    KatcpClientError)

reps = []

# Should use new multiple context with statement feature in python 2.7
# when we upgrade
with contextlib.nested(*contexts) as (
    k7_simulator_proc, client, k7_simulator_logfile, spead_rx):
    client.start()
    help_reply, help_informs = (req_retry(client, 'help'))

    meta_req = ['n_bls', 'n_chans', 'n_accs', 'xeng_raw']
    for heap in spead.iterheaps(spead_rx):
        spead_ig_before.update(heap)
        speadkeys = spead_ig_before.keys()
        logging.info('spead item group keys: %s' % str(speadkeys))
        if (all(k in speadkeys for k in meta_req) and
            spead_ig_before['xeng_raw'] is not None):
            logging.info('all required spead keys found')
            break


    reps.append(req_retry(client, 'test-target', '165.', '45.', '200.'))
    reps.append(req_retry(client, 'pointing-az', '165.'))
    reps.append(req_retry(client, 'pointing-el', '45.'))
    reps.append(req_retry(client, 'spead-issue'))
    t2 = time.time()

    #no_update = 1
    for heap in spead.iterheaps(spead_rx):
        spead_ig_after.update(heap)
        speadkeys = spead_ig_after.keys()
        logging.info('spead item group keys: %s' % str(speadkeys))
        if (all(k in speadkeys for k in meta_req) and
            spead_ig_after['xeng_raw'] is not None):
            logging.info('all required spead keys found')
            t2_ts = (spead_ig_after['timestamp'] /
                     spead_ig_after['scale_factor_timestamp']+
                     spead_ig_after['sync_time'])
            #break
            if t2_ts  > t2 + spead_ig_after['int_time'] : break
            
            # if no_update > 1:
            #     break
            # else:
            #     no_update = no_update + 1

        

spead_ig = spead_ig_before
xeng_before = np.zeros(spead_ig['xeng_raw'].shape[0:2], dtype=np.complex64)
xeng_before[:] = (spead_ig['xeng_raw'][:,:,0] / spead_ig['n_accs'] +
                  1j*spead_ig['xeng_raw'][:,:,1] / spead_ig['n_accs'])
        
spead_ig = spead_ig_after
xeng_after = np.zeros(spead_ig['xeng_raw'].shape[0:2], dtype=np.complex64)
xeng_after[:] = (spead_ig['xeng_raw'][:,:,0] / spead_ig['n_accs'] +
                 1j*spead_ig['xeng_raw'][:,:,1] / spead_ig['n_accs'])


translate_string = lambda s: string.replace(s, '\\n', '\n')
def print_args(inform):
    for arg in inform.arguments:
        print translate_string(arg)

# for rep in reps:
#     reply, informs = rep
#     for inf in informs:
#         print_args(inf)


def print_help():
    for inf in help_informs:
        print_args(inf)
        
        
import sys
sys.stdout.flush()
sys.stderr.flush()
#logging.info('spead_udp_port: %d' % spead_udp_port)

import pylab
pylab.figure(1)
pylab.clf()
pylab.plot(np.abs(xeng_before[:,0]))
pylab.plot(np.abs(xeng_after[:,0]))
pylab.xlim([180, 210])
pylab.grid(1)
pylab.show()
