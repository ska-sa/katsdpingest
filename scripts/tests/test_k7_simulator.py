import unittest2 as unittest
import os.path
import subprocess
import string
import random
import katcp
import spead
import time
import socket
import ConfigParser
import numpy as np

def retry(fn, exception, timeout=1, sleep_interval=0.025):
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

class fixtures(object):
    sim = None

# setup diliberately spelt wrong to keep nosetest out of it. Doing it
# since nose hides module level setup/teardown errors:
# https://github.com/nose-devs/nose/issues/506
def sertup():
    fixtures.sim = SimulatorSetup()
    fixtures.sim.setup_simulator_process()

def tearDown():
    try:
        fixtures.sim.tear_down_simulator()
    except BaseException, e:           # Work around nosetest nonsense
        print(e)

class SimulatorSetup(object):
    def setup_simulator_process(self, extra_sim_parms=[]):
        # simulator host and port
        device_host = '127.0.0.1' ; device_port = 2041 ; test_device_port = 2042
        # Test that there is not already a service listening on that
        # port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((device_host, device_port))
            raise RuntimeError('Service already running on port %d. Perhaps run kat-stop?')
        except socket.error: pass
        self.k7_simulator_logfile = open('k7_simulator.log', 'w+')
        #Find the k7_simulator executable
        k7_simulator_file = os.path.join(
            os.path.dirname(__file__), '..', 'k7_simulator.py')
        # k7_conf_file = os.path.join(
        #     os.path.dirname(__file__), '..', '..',
        #     'katcapture', 'conf', 'k7-local.conf')
        k7_conf_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'katcapture', 'conf')

        # Set up a process running the k7_simulator
        self.k7_simulator_proc = subprocess.Popen(
            [k7_simulator_file, '-p', '%d' %
             device_port, '-c', k7_conf_dir] + extra_sim_parms,
            stdout=self.k7_simulator_logfile, stderr=self.k7_simulator_logfile)

        # Set up katcp client for device and test interface
        self.k7_simulator_katcp = katcp.BlockingClient(device_host, device_port)
        self.k7_simulator_katcp.start()
        self.k7_testinterface_katcp = katcp.BlockingClient(
            device_host, test_device_port)
        self.k7_testinterface_katcp.start()
        self.k7_simulator_katcp.wait_connected(1.)
        self.k7_testinterface_katcp.wait_connected(1.)
        # Wait for katcp clients to become alive
        help_reply, help_informs = retry(
            lambda : self.k7_simulator_katcp.blocking_request(
                katcp.Message.request('help')),
            katcp.KatcpClientError)
        help_reply, help_informs = retry(
            lambda : self.k7_testinterface_katcp.blocking_request(
                katcp.Message.request('help')),
            katcp.KatcpClientError)

    def katcp_req(self, *req, **kwargs):
        timeout = kwargs.get('timeout')
        return self.k7_simulator_katcp.blocking_request(
            katcp.Message.request(*req), timeout=timeout)

    def katcp_test_req(self, *req, **kwargs):
        timeout = kwargs.get('timeout')
        return self.k7_testinterface_katcp.blocking_request(
            katcp.Message.request(*req), timeout=timeout)

    def tear_down_simulator(self):
        # Stop the katcp clients
        self.k7_simulator_katcp.stop(1)
        self.k7_testinterface_katcp.stop(1)
        self.k7_simulator_katcp.join(1)
        self.k7_testinterface_katcp.join(1)
        # Kill the k7_simulator process
        self.k7_simulator_proc.terminate()
        time.sleep(0.1)         # Give it a chance to exit gracefully
        self.k7_simulator_proc.kill()
        self.k7_simulator_logfile.close()

class ModuleSetupTestCase(unittest.TestCase):
    def setUp(self):
        """Set up correlator process if needed and get correlator  to 'standard' setup"""
        if fixtures.sim is None:
            sertup()
        fixtures.sim.katcp_test_req('hang-requests', 0)
        self.set_mode_quickly('c16n400M1k')
        self.set_standard_mode_delay()
        self.set_standard_mapping()

    def set_standard_mapping(self):
        for i in range(8):
            for pol in 'xy':
                chan = '%d%s' % (i, pol)
                fixtures.sim.katcp_req('label-input', chan, chan)

    def set_standard_mode_delay(self):
        fixtures.sim.katcp_test_req('set-test-sensor-value', 'mode-change-delay', '10')

    def set_mode_quickly(self, mode):
        fixtures.sim.katcp_test_req('set-test-sensor-value', 'mode-change-delay', '0')
        fixtures.sim.katcp_req('mode', mode)


class TestCorrelatorDataWBC(ModuleSetupTestCase):
    channels = 1024                    # c16n400M1k number of frequency channels
    mode = 'c16n400M1k'
    spead_udp_port = 7148

    def setUp(self):
        super(TestCorrelatorDataWBC, self).setUp()
        self.sim = fixtures.sim
        self.set_mode_quickly(self.mode)
        self.speadrx = spead.TransportUDPrx(
            self.spead_udp_port, pkt_count=1024, buffer_size=51200000)
        self.addCleanup(# Stop the spead client
                        self.speadrx.stop)

    def test_data(self):
        # Set dump rate to 100ms to speed stuff up
        self.sim.katcp_req('k7-accumulation-length', 100)
        # Set test target at az 165, el 45 deg with flux of 200
        self.sim.katcp_req('test-target', '165.', '45.', '200.')
        # Point 'antenna' away from test target
        self.sim.katcp_req('pointing-az', '280.')
        self.sim.katcp_req('pointing-el', '80.')
        away_group = self.get_spead_itemgroup(time_after=time.time())
        self.verify_shapes(away_group)

        # Point 'antenna' at the test target
        self.sim.katcp_req('pointing-az', '165.')
        self.sim.katcp_req('pointing-el', '45.')
        target_group_200 = self.get_spead_itemgroup(time_after=time.time())
        self.verify_shapes(target_group_200)

        # Increase flux of test target to 1000
        self.sim.katcp_req('test-target', '165.', '45.', '1000.')
        target_group_1000 = self.get_spead_itemgroup(time_after=time.time())
        self.verify_shapes(target_group_1000)

        away_max = np.max(np.abs(self.get_xeng(away_group)).flat)
        target_200_max = np.max(np.abs(self.get_xeng(target_group_200)).flat)
        target_1000_max = np.max(np.abs(self.get_xeng(target_group_1000)).flat)
        self.assertTrue(target_200_max > 2*away_max)
        self.assertTrue(target_1000_max > 4*target_200_max)

    def get_xeng(self, group):
        xeng_raw = group['xeng_raw']
        xeng = np.zeros(xeng_raw.shape[0:2], dtype=np.complex64)
        xeng[:] = (xeng_raw[:,:,0] / group['n_accs'] +
                  1j*xeng_raw[:,:,1] / group['n_accs'])
        return xeng


    def verify_shapes(self, group):
        n_stokes = 4            # Number of stokes polarisation parameters
        n_chans = self.channels           # Number of frequency channels
        self.assertEqual(group['n_chans'], n_chans)
        n_ants = group['n_ants']
        self.assertEqual(n_ants, 8) # Number of antennas supported by DBE7
        # Number of baselines (including self-correlations)
        n_bls = (n_ants)*(n_ants+1)/2*n_stokes
        self.assertEqual(group['n_bls'], n_bls)
        self.assertEqual(group['xeng_raw'].shape, (n_chans, n_bls, 2))


    def get_spead_itemgroup(self, time_after=None):
        max_spead_iters = 50
        itemgroup = spead.ItemGroup()
        meta_req = ['n_bls', 'n_chans', 'n_accs', 'xeng_raw',
                    'timestamp', 'scale_factor_timestamp', 'sync_time']
        def issue():
            (message, informs) = self.sim.katcp_req('capture-start', 'k7')
            if not message.reply_ok():
                raise RuntimeError('Error with katcp command')
        issue()
        for i, heap in enumerate(spead.iterheaps(self.speadrx)):
            if i == 0: issue()
            itemgroup.update(heap)
            speadkeys = itemgroup.keys()
            if all(
                k in speadkeys and itemgroup[k] is not None for k in meta_req):
                if time_after is None:
                    break
                timestamp = (itemgroup['timestamp'] /
                             itemgroup['scale_factor_timestamp'] +
                             itemgroup['sync_time'])
                min_timestamp = time_after + itemgroup['int_time']
                if timestamp > min_timestamp:
                    break
                if i > max_spead_iters:
                    raise Exception(
                        'Timestamp still too old by %fs in %d iterations' % (
                            min_timestamp - timestamp, i))
            if i > max_spead_iters:
                raise Exception(
                    'Unable to obtain required spead data in %d iterations'
                    % max_spead_iters)
        return itemgroup

class TestCorrelatorData_c16n13M4k(TestCorrelatorDataWBC):
    channels = 4096
    mode = 'c16n13M4k'
    spead_udp_port = 7140

class TestCorrelatorLabeling(ModuleSetupTestCase):
    def setUp(self):
        super(TestCorrelatorLabeling, self).setUp()

    def test_labels(self):
        reply, informs = fixtures.sim.katcp_req('label-input')
        # Check that we have the standard setup
        expected_map = {'0x':'0x',
                        '0y':'0y',
                        '1x':'1x',
                        '1y':'1y',
                        '2x':'2x',
                        '2y':'2y',
                        '3x':'3x',
                        '3y':'3y',
                        '4x':'4x',
                        '4y':'4y',
                        '5x':'5x',
                        '5y':'5y',
                        '6x':'6x',
                        '6y':'6y',
                        '7x':'7x',
                        '7y':'7y'}
        def get_map_from_informs(informs):
            input_map = {}
            for inform in informs:
                name, input = inform.arguments
                input_map[input] = name
            return input_map
        actual_map = get_map_from_informs(informs)
        # generate random channel names for the test
        choice_chars = string.letters+string.digits
        for k in expected_map.keys():
            expected_map[k] = ''.join(random.choice(choice_chars)
                                      for i in xrange(random.randint(1,12)))
        # Set the mappings on the correlator
        for input, name in expected_map.items():
            reply, informs = fixtures.sim.katcp_req('label-input', input, name)
            self.assertTrue(reply.reply_ok())

        # Now see what the correlator actually has
        reply, informs = fixtures.sim.katcp_req('label-input')
        actual_map = get_map_from_informs(informs)
        self.assertEqual(actual_map, expected_map)

class TestTestInterface(ModuleSetupTestCase):
    def setUp(self):
        super(TestTestInterface, self).setUp()
        self.sim = fixtures.sim
        self.sim.katcp_test_req('hang-requests', 0)

    def test_hang(self):
        # Check that the simulator is currently working
        self.assertTrue(self.sim.katcp_req('watchdog', timeout=0.25)[0].reply_ok())
        self.assertTrue(self.sim.katcp_req('label-input', timeout=0.25)[0].reply_ok())
        self.sim.katcp_test_req('hang-requests', 1)
        with self.assertRaises(RuntimeError):
            self.sim.katcp_req('label-input', timeout=0.25)
        # Watchdogs should still work
        self.assertTrue(self.sim.katcp_req('watchdog', timeout=0.25)[0].reply_ok())

    def test_mode_change_time(self):
        self.sim.katcp_test_req('set-test-sensor-value', 'mode-change-delay', '0')
        # Get us into a known mode
        reply, informs = self.sim.katcp_req('mode', 'ready')
        delay1 = 0.1
        self.sim.katcp_test_req('set-test-sensor-value', 'mode-change-delay',
                                str(delay1))
        start = time.time()
        reply, informs = self.sim.katcp_req('mode', 'c16n13M4k')
        duration = time.time() - start
        self.assertTrue(reply.reply_ok())
        # Mode change should take at least delay1 seconds
        self.assertGreaterEqual(duration, delay1)
        self.assertLess(duration, delay1+0.1)   # But really should not take much longer
        delay2 = 1.3
        self.sim.katcp_test_req('set-test-sensor-value', 'mode-change-delay',
                                str(delay2))
        start = time.time()
        reply, informs = self.sim.katcp_req('mode', 'c16n400M1k')
        duration = time.time() - start
        # Mode change should take at least delay2 seconds
        self.assertGreaterEqual(duration, delay2)
        self.assertLess(duration, delay2+0.1)   # But really should not take much longer

class TestReadyMode(ModuleSetupTestCase):
    """Test proper functioning of ready mode, i.e. does ?label-input break"""
    def setUp(self):
        super(TestReadyMode, self).setUp()

    def test_label(self):
        # Check that it is broken in ready mode
        self.set_mode_quickly('ready')
        reply, informs = fixtures.sim.katcp_req('label-input')
        self.assertFalse(reply.reply_ok())
        reply, informs = fixtures.sim.katcp_req('label-input', '0x', 'bananans')
        self.assertFalse(reply.reply_ok())
        # Now check that it actually would have worked in another mode!
        self.set_mode_quickly('c16n400M1k')
        reply, informs = fixtures.sim.katcp_req('label-input')
        self.assertTrue(reply.reply_ok())
        reply, informs = fixtures.sim.katcp_req('label-input', '0x', 'bananans')
        self.assertTrue(reply.reply_ok())
