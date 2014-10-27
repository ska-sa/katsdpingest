"""Test for the sigproc module."""

import unittest
import mock
import numpy as np
from katsdpingest import sigproc
from katsdpsigproc.test.test_accel import device_test, test_context, test_command_queue

class SigProcTestCases(unittest.TestCase):
    """Exercise the sigproc library under a number of varying conditions."""

    def setUp(self):
        """Create the basic environment, along with test vectors."""
        self.scale_factor = 3
        self.n_ts = 10
        self.n_chans = 512
        self.n_ants = 4
        self.n_bls = 2 * self.n_ants * (self.n_ants+1)

        self.spike_channel = np.random.randint(self.n_chans)
        self.spike_bls = np.random.randint(self.n_bls)
         # the channel and baseline into which to insert rfi spikes

        self.data = (np.random.random((self.n_chans, self.n_bls,2)) + 100).astype(np.int32)
         # produce some random data
        self.data_copy = self.data.copy()

        sigproc.ProcBlock.history = None
        sigproc.ProcBlock.current = self.data

    def testEmptyBlock(self):
        """Test a empty signal processing block."""
        pc = sigproc.ProcBlock()
        self.assertRaises(NotImplementedError, pc.proc)

    def testBlsOrdering(self):
        """Check that CorrProdRef is built correctly."""
        pc = sigproc.ProcBlock(n_ants=self.n_ants)
        self.assertEqual(len(pc.cpref._id_to_real), self.n_bls)

    def testBasicAssignment(self):
        """Test assignment to class variables."""
        pc = sigproc.ProcBlock()
        np.testing.assert_equal(pc.current,self.data)

    def testScaleDataType(self):
        """Test acceptable data type for scale."""
        data_wrong_type = np.ones((self.n_ts,self.n_chans,self.n_bls,2), dtype=np.float32)
        sigproc.ProcBlock.current = data_wrong_type[0]
        sc = sigproc.Scale(self.scale_factor)
        self.assertRaises(TypeError, sc.proc)

    def testScale(self):
        """Test basic scaling."""
        sc = sigproc.Scale(self.scale_factor)
        sc.proc()
        np.testing.assert_equal(self.data, np.float32(self.data_copy) / (1.0 * self.scale_factor))
        self.assertEqual(self.data.dtype, np.float32)

    def testVanVleck(self):
        """Not implemented yet..."""
        pass

    def testRFIThresholdDataType(self):
        """Test acceptable data type for rfi threshold."""
        rfi = sigproc.RFIThreshold2()
        data_wrong_type = np.ones((self.n_ts,self.n_chans,self.n_bls,2), dtype=np.int32)
        sigproc.ProcBlock.history = data_wrong_type
        sigproc.ProcBlock.current = data_wrong_type[0]
        self.assertRaises(TypeError, rfi.proc)

    def testRFIThreshold2(self):
        """Test simple RFI Thresholding..."""
        data = (np.random.random((self.n_chans, self.n_bls)) + 100).astype(np.complex64)
        data[self.spike_channel, self.spike_bls] += 100
        sigproc.ProcBlock.current = data
        rfi = sigproc.RFIThreshold()
        flags = np.unpackbits(rfi.proc()).reshape(self.n_chans, self.n_bls)
        self.assertEqual(flags[self.spike_channel, self.spike_bls],1)

class TestPrepare(unittest.TestCase):
    """Test :class:`katsdpingest.sigproc.Prepare`"""

    @device_test
    def testPrepare(self):
        """Basic test of data preparation"""
        channels = 73
        channel_range = (10, 55)
        keep_channels = channel_range[1] - channel_range[0]
        baselines = 91
        scale = 3.625

        rs = np.random.RandomState(seed=1)
        vis_in = rs.random_integers(-1000, 1000, (channels, baselines, 2)).astype(np.int32)
        permutation = rs.permutation(baselines).astype(np.uint16)

        template = sigproc.PrepareTemplate(test_context)
        prepare = template.instantiate(test_command_queue, channels, channel_range, baselines)
        prepare.ensure_all_bound()
        prepare.slots['vis_in'].buffer.set(test_command_queue, vis_in)
        prepare.slots['permutation'].buffer.set(test_command_queue, permutation)
        prepare(scale)
        weights = prepare.slots['weights'].buffer.get(test_command_queue)
        vis_out = prepare.slots['vis_out'].buffer.get(test_command_queue)

        self.assertEqual((baselines, channels), vis_out.shape)
        self.assertEqual((baselines, keep_channels), weights.shape)
        expected_vis = np.zeros_like(vis_out)
        expected_weights = np.zeros_like(weights)
        for i in range(channels):
            for j in range(baselines):
                value = (vis_in[i, j, 0] + 1j * vis_in[i, j, 1]) * scale
                row = permutation[j]
                expected_vis[row, i] = value
                if i >= channel_range[0] and i < channel_range[1]:
                    col = i - channel_range[0]
                    expected_weights[row, col] = 1.0
        np.testing.assert_equal(expected_vis, vis_out)
        np.testing.assert_equal(expected_weights, weights)

class TestAccum(unittest.TestCase):
    """Test :class:`katsdpingest.sigproc.Accum`"""

    @device_test
    def testSmall(self):
        """Hand-coded test data, to test various cases"""

        flag_scale = 2 ** -64
        # Host copies of arrays
        host = {
            'vis_in':       np.array([[1+2j, 2+5j, 3-3j, 2+1j, 4]], dtype=np.complex64),
            'weights_in':   np.array([[2.0, 4.0, 3.0]], dtype=np.float32),
            'flags_in':     np.array([[5, 0, 10, 0, 4]], dtype=np.uint8),
            'vis_out0':     np.array([[7-3j, 0+0j, 0+5j]], dtype=np.complex64).T,
            'weights_out0': np.array([[1.5, 0.0, 4.5]], dtype=np.float32).T,
            'flags_out0':   np.array([[1, 9, 0]], dtype=np.uint8).T
        }

        template = sigproc.AccumTemplate(test_context, 1)
        fn = template.instantiate(test_command_queue, 5, [1, 4], 1)
        fn.ensure_all_bound()
        for name, value in host.iteritems():
            fn.slots[name].buffer.set(test_command_queue, value)
        fn()

        expected = {
            'vis_out0':     np.array([[11+7j, (12-12j) * flag_scale, 6+8j]], dtype=np.complex64).T,
            'weights_out0': np.array([[3.5, 4.0 * flag_scale, 7.5]], dtype=np.float32).T,
            'flags_out0':   np.array([[0, 8, 0]], dtype=np.uint8).T
        }
        for name, value in expected.iteritems():
            actual = fn.slots[name].buffer.get(test_command_queue)
            np.testing.assert_equal(value, actual, err_msg=name + " does not match")

    @device_test
    def testBig(self):
        """Test with large random data against a simple CPU version"""
        flag_scale = 2 ** -64
        channels = 203
        baselines = 171
        channel_range = [7, 198]
        kept_channels = channel_range[1] - channel_range[0]
        outputs = 2
        rs = np.random.RandomState(1)

        vis_in = (rs.standard_normal((baselines, channels)) + rs.standard_normal((baselines, channels)) * 1j).astype(np.complex64)
        weights_in = rs.uniform(size=(baselines, kept_channels)).astype(np.float32)
        flags_in = rs.choice(4, (baselines, channels), p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8)
        vis_out = []
        weights_out = []
        flags_out = []
        for i in range(outputs):
            vis_out.append((rs.standard_normal((kept_channels, baselines)) + rs.standard_normal((kept_channels, baselines)) * 1j).astype(np.complex64))
            weights_out.append(rs.uniform(size=(kept_channels, baselines)).astype(np.float32))
            flags_out.append(rs.choice(4, (kept_channels, baselines), p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8))

        template = sigproc.AccumTemplate(test_context, outputs)
        fn = template.instantiate(test_command_queue, channels, channel_range, baselines)
        fn.ensure_all_bound()
        for (name, value) in [('vis_in', vis_in), ('weights_in', weights_in), ('flags_in', flags_in)]:
            fn.slots[name].buffer.set(test_command_queue, value)
        for (name, value) in [('vis_out', vis_out), ('weights_out', weights_out), ('flags_out', flags_out)]:
            for i in range(outputs):
                fn.slots[name + str(i)].buffer.set(test_command_queue, value[i])
        fn()

        # Perform the operation on the host
        kept_vis = vis_in[:, channel_range[0] : channel_range[1]]
        kept_flags = flags_in[:, channel_range[0] : channel_range[1]]
        flagged_weights = weights_in * ((kept_flags == 0) + flag_scale)
        for i in range(outputs):
            vis_out[i] += (kept_vis * flagged_weights).T
            weights_out[i] += flagged_weights.T
            flags_out[i] = np.bitwise_and(flags_out[i], kept_flags.T)

        # Verify results
        for (name, value) in [('vis_out', vis_out), ('weights_out', weights_out), ('flags_out', flags_out)]:
            for i in range(outputs):
                actual = fn.slots[name + str(i)].buffer.get(test_command_queue)
                np.testing.assert_allclose(value[i], actual)

class TestPostproc(unittest.TestCase):
    """Tests for :class:`katsdpingest.sigproc.Postproc`"""

    def test_bad_cont_factor(self):
        """Test with a continuum factor that does not divide into the channel count"""
        template = mock.sentinel.template
        template.cont_factor = 8
        self.assertRaises(ValueError, sigproc.Postproc, template, mock.sentinel.command_queue, 12, 8)

    @device_test
    def testPostproc(self):
        """Test with random data against a CPU implementation"""

        channels = 1024
        baselines = 512
        cont_factor = 16
        cont_channels = channels // cont_factor
        rs = np.random.RandomState(1)
        vis_in = (rs.standard_normal((channels, baselines)) + rs.standard_normal((channels, baselines)) * 1j).astype(np.complex64)
        weights_in = rs.uniform(0.5, 2.0, (channels, baselines)).astype(np.float32)
        flags_in = rs.choice(4, (channels, baselines), p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8)
        # Ensure that we test the case of none flagged and all flagged when
        # doing continuum reduction
        flags_in[:, 123] = 1
        flags_in[:, 234] = 0

        template = sigproc.PostprocTemplate(test_context, cont_factor)
        fn = sigproc.Postproc(template, test_command_queue, channels, baselines)
        fn.ensure_all_bound()
        fn.slots['vis'].buffer.set(test_command_queue, vis_in)
        fn.slots['weights'].buffer.set(test_command_queue, weights_in)
        fn.slots['flags'].buffer.set(test_command_queue, flags_in)
        fn()

        # Compute expected spectral values
        expected_vis = vis_in / weights_in
        expected_weights = weights_in * (flags_in == 0) # Flagged visibilities have their weights set to zero

        # Compute expected continuum values.
        indices = range(0, channels, cont_factor)
        cont_weights = np.add.reduceat(weights_in, indices, axis=0)
        cont_vis = np.add.reduceat(vis_in, indices, axis=0) / cont_weights
        cont_flags = np.bitwise_and.reduceat(flags_in, indices, axis=0)
        cont_weights *= (cont_flags == 0)

        # Verify results
        np.testing.assert_allclose(expected_vis, fn.slots['vis'].buffer.get(test_command_queue))
        np.testing.assert_allclose(expected_weights, fn.slots['weights'].buffer.get(test_command_queue))
        np.testing.assert_allclose(cont_vis, fn.slots['cont_vis'].buffer.get(test_command_queue), rtol=1e-5)
        np.testing.assert_allclose(cont_weights, fn.slots['cont_weights'].buffer.get(test_command_queue), rtol=1e-5)
        np.testing.assert_equal(cont_flags, fn.slots['cont_flags'].buffer.get(test_command_queue))
