"""Test for the sigproc module."""

import unittest
import mock
import numpy as np
from katsdpingest import sigproc
from katsdpsigproc import accel, tune
import katsdpsigproc.rfi.device as rfi
import katsdpsigproc.rfi.host as rfi_host
from katsdpsigproc.test.test_accel import device_test, test_context, test_command_queue, force_autotune

def reduce_flags(flags, axis):
    """Reduction by logical AND along an axis. This is necessary because
    `np.bitwise_and.identity` is `incorrect`__.

    .. __: https://github.com/numpy/numpy/issues/5250
    """
    return np.bitwise_not(
            np.bitwise_or.reduce(np.bitwise_not(flags), axis))

def reduceat_flags(flags, indices, axis):
    """Segmented reduction by logical AND along an axis. See
    :func:`reduce_flags` for an explanation of why this is needed."""
    return np.bitwise_not(
            np.bitwise_or.reduceat(np.bitwise_not(flags), indices, axis))

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
        prepare.set_scale(scale)
        prepare()
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

    @device_test
    @force_autotune
    def testAutotune(self):
        sigproc.PrepareTemplate(test_context)

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

    @device_test
    @force_autotune
    def testAutotune(self):
        sigproc.AccumTemplate(test_context, 2)

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
        cont_flags = reduceat_flags(flags_in, indices, axis=0)
        cont_weights *= (cont_flags == 0)

        # Verify results
        np.testing.assert_allclose(expected_vis, fn.slots['vis'].buffer.get(test_command_queue), rtol=1e-5)
        np.testing.assert_allclose(expected_weights, fn.slots['weights'].buffer.get(test_command_queue), rtol=1e-5)
        np.testing.assert_allclose(cont_vis, fn.slots['cont_vis'].buffer.get(test_command_queue), rtol=1e-5)
        np.testing.assert_allclose(cont_weights, fn.slots['cont_weights'].buffer.get(test_command_queue), rtol=1e-5)
        np.testing.assert_equal(cont_flags, fn.slots['cont_flags'].buffer.get(test_command_queue))

    @device_test
    @force_autotune
    def testAutotune(self):
        sigproc.PostprocTemplate(test_context, 16)

class TestIngestOperation(unittest.TestCase):
    flag_value = 1 << sigproc.IngestTemplate.flag_names.index('detected_rfi')

    @mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.stub_autotuner)
    @mock.patch('katsdpsigproc.accel.build', spec=True)
    def testDescriptions(self, *args):
        channels = 128
        channel_range = (16, 96)
        baselines = 192

        context = mock.Mock()
        command_queue = mock.Mock()
        background_template = rfi.BackgroundMedianFilterDeviceTemplate(
                context, width=13)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(
                context, 10240)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
                context, n_sigma=11.0, transposed=True, flag_value=self.flag_value)
        flagger_template = rfi.FlaggerDeviceTemplate(
                background_template, noise_est_template, threshold_template)
        template = sigproc.IngestTemplate(context, flagger_template, 16)
        fn = template.instantiate(command_queue, channels, channel_range, baselines)

        expected = [
            ('ingest', 'class=katsdpingest.sigproc.IngestOperation', 'unknown'),
            ('ingest:prepare', 'baselines=192, channel_range=(16, 96), channels=128, class=katsdpingest.sigproc.Prepare, scale=1.0', 'unknown'),
            ('ingest:zero_spec_vis', 'class=katsdpsigproc.fill.Fill, ctype=float2, dtype=complex64, shape=(80, 192), value=0j', 'unknown'),
            ('ingest:zero_spec_weights', 'class=katsdpsigproc.fill.Fill, ctype=float, dtype=float32, shape=(80, 192), value=0.0', 'unknown'),
            ('ingest:zero_spec_flags', 'class=katsdpsigproc.fill.Fill, ctype=unsigned char, dtype=uint8, shape=(80, 192), value=255', 'unknown'),
            ('ingest:transpose_vis', 'class=katsdpsigproc.transpose.Transpose, ctype=float2, dtype=complex64, shape=(192, 128)', 'unknown'),
            ('ingest:flagger', 'class=katsdpsigproc.rfi.device.FlaggerDevice', 'unknown'),
            ('ingest:flagger:background', 'baselines=192, channels=128, class=katsdpsigproc.rfi.device.BackgroundMedianFilterDevice, width=13', 'unknown'),
            ('ingest:flagger:transpose_deviations', 'class=katsdpsigproc.transpose.Transpose, ctype=float, dtype=float32, shape=(128, 192)', 'unknown'),
            ('ingest:flagger:noise_est', 'baselines=192, channels=128, class=katsdpsigproc.rfi.device.NoiseEstMADTDevice, max_channels=10240', 'unknown'),
            ('ingest:flagger:threshold', 'baselines=192, channels=128, class=katsdpsigproc.rfi.device.ThresholdSimpleDevice, flag_value=16, n_sigma=11.0, transposed=True', 'unknown'), ('ingest:flagger:transpose_flags', 'class=katsdpsigproc.transpose.Transpose, ctype=unsigned char, dtype=uint8, shape=(192, 128)', 'unknown'), ('ingest:accum', 'baselines=192, channel_range=(16, 96), channels=128, class=katsdpingest.sigproc.Accum, outputs=1', 'unknown'), ('ingest:postproc', 'baselines=192, channels=80, class=katsdpingest.sigproc.Postproc, cont_factor=16', 'unknown')
        ]
        self.assertEqual(expected, fn.descriptions())

    def runHost(self, vis, scale, permutation, cont_factor, channel_range, n_sigma):
        """Simple CPU implementation. All inputs and outputs are
        channel-major.

        Parameters
        ----------
        vis : array-like
            Input dump visibilities (first axis being time)
        scale : float
            Scale factor for integral visibilities
        permutation : sequence
            Maps input baseline numbers to output numbers
        cont_factor : int
            Number of spectral channels per continuum channel
        n_sigma : float
            Significance level for flagger

        Returns
        -------
        tuple
            (spec vis, spec weights, spec flags, cont vis, cont weights, cont flags)
        """
        background = rfi_host.BackgroundMedianFilterHost(width=13)
        noise_est = rfi_host.NoiseEstMADHost()
        threshold = rfi_host.ThresholdSimpleHost(n_sigma=n_sigma, flag_value=self.flag_value)
        flagger = rfi_host.FlaggerHost(background, noise_est, threshold)

        vis = np.asarray(vis).astype(np.float32)
        # Scaling, and combine real and imaginary elements
        vis = vis[..., 0] * scale + vis[..., 1] * (1j * scale)
        # Baseline permutation
        vis_tmp = vis.copy()
        vis[..., permutation] = vis_tmp
        # Compute weights (currently just unity)
        weights = np.ones(vis.shape, dtype=np.float32)
        # Compute flags
        flags = np.empty(vis.shape, dtype=np.uint8)
        for i in range(len(vis)):
            flags[i, ...] = flagger(vis[i, ...])
        # Apply flags to weights
        weights *= (flags == 0).astype(np.float32) + 2**-64

        # Time accumulation
        vis = np.sum(vis * weights, axis=0)
        weights = np.sum(weights, axis=0)
        flags = reduce_flags(flags, axis=0)

        # Clip to the channel range
        rng = slice(channel_range[0], channel_range[1])
        vis = vis[rng, ...]
        weights = weights[rng, ...]
        flags = flags[rng, ...]

        # Continuum accumulation
        indices = range(0, vis.shape[0], cont_factor)
        cont_vis = np.add.reduceat(vis, indices, axis=0)
        cont_weights = np.add.reduceat(weights, indices, axis=0)
        cont_flags = reduceat_flags(flags, indices, axis=0)

        # Division by weight, and set weight to zero where flagged
        vis /= weights
        weights *= (flags == 0)
        cont_vis /= cont_weights
        cont_weights *= (cont_flags == 0)
        return (vis, weights, flags, cont_vis, cont_weights, cont_flags)

    def testRandom(self):
        """Test with random data against a CPU implementation"""
        channels = 128
        channel_range = (16, 96)
        baselines = 192
        cont_factor = 4
        scale = 1.0 / 64
        dumps = 4
        # Use a very low significance so that there will still be about 50%
        # flags after averaging
        n_sigma = -1.0

        rs = np.random.RandomState(seed=1)
        vis_in = rs.random_integers(-1000, 1000, (dumps, channels, baselines, 2)).astype(np.int32)
        permutation = rs.permutation(baselines).astype(np.uint16)

        background_template = rfi.BackgroundMedianFilterDeviceTemplate(
                test_context, width=13)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(
                test_context, 10240)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
                test_context, n_sigma=n_sigma, transposed=True, flag_value=self.flag_value)
        flagger_template = rfi.FlaggerDeviceTemplate(
                background_template, noise_est_template, threshold_template)
        template = sigproc.IngestTemplate(test_context, flagger_template, cont_factor)
        fn = template.instantiate(test_command_queue, channels, channel_range, baselines)
        fn.ensure_all_bound()
        fn.set_scale(scale)
        fn.slots['permutation'].buffer.set(test_command_queue, permutation)

        fn.start_sum()
        for i in range(dumps):
            fn.slots['vis_in'].buffer.set(test_command_queue, vis_in[i])
            fn()
        fn.end_sum()

        expected = self.runHost(vis_in, scale, permutation, cont_factor, channel_range, n_sigma)
        (expected_spec_vis, expected_spec_weights, expected_spec_flags,
                expected_cont_vis, expected_cont_weights, expected_cont_flags) = expected
        spec_vis = fn.slots['spec_vis'].buffer.get(test_command_queue)
        spec_weights = fn.slots['spec_weights'].buffer.get(test_command_queue)
        spec_flags = fn.slots['spec_flags'].buffer.get(test_command_queue)
        cont_vis = fn.slots['cont_vis'].buffer.get(test_command_queue)
        cont_weights = fn.slots['cont_weights'].buffer.get(test_command_queue)
        cont_flags = fn.slots['cont_flags'].buffer.get(test_command_queue)

        np.testing.assert_allclose(expected_spec_vis, spec_vis, rtol=1e-5)
        np.testing.assert_allclose(expected_spec_weights, spec_weights, rtol=1e-5)
        np.testing.assert_equal(expected_spec_flags, spec_flags)
        np.testing.assert_allclose(expected_cont_vis, cont_vis, rtol=1e-5)
        np.testing.assert_allclose(expected_cont_weights, cont_weights, rtol=1e-5)
        np.testing.assert_equal(expected_cont_flags, cont_flags)
