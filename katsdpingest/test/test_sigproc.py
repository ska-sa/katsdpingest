# coding: utf-8
"""Test for the sigproc module."""

from __future__ import print_function, division, absolute_import
import mock
import numpy as np
from katsdpingest import sigproc
from katsdpingest.utils import Range
from katsdpsigproc import tune
import katsdpsigproc.rfi.device as rfi
import katsdpsigproc.rfi.host as rfi_host
from katsdpsigproc.test.test_accel import device_test, force_autotune
from nose.tools import *


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


class TestPrepare(object):
    """Test :class:`katsdpingest.sigproc.Prepare`"""

    @device_test
    def test_prepare(self, context, queue):
        """Basic test of data preparation"""
        channels = 73
        in_baselines = 99
        out_baselines = 91
        n_accs = 11

        rs = np.random.RandomState(seed=1)
        vis_in = rs.random_integers(-1000, 1000, (channels, in_baselines, 2)).astype(np.int32)
        permutation = rs.permutation(in_baselines).astype(np.int16)
        permutation[permutation >= out_baselines] = -1

        template = sigproc.PrepareTemplate(context)
        prepare = template.instantiate(queue, channels, in_baselines, out_baselines)
        prepare.ensure_all_bound()
        prepare.buffer('vis_in').set(queue, vis_in)
        prepare.buffer('permutation').set(queue, permutation)
        prepare.n_accs = n_accs
        prepare()
        vis_out = prepare.buffer('vis_out').get(queue)

        assert_equal((out_baselines, channels), vis_out.shape)
        expected_vis = np.zeros_like(vis_out)
        scale = np.float32(1 / n_accs)
        for i in range(channels):
            for j in range(in_baselines):
                value = (vis_in[i, j, 0] + 1j * vis_in[i, j, 1]) * scale
                row = permutation[j]
                if row >= 0:
                    expected_vis[row, i] = value
        np.testing.assert_equal(expected_vis, vis_out)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        sigproc.PrepareTemplate(context)


class TestAutoWeights(object):
    """Test :class:`katsdpingest.sigproc.AutoWeights`"""

    @device_test
    def test_random(self, context, queue):
        """Basic test using random data"""
        channels = 73
        channel_range = Range(15, 70)
        inputs = 17
        baselines = 101
        n_accs = 12

        rs = np.random.RandomState(seed=1)
        vis = (rs.standard_normal((baselines, channels)) +
               rs.standard_normal((baselines, channels)) * 1j).astype(np.complex64)
        baseline_map = rs.permutation(baselines)[:inputs].astype(np.uint16)

        template = sigproc.AutoWeightsTemplate(context)
        fn = template.instantiate(queue, channels, channel_range, inputs, baselines)
        fn.n_accs = n_accs
        fn.ensure_all_bound()
        fn.buffer('vis').set(queue, vis)
        fn.buffer('baseline_map').set(queue, baseline_map)
        fn.buffer('weights').zero(queue)
        fn()
        weights = fn.buffer('weights').get(queue)

        assert_equal((inputs, len(channel_range)), weights.shape)
        expected = np.zeros_like(weights)
        scale = np.float32(np.sqrt(n_accs))
        for i in range(inputs):
            expected[i, :] = scale / vis[baseline_map[i], channel_range.asslice()].real
        np.testing.assert_allclose(expected, weights)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        sigproc.AutoWeightsTemplate(context)


class TestInitWeights(object):
    """Test :class:`katsdpingest.sigproc.InitWeights`"""

    @device_test
    def test_random(self, context, queue):
        """Basic test using random data"""
        channels = 73
        inputs = 17
        baselines = 101

        rs = np.random.RandomState(seed=1)
        auto_weights = rs.uniform(1.0, 2.0, size=(inputs, channels)).astype(np.float32)
        input_map = rs.randint(0, inputs - 1, size=(baselines, 2)).astype(np.uint16)

        template = sigproc.InitWeightsTemplate(context)
        fn = template.instantiate(queue, channels, inputs, baselines)
        fn.ensure_all_bound()
        fn.buffer('auto_weights').set(queue, auto_weights)
        fn.buffer('input_map').set(queue, input_map)
        fn.buffer('weights').zero(queue)
        fn()
        weights = fn.buffer('weights').get(queue)

        assert_equal((baselines, channels), weights.shape)
        expected = np.zeros_like(weights)
        for i in range(baselines):
            expected[i, :] = auto_weights[input_map[i, 0], :] * auto_weights[input_map[i, 1], :]
        np.testing.assert_allclose(expected, weights)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        sigproc.InitWeightsTemplate(context)


class TestAccum(object):
    """Test :class:`katsdpingest.sigproc.Accum`"""

    @device_test
    def test_small(self, context, queue):
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

        template = sigproc.AccumTemplate(context, 1)
        fn = template.instantiate(queue, 5, Range(1, 4), 1)
        fn.ensure_all_bound()
        for name, value in host.iteritems():
            fn.buffer(name).set(queue, value)
        fn()

        expected = {
            'vis_out0':     np.array([[11+7j, (12-12j) * flag_scale, 6+8j]], dtype=np.complex64).T,
            'weights_out0': np.array([[3.5, 4.0 * flag_scale, 7.5]], dtype=np.float32).T,
            'flags_out0':   np.array([[0, 8, 0]], dtype=np.uint8).T
        }
        for name, value in expected.iteritems():
            actual = fn.buffer(name).get(queue)
            np.testing.assert_equal(value, actual, err_msg=name + " does not match")

    @device_test
    def test_big(self, context, queue):
        """Test with large random data against a simple CPU version"""
        flag_scale = 2 ** -64
        channels = 203
        baselines = 171
        channel_range = Range(7, 198)
        kept_channels = len(channel_range)
        outputs = 2
        rs = np.random.RandomState(1)

        vis_in = (rs.standard_normal((baselines, channels)) +
                  rs.standard_normal((baselines, channels)) * 1j).astype(np.complex64)
        weights_in = rs.uniform(size=(baselines, kept_channels)).astype(np.float32)
        flags_in = rs.choice(4, (baselines, channels), p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8)
        vis_out = []
        weights_out = []
        flags_out = []
        for i in range(outputs):
            vis_out.append((rs.standard_normal((kept_channels, baselines)) +
                            rs.standard_normal((kept_channels, baselines)) * 1j)
                           .astype(np.complex64))
            weights_out.append(rs.uniform(size=(kept_channels, baselines)).astype(np.float32))
            flags_out.append(rs.choice(4, (kept_channels, baselines),
                             p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8))

        template = sigproc.AccumTemplate(context, outputs)
        fn = template.instantiate(queue, channels, channel_range, baselines)
        fn.ensure_all_bound()
        for (name, value) in [('vis_in', vis_in), ('weights_in', weights_in),
                              ('flags_in', flags_in)]:
            fn.buffer(name).set(queue, value)
        for (name, value) in [('vis_out', vis_out), ('weights_out', weights_out),
                              ('flags_out', flags_out)]:
            for i in range(outputs):
                fn.buffer(name + str(i)).set(queue, value[i])
        fn()

        # Perform the operation on the host
        kept_vis = vis_in[:, channel_range.start : channel_range.stop]
        kept_flags = flags_in[:, channel_range.start : channel_range.stop]
        flagged_weights = weights_in * ((kept_flags == 0) + flag_scale)
        for i in range(outputs):
            vis_out[i] += (kept_vis * flagged_weights).T
            weights_out[i] += flagged_weights.T
            flags_out[i] = np.bitwise_and(flags_out[i], kept_flags.T)

        # Verify results
        for (name, value) in [('vis_out', vis_out), ('weights_out', weights_out),
                              ('flags_out', flags_out)]:
            for i in range(outputs):
                actual = fn.buffer(name + str(i)).get(queue)
                np.testing.assert_allclose(value[i], actual, 1e-5)

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        sigproc.AccumTemplate(context, 2)


class TestPostproc(object):
    """Tests for :class:`katsdpingest.sigproc.Postproc`"""

    def test_bad_cont_factor(self):
        """Test with a continuum factor that does not divide into the channel count"""
        template = mock.sentinel.template
        mock.sentinel.command_queue.context = mock.sentinel.context
        assert_raises(ValueError, sigproc.Postproc, template, mock.sentinel.command_queue, 12, 8, 8)

    @device_test
    def test_postproc(self, context, queue):
        """Test with random data against a CPU implementation"""
        channels = 1024
        baselines = 512
        cont_factor = 16
        rs = np.random.RandomState(1)
        vis_in = (rs.standard_normal((channels, baselines)) +
                  rs.standard_normal((channels, baselines)) * 1j).astype(np.complex64)
        weights_in = rs.uniform(0.5, 2.0, (channels, baselines)).astype(np.float32)
        flags_in = rs.choice(4, (channels, baselines), p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8)
        # Ensure that we test the case of none flagged and all flagged when
        # doing continuum reduction
        flags_in[:, 123] = 1
        flags_in[:, 234] = 0

        template = sigproc.PostprocTemplate(context)
        fn = sigproc.Postproc(template, queue, channels, baselines, cont_factor)
        fn.ensure_all_bound()
        fn.buffer('vis').set(queue, vis_in)
        fn.buffer('weights').set(queue, weights_in)
        fn.buffer('flags').set(queue, flags_in)
        fn()

        # Compute expected spectral values
        expected_vis = vis_in / weights_in

        # Compute expected continuum values.
        indices = range(0, channels, cont_factor)
        cont_weights = np.add.reduceat(weights_in, indices, axis=0)
        cont_vis = np.add.reduceat(vis_in, indices, axis=0) / cont_weights
        cont_flags = reduceat_flags(flags_in, indices, axis=0)

        # Flagged visibilities have their weights re-scaled
        expected_weights = weights_in * ((flags_in != 0) * np.float32(2**64) + 1)
        cont_weights *= (cont_flags != 0) * np.float32(2**64) + 1

        # Verify results
        np.testing.assert_allclose(expected_vis, fn.buffer('vis').get(queue), rtol=1e-5)
        np.testing.assert_allclose(expected_weights, fn.buffer('weights').get(queue), rtol=1e-5)
        np.testing.assert_allclose(cont_vis, fn.buffer('cont_vis').get(queue), rtol=1e-5)
        np.testing.assert_allclose(cont_weights, fn.buffer('cont_weights').get(queue), rtol=1e-5)
        np.testing.assert_equal(cont_flags, fn.buffer('cont_flags').get(queue))

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        sigproc.PostprocTemplate(context)


class TestCompressWeights(object):
    """Tests for :class:`katsdpingest.sigproc.CompressWeights`"""
    @device_test
    def test_simple(self, context, queue):
        """Test with random data against a CPU implementation"""
        channels = 123
        baselines = 235
        rs = np.random.RandomState(1)
        weights_in = rs.uniform(0.01, 1000.0, (channels, baselines)).astype(np.float32)

        template = sigproc.CompressWeightsTemplate(context)
        fn = template.instantiate(queue, channels, baselines)
        fn.ensure_all_bound()
        fn.buffer('weights_in').set(queue, weights_in)
        fn.buffer('weights_out').zero(queue)
        fn.buffer('weights_channel').zero(queue)
        fn()

        expected_channel = np.max(weights_in, axis=1) * np.float32(1.0 / 255.0)
        scale = np.reciprocal(expected_channel)[..., np.newaxis]
        expected_out = (weights_in * scale).astype(np.uint8)
        np.testing.assert_allclose(expected_channel, fn.buffer('weights_channel').get(queue), rtol=1e-5)
        np.testing.assert_equal(expected_out, fn.buffer('weights_out').get(queue))

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        sigproc.CompressWeightsTemplate(context)


class TestIngestOperation(object):
    flag_value = 1 << sigproc.IngestTemplate.flag_names.index('ingest_rfi')

    @mock.patch('katsdpsigproc.tune.autotuner_impl', new=tune.stub_autotuner)
    @mock.patch('katsdpsigproc.accel.build', spec=True)
    def test_descriptions(self, *args):
        channels = 128
        channel_range = Range(16, 96)
        cbf_baselines = 220
        inputs = 32
        baselines = 192

        context = mock.Mock()
        command_queue = mock.Mock()
        background_template = rfi.BackgroundMedianFilterDeviceTemplate(
                context, width=13)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(
                context, 10240)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
                context, transposed=True, flag_value=self.flag_value)
        flagger_template = rfi.FlaggerDeviceTemplate(
                background_template, noise_est_template, threshold_template)
        template = sigproc.IngestTemplate(context, flagger_template, [8, 12])
        fn = template.instantiate(
                command_queue, channels, channel_range, inputs,
                cbf_baselines, baselines,
                8, 16, [(0, 8), (10, 22)],
                threshold_args={'n_sigma': 11.0})

        expected = [
            ('ingest', {'class': 'katsdpingest.sigproc.IngestOperation'}),
            ('ingest:prepare', {'channels': 128, 'class': 'katsdpingest.sigproc.Prepare', 'in_baselines': 220, 'out_baselines': 192, 'n_accs': 1}),
            ('ingest:auto_weights', {'baselines': 192, 'channel_range': (16, 96), 'channels': 128, 'class': 'katsdpingest.sigproc.AutoWeights', 'inputs': 32, 'n_accs': 1}),
            ('ingest:init_weights', {'baselines': 192, 'channels': 80, 'class': 'katsdpingest.sigproc.InitWeights', 'inputs': 32}),
            ('ingest:zero_spec', {'class': 'katsdpingest.sigproc.Zero'}),
            ('ingest:zero_spec:zero_vis', {'class': 'katsdpsigproc.fill.Fill', 'ctype': 'float2', 'dtype': 'complex64', 'shape': (80, 192), 'value': 0j}),
            ('ingest:zero_spec:zero_weights', {'class': 'katsdpsigproc.fill.Fill', 'ctype': 'float', 'dtype': 'float32', 'shape': (80, 192), 'value': 0.0}),
            ('ingest:zero_spec:zero_flags', {'class': 'katsdpsigproc.fill.Fill', 'ctype': 'unsigned char', 'dtype': 'uint8', 'shape': (80, 192), 'value': 255}),
            ('ingest:zero_sd_spec', {'class': 'katsdpingest.sigproc.Zero'}),
            ('ingest:zero_sd_spec:zero_vis', {'class': 'katsdpsigproc.fill.Fill', 'ctype': 'float2', 'dtype': 'complex64', 'shape': (80, 192), 'value': 0j}),
            ('ingest:zero_sd_spec:zero_weights', {'class': 'katsdpsigproc.fill.Fill', 'ctype': 'float', 'dtype': 'float32', 'shape': (80, 192), 'value': 0.0}),
            ('ingest:zero_sd_spec:zero_flags', {'class': 'katsdpsigproc.fill.Fill', 'ctype': 'unsigned char', 'dtype': 'uint8', 'shape': (80, 192), 'value': 255}),
            ('ingest:transpose_vis', {'class': 'katsdpsigproc.transpose.Transpose', 'ctype': 'float2', 'dtype': 'complex64', 'shape': (192, 128)}),
            ('ingest:flagger', {'class': 'katsdpsigproc.rfi.device.FlaggerDevice'}),
            ('ingest:flagger:background', {'baselines': 192, 'channels': 128, 'class': 'katsdpsigproc.rfi.device.BackgroundMedianFilterDevice', 'width': 13}),
            ('ingest:flagger:transpose_deviations', {'class': 'katsdpsigproc.transpose.Transpose', 'ctype': 'float', 'dtype': 'float32', 'shape': (128, 192)}),
            ('ingest:flagger:noise_est', {'baselines': 192, 'channels': 128, 'class': 'katsdpsigproc.rfi.device.NoiseEstMADTDevice', 'max_channels': 10240}),
            ('ingest:flagger:threshold', {'baselines': 192, 'channels': 128, 'class': 'katsdpsigproc.rfi.device.ThresholdSimpleDevice', 'flag_value': 16, 'n_sigma': 11.0, 'transposed': True}),
            ('ingest:flagger:transpose_flags', {'class': 'katsdpsigproc.transpose.Transpose', 'ctype': 'unsigned char', 'dtype': 'uint8', 'shape': (192, 128)}),
            ('ingest:accum', {'baselines': 192, 'channel_range': (16, 96), 'channels': 128, 'class': 'katsdpingest.sigproc.Accum', 'outputs': 2}),
            ('ingest:finalise', {'class': 'katsdpingest.sigproc.Finalise'}),
            ('ingest:finalise:postproc', {'baselines': 192, 'channels': 80, 'class': 'katsdpingest.sigproc.Postproc', 'cont_factor': 8}),
            ('ingest:finalise:compress_weights_spec', {'baselines': 192, 'channels': 80, 'class': 'katsdpingest.sigproc.CompressWeights'}),
            ('ingest:finalise:compress_weights_cont', {'baselines': 192, 'channels': 10, 'class': 'katsdpingest.sigproc.CompressWeights'}),
            ('ingest:sd_finalise', {'class': 'katsdpingest.sigproc.Finalise'}),
            ('ingest:sd_finalise:postproc', {'baselines': 192, 'channels': 80, 'class': 'katsdpingest.sigproc.Postproc', 'cont_factor': 16}),
            ('ingest:sd_finalise:compress_weights_spec', {'baselines': 192, 'channels': 80, 'class': 'katsdpingest.sigproc.CompressWeights'}),
            ('ingest:sd_finalise:compress_weights_cont', {'baselines': 192, 'channels': 5, 'class': 'katsdpingest.sigproc.CompressWeights'}),
            ('ingest:timeseries', {'class': 'katsdpsigproc.maskedsum.MaskedSum', 'shape': (80, 192)}),
            ('ingest:percentile0', {'class': 'katsdpsigproc.percentile.Percentile5', 'column_range': (0, 8), 'is_amplitude': False, 'max_columns': 8, 'shape': (80, 192)}),
            ('ingest:percentile0_flags', {'class': 'katsdpsigproc.reduce.HReduce', 'column_range': (0, 8), 'ctype': 'unsigned char', 'dtype': np.uint8, 'extra_code': '', 'identity': '0', 'op': 'a | b', 'shape': (80, 192)}),
            ('ingest:percentile1', {'class': 'katsdpsigproc.percentile.Percentile5', 'column_range': (10, 22), 'is_amplitude': False, 'max_columns': 12, 'shape': (80, 192)}),
            ('ingest:percentile1_flags', {'class': 'katsdpsigproc.reduce.HReduce', 'column_range': (10, 22), 'ctype': 'unsigned char', 'dtype': np.uint8, 'extra_code': '', 'identity': '0', 'op': 'a | b', 'shape': (80, 192)})
        ]
        self.maxDiff = None
        assert_equal(expected, fn.descriptions())

    def finalise_host(self, vis, flags, weights):
        """Does the final steps of run_host_basic, for either the continuum or spectral
        product. The inputs are modified in-place.
        """
        vis /= weights
        weights *= (flags != 0) * np.float32(2**64) + 1
        weights_channel = np.max(weights, axis=1) * np.float32(1.0 / 255.0)
        inv_weights_channel = np.float32(1.0) / weights_channel
        weights = (weights * inv_weights_channel[..., np.newaxis]).astype(np.uint8)
        return vis, flags, weights, weights_channel


    def run_host_basic(self, vis, n_accs, permutation, baseline_map, input_map,
                       cont_factor, channel_range, n_sigma):
        """Simple CPU implementation. All inputs and outputs are channel-major.
        There is no support for separate cadences for main and signal display
        products; instead, call the function twice with different time slices.
        No signal display calculations are performed.

        Parameters
        ----------
        vis : array-like
            Input dump visibilities (first axis being time)
        n_accs : int
            Number of correlations accumulated in `vis`
        permutation : sequence
            Maps input baseline numbers to output numbers (with -1 indicating discard)
        baseline_map : sequence
            Maps input signal to its post-permutation baseline
        input_map : sequence of pairs
            Maps baseline to pair of input signals
        cont_factor : int
            Number of spectral channels per continuum channel
        channel_range: 2-tuple of int
            Range of channels to retain in the output
        n_sigma : float
            Significance level for flagger

        Returns
        -------
        dictionary, with the following keys:

            - spec_vis, spec_weights, spec_flags
            - cont_vis, cont_weights, cont_flags
        """
        background = rfi_host.BackgroundMedianFilterHost(width=13)
        noise_est = rfi_host.NoiseEstMADHost()
        threshold = rfi_host.ThresholdSimpleHost(n_sigma=n_sigma, flag_value=self.flag_value)
        flagger = rfi_host.FlaggerHost(background, noise_est, threshold)

        vis = np.asarray(vis).astype(np.float32)
        # Scaling, and combine real and imaginary elements
        vis = vis[..., 0] + vis[..., 1] * 1j
        vis *= np.float32(1.0 / n_accs)
        # Baseline permutation
        new_baselines = np.sum(np.asarray(permutation) != -1)
        new_vis = np.empty(vis.shape[:-1] + (new_baselines,), np.complex64)
        weights = np.empty(new_vis.shape, np.float32)
        for old_idx, new_idx in enumerate(permutation):
            if new_idx != -1:
                new_vis[..., new_idx] = vis[..., old_idx]
        vis = new_vis
        # Compute initial weights
        for i in range(new_baselines):
            bl1 = baseline_map[input_map[i][0]]
            bl2 = baseline_map[input_map[i][1]]
            with np.errstate(divide='ignore'):
                weights[..., i] = np.float32(n_accs) / ((vis[..., bl1].real * vis[..., bl2].real))
                weights = np.where(np.isfinite(weights), weights, np.float32(2**-64))
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
        rng = slice(channel_range.start, channel_range.stop)
        vis = vis[rng, ...]
        weights = weights[rng, ...]
        flags = flags[rng, ...]

        # Continuum accumulation
        indices = range(0, vis.shape[0], cont_factor)
        cont_vis = np.add.reduceat(vis, indices, axis=0)
        cont_weights = np.add.reduceat(weights, indices, axis=0)
        cont_flags = reduceat_flags(flags, indices, axis=0)

        # Finalisation
        spec_vis, spec_flags, spec_weights, spec_weights_channel = \
            self.finalise_host(vis, flags, weights)
        cont_vis, cont_flags, cont_weights, cont_weights_channel = \
            self.finalise_host(cont_vis, cont_flags, cont_weights)
        return {
            'spec_vis': spec_vis,
            'spec_flags': spec_flags,
            'spec_weights': spec_weights,
            'spec_weights_channel': spec_weights_channel,
            'cont_vis': cont_vis,
            'cont_flags': cont_flags,
            'cont_weights': cont_weights,
            'cont_weights_channel': cont_weights_channel
        }

    def run_host(
            self, vis, n_vis, n_sd_vis, scale, permutation, baseline_map, input_map,
            cont_factor, sd_cont_factor, channel_range,
            n_sigma, timeseries_weights, percentile_ranges):
        """Simple CPU implementation. All inputs and outputs are channel-major.
        There is no support for separate cadences for main and signal display
        products; instead, call the function twice with different time slices.

        Parameters
        ----------
        vis : array-like
            Input dump visibilities (first axis being time)
        n_vis : int
            number of dumps to use for main calculations
        n_sd_vis : int
            number of dumps to use for signal display calculations
        scale : float
            Scale factor for integral visibilities
        permutation : sequence
            Maps input baseline numbers to output numbers (with -1 indicating discard)
        baseline_map : sequence
            Maps input signal to its post-permutation baseline
        input_map : sequence of pairs
            Maps baseline to pair of input signals
        cont_factor : int
            Number of spectral channels per continuum channel
        sd_cont_factor : int
            Number of spectral channels per continuum channel, for signal displays
        channel_range: 2-tuple of int
            Range of channels to retain in the output
        n_sigma : float
            Significance level for flagger
        timeseries_weights : 1D array of float
            Weights for masked timeseries averaging
        percentile_ranges : list of 2-tuples of int
            Range of baselines (after permutation) for each percentile product

        Returns
        -------
        dictionary, with the following keys:

            - spec_vis, spec_weights, spec_flags
            - cont_vis, cont_weights, cont_flags
            - sd_spec_vis, sd_spec_weights, sd_spec_flags
            - sd_cont_vis, sd_cont_weights, sd_cont_flags
            - timeseries
            - percentileN (where N is a non-negative integer)
        """
        expected = self.run_host_basic(
            vis[:n_vis], scale, permutation, baseline_map, input_map,
            cont_factor, channel_range, n_sigma)
        sd_expected = self.run_host_basic(
            vis[:n_sd_vis], scale, permutation, baseline_map, input_map,
            sd_cont_factor, channel_range, n_sigma)
        for (name, value) in sd_expected.iteritems():
            expected['sd_' + name] = value

        # Time series
        expected['timeseries'] = \
            np.sum(expected['sd_spec_vis'] * timeseries_weights[..., np.newaxis], axis=0)

        # Percentiles
        for i, (start, end) in enumerate(percentile_ranges):
            if start != end:
                percentile = np.percentile(
                    np.abs(expected['sd_spec_vis'][..., start:end]),
                    [0, 100, 25, 75, 50], axis=1, interpolation='lower')
                flags = np.bitwise_or.reduce(
                    expected['sd_spec_flags'][..., start:end], axis=1)
            else:
                percentile = \
                    np.tile(np.nan, (5, expected['sd_spec_vis'].shape[0])).astype(np.float32)
                flags = np.zeros(expected['sd_spec_flags'].shape[0], np.uint8)
            expected['percentile{0}'.format(i)] = percentile
            expected['percentile{0}_flags'.format(i)] = flags

        return expected

    @device_test
    def test_random(self, context, queue):
        """Test with random data against a CPU implementation"""
        channels = 128
        channel_range = Range(16, 96)
        kept_channels = len(channel_range)
        cbf_baselines = 220
        baselines = 192
        inputs = 32
        cont_factor = 4
        sd_cont_factor = 8
        n_accs = 64
        dumps = 4
        sd_dumps = 3   # Must currently be <= dumps, but could easily be fixed
        percentile_ranges = [(0, 10), (32, 40), (0, 0), (180, 192)]
        # Use a very low significance so that there will still be about 50%
        # flags after averaging
        n_sigma = -1.0

        rs = np.random.RandomState(seed=1)
        vis_in = \
            rs.random_integers(-1000, 1000, (dumps, channels, cbf_baselines, 2)).astype(np.int32)
        permutation = rs.permutation(cbf_baselines).astype(np.int16)
        permutation[permutation >= baselines] = -1
        baseline_map = rs.permutation(baselines)[:inputs].astype(np.uint16)
        input_map = rs.randint(0, inputs - 1, size=(baselines, 2)).astype(np.uint16)
        # Make sure input_map is consistent with baseline_map
        for i in range(inputs):
            input_map[baseline_map, :] = i
        # Make the designated autocorrelations look like autocorrelations (non-negative real)
        for baseline in baseline_map:
            orig_baseline = list(permutation).index(baseline)
            vis_in[..., orig_baseline, 0] = \
                rs.random_integers(0, 100, (dumps, channels)).astype(np.int32)
            vis_in[..., orig_baseline, 1].fill(0)
        timeseries_weights = rs.random_integers(0, 1, kept_channels).astype(np.float32)
        timeseries_weights /= np.sum(timeseries_weights)

        background_template = rfi.BackgroundMedianFilterDeviceTemplate(
                context, width=13)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(
                context, 10240)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
                context, transposed=True, flag_value=self.flag_value)
        flagger_template = rfi.FlaggerDeviceTemplate(
                background_template, noise_est_template, threshold_template)
        template = sigproc.IngestTemplate(context, flagger_template, [0, 8, 12])
        fn = template.instantiate(
                queue, channels, channel_range, inputs,
                cbf_baselines, baselines,
                cont_factor, sd_cont_factor, percentile_ranges,
                threshold_args={'n_sigma': n_sigma})
        fn.ensure_all_bound()
        fn.n_accs = n_accs
        fn.buffer('permutation').set(queue, permutation)
        fn.buffer('baseline_map').set(queue, baseline_map)
        fn.buffer('input_map').set(queue, input_map)
        fn.buffer('timeseries_weights').set(queue, timeseries_weights)

        data_keys = ['spec_vis', 'spec_weights', 'spec_weights_channel', 'spec_flags',
                     'cont_vis', 'cont_weights', 'cont_weights_channel', 'cont_flags']
        sd_keys = ['sd_spec_vis', 'sd_spec_weights', 'sd_spec_flags',
                   'sd_cont_vis', 'sd_cont_weights', 'sd_cont_flags',
                   'timeseries']
        for i in range(len(percentile_ranges)):
            sd_keys.append('percentile{0}'.format(i))
            sd_keys.append('percentile{0}_flags'.format(i))
        for name in data_keys + sd_keys:
            fn.buffer(name).zero(queue)

        actual = {}
        fn.start_sum()
        fn.start_sd_sum()
        for i in range(max(dumps, sd_dumps)):
            fn.buffer('vis_in').set(queue, vis_in[i])
            fn()
            if i + 1 == dumps:
                fn.end_sum()
                for name in data_keys:
                    actual[name] = fn.buffer(name).get(queue)
            if i + 1 == sd_dumps:
                fn.end_sd_sum()
                for name in sd_keys:
                    actual[name] = fn.buffer(name).get(queue)

        expected = self.run_host(
                vis_in, dumps, sd_dumps, n_accs, permutation, baseline_map, input_map,
                cont_factor, sd_cont_factor, channel_range, n_sigma,
                timeseries_weights, percentile_ranges)

        for name in data_keys + sd_keys:
            err_msg = '{0} is not equal'.format(name)
            if expected[name].dtype in (np.dtype(np.float32), np.dtype(np.complex64)):
                np.testing.assert_allclose(expected[name], actual[name], rtol=1e-5, atol=1e-5, err_msg=err_msg)
            elif name.endswith('_weights'):
                # Integer parts of weights can end up slightly different due to rounding
                np.testing.assert_allclose(expected[name], actual[name], atol=1, err_msg=err_msg)
            else:
                np.testing.assert_equal(expected[name], actual[name], err_msg=err_msg)
