# coding: utf-8
import pkg_resources
import numpy as np
from katsdpsigproc import accel, tune, fill, transpose, percentile, maskedsum, reduce


class ZeroTemplate(object):
    """Zeros a set of visibilities, weights and flags

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    """
    def __init__(self, context):
        self.context = context
        self.zero_vis = fill.FillTemplate(
                context, np.complex64, 'float2')
        self.zero_weights = fill.FillTemplate(
                context, np.float32, 'float')
        self.zero_flags = fill.FillTemplate(
                context, np.uint8, 'unsigned char')

    def instantiate(self, command_queue, channels, baselines):
        return Zero(self, command_queue, channels, baselines)


class Zero(accel.OperationSequence):
    def __init__(self, template, command_queue, channels, baselines):
        self.zero_vis = template.zero_vis.instantiate(
                command_queue, (channels, baselines))
        self.zero_weights = template.zero_weights.instantiate(
                command_queue, (channels, baselines))
        self.zero_flags = template.zero_flags.instantiate(
                command_queue, (channels, baselines))
        self.zero_flags.set_value(0xff)
        operations = [
                ('zero_vis', self.zero_vis),
                ('zero_weights', self.zero_weights),
                ('zero_flags', self.zero_flags)
        ]
        compounds = {
                'vis': ['zero_vis:data'],
                'weights': ['zero_weights:data'],
                'flags': ['zero_flags:data']
        }
        super(Zero, self).__init__(command_queue, operations, compounds)


class PrepareTemplate(object):
    """Handles first-stage data processing on a compute device:

    - Conversion to floating point
    - Scaling
    - Transposition
    - Baseline reordering

    When instantiating the template, one specifies a total number of channels,
    as well as a subrange of that total for which weights will be generated.
    At present weights are always 1.0, but this may change.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - block: number of workitems per workgroup in each dimension
        - vtx, vty: number of elements handled by each workitem, per dimension
    """

    autotune_version = 1

    def __init__(self, context, tuning=None):
        if tuning is None:
            tuning = self.autotune(context)
        self.block = tuning['block']
        self.vtx = tuning['vtx']
        self.vty = tuning['vty']
        program = accel.build(
            context, 'ingest_kernels/prepare.mako',
            {'block': self.block, 'vtx': self.vtx, 'vty': self.vty},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('prepare')

    @classmethod
    @tune.autotuner(test={'block': 16, 'vtx': 2, 'vty': 3})
    def autotune(cls, context):
        queue = context.create_tuning_command_queue()
        baselines = 1024
        channels = 2048
        channel_range = (128, channels - 128)
        kept_channels = channel_range[1] - channel_range[0]

        vis_in = accel.DeviceArray(context, (channels, baselines, 2), np.int32)
        vis_out = accel.DeviceArray(context, (baselines, channels), np.complex64)
        permutation = accel.DeviceArray(context, (baselines,), np.int16)
        weights = accel.DeviceArray(context, (baselines, kept_channels), np.float32)
        vis_in.set(queue, np.zeros(vis_in.shape, np.int32))
        permutation.set(queue, np.arange(baselines).astype(np.int16))

        def generate(block, vtx, vty):
            local_mem = (block * vtx + 1) * (block * vty) * 8
            if local_mem > 32768:
                # Skip configurations using lots of lmem
                raise RuntimeError('too much local memory')
            fn = cls(context, {
                'block': block,
                'vtx': vtx,
                'vty': vty}).instantiate(queue, channels, channel_range, baselines, baselines)
            fn.bind(vis_in=vis_in, vis_out=vis_out,
                    permutation=permutation, weights=weights)
            return tune.make_measure(queue, fn)
        return tune.autotune(
            generate,
            block=[8, 16, 32],
            vtx=[1, 2, 3, 4],
            vty=[1, 2, 3, 4])

    def instantiate(self, command_queue, channels, channel_range, in_baselines, out_baselines):
        return Prepare(self, command_queue, channels, channel_range, in_baselines, out_baselines)


class Prepare(accel.Operation):
    """Concrete instance of :class:`PrepareTemplate`.

    .. rubric:: Slots

    **vis_in** : channels × in_baselines × 2, int32
        Input visibilities
    **permutation** : in_baselines, int16
        Permutation mapping original to new baseline index, or -1 to discard
    **vis_out** : out_baselines × channels, complex64
        Transformed visibilities
    **weights** : out_baselines × kept-channels, float32
        Weights corresponding to visibilities

    Parameters
    ----------
    template : :class:`PrepareTemplate`
        Template containing the code
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    channels : int
        Number of channels
    channel_range : tuple of two ints
        Half-open interval of channels that will be written to **weights**
    in_baselines : int
        Number of baselines in the input
    out_baselines : int
        Number of baselines in the output
    """
    def __init__(self, template, command_queue, channels, channel_range,
                 in_baselines, out_baselines):
        if in_baselines < out_baselines:
            raise ValueError('Baselines can only be discarded, not amplified')
        super(Prepare, self).__init__(command_queue)
        tilex = template.block * template.vtx
        tiley = template.block * template.vty
        self.template = template
        self.channels = channels
        self.channel_range = channel_range
        self.in_baselines = in_baselines
        self.out_baselines = out_baselines
        self.scale = 1.0
        padded_channels = accel.Dimension(channels, tiley)
        padded_in_baselines = accel.Dimension(in_baselines, tilex)
        complex_parts = accel.Dimension(2, exact=True)
        self.slots['vis_in'] = accel.IOSlot(
                (padded_channels, padded_in_baselines, complex_parts), np.int32)
        # For output we do not need to pad the baselines, because the
        # permutation requires that we do range checks anyway.
        self.slots['vis_out'] = accel.IOSlot(
                (out_baselines, padded_channels), np.complex64)
        # Channels need to be range-checked anywhere here, so no padding
        self.slots['weights'] = accel.IOSlot(
                (out_baselines, channel_range[1] - channel_range[0]), np.float32)
        self.slots['permutation'] = accel.IOSlot((in_baselines,), np.int16)

    def set_scale(self, scale):
        self.scale = scale

    def _run(self):
        vis_in = self.buffer('vis_in')
        permutation = self.buffer('permutation')
        vis_out = self.buffer('vis_out')
        weights = self.buffer('weights')

        block = self.template.block
        tilex = block * self.template.vtx
        tiley = block * self.template.vty
        xblocks = accel.divup(self.in_baselines, tilex)
        yblocks = accel.divup(self.channels, tiley)
        self.command_queue.enqueue_kernel(
                self.template.kernel,
                [
                    vis_out.buffer,
                    weights.buffer,
                    vis_in.buffer,
                    permutation.buffer,
                    np.int32(vis_out.padded_shape[1]),
                    np.int32(weights.padded_shape[1]),
                    np.int32(vis_in.padded_shape[1]),
                    np.int32(self.channel_range[0]),
                    np.int32(self.channel_range[1]),
                    np.int32(self.in_baselines),
                    np.float32(self.scale)
                ],
                global_size=(xblocks * block, yblocks * block),
                local_size=(block, block))

    def parameters(self):
        return {
            'channels': self.channels,
            'channel_range': self.channel_range,
            'in_baselines': self.in_baselines,
            'out_baselines': self.out_baselines,
            'scale': self.scale
        }


class AccumTemplate(object):
    """Template for weighted visibility accumulation with flags. The
    inputs are in baseline-major order, while the outputs are in
    channel-major order. Support is provided for accumulating to multiple
    output sets.

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    outputs : int
        Number of outputs in which to accumulate
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - block: number of workitems per workgroup in each dimension
        - vtx, vty: number of elements handled by each workitem, per dimension
    """
    autotune_version = 1

    def __init__(self, context, outputs, tuning=None):
        if tuning is None:
            tuning = self.autotune(context, outputs)
        self.context = context
        self.block = tuning['block']
        self.vtx = tuning['vtx']
        self.vty = tuning['vty']
        self.outputs = outputs
        program = accel.build(
            context, 'ingest_kernels/accum.mako',
            {
                'block': self.block,
                'vtx': self.vtx,
                'vty': self.vty,
                'outputs': self.outputs},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('accum')

    @classmethod
    @tune.autotuner(test={'block': 8, 'vtx': 2, 'vty': 3})
    def autotune(cls, context, outputs):
        queue = context.create_tuning_command_queue()
        baselines = 1024
        channels = 2048
        channel_range = (128, channels - 128)
        kept_channels = channel_range[1] - channel_range[0]

        vis_in = accel.DeviceArray(context, (baselines, channels), np.complex64)
        weights_in = accel.DeviceArray(context, (baselines, kept_channels), np.float32)
        flags_in = accel.DeviceArray(context, (baselines, channels), np.uint8)
        out = {}
        for i in range(outputs):
            suffix = str(i)
            out['vis_out' + suffix] = accel.DeviceArray(
                context, (kept_channels, baselines), np.complex64)
            out['weights_out' + suffix] = accel.DeviceArray(
                context, (kept_channels, baselines), np.float32)
            out['flags_out' + suffix] = accel.DeviceArray(
                context, (kept_channels, baselines), np.uint8)

        rs = np.random.RandomState(seed=1)
        vis_in.set(queue, np.ones(vis_in.shape, np.complex64))
        weights_in.set(queue, np.ones(weights_in.shape, np.float32))
        flags_in.set(queue, rs.choice([0, 16], size=flags_in.shape,
                     p=[0.95, 0.05]).astype(np.uint8))

        def generate(block, vtx, vty):
            local_mem = (block * vtx + 1) * (block * vty) * 13
            if local_mem > 32768:
                # Skip configurations using lots of lmem
                raise RuntimeError('too much local memory')
            fn = cls(context, outputs, {
                'block': block,
                'vtx': vtx,
                'vty': vty}).instantiate(queue, channels, channel_range, baselines)
            fn.bind(vis_in=vis_in, weights_in=weights_in, flags_in=flags_in, **out)
            return tune.make_measure(queue, fn)
        return tune.autotune(
            generate,
            block=[8, 16, 32],
            vtx=[1, 2, 3, 4],
            vty=[1, 2, 3, 4])

    def instantiate(self, command_queue, channels, channel_range, baselines):
        return Accum(self, command_queue, channels, channel_range, baselines)


class Accum(accel.Operation):
    """Concrete instance of :class:`AccumTemplate`.

    .. rubric:: Slots

    In the outputs, *N* is an index starting from zero.

    **vis_in** : baselines × channels, complex64
        Input visibilities
    **weights_in** : baselines × kept-channels, float32
        Input weights
    **flags_in** : baselines × channels, uint8
        Input flags: non-zero values cause downweighting by 2^-64
    **vis_outN** : kept-channels × baselines, complex64
        Incremented by weight × visibility
    **weights_outN** : kept-channels × baselines, float32
        Incremented by (computed) weight
    **flags_outN** : kept-channels × baselines, uint8
        ANDed with the input flags

    Here *kept-channels* indicates the number of channels in `channel_range`.

    Parameters
    ----------
    template : :class:`AccumTemplate`
        Template containing the code
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    channels : int
        Number of channels
    change_range : tuple of two ints
        Half-open interval of channels that will appear in the output and in **weights_in**
    baselines : int
        Number of baselines
    """

    def __init__(self, template, command_queue, channels, channel_range, baselines):
        super(Accum, self).__init__(command_queue)
        tilex = template.block * template.vtx
        tiley = template.block * template.vty
        self.template = template
        self.channels = channels
        self.channel_range = channel_range
        self.baselines = baselines
        kept_channels = channel_range[1] - channel_range[0]
        padded_kept_channels = accel.Dimension(kept_channels, tilex)
        padded_baselines = accel.Dimension(baselines, tiley)
        padded_channels = accel.Dimension(
            channels,
            min_padded_size=max(channels, padded_kept_channels.min_padded_size + channel_range[0]))
        self.slots['vis_in'] = accel.IOSlot(
                (padded_baselines, padded_channels), np.complex64)
        self.slots['weights_in'] = accel.IOSlot(
                (padded_baselines, padded_kept_channels), np.float32)
        self.slots['flags_in'] = accel.IOSlot(
                (padded_baselines, padded_channels), np.uint8)
        for i in range(self.template.outputs):
            label = str(i)
            self.slots['vis_out' + label] = accel.IOSlot(
                (padded_kept_channels, padded_baselines), np.complex64)
            self.slots['weights_out' + label] = accel.IOSlot(
                (padded_kept_channels, padded_baselines), np.float32)
            self.slots['flags_out' + label] = accel.IOSlot(
                (padded_kept_channels, padded_baselines), np.uint8)

    def _run(self):
        buffer_names = []
        for i in range(self.template.outputs):
            label = str(i)
            buffer_names.extend(['vis_out' + label, 'weights_out' + label, 'flags_out' + label])
        buffer_names.extend(['vis_in', 'weights_in', 'flags_in'])
        buffers = [self.buffer(x) for x in buffer_names]
        args = [x.buffer for x in buffers] + [
            np.int32(buffers[0].padded_shape[1]),
            np.int32(buffers[-3].padded_shape[1]),
            np.int32(buffers[-2].padded_shape[1]),
            np.int32(self.channel_range[0])]

        kept_channels = self.channel_range[1] - self.channel_range[0]
        block = self.template.block
        tilex = block * self.template.vtx
        tiley = block * self.template.vty
        xblocks = accel.divup(kept_channels, tilex)
        yblocks = accel.divup(self.baselines, tiley)

        self.command_queue.enqueue_kernel(
                self.template.kernel,
                args,
                global_size=(xblocks * block, yblocks * block),
                local_size=(block, block))

    def parameters(self):
        return {
            'outputs': self.template.outputs,
            'channels': self.channels,
            'channel_range': self.channel_range,
            'baselines': self.baselines
        }


class PostprocTemplate(object):
    """Postprocessing performed on each output dump:

    - Accumulated visibility-weight product divided by weight
    - Weights for flagged outputs set to zero
    - Computation of continuum visibilities, weights and flags (flags are ANDed)

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgsx, wgsy: number of workitems per workgroup in each dimension
    """
    autotune_version = 2

    def __init__(self, context, tuning=None):
        if tuning is None:
            tuning = self.autotune(context)
        self.context = context
        self.wgsx = tuning['wgsx']
        self.wgsy = tuning['wgsy']
        program = accel.build(
            context, 'ingest_kernels/postproc.mako',
            {
                'wgsx': self.wgsx,
                'wgsy': self.wgsy,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('postproc')

    @classmethod
    @tune.autotuner(test={'wgsx': 32, 'wgsy': 8})
    def autotune(cls, context):
        queue = context.create_tuning_command_queue()
        baselines = 1024
        channels = 2048
        cont_factor = 16
        cont_channels = channels // cont_factor

        vis = accel.DeviceArray(context, (channels, baselines), np.complex64)
        weights = accel.DeviceArray(context, (channels, baselines), np.float32)
        flags = accel.DeviceArray(context, (channels, baselines), np.uint8)
        cont_vis = accel.DeviceArray(context, (cont_channels, baselines), np.complex64)
        cont_weights = accel.DeviceArray(context, (cont_channels, baselines), np.float32)
        cont_flags = accel.DeviceArray(context, (cont_channels, baselines), np.uint8)

        rs = np.random.RandomState(seed=1)
        vis.set(queue, np.ones(vis.shape, np.complex64))
        weights.set(queue, rs.uniform(1e-5, 4.0, weights.shape).astype(np.float32))
        flags.set(queue, rs.choice([0, 16], size=flags.shape, p=[0.95, 0.05]).astype(np.uint8))

        def generate(**tuning):
            fn = cls(context, tuning=tuning).instantiate(
                    queue, channels, baselines, cont_factor)
            fn.bind(vis=vis, weights=weights, flags=flags)
            fn.bind(cont_vis=cont_vis, cont_weights=cont_weights, cont_flags=cont_flags)
            return tune.make_measure(queue, fn)
        return tune.autotune(
            generate,
            wgsx=[8, 16, 32],
            wgsy=[8, 16, 32])

    def instantiate(self, context, channels, baselines, cont_factor):
        return Postproc(self, context, channels, baselines, cont_factor)


class Postproc(accel.Operation):
    """Concrete instance of :class:`PostprocTemplate`.

    .. rubric:: Slots

    **vis** : channels × baselines, complex64
        Sum of visibility times weight (on input), average visibility (on output)
    **weights** : channels × baselines, float32
        Sum of weights; on output, flagged results are set to zero
    **flags** : channels × baselines, uint8
        Flags (read-only)
    **cont_vis** : channels/cont_factor × baselines, complex64
        Output continuum visibilities
    **cont_weights** : channels/cont_factor × baselines, float32
        Output continuum weights
    **cont_flags** : channels/cont_factor × baselines, uint8
        Output continuum flags

    Parameters
    ----------
    template : :class:`PostprocTemplate`
        Template containing the code
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    channels : int
        Number of channels (must be a multiple of `template.cont_factor`)
    baselines : int
        Number of baselines
    cont_factor : int
        Number of spectral channels per continuum channel

    Raises
    ------
    ValueError
        If `channels` is not a multiple of `template.cont_factor`
    """
    def __init__(self, template, command_queue, channels, baselines, cont_factor):
        super(Postproc, self).__init__(command_queue)
        self.template = template
        self.channels = channels
        self.baselines = baselines
        self.cont_factor = cont_factor

        if channels % cont_factor:
            raise ValueError('Number of channels must be a multiple of the continuum factor')
        cont_channels = channels // cont_factor

        spectral_dims = (
            accel.Dimension(channels, cont_factor * template.wgsy),
            accel.Dimension(baselines, template.wgsx))
        cont_dims = (
            accel.Dimension(cont_channels, template.wgsy),
            spectral_dims[1])
        self.slots['vis'] = accel.IOSlot(spectral_dims, np.complex64)
        self.slots['weights'] = accel.IOSlot(spectral_dims, np.float32)
        self.slots['flags'] = accel.IOSlot(spectral_dims, np.uint8)
        self.slots['cont_vis'] = accel.IOSlot(cont_dims, np.complex64)
        self.slots['cont_weights'] = accel.IOSlot(cont_dims, np.float32)
        self.slots['cont_flags'] = accel.IOSlot(cont_dims, np.uint8)

    def _run(self):
        buffer_names = ['vis', 'weights', 'flags', 'cont_vis', 'cont_weights', 'cont_flags']
        buffers = [self.buffer(name) for name in buffer_names]
        args = [x.buffer for x in buffers] + [
            np.int32(self.cont_factor),
            np.int32(buffers[0].padded_shape[1])]
        xblocks = accel.divup(self.baselines, self.template.wgsx)
        yblocks = accel.divup(self.channels, self.template.wgsy * self.cont_factor)
        self.command_queue.enqueue_kernel(
            self.template.kernel, args,
            global_size=(xblocks * self.template.wgsx, yblocks * self.template.wgsy),
            local_size=(self.template.wgsx, self.template.wgsy))

    def parameters(self):
        return {
            'cont_factor': self.cont_factor,
            'channels': self.channels,
            'baselines': self.baselines
        }


class IngestTemplate(object):
    """Template for the entire on-device ingest processing

    Parameters
    ----------
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context for which kernels will be compiled
    flagger : :class:`katsdpsigproc.rfi.device.FlaggerTemplateDevice`
        Template for RFI flagging. It must have transposed flag outputs.
    percentile_sizes : list of int
        Set of number of baselines per percentile calculation. These do not need to exactly
        match the actual sizes of the ranges passed to the instance. The smaller template
        that is big enough will be used, so an exact match is best for good performance,
        and there must be one that is at least big enough.
    """

    flag_names = ['reserved0', 'static', 'cam', 'reserved3', 'ingest_rfi',
                  'predicted_rfi', 'cal_rfi', 'reserved7']

    def __init__(self, context, flagger, percentile_sizes):
        self.context = context
        self.prepare = PrepareTemplate(context)
        self.zero = ZeroTemplate(context)
        self.transpose_vis = transpose.TransposeTemplate(
                context, np.complex64, 'float2')
        self.flagger = flagger
        self.accum = AccumTemplate(context, 2)
        self.postproc = PostprocTemplate(context)
        self.timeseries = maskedsum.MaskedSumTemplate(context)
        self.percentiles = [percentile.Percentile5Template(
            context, max(size, 1), False) for size in percentile_sizes]
        self.percentiles_flags = reduce.HReduceTemplate(
            context, np.uint8, 'unsigned char', 'a | b', '0')
        # These last two are used when the input is empty (zero baselines)
        self.nan_percentiles = fill.FillTemplate(context, np.float32, 'float')
        self.zero_percentiles_flags = fill.FillTemplate(context, np.uint8, 'unsigned char')

    def instantiate(
            self, command_queue, channels, channel_range,
            cbf_baselines, baselines, cont_factor, sd_cont_factor, percentile_ranges,
            background_args={}, noise_est_args={}, threshold_args={}):
        return IngestOperation(
                self, command_queue, channels, channel_range,
                cbf_baselines, baselines, cont_factor, sd_cont_factor, percentile_ranges,
                background_args, noise_est_args, threshold_args)


class IngestOperation(accel.OperationSequence):
    """Concrete instance of :class:`IngestTemplate`.

    .. rubric:: Input slots

    **vis_in** : channels × baselines × 2, int32
        Input visibilities from the correlator
    **permutation** : baselines, int16
        Permutation mapping original to new baseline index
    **timeseries_weights** : kept-channels, float32
        Per-channel weights for timeseries averaging

    .. rubric:: Output slots

    **spec_vis** : kept-channels × baselines, complex64
        Spectral visibilities
    **spec_weights** : kept-channels × baselines, float32
        Spectral weights
    **spec_flags** : kept-channels × baselines, uint8
        Spectral flags
    **cont_vis** : kept-channels/`cont_factor` × baselines, complex64
        Continuum visibilities
    **cont_weights** : kept-channels/`cont_factor` × baselines, float32
        Continuum weights
    **cont_flags** : kept-channels/`cont_factor` × baselines, uint8
        Continuum flags
    **sd_spec_vis**, **sd_spec_weights**, **sd_spec_flags**, **sd_cont_vis**, **sd_cont_weights**, **sd_cont_flags**
        Signal display versions of the above
    **timeseries** : kept-channels, complex64
        Weights sum over channels of **sd_spec_vis**
    **percentileN** : 5 × kept-channels, float32 (where *N* is 0, 1, ...)
        Percentiles for each selected set of baselines (see `katsdpsigproc.percentile.Percentile5`)
    **percentileN**_flags : kept-channels, uint8 (where *N* is 0, 1, ...)
        For each channel, the bitwise OR of the flags from the corresponding set of baselines in **percentileN**

    .. rubric:: Scratch slots

    These are subject to change and so are not documented at this time.

    Parameters
    ----------
    template : :class:`IngestTemplate`
        Template containing the code
    command_queue : :class:`katsdpsigproc.cuda.CommandQueue` or :class:`katsdpsigproc.opencl.CommandQueue`
        Command queue for the operation
    channels : int
        Number of channels
    channel_range : tuple of two ints
        Half-open interval of channels that will be written to **weights**
    cbf_baselines : int
        Number of baselines received from CBF
    baselines : int
        Number of baselines, after antenna masking
    cont_factor : int
        Number of spectral channels per continuum channel
    sd_cont_factor : int
        Number of spectral channels to average together for signal displays
    percentile_ranges : list of 2-tuples of ints
        Column range for each set of baselines (post-permutation) for which
        percentiles will be computed.
    background_args : dict, optional
        Extra keyword arguments to pass to the background instantiation
    noise_est_args : dict, optional
        Extra keyword arguments to pass to the noise estimation instantiation
    threshold_args : dict, optional
        Extra keyword arguments to pass to the threshold instantiation
    """
    def __init__(
            self, template, command_queue, channels, channel_range,
            cbf_baselines, baselines,
            cont_factor, sd_cont_factor, percentile_ranges,
            background_args={}, noise_est_args={}, threshold_args={}):
        kept_channels = channel_range[1] - channel_range[0]
        self.template = template
        self.prepare = template.prepare.instantiate(
                command_queue, channels, channel_range, cbf_baselines, baselines)
        self.zero_spec = template.zero.instantiate(command_queue, kept_channels, baselines)
        self.zero_sd_spec = template.zero.instantiate(command_queue, kept_channels, baselines)
        # TODO: a single transpose+absolute value kernel uses less memory
        self.transpose_vis = template.transpose_vis.instantiate(
                command_queue, (baselines, channels))
        self.flagger = template.flagger.instantiate(
                command_queue, channels, baselines, background_args, noise_est_args, threshold_args)
        self.accum = template.accum.instantiate(
                command_queue, channels, channel_range, baselines)
        self.postproc = template.postproc.instantiate(
                command_queue, kept_channels, baselines, cont_factor)
        self.sd_postproc = template.postproc.instantiate(
                command_queue, kept_channels, baselines, sd_cont_factor)
        self.timeseries = template.timeseries.instantiate(
                command_queue, (kept_channels, baselines))
        self.percentiles = []
        self.percentiles_flags = []
        for prange in percentile_ranges:
            size = prange[1] - prange[0]
            ptemplate = None
            if size > 0:
                # Find the smallest match
                for t in template.percentiles:
                    if t.max_columns >= size:
                        if ptemplate is None or t.max_columns < ptemplate.max_columns:
                            ptemplate = t
                if ptemplate is None:
                    raise ValueError(
                        'Baseline range {0} is too large for any template'.format(prange))
                self.percentiles.append(
                    ptemplate.instantiate(command_queue, (kept_channels, baselines), prange))
                self.percentiles_flags.append(
                    template.percentiles_flags.instantiate(
                        command_queue, (kept_channels, baselines), prange))
            else:
                self.percentiles.append(
                    template.nan_percentiles.instantiate(command_queue, (5, kept_channels)))
                self.percentiles[-1].set_value(np.nan)
                self.percentiles_flags.append(
                    template.zero_percentiles_flags.instantiate(command_queue, (kept_channels,)))
                self.percentiles_flags[-1].set_value(0)

        # The order of these does not matter, since the actual sequencing is
        # done by methods in this class.
        operations = [
                ('prepare', self.prepare),
                ('zero_spec', self.zero_spec),
                ('zero_sd_spec', self.zero_sd_spec),
                ('transpose_vis', self.transpose_vis),
                ('flagger', self.flagger),
                ('accum', self.accum),
                ('postproc', self.postproc),
                ('sd_postproc', self.sd_postproc),
                ('timeseries', self.timeseries)
        ]
        for i in range(len(self.percentiles)):
            name = 'percentile{0}'.format(i)
            operations.append((name, self.percentiles[i]))
            operations.append((name + '_flags', self.percentiles_flags[i]))

        # TODO: eliminate transposition of flags, which aren't further used
        assert 'flags_t' in self.flagger.slots
        compounds = {
                'vis_in':       ['prepare:vis_in'],
                'permutation':  ['prepare:permutation'],
                'vis_t':        ['prepare:vis_out', 'transpose_vis:src', 'accum:vis_in'],
                'weights':      ['prepare:weights', 'accum:weights_in'],
                'vis_mid':      ['transpose_vis:dest', 'flagger:vis'],
                'deviations':   ['flagger:deviations'],
                'noise':        ['flagger:noise'],
                'flags':        ['flagger:flags_t', 'accum:flags_in'],
                'spec_vis':     ['accum:vis_out0', 'zero_spec:vis', 'postproc:vis'],
                'spec_weights': ['accum:weights_out0', 'zero_spec:weights', 'postproc:weights'],
                'spec_flags':   ['accum:flags_out0', 'zero_spec:flags', 'postproc:flags'],
                'cont_vis':     ['postproc:cont_vis'],
                'cont_weights': ['postproc:cont_weights'],
                'cont_flags':   ['postproc:cont_flags'],
                'sd_spec_vis':  ['accum:vis_out1', 'zero_sd_spec:vis',
                                 'sd_postproc:vis', 'timeseries:src'],
                'sd_spec_weights': ['accum:weights_out1', 'zero_sd_spec:weights',
                                    'sd_postproc:weights'],
                'sd_spec_flags': ['accum:flags_out1', 'zero_sd_spec:flags', 'sd_postproc:flags'],
                'sd_cont_vis':  ['sd_postproc:cont_vis'],
                'sd_cont_weights': ['sd_postproc:cont_weights'],
                'sd_cont_flags': ['sd_postproc:cont_flags'],
                'timeseries_weights': ['timeseries:mask'],
                'timeseries':   ['timeseries:dest']
        }
        for i in range(len(self.percentiles)):
            name = 'percentile{0}'.format(i)
            # The :data slots are used when NaN-filling. Unused slots are ignored,
            # so it is safe to just list all the variations.
            compounds['sd_spec_vis'].append(name + ':src')
            compounds['sd_spec_flags'].append(name + '_flags:src')
            compounds[name] = [name + ':dest', name + ':data']
            compounds[name + '_flags'] = [name + '_flags:dest', name + '_flags:data']

        aliases = {
                'scratch1': ['vis_in', 'vis_mid', 'flagger:deviations_t', 'flagger:flags']
        }

        super(IngestOperation, self).__init__(command_queue, operations, compounds, aliases)

    def set_scale(self, scale):
        self.prepare.set_scale(scale)

    def _run(self):
        """Process a single input dump"""
        self.prepare()
        self.transpose_vis()
        self.flagger()
        self.accum()

    def start_sum(self, **kwargs):
        """Reset accumulation buffers for a new output dump"""
        self.bind(**kwargs)
        self.ensure_all_bound()

        self.zero_spec()

    def end_sum(self):
        """Perform postprocessing for an output dump. This only does
        on-device processing; it does not transfer the results to the host.
        """
        self.postproc()

    def start_sd_sum(self, **kwargs):
        """Reset accumulation buffers for a new signal display dump"""
        self.bind(**kwargs)
        self.ensure_all_bound()

        self.zero_sd_spec()

    def end_sd_sum(self):
        """Perform postprocessing for a signal display dump. This only does
        on-device processing; it does not transfer the results to the host.
        """
        self.sd_postproc()
        self.timeseries()
        for p, f in zip(self.percentiles, self.percentiles_flags):
            p()
            f()

    def descriptions(self):
        """Generate descriptions of all the components.

        Each description is a 2-tuple consisting of a component name and
        a dictionary of parameters describing the operation."""
        def generate(operation, name):
            parameters = dict(operation.parameters())
            parameters['class'] = (operation.__class__.__module__ + '.' +
                                   operation.__class__.__name__)
            yield (name, parameters)
            if isinstance(operation, accel.OperationSequence):
                for child_name, child_op in operation.operations.iteritems():
                    for d in generate(child_op, child_name):
                        yield (name + ':' + d[0], d[1])
        return list(generate(self, 'ingest'))
