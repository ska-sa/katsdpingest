# coding: utf-8
import logging
import inspect
import time
import pkg_resources

import numpy as np
import scipy.signal as signal

import katpoint
from katsdpdisp import CorrProdRef
from .vanvleck import create_correction
from .antsol import stefcal
from .__init__ import __version__ as revision
import katsdpsigproc.rfi.host
import katsdpsigproc.rfi.device
from katsdpsigproc import accel, tune, fill, transpose

try:
    context = accel.create_some_context(False)
except RuntimeError:
    context = None

class ProcBlock(object):
    """A generic processing block for use in the Kat-7 online system.

    A processing block will have access to the current data frame and
    a number of history frames determined by the top level script (basically limited by memory constraints).
   
    A frame is currently defined as the complex spectral correlation output of the correlator for a single time interval.
    (Nchans, Nbaselines, 2, dtype)

    The dtype is not specified but is typically np.int32 (direct from correlator), or np.float32 (post scaling).
 
    It is expected that a processing block will require only minimal state, beyond that which is
    avaiable via the history.

    Support for both inline modification and production of new data products is provided.
    """
    current = None
    history = None
    logger = logging.getLogger("kat.k7capture.sigproc")
     # class attributes for storing references to numpy arrays contaning current and historical data
     # current is purely for convenience as current == history[0]
    _proc_times = []
    def __init__(self, **kwargs):
        self.cpref = CorrProdRef(**kwargs)
        self.expected_dtype = None
        self._extracted_arguments = self._extract_args(inspect.currentframe(1))
         # extract the arguments from the parent __init__

    def _extract_args(self, frame):
        args, _, _, values = inspect.getargvalues(frame)
        return ", ".join(["%s=%s" % (k,values[k]) for k in args if k is not 'self'])

    def proc(self, *args, **kwargs):
        """Process a single time instance of data."""
        if self.current is None:
            self.logger.warning("Warning: Data references have not been set up. This is almost certainly in error. The data exists as an attribute on the ProcBlock base class.")
        if self.expected_dtype is not None and self.current.dtype != self.expected_dtype:
            smsg = "%s expects data of type %s not %s" % (self.__class__.__name__,str(self.expected_dtype),str(self.current.dtype))
            self.logger.error(smsg)
            raise(TypeError,smsg)
        st = time.time()
        retval = self._proc(*args, **kwargs)
        self._proc_times.append(time.time() - st)
        return retval

    def _proc(self):
        raise NotImplementedError("Method must be overridden by subclasses.")

    def __str__(self):
        """Accumulated processing statistics of this block."""
        descr = ["Processing Block: %s" % self.__class__.__name__]
        descr.append("Parameters: " + ", ".join(["%s == %s" % (k,v) for k,v in vars(self).iteritems() if not k.startswith("_")]))
        if len(self._proc_times) > 0:
            descr.append("Processed %i frame(s) in %.3fs. Last: %.3fs, Avg: %.3fs" % (len(self._proc_times), np.sum(self._proc_times), self._proc_times[-1], np.average(self._proc_times)))
        else:
            descr.append("No frames processed.")
        if self.current is not None:
            descr.append("Current data has shape %s and type %s" % (self.current.shape, self.current.dtype))
        else:
            descr.append("No current data.")
        return "\n".join(descr)

    def description(self):
        """Compact representation of Processing Block used in process logs in the hdf5 file."""
        process = self.__class__.__name__
        if revision == 'unknown':
            svn_rev = 0
        else:
            svn_rev = int(revision[1:])

        return (process, self._extracted_arguments, svn_rev)

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
    def __init__(self, context, tuning=None):
        if tuning is None:
            tuning = self.autotune(context)
        self.block = tuning['block']
        self.vtx = tuning['vtx']
        self.vty = tuning['vty']
        program = accel.build(context, 'ingest_kernels/prepare.mako',
                {'block': self.block, 'vtx': self.vtx, 'vty': self.vty},
                extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('prepare')

    @classmethod
    @tune.autotuner(test={'block': 16, 'vtx': 1, 'vty': 1})
    def autotune(cls, context):
        # TODO: do real autotuning
        return {'block': 16, 'vtx': 1, 'vty': 1}

    def instantiate(self, command_queue, channels, channel_range, baselines):
        return Prepare(self, command_queue, channels, channel_range, baselines)

class Prepare(accel.Operation):
    """Concrete instance of :class:`PrepareTemplate`.

    .. rubric:: Slots

    **vis_in** : channels × baselines × 2, int32
        Input visibilities
    **permutation** : baselines, uint16
        Permutation mapping original to new baseline index
    **vis_out** : baselines × channels, complex64
        Transformed visibilities
    **weights** : baselines × kept-channels, float32
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
    baselines : int
        Number of baselines
    """
    def __init__(self, template, command_queue, channels, channel_range, baselines):
        super(Prepare, self).__init__(command_queue)
        tilex = template.block * template.vtx
        tiley = template.block * template.vty
        self.template = template
        self.channels = channels
        self.channel_range = channel_range
        self.baselines = baselines
        self.scale = 1.0
        padded_channels = accel.Dimension(channels, tiley)
        padded_baselines = accel.Dimension(baselines, tilex)
        complex_parts = accel.Dimension(2, exact=True)
        self.slots['vis_in'] = accel.IOSlot(
                (padded_channels, padded_baselines, complex_parts), np.int32)
        # For output we do not need to pad the baselines, because the
        # permutation requires that we do range checks anyway.
        self.slots['vis_out'] = accel.IOSlot(
                (baselines, padded_channels), np.complex64)
        # Channels need to be range-checked anywhere here, so no padding
        self.slots['weights'] = accel.IOSlot(
                (baselines, channel_range[1] - channel_range[0]), np.float32)
        self.slots['permutation'] = accel.IOSlot((baselines,), np.uint16)

    def set_scale(self, scale):
        self.scale = scale

    def _run(self):
        vis_in = self.slots['vis_in'].buffer
        permutation = self.slots['permutation'].buffer
        vis_out = self.slots['vis_out'].buffer
        weights = self.slots['weights'].buffer

        block = self.template.block
        tilex = block * self.template.vtx
        tiley = block * self.template.vty
        xblocks = accel.divup(self.baselines, tilex)
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
                    np.int32(self.baselines),
                    np.float32(self.scale)
                ],
                global_size = (xblocks * block, yblocks * block),
                local_size = (block, block))

    def parameters(self):
        return {
            'channels': self.channels,
            'channel_range': self.channel_range,
            'baselines': self.baselines,
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
    def __init__(self, context, outputs, tuning=None):
        if tuning is None:
            tuning = self.autotune(context)
        self.context = context
        self.block = tuning['block']
        self.vtx = tuning['vtx']
        self.vty = tuning['vty']
        self.outputs = outputs
        program = accel.build(context, 'ingest_kernels/accum.mako',
            {
                'block': self.block,
                'vtx': self.vtx,
                'vty': self.vty,
                'outputs': self.outputs},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('accum')

    @classmethod
    @tune.autotuner(test={'block': 16, 'vtx': 1, 'vty': 1})
    def autotune(cls, context):
        # TODO: do real autotuning
        return {'block': 16, 'vtx': 1, 'vty': 1}

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
        padded_channels = accel.Dimension(channels,
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
        buffers = [self.slots[x].buffer for x in buffer_names]
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
                global_size = (xblocks * block, yblocks * block),
                local_size = (block, block))

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
    cont_factor : int
        Number of spectral channels per continuum channel
    tuning : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - wgsx, wgsy: number of workitems per workgroup in each dimension
    """
    def __init__(self, context, cont_factor, tuning=None):
        if tuning is None:
            tuning = self.autotune(context, cont_factor)
        self.context = context
        self.cont_factor = cont_factor
        self.wgsx = tuning['wgsx']
        self.wgsy = tuning['wgsy']
        program = accel.build(context, 'ingest_kernels/postproc.mako',
            {
                'wgsx': self.wgsx,
                'wgsy': self.wgsy,
                'cont_factor': cont_factor
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('postproc')

    @classmethod
    @tune.autotuner(test={'wgsx': 32, 'wgsy': 8})
    def autotune(cls, context, cont_factor):
        # TODO: do real autotuning
        return {'wgsx': 32, 'wgsy': 8}

    def instantiate(self, context, channels, baselines):
        return Postproc(self, context, channels, baselines)

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

    Raises
    ------
    ValueError
        If `channels` is not a multiple of `template.cont_factor`
    """
    def __init__(self, template, command_queue, channels, baselines):
        super(Postproc, self).__init__(command_queue)
        self.template = template
        self.channels = channels
        self.baselines = baselines

        if channels % template.cont_factor:
            raise ValueError('Number of channels must be a multiple of the continuum factor')
        cont_channels = channels // template.cont_factor

        spectral_dims = (
            accel.Dimension(channels, template.cont_factor * template.wgsy),
            accel.Dimension(baselines, template.wgsx))
        cont_dims = (
            accel.Dimension(cont_channels, template.wgsy),
            spectral_dims[1])
        self.slots['vis'] =     accel.IOSlot(spectral_dims, np.complex64)
        self.slots['weights'] = accel.IOSlot(spectral_dims, np.float32)
        self.slots['flags'] =   accel.IOSlot(spectral_dims, np.uint8)
        self.slots['cont_vis'] =     accel.IOSlot(cont_dims, np.complex64)
        self.slots['cont_weights'] = accel.IOSlot(cont_dims, np.float32)
        self.slots['cont_flags'] =   accel.IOSlot(cont_dims, np.uint8)

    def _run(self):
        buffer_names = ['vis', 'weights', 'flags', 'cont_vis', 'cont_weights', 'cont_flags']
        buffers = [self.slots[name].buffer for name in buffer_names]
        args = [x.buffer for x in buffers] + [np.int32(buffers[0].padded_shape[1])]
        xblocks = accel.divup(self.baselines, self.template.wgsx)
        yblocks = accel.divup(self.channels, self.template.wgsy * self.template.cont_factor)
        self.command_queue.enqueue_kernel(
            self.template.kernel, args,
            global_size=(xblocks * self.template.wgsx, yblocks * self.template.wgsy),
            local_size=(self.template.wgsx, self.template.wgsy))

    def parameters(self):
        return {
            'cont_factor': self.template.cont_factor,
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
    cont_factor : int
        Number of spectral channels per continuum channel
    """

    flag_names = ['reserved0','static','cam','reserved3','detected_rfi','predicted_rfi','reserved6','reserved7']
    flag_descriptions = [
            'reserved - bit 0',
            'predefined static flag list',
            'flag based on live CAM information',
            'reserved - bit 3',
            'RFI detected in the online system',
            'RFI predicted from space based pollutants',
            'reserved - bit 6','reserved - bit 7']

    def __init__(self, context, flagger, cont_factor):
        self.context = context
        self.prepare = PrepareTemplate(context)
        # TODO: create a zero-fill template that handles all these cases
        self.zero_vis_accum = fill.FillTemplate(
                context, np.complex64, 'float2')
        self.zero_weights_accum = fill.FillTemplate(
                context, np.float32, 'float')
        self.zero_flags_accum = fill.FillTemplate(
                context, np.uint8, 'unsigned char')
        self.transpose_vis = transpose.TransposeTemplate(
                context, np.complex64, 'float2')
        self.flagger = flagger
        self.accum = AccumTemplate(context, 1)
        self.postproc = PostprocTemplate(context, cont_factor)

    def instantiate(self, command_queue, channels, channel_range, baselines):
        return IngestOperation(self, command_queue, channels, channel_range, baselines)

class IngestOperation(accel.OperationSequence):
    """Concrete instance of :class:`IngestTemplate`.

    .. rubric:: Input slots

    **vis_in** : channels × baselines × 2, int32
        Input visibilities from the correlator
    **permutation** : baselines, uint16
        Permutation mapping original to new baseline index

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

    .. rubric:: Scratch slots

    These are subject to change and so are not documented at this time.
    """
    def __init__(self, template, command_queue, channels, channel_range, baselines):
        kept_channels = channel_range[1] - channel_range[0]
        self.template = template
        self.prepare = template.prepare.instantiate(
                command_queue, channels, channel_range, baselines)
        self.zero_spec_vis = template.zero_vis_accum.instantiate(
                command_queue, (kept_channels, baselines))
        self.zero_spec_weights = template.zero_weights_accum.instantiate(
                command_queue, (kept_channels, baselines))
        self.zero_spec_flags = template.zero_flags_accum.instantiate(
                command_queue, (kept_channels, baselines))
        self.zero_spec_flags.set_value(0xff)
        # TODO: a single transpose+absolute value kernel uses less memory
        self.transpose_vis = template.transpose_vis.instantiate(
                command_queue, (baselines, channels))
        self.flagger = template.flagger.instantiate(
                command_queue, channels, baselines)
        self.accum = template.accum.instantiate(
                command_queue, channels, channel_range, baselines)
        self.postproc = template.postproc.instantiate(
                command_queue, kept_channels, baselines)

        # The order of these does not matter, since the actual sequencing is
        # done by methods in this class.
        operations = [
                ('prepare', self.prepare),
                ('zero_spec_vis', self.zero_spec_vis),
                ('zero_spec_weights', self.zero_spec_weights),
                ('zero_spec_flags', self.zero_spec_flags),
                ('transpose_vis', self.transpose_vis),
                ('flagger', self.flagger),
                ('accum', self.accum),
                ('postproc', self.postproc)
        ]
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
                'spec_vis':     ['accum:vis_out0', 'zero_spec_vis:data', 'postproc:vis'],
                'spec_weights': ['accum:weights_out0', 'zero_spec_weights:data', 'postproc:weights'],
                'spec_flags':   ['accum:flags_out0', 'zero_spec_flags:data', 'postproc:flags'],
                'cont_vis':     ['postproc:cont_vis'],
                'cont_weights': ['postproc:cont_weights'],
                'cont_flags':   ['postproc:cont_flags']
        }

        aliases = {
                'scratch1': ['flagger:deviations_t', 'vis_mid', 'flagger:flags']
        }
        # TODO: aliasing of buffers

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

        self.zero_spec_vis()
        self.zero_spec_weights()
        self.zero_spec_flags()

    def end_sum(self, **kwargs):
        """Perform postprocessing for an output dump. This only does
        on-device processing; it does not transfer the results to the host.
        """
        self.postproc()

    def descriptions(self):
        """Generate descriptions of all the components, for the process log.
        Each description is a 3-tuple consisting of a component name, a
        string describing the parameters, and a version string."""
        def generate(operation, name):
            try:
                revision = operation.__class__.__module__.__version__
            except AttributeError:
                revision = 'unknown'
            parameters = dict(operation.parameters())
            parameters['class'] = operation.__class__.__module__ + '.' + operation.__class__.__name__
            yield (
                name,
                ', '.join(['%s=%s' % x for x in sorted(parameters.iteritems())]),
                revision)
            if isinstance(operation, accel.OperationSequence):
                for child_name, child_op in operation.operations.iteritems():
                    for d in generate(child_op, child_name):
                        yield (name + ':' + d[0], d[1], d[2])
        return list(generate(self, 'ingest'))


class VanVleckOutOfRangeError(Exception):
    """Input power is out of range of correction function."""
    pass

class VanVleck(ProcBlock):
    """Perform Van Vleck (quantisation) correction on the incoming data.

    This currently only corrects the autocorrelation values.

    Parameters
    ----------
    accum_per_int : int, optional
        Number of accumulations per dump / integration

    """
    def __init__(self, accum_per_int=390625, *args, **kwargs):
        super(VanVleck, self).__init__(*args, **kwargs)
        self.expected_dtype = np.float32
        self.correct_mean, self.correct_std = create_correction(accum_per_int)

    def _proc(self):
        # Iterate over the auto correlations
        for auto in self.cpref.autos:
            # Since autocorrelations are real, ignore the imaginary part of visibilities
            # Also take half the power to account for complex input data (correct_mean expects real data)
            auto_power = 0.5 * self.current[:, auto, 0]
            # Check whether input power is in the expected range (could indicate that correction was already done)
            if np.any(auto_power < 0.0) or np.any(auto_power > 49.0):
                raise VanVleckOutOfRangeError('Power out of range - bad indexing or Van Vleck correction already done')
            self.current[:, auto, 0] = 2.0 * self.correct_mean(auto_power)
        # Contents of current data are updated in-place
        return None


###################################################################################################################

class InjectedNoiseCal(ProcBlock):
    """Produce weights for gain calibration based on injected noise.
    
    This method expects a block of historical data in which the noise diode is
    on for some integer number of dumps. A single gaincal number is produced for this block.
    The block is later averaged and this number is associated with it.

    Parameters
    ----------
    nd_power : array(float)
        The noise diode temperatures in K for each antenna polarisation.
    int_dumps : int
        The number of dumps that make a single integration.
    nd_on_dumps : int
        The number of consecutive dumps in the integration during which the noise diode was firing.

    """
    def __init__(self, nd_power, int_dumps, nd_on_dumps, *args, **kwargs):
        self.nd_power = nd_power
        self.int_dumps = int_dumps
        self.nd_on_dumps = nd_on_dumps
        super(InjectedNoiseCal, self).__init__(*args, **kwargs)

    def _proc(self):
        """Process a block of dumps to produce a counts per kelvin value for each baseline.

        For each autocorrelation (Nants * 2) product (per antenna, per pol) we know the power in K of the noise diode.
        We locate the dumps during which the noise diode was firing and measure the avg power over this interval.
        Off power is measured either side of the noise diode on interval (if firing is in the first dump then only the trailing dump is selected.)
        This allows us to calculate a counts per K conversion.
        Autocorrelations are assigned the average of the number for the two antenna/pols producing it.

        """
        if self.nd_on_dumps > 1:
            print "Support is currently limited to single dump."
            return

        if self.history.shape[0] != self.int_dumps:
            print "Insufficient history (%i/%i)." % (self.history.shape[0],self.int_dumps)

        weights = np.zeros(self.history.shape[1:],dtype=np.float32)
         # iterate over the auto correlations
        for i,auto in enumerate(self.cpref.autos):
            auto_power = np.abs(self.history[:,:,auto])
            nd_pos = np.argmax(auto_power, axis=0)
             # find the spectral positions of the noise diode
            c_per_k = (auto_power[nd_pos][0] - auto_power[(nd_pos+1) % self.int_dumps][0]) / self.nd_power
            weights[:,auto] = c_per_k

        for cross in [x for x in range(self.history.shape[-1]) if x not in self.cpref.autos]:
            (i1,i2) = self.cpref.get_auto_from_cross(cross)
            weights[:,cross] = np.average([weights[:,i1],weights[:,i2]])

        return weights


class AntennaGains(ProcBlock):
    """Determine antenna-based gain corrections for quality assurance and beamformer weights.

    Parameters
    ----------
    solution_interval : float, optional
        The minimum duration of a solution interval, in seconds
    ref_ant : string, optional
        Name of desired ref_ant, or comma-separated list of names in order of
        preference (default is a KAT-7 specific list provided by Tom Mauch)

    """
    def __init__(self, solution_interval=60.0,
                 ref_ant='ant7,ant6,ant4,ant2,ant5,ant1,ant3', *args, **kwargs):
        super(AntennaGains, self).__init__(*args, **kwargs)
        self.solution_interval = solution_interval
        self.ref_ant = ref_ant.split(',')
        self._reset_state()

    def _reset_state(self, target=None, track_start=0):
        """Reset state of processing block (vis buffer + target info)."""
        self.target = '' if target is None else target
        self.track_start = track_start
        self.vis_buffer = []

    def _proc(self, current_dbe_target, dbe_target_since, current_ant_activities,
              ant_activities_since, script_ants, center_freq, bandwidth, output_sensors):
        """Determine antenna gain corrections based on target and antenna sensors."""
        # Use start of last dump as reference for current data
        last_dump = output_sensors["last-dump-timestamp"].value()
        # Restrict ourselves to script antennas that update their activities
        available_ants = [ant for ant in current_ant_activities]
        if script_ants is not None:
            available_ants = [ant for ant in available_ants if ant in script_ants]
        # Only continue if all script antennas have been tracking for the entire dump (and make sure we have 4 antennas to keep solver happy)
        tracking_ants = [ant for ant in available_ants if current_ant_activities[ant] == 'track' and ant_activities_since[ant] < last_dump]
        required_ants = max(len(available_ants), 4)
        if len(tracking_ants) < required_ants:
            # Reset the state the moment antennas slew off a target
            self.logger.log(logging.INFO if self.track_start > 0 else logging.DEBUG,
                            "AntennaGains: Resetting state because only %d antennas are tracking (need %d)" % (len(tracking_ants), required_ants))
            self._reset_state()
            return
        # Ensure we have a valid target suitable for gain cal (point source with known spectrum)
        target = katpoint.Target(current_dbe_target)
        if ('gaincal' not in target.tags) and ('bpcal' not in target.tags):
            self.logger.debug("AntennaGains: Quitting because target '%s' is not a gain / bandpass calibrator" % (target.description,))
            return
        num_chans = self.current.shape[0]
        channel_width = bandwidth / num_chans
        channel_freqs = center_freq - channel_width * (np.arange(num_chans) - num_chans / 2)
        point_source_spectrum = target.flux_density(channel_freqs / 1e6)
        if np.isnan(point_source_spectrum).any():
            self.logger.debug("AntennaGains: Quitting because target '%s' does not have complete spectral model over range %g - %g MHz - missing freqs: %s" %
                              (target.description, channel_freqs.min() / 1e6, channel_freqs.max() / 1e6, channel_freqs[np.isnan(point_source_spectrum)] / 1e6))
            return
        # Initialise state once we see a new valid target
        if self.target != current_dbe_target:
            self.logger.info("AntennaGains: Initiating data collection on calibrator source '%s' tracked by antennas %s" % (target.description, tracking_ants))
            self._reset_state(current_dbe_target, last_dump)
        # Buffer the visibilities and return if we don't have a full solution interval
        self.vis_buffer.append(self.current)
        if (self.track_start == 0) or (last_dump - self.track_start < self.solution_interval):
            self.logger.debug("AntennaGains: Quitting because solution interval not reached yet (%d seconds so far, need %d)" %
                              (last_dump - self.track_start, self.solution_interval))
            return
        # Pick reference antenna (go for preferred ones first)
        for ant in self.ref_ant:
            try:
                ref_ant_index = tracking_ants.index(ant)
                break
            except ValueError:
                pass
        else:
            ref_ant_index = 0
        self.logger.debug("AntennaGains: Picked '%s' as reference antenna" % (tracking_ants[ref_ant_index],))
        for pol in ['h', 'v']:
            # Restrict ourselves to cross-correlations involving tracking antennas and current polarisation
            select_bl = [(bl[0][:-1] in tracking_ants and bl[0][-1] == pol and bl[0][:-1] != bl[1][:-1] and
                          bl[1][:-1] in tracking_ants and bl[1][-1] == pol) for bl in self.cpref.bls_ordering]
            if not any(select_bl):
                continue
            # Normalise visibilities by source model (simple point source with known spectrum)
            norm_vis = np.dstack(self.vis_buffer).mean(axis=2)[:, np.array(select_bl)] / point_source_spectrum[:, np.newaxis]
            antA = [tracking_ants.index(bl[0][:-1]) for n, bl in enumerate(self.cpref.bls_ordering) if select_bl[n]]
            antB = [tracking_ants.index(bl[1][:-1]) for n, bl in enumerate(self.cpref.bls_ordering) if select_bl[n]]
            # Augment visibilities with complex conjugate values (i.e. add swapped baseline pairs)
            augm_vis, augm_antA, augm_antB = np.hstack([norm_vis, norm_vis.conj()]), np.r_[antA, antB], np.r_[antB, antA]
            # Solve for gains per channel, invert them to get gain corrections and emit sensor values
            self.logger.info("AntennaGains: Solving for %dx%d complex gains" % (num_chans, len(tracking_ants)))
            gains_per_channel = stefcal(augm_vis, len(tracking_ants), augm_antA, augm_antB, ref_ant=ref_ant_index)
            corrections = 1. / gains_per_channel
            for n, ant in enumerate(tracking_ants):
                correct_str = ' '.join([("%5.3f%+5.3fj" % (corrections[chan, n].real, corrections[chan, n].imag)) for chan in range(num_chans)])
                #output_sensors['antenna-gain-corrections'].set_value(ant + pol + ' ' + correct_str)
                sens_name = '{0}{1}-gain-correction-per-channel'.format(ant,pol)
                output_sensors[sens_name].set_value(correct_str)

            self.logger.info("AntennaGains: Updated gain corrections for pol '%s' on target '%s' - average magnitude: %s" %
                             (pol, target.name, ' '.join([('%5.3f' % g) for g in np.abs(corrections).mean(axis=0)])))
        self._reset_state()
