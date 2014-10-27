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
from katsdpsigproc import accel

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

    A flag array is used to refer to values within the currently linked data that should be flagged. The flag array is eventually finalised
    into a packed int8 array hence the requirement for the flag column to be a multiple of 8. Currently the following bits are specified:
        0 - reserved
        1 - predefined static flag list
        2 - flag based on live CAM information
        3 - reserved
        4 - RFI detected in the online system
        5 - RFI predicted from space based pollutants
        6 - reserved
        7 - reserved

    NOTE: Currently np.packbits is used to pack the boolean flag array into an integer value. This means that the flags
          listed above are in MSB *first* order. e.g.. setting flag 4 (flag[:,:,4]=1) will toggle the 4th bit and produce an
          integer value of 8 (00001000) once packed.
    """
    current = None
    history = None
    flags = None
    logger = logging.getLogger("kat.k7capture.sigproc")
     # class attributes for storing references to numpy arrays contaning current and historical data
     # current is purely for convenience as current == history[0]
    _proc_times = []
    def __init__(self, **kwargs):
        self.cpref = CorrProdRef(**kwargs)
        self.expected_dtype = None
        self.flag_names = ['reserved0','static','cam','reserved3','detected_rfi','predicted_rfi','reserved6','reserved7']
        self.flag_descriptions = ['reserved - bit 0','predefined static flag list','flag based on live CAM information',
                                  'reserved - bit 3','RFI detected in the online system','RFI predicted from space based pollutants',
                                  'reserved - bit 6','reserved - bit 7']
        self._extracted_arguments = self._extract_args(inspect.currentframe(1))
         # extract the arguments from the parent __init__

    def _extract_args(self, frame):
        args, _, _, values = inspect.getargvalues(frame)
        return ", ".join(["%s=%s" % (k,values[k]) for k in args if k is not 'self'])

    def finalise_flags(self):
        """Packs flags into optimal structure and returns them to caller."""
        if self.flags is None:
            self.logger.error("No flags yet defined. Using init_flags to setup the basic flag table.")
        return np.packbits(self.flags).reshape(self.current.shape)

    def init_flags(self, flag_size=8):
        """Initialise a flag array based on the shape of current.

        Typically this is called after updating the current reference to the current data.
        Flags are stored in the final axis of the array (with size as given in flag_size).
        Flag size must be a multiple of 8 to allow packbits to work unambiguously.

        """
        if flag_size % 8 != 0:
            self.logger.error("given flag_size (%i) is not a multiple of 8." % flag_size)
        if self.current is None:
            self.logger.error("No current data. Unable to create flag table until ProcBlock.current has been initialised.")
        self.flags = np.zeros(list(self.current.shape) + [flag_size], dtype=np.int8)

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
    tune : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - block: number of workitems per workgroup in each dimension
        - vtx, vty: number of elements handled by each workitem, per dimension
    """
    def __init__(self, context, tune=None):
        if tune is None:
            tune = self.autotune(context)
        self.block = tune['block']
        self.vtx = tune['vtx']
        self.vty = tune['vty']
        program = accel.build(context, 'ingest_kernels/prepare.mako',
                {'block': self.block, 'vtx': self.vtx, 'vty': self.vty},
                extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('prepare')

    def autotune(self, context):
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
    **weigths** : baselines × kept-channels, float32
        Weights corresponding to visibilities

    Parameters
    ----------
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
        self.slots['vis_in'] = accel.IOSlot(
                (channels, baselines, 2), np.int32,
                (tiley, tilex, 1))
        # For output we do not need to pad the baselines, because the
        # permutation requires that we do range checks anyway.
        self.slots['vis_out'] = accel.IOSlot(
                (baselines, channels), np.complex64,
                (1, tiley))
        # Channels need to be range-checked anywhere here, so no padding
        self.slots['weights'] = accel.IOSlot(
                (baselines, channel_range[1] - channel_range[0]), np.float32)
        self.slots['permutation'] = accel.IOSlot((baselines,), np.uint16)

    def __call__(self, scale, **kwargs):
        self.bind(**kwargs)
        self.ensure_all_bound()

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
                    np.float32(scale)
                ],
                global_size = (xblocks * block, yblocks * block),
                local_size = (block, block))

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
    tune : mapping, optional
        Kernel tuning parameters; if omitted, will autotune. The possible
        parameters are

        - block: number of workitems per workgroup in each dimension
        - vtx, vty: number of elements handled by each workitem, per dimension
    """
    def __init__(self, context, outputs, tune=None):
        if tune is None:
            tune = self.autotune(context)
        self.block = tune['block']
        self.vtx = tune['vtx']
        self.vty = tune['vty']
        self.outputs = outputs
        program = accel.build(context, 'ingest_kernels/accum.mako',
            {
                'block': self.block,
                'vtx': self.vtx,
                'vty': self.vty,
                'outputs': self.outputs},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('accum')

    def autotune(self, context):
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
        self.slots['vis_in'] = accel.IOSlot(
                (baselines, channels), np.complex64,
                (tiley, tilex))
        self.slots['weights_in'] = accel.IOSlot(
                (baselines, kept_channels), np.float32,
                (tiley, tilex))
        self.slots['flags_in'] = accel.IOSlot(
                (baselines, channels), np.uint8,
                (tiley, tilex))
        for i in range(self.template.outputs):
            label = str(i)
            self.slots['vis_out' + label] = accel.IOSlot(
                (kept_channels, baselines), np.complex64,
                (tilex, tiley))
            self.slots['weights_out' + label] = accel.IOSlot(
                (kept_channels, baselines), np.float32,
                (tilex, tiley))
            self.slots['flags_out' + label] = accel.IOSlot(
                (kept_channels, baselines), np.uint8,
                (tilex, tiley))

    def __call__(self, **kwargs):
        self.bind(**kwargs)
        self.ensure_all_bound()

        buffer_names = []
        for i in range(self.template.outputs):
            label = str(i)
            buffer_names.extend(['vis_out' + label, 'weights_out' + label, 'flags_out' + label])
        buffer_names.extend(['vis_in', 'weights_in', 'flags_in'])
        buffers = [self.slots[x].buffer for x in buffer_names]
        # Arguments are structured as a list of buffers followed by a list of
        # their strides in the same order.
        args = [x.buffer for x in buffers] + \
               [np.int32(x.padded_shape[1]) for x in buffers] + \
               [np.int32(self.channel_range[0])]

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


class IngestTemplate(object):
    """Template for the entire on-device ingest processing"""

    def __init__(self, context, flagger):
        self.context = context
        self.prepare = PrepareTemplate(context)
        self.zero_vis_accum = accel.fill.FillTemplate(
                context, np.complex64, 'float2')
        self.transpose_float32 = accel.transpose.TransposeTemplate(
                context, np.float32, 'float')
        self.transpose_complex64 = accel.transpose.TransposeTemplate(
                context, np.complex64, 'float2')
        self.flagger = flagger

    def instantiate(self, command_queue, channels, channel_range, baselines):
        return IngestOperation(self, command_queue, channels, channel_range, baselines)

class IngestOperation(accel.OperationSequence):
    def __init__(self, template, command_queue, channels, channel_range, baselines):
        keep_channels = channel_range[1] - channel_range[0]
        self.zero_vis_accum = self.template.fill.instantiate(
                command_queue, (baselines, keep_channels))
        self.prepare = self.template.prepare.instantiate(
                command_queue, channels, channel_range, baselines)
        self.transpose_vis_amp = self.template.transpose_float32.instantiate(
                command_queue, (baselines, channels))
        self.transpose_vis_accum = self.template.transpose_complex64.instantiate(
                command_queue, (baselines, keep_channels))
        self.flagger = self.template.flagger.instantiate(
                command_queue, channels, baselines)

        operations = [
                ('prepare', self.prepare),
                ('zero_vis_accum', self.zero_vis_accum),
                ('transpose_vis_amp', self.transpose_vis_amp),
                ('flagger', self.flagger)]
        compounds = {
                'vis_in':      ['prepare:vis_in'],
                'vis_accum_t': ['prepare:vis_accum_t', 'zero_vis_accum:data', 'transpose_vis_accum:src'],
                'vis_accum':   ['transpose_vis_accum:dest'],
                'vis_amp_t':   ['prepare:vis_amp_t', 'transpose_vis_amp:src'],
                'vis_amp':     ['transpose_vis_amp:dest', 'flagger:vis'],
                'flags':       ['flagger:flags']}
        super(IngestOperation, self).__init__(self, command_queue, operations, compounds)

class Scale(ProcBlock):
    """Trivial block to perform data scaling and type conversion.

    Parameters
    ----------
    scale_factor : float
        The scale factor to use.

    """
    def __init__(self, scale_factor, *args, **kwargs):
        super(Scale, self).__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.expected_dtype = np.int32

    def _proc(self):
        self.current.dtype = np.float32
        self.current[:] = (np.float32(self.current.view(np.int32)) / (1.0 * self.scale_factor))[:]
         # avoid making a new current object. Just replace contents.
        return None


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

class RFIThreshold2(ProcBlock):
    """Simple RFI flagging through thresholding.

    Trivial thresholder that looks for n sigma deviations from the average
    of the supplied frame.

    Parameters
    ----------
    n_sigma : float
       The number of std deviations allowed

    """
    def __init__(self, n_sigma=11.0, spike_width=6, *args, **kwargs):
        super(RFIThreshold2, self).__init__(*args, **kwargs)
        width = 2 * spike_width + 1    # katsdpsigproc takes the median filter width
        if context:
            background = katsdpsigproc.rfi.device.BackgroundMedianFilterDeviceTemplate(context, width)
            noise_est = katsdpsigproc.rfi.device.NoiseEstMADTDeviceTemplate(context, 10240)
            threshold = katsdpsigproc.rfi.device.ThresholdSimpleDeviceTemplate(context, n_sigma, True)
            self.flagger = katsdpsigproc.rfi.device.FlaggerHostFromDevice(
                    katsdpsigproc.rfi.device.FlaggerDeviceTemplate(background, noise_est, threshold),
                    context.create_command_queue())
        else:
            self.logger.debug("RFI Threshold: CUDA/OpenCL not found, falling back to CPU")
            background = katsdpsigproc.rfi.host.BackgroundMedianFilterHost(width)
            noise_est = katsdpsigproc.rfi.host.NoiseEstMADHost()
            threshold = katsdpsigproc.rfi.host.ThresholdMADHost(n_sigma)
            self.flagger = katsdpsigproc.rfi.host.FlaggerHost(background, noise_est, threshold)
        self.expected_dtype = np.complex64

    def _proc(self):
        self.logger.debug("RFI Threshold: Processing block of shape %s" % str(self.current.shape))

        if self.flags is None:
            self.init_flags()

        flags = self.flagger(self.current)
        self.flags[..., 4] = flags


class RFIThreshold(ProcBlock):
    """Simple RFI flagging through thresholding.

    Trivial thresholder that looks for n sigma deviations from the average
    of the supplied frame.

    Parameters
    ----------
    n_sigma : float
       The number of std deviations allowed

    """
    def __init__(self,axis=0, *args, **kwargs):
        super(RFIThreshold, self).__init__(*args, **kwargs)
        self.axis = axis

    def _proc(self):

        flags = np.zeros(self.current.shape)
        for bl_index in range(self.current.shape[1]):
            spectral_data = np.abs(self.current[:,bl_index])
            spectral_data = np.atleast_1d(spectral_data)
            kernel = np.ones(spectral_data.ndim, dtype='int32')
            kernel[self.axis] = 3
            # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
            filtered_data = signal.medfilt(spectral_data, kernel)
            # The deviation is measured relative to the local median in the signal
            abs_dev = spectral_data - filtered_data
            # Identify outliers (again based on normal assumption), and replace them with local median
            outliers = (abs_dev > np.std(abs_dev)*2.3)
            flags[:,bl_index] = outliers

        return np.packbits(flags.astype(np.int8))

######################################################################################################
    #if self.history.shape[0] > 1:
         # in this case we want to use available historical data rather than just the current frame.
         # self.history has shape (Ntimestamps, Nchannels, Nbaselines)
        #    m = np.mean(np.abs(self.history[1:]), axis=0)
         #   s = np.std(np.abs(self.history[1:]), axis=0)
             # these array now have shape (Nchannels, Nbaselines)
    #    else:
    #        m = np.mean(np.abs(self.current), axis=0)
    #        s = np.std(np.abs(self.current), axis=0)
             # these arrays have shape (Nbaselines). i.e. a single point per spectrum
    #    flags = np.abs(self.current) >= (m + self.n_sigma * s)

        ### End of section to replace

#        return np.packbits(flags.astype(np.int8))

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
