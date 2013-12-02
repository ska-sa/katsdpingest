import logging
import inspect
import time

import numpy as np
import scipy.signal as signal

import katpoint
from katsdpdisp import CorrProdRef
from .vanvleck import create_correction
from .antsol import stefcal
from .__init__ import __version__ as revision


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
    def __init__(self, axis=0, n_sigma=11.0, spike_width=6, *args, **kwargs):
        super(RFIThreshold2, self).__init__(*args, **kwargs)
        self.n_sigma = n_sigma
        self.spike_width = spike_width
        self.axis = axis
        self.expected_dtype = np.complex64

    def _proc(self):
        self.logger.debug("RFI Threshold: Processing block of shape %s" % str(self.current.shape))

        if self.flags is None:
            self.init_flags()

        for bl_index in range(self.current.shape[1]):
            spectral_data = np.abs(self.current[:,bl_index])
            spectral_data = np.atleast_1d(spectral_data)
            kernel_size = 2 * max(int(self.spike_width), 0) + 1
            # Median filter data along the desired axis, with given kernel size
            kernel = np.ones(spectral_data.ndim, dtype='int32')
            kernel[self.axis] = kernel_size

            # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
            filtered_data = np.asarray(signal.medfilt(spectral_data, kernel), spectral_data.dtype)

            # The deviation is measured relative to the local median in the signal
            abs_dev = np.abs(spectral_data - filtered_data)

            # Calculate median absolute deviation (MAD)
            med_abs_dev = np.expand_dims(np.median(abs_dev[abs_dev>0], self.axis), self.axis)

            #med_abs_dev = signal.medfilt(abs_dev, kernel)
            # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
            estm_stdev = 1.4826 * med_abs_dev

            # Identify outliers (again based on normal assumption), and replace them with local median
            #outliers = ( abs_dev > self.n_sigma * estm_stdev)
            #print outliers
            # Identify only positve outliers
            outliers = (spectral_data - filtered_data > self.n_sigma*estm_stdev)

            self.flags[:,bl_index,4] = outliers
             # set appropriate flag bit for detected RFI


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
