import time
import numpy as np
from katsdisp.data import CorrProdRef
import scipy.signal as signal
import logging
from .vanvleck import create_correction


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

    *** Preliminary ***

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

    *** End Preliminary ***

    """
    current = None
    history = None
    flags = None
    logger = logging.getLogger("katcapture.sigproc")
     # class attributes for storing references to numpy arrays contaning current and historical data
     # current is purely for convenience as current == history[0]
    _proc_times = []
    def __init__(self, **kwargs):
        self.cpref = CorrProdRef(**kwargs)
        self.flag_names = ['reserved0','static','cam','reserved3','detected_rfi','predicted_rfi','reserved6','reserved7']
        self.flag_descriptions = ['reserved - bit 0','predefined static flag list','flag based on live CAM information',
                                  'reserved - bit 3','RFI detected in the online system','RFI predicted from space based pollutants',
                                  'reserved - bit 6','reserved - bit 7']
        self.expected_dtype = None

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
        descr.append("Parameters: " + ", ".join(["%s == %s" % (k,v) for k,v in self.__dict__.iteritems() if not k.startswith("_")]))
        if len(self._proc_times) > 0:
            descr.append("Processed %i frame(s) in %.3fs. Last: %.3fs, Avg: %.3fs" % (len(self._proc_times), np.sum(self._proc_times), self._proc_times[-1], np.average(self._proc_times)))
        else:
            descr.append("No frames processed.")
        if self.current is not None:
            descr.append("Current data has shape %s and type %s" % (self.current.shape, self.current.dtype))
        else:
            descr.append("No current data.")
        return "\n".join(descr)


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
    def __init__(self, axis=0, n_sigma=11.0, spike_width=3, *args, **kwargs):
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

class GainCal(ProcBlock):
    """Produce weights for gain calibration.
    
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
        super(GainCal, self).__init__(*args, **kwargs)

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
