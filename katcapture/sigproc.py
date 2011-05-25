import time
import numpy as np

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
     # class attributes for storing references to numpy arrays contaning current and historical data
     # current is purely for convenience as current == history[0]
    _proc_times = []
    def __init__(self):
        pass

    def proc(self, *args, **kwargs):
        """Process a single time instance of data."""
        if self.current is None:
            print "Warning: Data references have not been set up. This is almost certainly in error. The data exists as an attribute on the ProcBlock base class."
        st = time.time()
        retval = self._proc(*args, **kwargs)
        self._proc_times.append(time.time() - st)
        return retval

    def _proc(self):
        raise NotImplementedError("Method must be overridden by subclasses.")

    def stats(self):
        """Accumulated processing statistics of this block."""
        if len(self._proc_times) > 0:
            print "Processed %i frame(s) in %.3fs. Last: %.3fs, Avg: %.3fs\n" % (len(self._proc_times), np.sum(self._proc_times), self._proc_times[-1], np.average(self._proc_times))
        else:
            print "No frames processed."

class VanVleck(ProcBlock):
    """Perform van vleck corrections on the incoming data."""
    pass

class Scale(ProcBlock):
    """Trivial block to perform data scaling and type conversion.

    Parameters
    ==========
    scale_factor : float
        The scale factor to use.
    """
    def __init__(self, scale_factor, *args, **kwargs):
        self.scale_factor = scale_factor
        super(Scale, self).__init__(*args, **kwargs)

    def _proc(self):
        self.current[:] = (np.float32(self.current) / (1.0 * self.scale_factor))[:]
         # avoid making a new current object. Just replace contents.
        return None

class RFIThreshold(ProcBlock):
    """Simple RFI flagging through thresholding.
    
    Trivial thresholder that looks for n sigma deviations from the average
    of the supplied frame.
    
    Parameters
    ==========
    n_sigma : int
       The number of std deviations allowed
    """
    def __init__(self, n_sigma, *args, **kwargs):
        self.n_sigma = n_sigma
        super(RFIThreshold, self).__init__(*args, **kwargs)

    def _proc(self, use_history=False):
        print "\nRFI Threshold: Processing block of shape",self.current.shape,"and history of length",self.history.shape[0],"\n"

        #### Replace the code in this method with your own.
        #### You need to produce a flags array that has shape (Nchannels, Nbaselines)
        #### State can be stored in the class

        if use_history:
         # in this case we want to use available historical data rather than just the current frame.
         # self.history has shape (Ntimestamps, Nchannels, Nbaselines)
            m = np.mean(self.history, axis=0)
            s = np.std(self.history, axis=0)
             # these array now have shape (Nchannels, Nbaselines)
        else:
            m = np.mean(self.current, axis=0)
            s = np.std(self.current, axis=0)
             # these arrays have shape (Nbaselines). i.e. a single point per spectrum
        flags = self.current >= (m + self.n_sigma * s)

        ### End of section to replace

        return np.packbits(flags.astype(np.int8))

class GainCal(ProcBlock):
    """Produce weights for gain calibration."""
    pass





