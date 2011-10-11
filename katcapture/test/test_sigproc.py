"""Test for the sigproc module."""

import unittest
import numpy as np
from katcapture import sigproc

class SigProcTestCases(unittest.TestCase):
    """Exercise the sigproc library under a number of varying conditions."""

    def setUp(self):
        """Create the basic environment, along with test vectors."""
        self.scale_factor = 2
        self.n_ts = 10
        self.n_chans = 1024
        self.n_bls = 144
        self.spike_channel = 123
        self.spike_bls = 13
         # the channel and baseline into which to insert rfi spikes
        self.rfi_threshold = 3
         # no of sigma for rfi threshold

        self.data = (np.random.random((self.n_ts, self.n_chans, self.n_bls,2)) * 98).astype(np.int32).astype(np.float32)
         # produce some random data
        self.data_copy = self.data.copy()
        self.data_copy_complex = self.data.copy().view(np.complex64).squeeze()
        #self.data_ones = np.ones((self.n_ts,self.n_chans,self.n_bls), dtype=np.complex64)
        #self.data_short = np.ones((self.n_ts,self.n_chans), dtype=np.complex64)

        sigproc.ProcBlock.history = self.data
        sigproc.ProcBlock.current = self.data[0]

    def testEmptyBlock(self):
        """Test a empty signal processing block."""
        pc = sigproc.ProcBlock()
        self.assertRaises(NotImplementedError, pc.proc)

    def testBlsOrdering(self):
        """Check that CorrProdRef is built correctly."""
        pc = sigproc.ProcBlock(bls_ordering=None)
        self.assertEqual(len(pc.cpref._id_to_real), self.n_bls)

    def testBasicAssignment(self):
        """Test assignment to class variables."""
        pc = sigproc.ProcBlock()
        np.testing.assert_equal(pc.current,pc.history[0])

    def testScaleDataType(self):
        """Test acceptable data type for scale."""
        data_wrong_type = np.ones((self.n_ts,self.n_chans,self.n_bls,2), dtype=np.int32)
        sigproc.ProcBlock.history = data_wrong_type
        sigproc.ProcBlock.current = data_wrong_type[0]
        sc = sigproc.Scale(self.scale_factor)
        self.assertRaises(TypeError, sc.proc)

    def testScale(self):
        """Test basic scaling."""
        sc = sigproc.Scale(self.scale_factor)
        sc.proc()
        np.testing.assert_equal(self.data[0], self.data_copy[0] / self.scale_factor)

    def testVanVleck(self):
        """Not implemented yet..."""
        pass

    def testRFIThresholdDataType(self):
        """Test acceptable data type for rfi threshold."""
        rfi = sigproc.RFIThreshold(self.rfi_threshold)
        self.assertRaises(TypeError, rfi.proc)

    def testRFIThreshold(self):
        """Test simple RFI thresholding..."""
        bls_std = np.std(abs(self.data_copy_complex[1:,self.spike_channel,self.spike_bls]))
        bls_mean = np.mean(abs(self.data_copy_complex[1:,self.spike_channel,self.spike_bls]))
        self.data_spike_large = self.data_copy_complex.copy()
        self.data_spike_large[0,self.spike_channel,self.spike_bls] = bls_mean + (self.rfi_threshold*1.01) * bls_std
        self.data_spike_small = self.data_copy_complex.copy()
        self.data_spike_small[0,self.spike_channel,self.spike_bls] = bls_mean + (self.rfi_threshold*0.99) * bls_std
         # very simple test vectors for basic rfi thresholding

         # test for detection
        sigproc.ProcBlock.history = self.data_spike_large
        sigproc.ProcBlock.current = self.data_spike_large[0]
        rfi = sigproc.RFIThreshold(self.rfi_threshold)
        flags = np.unpackbits(rfi.proc()).reshape(self.n_chans, self.n_bls)
        self.assertEqual(flags[self.spike_channel, self.spike_bls], 1)

         # test for no detection
        sigproc.ProcBlock.history = self.data_spike_small
        sigproc.ProcBlock.current = self.data_spike_small[0]
        flags = np.unpackbits(rfi.proc()).reshape(self.n_chans, self.n_bls)
        self.assertEqual(flags[self.spike_channel, self.spike_bls], 0)


