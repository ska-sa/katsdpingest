"""Test for the sigproc module."""

import unittest
import numpy as np
from katcapture import sigproc

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
