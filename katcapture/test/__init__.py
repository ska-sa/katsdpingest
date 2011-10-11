"""Unit testing for katcapture.

Currently this primarily tests the sigproc library that is part of katcapture. Testing of the capture and
simulator scripts requires some infrastructure to run and is probably best done by a human."""

import unittest
import test_sigproc

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_sigproc))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
