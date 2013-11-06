"""Unit testing for katsdpingest.

Currently this primarily tests the sigproc library that is part of katsdpingest. Testing of the capture and
simulator scripts requires some infrastructure to run and is probably best done by a human."""

import unittest

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_sigproc))
    testsuite.addTests(loader.loadTestsFromModule(test_telescope_model))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
