"""Tests for the dbe simulator module."""

import unittest2 as unittest

import os
import katcapture

from katcore.sim import dbe_kat7
from katcore.testutils import SimulatorTestMixin
from katcp import Message

from katcapture.simulator import dbe7_simulator
from katcapture.simulator import dbe7_simulator_model

EXPECTED_SENSOR_LIST = [
    ('bandwidth', 'The bandwidth currently available', 'Hz', 'integer', '0', '400000000'),
    ('channels', 'the number of frequency channels that the correlator provides when in the current mode.', 'Hz', 'integer', '0', '10000'),
    ('corr.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('ntp.synchronised', 'clock good', '', 'boolean'),
    ('roach030279.1x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030203.3y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030276.4y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030266.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030267.6x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030276.4x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030268.0x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030277.7y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030269.5x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030267.6x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030277.7y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030268.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('destination_ip', 'The current destination address for data and metadata.', '', 'string'),
    ('roach030265.2x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030268.0y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030203.3x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030265.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030269.5y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030267.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('roach030277.7x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030265.2y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030203.3x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030268.0x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030203.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030273.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030265.2y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030269.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('roach030269.5y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030279.1y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030274.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030277.7y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030203.3x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030268.0y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030277.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('roach030263.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030265.2x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('mode', 'Current DBE operating mode', '', 'string'),
    ('roach030267.6x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030277.7x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('sync_time', 'Last sync time in epoch seconds.', 'seconds', 'integer', '0', '4294967296'),
    ('roach030265.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('roach030268.0y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030203.3x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030279.1x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030271.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030268.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030275.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030196.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030265.2y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030276.4y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030267.6x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030279.1y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030277.7x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030267.6y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030279.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030268.0y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030279.1y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030269.5y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030279.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('roach030203.3x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030269.5x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030268.0y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030279.1x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030276.4y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030276.4y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030267.6y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('tone_freq', 'The frequency of the injected tone in Hz.', '', 'integer', '0', '4294967296'),
    ('roach030277.7x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030269.5y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030269.5x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030276.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('centerfrequency', 'current selected center frequency', 'Hz', 'integer', '0', '400000000'),
    ('roach030265.2y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030267.6x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030268.0x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030267.6y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030265.2x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030279.1y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030265.2x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030276.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('roach030277.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030265.2x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030270.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030277.7y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030267.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030265.2y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030268.0x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030276.4x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030267.6y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030279.1x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030276.4x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030276.4y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030267.6y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030276.4x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030203.3y.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030279.1x.adc.overrange', 'adc overrange indicator', '', 'boolean'),
    ('roach030277.7y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030203.3y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030268.0x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '1'),
    ('roach030276.4x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030203.3y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030279.1y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030269.5x.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),
    ('roach030269.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('roach030269.5x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030203.fpga.synchronised', 'signal processing clock stable', '', 'boolean'),
    ('roach030203.3y.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030277.7x.fft.overrange', 'fft overrange indicator', '', 'boolean'),
    ('roach030269.5y.adc.power', 'approximate input signal strength', '', 'float', '-1e+09', '0'),

]

EXPECTED_REQUEST_LIST = [
    ('capture-list', 'list available data streams (?capture-list)'),
    ('capture-start', 'Start a capture (?capture-start k7 [time]). Mostly a dummy, does a spead_issue.'),
    ('capture-start', 'Start a capture (?capture-start k7 [time]). Mostly a dummy, does a spead_issue.'),
    ('capture-stop', 'For compatibility with dbe_proxy. Does nothing :).'),
    ('cycle-nd', 'Fire the noise diode with the requested duty cycle. Set to 0 to disable.'),
    ('fire-nd', 'Insert noise diode spike into output data.'),
    ('k7-accumulation-length', 'Set the accumulation length. (?k7-accumlation-length accumulation-period)'),
    ('k7-adc-snap-shot', 'retrieve an adc snapshot (?k7-adc-snap-shot [pps|now] threshold input+)'),
    ('k7-delay', 'set the delay and fringe correction (?k7-delay board-input time delay-value delay-rate fringe-offset fringe-rate)'),
    ('k7-frequency-select', 'select a frequency for fine channelisation (?k7-frequency-select center-frequency)'),
    ('k7-gain', 'Dummy for compatibility: sets the digital gain (?k7-gain board-input values).'),
    ('label-input', 'Label the specified input with a string.'),
    ('mode', 'mode change command (?mode [new-mode])'),
    ('pointing-az', 'Sets the current simulator azimuth pointing.'),
    ('pointing-el', 'Sets the current simulator elevation pointing.'),
    ('start-tx', 'Start the data stream.'),
    ('stop-tx', 'Stop the data stream.'),
    ('test-target', 'Add a test target to the simulator. ?test-target <az> <el> [<flux_scale>]'),
# TODO    ('k7-snap-shot', 'Return a snap shot of the data from a DBE snap block or other source.'),
]


class TestDbeKat7(unittest.TestCase, SimulatorTestMixin):
    def setUp(self):
        corr_confdir = os.path.join(os.path.dirname(katcapture.__file__), 'conf')
        self.addCleanup(self.tear_down_threads)
        self.proxy = self.add_device(
            dbe7_simulator.DBE7DeviceServer,
            model=dbe7_simulator_model.K7CorrelatorModel(corr_confdir))
        self.client = self.add_client(self.proxy._sock.getsockname())

    def test_sensor_list(self):
        self.client.test_sensor_list(EXPECTED_SENSOR_LIST)

    def test_help(self):
        self.client.test_help(EXPECTED_REQUEST_LIST)

    def test_capture(self):
        reply, informs = self.client.blocking_request(Message.request("capture-list"))
        self.assertEqual(reply, Message.reply("capture-list", "ok", 1))
        self.assertEqual(informs, [Message.inform("capture-list", "k7", "127.0.0.1", 7148)])

        self.client.assert_request_succeeds("capture-start", "k7")
        self.client.assert_request_succeeds("capture-stop", "k7")

    def test_label_input(self):
        self.client.assert_request_fails("label-input", "0z", status_equals="fail")
        self.client.assert_request_fails("label-input", "0z", "ant1H", status_equals="fail")
        self.client.assert_request_succeeds("label-input", "0x", args_equal=["0x"])
        self.client.assert_request_succeeds("label-input", "0x", "ant1H", args_equal=["ant1H"])
        self.client.assert_request_succeeds("label-input", "0x", args_equal=["ant1H"])

    def test_k7_accumulation_length(self):
        self.client.assert_request_succeeds("k7-accumulation-length", 1.0)

    @unittest.skip
    # Request not currently implemented
    def test_k7_snap_shot(self):
        self.client.assert_request_succeeds("k7-snap-shot", "adc", "0x")
        self.client.assert_request_succeeds("k7-snap-shot", "quant", "1y")

    def test_gain_setting(self):
        self.client.assert_request_succeeds("k7-gain", "0x", *(["2000"] * 1024))

    def test_complex_gain_setting(self):
        self.client.assert_request_succeeds("k7-gain", "0x", *(["2000+1000j"] * 1024))


