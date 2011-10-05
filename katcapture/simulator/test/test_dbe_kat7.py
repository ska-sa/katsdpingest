"""Tests for the dbe simulator module."""

import unittest
from katcore.sim import dbe_kat7
from katcore.testutils import SimulatorTestMixin
from katcp import Message

EXPECTED_SENSOR_LIST = [
    ('corr.lru.available', 'line replacement unit operational', '', 'boolean'),
    ('ntp.synchronised', 'clock good', '', 'boolean'),
    ('roach030204.0x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '65000'),
    ('roach030204.0x.adc.overrange', 'overrange indicator', '', 'boolean'),
    ('roach030204.0x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030204.0x.fft.overrange', 'overrange indicator', '', 'boolean'),
    ('roach030204.0y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '65000'),
    ('roach030204.0y.adc.overrange', 'overrange indicator', '', 'boolean'),
    ('roach030204.0y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030204.0y.fft.overrange', 'overrange indicator', '', 'boolean'),
    ('roach030205.1x.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '65000'),
    ('roach030205.1x.adc.overrange', 'overrange indicator', '', 'boolean'),
    ('roach030205.1x.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030205.1x.fft.overrange', 'overrange indicator', '', 'boolean'),
    ('roach030205.1y.adc.amplitude', 'approximate input signal strength', '', 'float', '0', '65000'),
    ('roach030205.1y.adc.overrange', 'overrange indicator', '', 'boolean'),
    ('roach030205.1y.adc.terminated', 'adc disabled', '', 'boolean'),
    ('roach030205.1y.fft.overrange', 'overrange indicator', '', 'boolean'),
]

EXPECTED_REQUEST_LIST = [
    ("capture-list", "List the available streams."),
    ("capture-start", "Instruct the named stream to emit data, optionally starting at a given time."),
    ("capture-stop", "Instruct the named stream to stop emitting data, optionally at a given time."),
    ("capture-sync", "Re-emit the header packets for an active stream."),
    ("label-input", "Label an ADC board input or retrieve the current label for an input."),
    ("k7-accumulation-length", "Set the KAT7 correlator accumulation period."),
    ("k7-delay", "Set the KAT7 correlator delay parameters."),
    ("k7-gain", "Set the KAT7 correlator gain factors."),
    ("k7-snap-shot", "Return a snap shot of the data from a DBE snap block or other source."),
    ("delay-list", "Return the current delay parameters for each input."),
    ("gain-list", "Return the current gain parameters for each input."),
]


class TestDbeKat7(unittest.TestCase, SimulatorTestMixin):
    def setUp(self):
        self.set_up_device_and_client(dbe_kat7.DbeKat7Device, dbe_kat7.DbeKat7Model, dbe_kat7.DbeKat7TestDevice)

    def tearDown(self):
        self.tear_down_threads()

    def test_sensor_list(self):
        self.client.test_sensor_list(EXPECTED_SENSOR_LIST)

    def test_help(self):
        self.client.test_help(EXPECTED_REQUEST_LIST)

    def test_capture(self):
        reply, informs = self.client.blocking_request(Message.request("capture-list"))
        self.assertEqual(reply, Message.reply("capture-list", "ok", 1))
        self.assertEqual(informs, [Message.inform("capture-list", "k7", "127.0.0.1", 7148)])

        self.client.assert_request_succeeds("capture-start", "k7")
        self.client.assert_request_succeeds("capture-sync", "k7")
        self.client.assert_request_succeeds("capture-stop", "k7")

    def test_label_input(self):
        self.client.assert_request_fails("label-input", "0z", status_equals="fail")
        self.client.assert_request_fails("label-input", "0z", "ant1H", status_equals="fail")
        self.client.assert_request_succeeds("label-input", "0x", args_equal=["0x"])
        self.client.assert_request_succeeds("label-input", "0x", "ant1H", args_equal=["ant1H"])
        self.client.assert_request_succeeds("label-input", "0x", args_equal=["ant1H"])

    def test_k7_accumulation_length(self):
        self.client.assert_request_succeeds("k7-accumulation-length", 1.0)

    def test_k7_snap_shot(self):
        self.client.assert_request_succeeds("k7-snap-shot", "adc", "0x")
        self.client.assert_request_succeeds("k7-snap-shot", "quant", "1y")

    def test_delay_setting(self):
        self.client.assert_request_succeeds("k7-delay", "0x", 12345.0, 0.0, 1.0, 2.0, 3.0)
        reply, informs = self.client.blocking_request(Message.request("delay-list"))
        self.assertEqual(reply, Message.reply("delay-list", "ok", 1))
        self.assertEqual(informs, [Message.inform("delay-list", 0, "x", 0.0, 1.0, 2.0, 3.0)])

    def test_gain_setting(self):
        self.client.assert_request_succeeds("k7-gain", "0x", *(["2000"] * 1024))
        reply, informs = self.client.blocking_request(Message.request("gain-list"))
        self.assertEqual(reply, Message.reply("gain-list", "ok", 1))
        self.assertEqual(informs, [Message.inform("gain-list", 0, "x", *(["2000"] * 1024))])

    def test_complex_gain_setting(self):
        self.client.assert_request_succeeds("k7-gain", "0x", *(["2000+1000j"] * 1024))
        reply, informs = self.client.blocking_request(Message.request("gain-list"))
        self.assertEqual(reply, Message.reply("gain-list", "ok", 1))
        self.assertEqual(informs, [Message.inform("gain-list", 0, "x", *(["2000+1000j"] * 1024))])
