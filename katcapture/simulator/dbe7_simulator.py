import time
import logging
import sys

from katcp import Sensor, Message
from katcore.dev.base import Device
# XXX TODO
# Evil hack to make kattypes support variable number of
# arguments. Should be ported to katcp when we start working on it
import monkeypatch_kattypes

from katcp.kattypes import request, return_reply
from katcp.kattypes import Str, Int, Float, Discrete, Timestamp

activitylogger = logging.getLogger('activity')
log_name = 'kat.k7simulator'
logger = logging.getLogger(log_name)

# Requests to nix:
#
# capture_destination -> Removed
# cycle_nd -> Should go to sim interface, actually listen to rfe3 simulator?
# fire_nd -> Should go to sim interface, actually listen to rfe3 simulator?
# poco_accumulation_length -> k7_accumulation_length
# poco_gain -> k7_gain
# pointing_az -> Should go to sim interface, interaction with other simulators?
# pointing_el -> Should go to sim interface, interaction with other simulators?
# set_dump_rate -> Removed
# spead_issue -> Removed
# start_tx -> Should go to sim interface
# stop_tx -> Should go to sim interface
# test_target -> Should move to sim interface


# Requests to add:
#
# capture_list -> Done
# define -> leave?
# dict -> leave?
# dispatch -> leave?
# enable_sensors -> leave?
# get -> leave?
# job -> leave?
# k7_accumulation_length -> Done
# k7_adc_snap_shot -> Done
# k7_delay -> Done, stubbily
# k7_frequency_select TODO
# k7_gain -> Done
# k7_snap_shot TODO
# log_record -> leave?
# notice -> leave?
# process -> leave?
# roach -> leave?
# sensor -> leave?
# set -> leave?
# sm -> leave?
# sync_now -> leave?
# system_info -> leave?
# version -> TODO
# version_list -> TODO
# watchannounce -> leave?
# xport -> leave?


class SimulatorDeviceServer(Device):

    VERSION_INFO = ("k7-simulator",0,1)
    BUILD_INFO = ("k7-simulator",0,1,"rc1")

    @request(Str())
    @return_reply(Str())
    def request_mode(self, sock, mode):
        """mode change command (?mode [new-mode])

        Currently a dummy operation in simulator, just pauses to test
        proxy timeouts and changes mode sensor.
        """

        def progress_callback(smsg):
            self.log.info(smsg)
        self._model.set_mode(mode, progress_callback)
        smsg = 'Correlator mode changed to ' + mode
        activitylogger.info(smsg)

        return ('ok', smsg)

    @return_reply(Str())
    def request_start_tx(self, sock, msg):
        """Start the data stream."""
        self._model._thread_paused = False
        smsg = "Data stream started."
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Str())
    @return_reply(Str())
    def request_k7_accumulation_length(self, sock, period):
        """Set the accumulation length. (?k7-accumlation-length accumulation-period)"""
        self._model.set_dump_period(float(period) / 1000.0)
        dump_period = self._model.get_dump_period()
        time.sleep(dump_period)
        smsg = "Set accumulation period to %f s" % dump_period
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Discrete(('pps', 'now')), Int(min=0, max=127), Str(multiple=True),
             include_msg=True)
    @return_reply()
    def request_k7_adc_snap_shot(self, sock, req_msg, when, level, *inputs):
        """retrieve an adc snapshot (?k7-adc-snap-shot [pps|now] threshold input+)

        The simulator returns some dummy values.
        """
        for input, timestamp, values in self._model.get_adc_snap_shot(
                when, level, inputs):
            imsg = Message.inform(req_msg.name, input, '%d' % timestamp, *(
                "%d" % v for v in values))
            self.reply_inform(sock, imsg, req_msg)

        return ('ok', )

    @request(Str(), Timestamp(), Timestamp(), Float(), Float(), Float())
    @return_reply()
    def request_k7_delay(self, sock, board_input, time_, delay,
                         delay_rate, fringe_offset, fringe_rate):
        """set the delay and fringe correction (?k7-delay board-input time delay-value delay-rate fringe-offset fringe-rate)

        Stubby simulator dummy for compatibility
        """
        return('ok', )


    @request(Int(min=0, max=400000000))
    @return_reply(Int())
    def request_k7_frequency_select(self, sock, centre_frequency):
        """select a frequency for fine channelisation (?k7-frequency-select center-frequency)

        Frequency in Hertz. The actual centre frequency will be
        quantized to the closest available frequency bin, and is
        returned (also Hz).

        Has no effect in the simulator

        """
        actual_centre_frequency = self._model.k7_frequency_select(centre_frequency)
        return ('ok', actual_centre_frequency)

    @request(Str(), Str())
    @return_reply(Str())
    def request_k7_gain(self, sock, input, gains):
        """Dummy for compatibility: sets the digital gain (?k7-gain board-input values)."""
        # TODO Check that valid inputs were specified
        # TODO Check format of 'gains'. Talk to DBE team?
        return ("ok","OK")

    @request(Float(optional=True, default=5.0))
    @return_reply(Str())
    def request_fire_nd(self, sock, duration):
        """Insert noise diode spike into output data."""
        self._model.noise_diode = duration
        return ("ok","Fired")

    @request(Float(optional=True, default=0.5))
    @return_reply(Str())
    def request_cycle_nd(self, sock, duty_cycle):
        """Fire the noise diode with the requested duty cycle. Set to 0 to disable."""
        self._model.nd_duty_cycle = duty_cycle
        return("ok","Duty cycle firing enabled")


    @request(Float(),Float(),Float(optional=True,default=20.0))
    @return_reply(Str())
    def request_test_target(self, sock, az, el, flux):
        """Add a test target to the simulator. ?test-target <az> <el> [<flux_scale>]"""
        self._model.set_target_az(az)
        self._model.set_target_el(el)
        self._model.set_target_flux(flux)
        smsg = "Target set to (%f, %f, %f)" % (az, el, flux)
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Float())
    @return_reply(Str())
    def request_pointing_el(self, sock, el):
        """Sets the current simulator elevation pointing."""
        self._model.set_test_el(el)
        smsg = "Pointing elevation set to %f" % el
        logger.info(smsg)
        return ("ok", smsg)

    @request(Float())
    @return_reply(Str())
    def request_pointing_az(self, sock, az):
        """Sets the current simulator azimuth pointing."""
        self._model.set_test_az(az)
        smsg = "Pointing azimuth set to %f" % az
        logger.info(smsg)
        return ("ok", smsg)


    @return_reply(Str())
    def request_label_input(self, sock, msg):
        """Label the specified input with a string."""
        if not msg.arguments:
            for (inp,label) in sorted(self._model.labels.iteritems()):
                self.reply_inform(sock, Message.inform("label-input", label, inp), msg)
            return ("ok",str(len(self._model.labels)))
        else:
            inp = msg.arguments[0]
            label = msg.arguments[1]

        try:
            self._model.set_antenna_mapping(inp, label)
            return ('ok', label)
        except ValueError:
            return ("fail","Unknown input %s. Should be one of %s." % (
                inp, ', '.join(self._model.config.get_unmapped_channel_names())) )


    @request(Str(), Float(optional=True))
    @return_reply(Str())
    def request_capture_start(self, sock, destination, time_):
        """Start a capture (?capture-start k7 [time]). Mostly a dummy, does a spead_issue."""
        self._model.spead_issue()
        smsg = "SPEAD meta packets sent to %s" % (self._model.config['rx_meta_ip_str'])
        activitylogger.info("k7simulator: %s" % smsg)
        return ("ok",smsg)

    @request(Str(optional=True))
    @return_reply(Str())
    def request_capture_stop(self, sock, destination):
        """For compatibility with dbe_proxy. Does nothing :)."""
        self._model.send_stop()
        smsg = "Capture stopped. (dummy)"
        activitylogger.info(smsg)
        return ("ok", smsg)

    @return_reply()
    def request_capture_list(self, sock, req_msg):
        """list available data streams (?capture-list)"""
        smsg = 'k7 %s %d' % (self._model.config['rx_meta_ip'],
                             self._model.config['rx_udp_port'])
        self.reply_inform(
            sock, Message.inform(req_msg.name, smsg), req_msg)
        return ("ok", )


    @return_reply(Str())
    def request_stop_tx(self, sock, msg):
        """Stop the data stream."""
        self._model._thread_paused = True
        self._model.send_stop()
        smsg = "Data stream stopped."
        activitylogger.info(smsg)
        return ("ok", smsg)

