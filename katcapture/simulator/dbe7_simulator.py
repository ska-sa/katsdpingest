import time
import logging
import sys

from katcp import Sensor, Message
from katcore.dev.base import Device
from katcp.kattypes import request, return_reply, Str, Int, Float

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
# capture_list
# define
# dict
# dispatch
# enable_sensors
# get
# job
# k7_accumulation_length -> Done
# k7_adc_snap_shot
# k7_delay
# k7_frequency_select
# k7_gain -> Done
# k7_snap_shot
# log_record
# notice
# process
# roach
# sensor
# set
# sm
# sync_now
# system_info
# version
# version_list
# watchannounce
# xport



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
        valid_modes = ('nbc', 'wbc')
        mode_delay = 10         # Mode change delay in seconds
        if mode not in valid_modes:
            smsg = 'Invalid correlator mode selected. Valid modes are: %s' %(
                ','.join(valid_modes))
            activitylogger.error(smsg)
            return ('fail', smsg)
        logger.info('Sleeping %f s for dbe7 mode change' % mode_delay)
        for i in range(mode_delay):
            time.sleep(1)
            smsg = 'Doing some mode changing stuff on DBE %d' % (i+1)
            self.log.info(smsg)
        self.get_sensor('mode').set_value(mode, Sensor.NOMINAL, time.time())
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
        self._model.dump_period = float(period) / 1000.0
        time.sleep(self._model.dump_period)
        smsg = "Set accumulation period to %f s" % self._model.dump_period
        activitylogger.info(smsg)
        return ("ok", smsg)

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

    @request(Str(), Str())
    @return_reply(Str())
    def request_k7_gain(self, sock, msg1, msg2):
        """Dummy for compatibility: sets the digital gain (?k7-gain board-input values)."""
        return ("ok","OK")

    @request(Float(),Float(),Float(optional=True,default=20.0))
    @return_reply(Str())
    def request_test_target(self, sock, az, el, flux):
        """Add a test target to the simulator. ?test-target <az> <el> [<flux_scale>]"""
        self._model.target_az = az
        self._model.target_el = el
        self._model.target_flux = flux
        smsg = "Target set to (%f, %f, %f)" % (az, el, flux)
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Float())
    @return_reply(Str())
    def request_pointing_el(self, sock, el):
        """Sets the current simulator elevation pointing."""
        self._model.test_el = el
        return ("ok","Elevation set to %f" % el)

    @request(Float())
    @return_reply(Str())
    def request_pointing_az(self, sock, az):
        """Sets the current simulator azimuth pointing."""
        self._model.test_az = az
        return ("ok","Azimuth set to %f" % az)


    @return_reply(Str())
    def request_label_input(self, sock, msg):
        """Label the specified input with a string."""
        if not msg.arguments:
            for (inp,label) in sorted(self._model.labels.iteritems()):
                self.reply_inform(sock, Message.inform("label-input", label, inp,'roachXXXXXX'), msg)
            return ("ok",str(len(self._model.labels)))
        else:
            inp = msg.arguments[0]
            label = msg.arguments[1]

        if self._model.labels.has_key(inp):
            self._model.labels[inp] = label
            self._model.update_bls_ordering()
            return ("ok","Label set.")
        return ("fail","Input %s does not follow \d[x|y] form" % inp)


    @request(Str(), Float(optional=True))
    @return_reply(Str())
    def request_capture_start(self, sock, destination, time_):
        """Start a capture (?capture-start k7 [time]). Mostly a dummy, does a spead_issue."""
        self._model.spead_issue()
        smsg = "SPEAD meta packets sent to %s" % (self._model.config['rx_meta_ip'])
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

    @return_reply(Str())
    def request_stop_tx(self, sock, msg):
        """Stop the data stream."""
        self._model._thread_paused = True
        self._model.send_stop()
        smsg = "Data stream stopped."
        activitylogger.info(smsg)
        return ("ok", smsg)

