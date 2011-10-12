import time
import logging
import sys

from katcp import Sensor, Message
from katcore.dev.base import Device
from katcp.kattypes import request, return_reply, Str, Int, Float

activitylogger = logging.getLogger('activity')
log_name = 'kat.k7simulator'
logger = logging.getLogger(log_name)

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
    def request_spead_issue(self, sock, msg):
        """Issue the SPEAD meta packets..."""
        self._model.spead_issue()
        smsg = "SPEAD meta packets sent to %s" % (self._model.config['rx_meta_ip'])
        activitylogger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_start_tx(self, sock, msg):
        """Start the data stream."""
        self._model._thread_paused = False
        smsg = "Data stream started."
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Int())
    @return_reply(Str())
    def request_set_dump_rate(self, sock, rate):
        """Set the dump rate in Hz. Default is 1."""
        self._model.dump_period = 1.0 / int(rate)
        smsg = "Dump rate set to %i Hz" % rate
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Str())
    @return_reply(Str())
    def request_poco_accumulation_length(self, sock, period):
        """Set the period in ms. Default is 1000."""
        self._model.dump_period = 1000.0 / float(period)
        smsg = "Dump period set to %s ms" % period
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
    def request_poco_gain(self, sock, msg1, msg2):
        """Dummy for compatibility."""
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

    @request(Str(),Str(),Int())
    @return_reply(Str())
    def request_capture_destination(self, sock, destination, ip, port):
        """Dummy command to enable ff compatibility."""
        return ("ok","Destination OK")

    @return_reply(Str(optional=True))
    def request_capture_start(self, sock, destination):
        """For compatibility with dbe_proxy. Same as spead_issue."""
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

