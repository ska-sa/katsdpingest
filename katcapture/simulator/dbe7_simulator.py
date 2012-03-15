from __future__ import with_statement
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

# Request to update:
#
# label-input (see https://katfs.kat.ac.za/mantis/view.php?id=1394 ) TODO

# Requests to add:
#
# capture_list -> DONE
# k7_accumulation_length -> DONE
# k7_adc_snap_shot -> DONE
# k7_delay -> DONE, stubbily
# k7_frequency_select TODO
# k7_gain -> DONE
# k7_snap_shot TODO
# version -> TODO
# version_list -> TODO
# label-clear (https://katfs.kat.ac.za/mantis/view.php?id=1394) TODO

# Hardware request to leave out of simulator
#
# define -> leave?
# dict -> leave?
# dispatch -> leave?
# enable_sensors -> leave?
# get -> leave?
# job -> leave?
# log_record -> leave?
# notice -> leave?
# process -> leave?
# roach -> leave?
# sensor -> leave?
# set -> leave?
# sm -> leave?
# sync_now -> leave?
# system_info -> leave?
# watchannounce -> leave?
# xport -> leave?

class SimulatorDeviceServer(Device):

    VERSION_INFO = ("k7-simulator",0,1)
    BUILD_INFO = ("k7-simulator",0,1,"rc1")

    def setup_sensors(self):
        super(SimulatorDeviceServer, self).setup_sensors()
        # Subscribe to model signals indicating a change in sensors
        self._model.sensor_changed.connect(self.handle_sensor_update)
        self._model.sensor_removed.connect(self.handle_sensor_remove)
        self._model.sensor_added.connect(self.handle_sensor_add)

    def handle_sensor_add(self, model, sensor, **kwargs):
        assert(model is self._model)
        logger.debug('New sensor %s added signal.' % sensor.name)
        self.add_sensor(sensor)
        self.issue_device_changed('sensor-list')

    def handle_sensor_remove(self, model, sensor, **kwargs):
        assert(model is self._model)
        logger.info('Sensor %s removed signal.' % sensor.name)
        self.remove_sensor(sensor)
        self.issue_device_changed('sensor-list')

    def handle_sensor_update(self, model, sensor, old_sensor, **kwargs):
        assert(model is self._model)
        logger.debug('Sensor %s changed signal.' % sensor.name)
        assert(sensor.name == old_sensor.name)
        self.replace_sensor(sensor, old_sensor)
        self.issue_device_changed('sensor-list')

    def replace_sensor(self, sensor, old_sensor=None):
        # TODO Perhaps one wants to put this method in the katcp
        # server? Or at least the device class.
        if not old_sensor is None:
            assert(self.get_sensor(sensor.name) is old_sensor)
        else:
            old_sensor = self.get_sensor(sensor.name)
        # Can't call remove_sensor() since it whacks the strategies
        del self._sensors[sensor.name]
        # Get all the old sensor's strategies
        sensor_strategies = []
        with self._strat_lock:
            # Loop over the per-sock strategies
            for sock_strategies in self._strategies.values():
                for strat_sensor, strategy in list(sock_strategies.items()):
                    if strat_sensor.name == sensor.name:
                        # Replace the strategy's sensor with the new one
                        strategy.detach()
                        # Naughty private variable access. Del to make
                        # sure an error is raised if we set a
                        # non-existent variable
                        del(strategy._sensor)
                        strategy._sensor = sensor
                        sensor_strategies.append(strategy)

        # Can't call strat.attach while _strat_lock is held since it
        # may try and write to a disconnected socket, which in turn
        # calls the on_disconnect handler which in turn tries to
        # aquire _strat_lock so that it can remove the sensor
        # strategies of the dead socket.
        for strat in sensor_strategies:
            strat.attach()

        self.add_sensor(sensor)

    def handle_request(self, sock, msg):
        hang_requests = self._model.get_test_sensor('hang-requests').value()
        if not hang_requests  or msg.name == 'watchdog':
            return super(SimulatorDeviceServer, self).handle_request(sock, msg)
        # If hang_requests is set we never reply
        return

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

    def issue_device_changed(self, change):
        """
        Issue a device-changed inform to all clients.

        Parameter change indicates what part of the device has changed. e.g.

        issue_device_changed('sensor-list')

        will result in the inform

        #device-changed sensor-list.

        The change description is tested for validity; currently only
        'sensor-list' is considered valid.
        """
        if not change in ['sensor-list']:
            raise ValueError('Unknown change notification %s' % change)
        msg = Message.inform('device-changed', change)
        self.mass_inform(msg)
