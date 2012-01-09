import time

import upstream_correlator_conf

class ModelCorrConf(upstream_correlator_conf.CorrConf):
    def __init__(self, config_file):
        upstream_correlator_conf.CorrConf.__init__(self, config_file)
        # IRL sync_time will always be aligned with the PPS (pulse per
        # second), which by design always falls on a whole second. The
        # corr module spead header takes this assumption further, and
        # issues the sync time as an integer number of seconds, so we
        # have to be sure that sync_time is a whole number to prevent
        # stuff from breaking :)
        self._sync_time = int(time.time())
        self._antenna_mapping = self.get_unmapped_channel_names()

    def __getitem__(self, item):
        if item == 'sync_time':
            return self._get_sync_time()
        elif item =='antenna_mapping':
            return self._get_antenna_mapping()
        else:
            return self.config[item]

    def file_exists(self):
        return True

    def _get_sync_time(self):
        return self._sync_time

    def _get_antenna_mapping(self):
        return self._antenna_mapping

    def set_antenna_mapping(self, channel, ant_name):
        input_no = self._map_chan_to_input(channel)
        self._antenna_mapping[input_no] = ant_name

    def get_unmapped_channel_names(self):
        """Get the default channel names, e.g. 0x, 0y, etc."""
        ## Copy-pasted from _get_ant_mapping_list()
        ant_list=[]
        for a in range(self.config['n_ants']):
            for p in self.config['pols']:
                ant_list.append('%i%c'%(a,p))
        return ant_list[0:self.config['n_inputs']]
        ## End copy-paste

    def _map_chan_to_input(self, chan):
        """Map channel name (e.g. 0x, 3y) to input number"""
        try: return self.get_unmapped_channel_names().index(chan)
        except ValueError: raise ValueError('Unknown input channel %s' % chan)
