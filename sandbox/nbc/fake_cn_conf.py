import time

import cn_conf

class FakeCorrConf(cn_conf.CorrConf):
    def __init__(self, config_file):
        cn_conf.CorrConf.__init__(self, config_file)
        self._sync_time = time.time()

    def __getitem__(self, item):
        if item == 'sync_time':
            return self._get_sync_time()
        elif item =='antenna_mapping':
            return self._get_antenna_mapping()
        else:
            return self.config[item]


    def _get_sync_time(self):
        return self._sync_time

    def _get_antenna_mapping(self):
        return []
