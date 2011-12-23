import fake_cn_conf
import cn_conf
import spead_stuff

#corr_conf = fake_cn_conf.FakeCorrConf('config-nbc')
corr_conf = cn_conf.CorrConf('config-nbc')
fake_corr = spead_stuff.FakeCorrelator(corr_conf)
