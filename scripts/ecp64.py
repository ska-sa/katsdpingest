import logging
from collections import Counter, OrderedDict

import numpy as np
import spead2
from spead2 import recv


def channel_ordering(num_chans):
    pairs = np.arange(num_chans).reshape(-1, 2)
    first_half =  pairs[:num_chans // 4]
    second_half = pairs[num_chans // 4:]
    return np.c_[first_half, second_half].ravel()


def unpack_bits(x, partition):
    sizes = np.asarray(partition)
    shifts = np.cumsum(np.r_[0, partition[::-1][:-1]])[::-1]
    return [(x >> shift) & int(size * '1', 2)
            for size, shift in zip(sizes, shifts)]


ip = '239.10.3.10'
ports = (7150, 7151, 7152, 7153)
num_chans = 128
sampling_rate = 1712e6
bandwidth = 0.5 * sampling_rate
centre_freq = 0.75 * sampling_rate
test_tone = 912e6

chans = channel_ordering(num_chans)
freqs = centre_freq + bandwidth / num_chans * (np.arange(num_chans) - num_chans / 2)

logging.basicConfig(level='DEBUG')
rx = recv.Stream(spead2.ThreadPool())
for port in ports:
    rx.add_udp_reader(port, bind_hostname=ip, max_size=1536)
ig = spead2.ItemGroup()

# Hack to create descriptors locally (Marc's DMC will provide them eventually)
ig.add_item(id=int('0x1600', 16), name='timestamp',
            description='Local digitiser timestamp at start of accumulation',
            shape=(), format=[('u', 48)])
ig.add_item(id=int('0x3101', 16), name='digitiser_id',
            description='Digitiser serial number, type, receptor ID, pol ID',
            shape=(), format=[('u', 48)])
ig.add_item(id=int('0x3102', 16), name='digitiser_status',
            description='Accumulator / FFT / ADC saturation, noise diode status',
            shape=(), format=[('u', 48)])
ig.add_item(id=int('0x3301', 16), name='data_vv',
            description='Autocorrelation VV*',
            shape=(num_chans,), dtype='>u4')
ig.add_item(id=int('0x3302', 16), name='data_hh',
            description='Autocorrelation HH*',
            shape=(num_chans,), dtype='>u4')
ig.add_item(id=int('0x3303', 16), name='data_vh',
            description='Crosscorrelation VH* (real followed by imag)',
            shape=(2 * num_chans,), dtype='>i4')

old_versions = Counter({name: item.version for (name, item) in ig.items()})

# plt.close('all')
# fig, axgrid = plt.subplots(2, 2, num=1)
# axgrid[0][0].set_title('HH')
# axgrid[0][0].set_xticklabels([])
# axgrid[0][0].plot(freqs / 1e6, 30 * np.ones_like(freqs))
# axgrid[0][0].axvline(test_tone / 1e6, color='k', linestyle='--')
# axgrid[0][0].set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
# axgrid[0][0].set_ylim(45, 65)
# axgrid[0][0].set_ylabel('Power (dB)')
# axgrid[0][1].set_title('VV')
# axgrid[0][1].set_xticklabels([])
# axgrid[0][1].yaxis.tick_right()
# axgrid[0][1].yaxis.set_label_position('right')
# axgrid[0][1].plot(freqs / 1e6, 30 * np.ones_like(freqs))
# axgrid[0][1].axvline(test_tone / 1e6, color='k', linestyle='--')
# axgrid[0][1].set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
# axgrid[0][1].set_ylim(45, 65)
# axgrid[0][1].set_ylabel('Power (dB)')
# axgrid[1][0].set_title('ReVH')
# axgrid[1][0].set_xlabel('Frequency (MHz)')
# axgrid[1][0].plot(freqs / 1e6, np.zeros_like(freqs))
# axgrid[1][0].axvline(test_tone / 1e6, color='k', linestyle='--')
# axgrid[1][0].set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
# axgrid[1][0].set_ylim(-500000, 500000)
# axgrid[1][0].set_ylabel('Amplitude (linear)')
# axgrid[1][1].set_title('ImVH')
# axgrid[1][1].yaxis.tick_right()
# axgrid[1][1].yaxis.set_label_position('right')
# axgrid[1][1].plot(freqs / 1e6, np.zeros_like(freqs))
# axgrid[1][1].axvline(test_tone / 1e6, color='k', linestyle='--')
# axgrid[1][1].set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
# axgrid[1][1].set_ylim(-500000, 500000)
# axgrid[1][1].set_xlabel('Frequency (MHz)')
# axgrid[1][1].set_ylabel('Amplitude (linear)')

raw_data = OrderedDict()

for heap in rx:
    # print heap
    ig.update(heap)
    if 'timestamp' not in ig:
        continue
    new_versions = Counter({name: item.version for (name, item) in ig.items()})
    new_items = (new_versions - old_versions).keys()
    old_versions = new_versions

    timestamp = ig['timestamp'].value / sampling_rate
    dig_id = ig['digitiser_id'].value
    dig_status = ig['digitiser_status'].value
    dig_serial, dig_type, receptor, pol = unpack_bits(dig_id, (24, 8, 14, 2))
    saturation, nd_on = unpack_bits(dig_status, (8, 1))
    stream = [s[5:] for s in new_items if s.startswith('data_')][0]
    print receptor, timestamp, nd_on, stream
    key = (receptor, timestamp)
    fields = raw_data.get(key, {})
    fields['saturation'] = saturation
    fields['nd_on'] = nd_on
    if stream == 'vh':
        fields['revh'] = ig['data_' + stream].value[:num_chans][chans]
        fields['imvh'] = ig['data_' + stream].value[num_chans:][chans]
    else:
        fields[stream] = ig['data_' + stream].value[chans]
    raw_data[key] = fields

timestamps = []
saturation = []
nd_on = []
data_hh = []
data_vv = []
data_vh = []
for key, value in raw_data.items():
    receptor = 'm%03d' % (key[0],)
    if len(value) < 6:
        continue
    timestamps.append(key[1])
    nd_on.append(value['nd_on'])
    saturation.append(value['saturation'])
    data_hh.append(value['hh'])
    data_vv.append(value['vv'])
    data_vh.append(value['revh'] + 1.0j * value['imvh'])

np.savez('ecp64.npz', ts=timestamps, sat=saturation, nd=nd_on,
         hh=data_hh, vv=data_vv, vh=data_vh, freqs=freqs,
         ds=dig_serial, dt=dig_type, receptor=receptor)
