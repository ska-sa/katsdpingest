/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/SPEADReceiver.h"
#include "sys/time.h"

#include "spead2/recv_udp.h"
#include "spead2/recv_live_heap.h"
#include "spead2/recv_ring_stream.h"
#include "spead2/common_endian.h"



void spip::SPEADReceiver::parse_metadata (const spead2::recv::item &item)
{
  // this is an item descriptor with absolute address mode
  if (item.id == 0x5)
  {

  }
  else if (item.id == 0x1009)
  {
    n_chans = item_ptr_48u (item.ptr);
    cerr << "n_chans=" << n_chans << endl;
  }
  else if (item.id == 0x1020)
  {
    requant_bits = item_ptr_48u (item.ptr);
    cerr << "requant_bits=" << requant_bits << endl;
  }
  else if (item.id == 0x101e)
  {
    fft_shift = (((uint64_t *) item.ptr)[0]) > 16;
    cerr << "fft_shift=" << fft_shift << endl;
  }
  else if (item.id == 0x1021)
  {
    feng_pkt_len = item_ptr_48u (item.ptr);
    cerr << "feng_pkt_len=" << feng_pkt_len << endl;
  }
  else if (item.id == 0x1024)
  {
    rx_udp_ip_str.resize(item.length);
    memcpy (&rx_udp_ip_str[0], item.ptr, item.length);
    cerr << "rx_udp_ip_str=" << rx_udp_ip_str << endl;
  }
  else if (item.id == 0x1011)
  {
    centre_freq = item_ptr_48u (item.ptr);
    cerr << "centre_freq=" << centre_freq << endl;
  }
  else if (item.id == 0x1007)
  {
    adc_sample_rate = item_ptr_48u (item.ptr);
    cerr << "adc_sample_rate=" << adc_sample_rate << endl;
  }
  else if (item.id == 0x100a)
  {
    n_ants = item_ptr_48u (item.ptr);
    cerr << "n_ants=" << n_ants << endl;
  }
  else if (item.id == 0x1013)
  {
    bandwidth = item_ptr_64f (item.ptr);
    cerr << "bandwidth=" << bandwidth << endl;
  }
  else if (item.id == 0x1022)
  {
    rx_udp_port = item_ptr_48u (item.ptr);
    cerr << "rx_udp_port=" << rx_udp_port << endl;
  }
  else if (item.id == 0x1045)
  {
    adc_bits = item_ptr_48u (item.ptr);
    cerr << "adc_bits=" << adc_bits << endl;
  }
  else if (item.id == 0x100f)
  {
    n_bengs = item_ptr_48u (item.ptr);
    cerr << "n_bengs=" << n_bengs << endl;
  }
  else if (item.id == 0x101f)
  {
    xeng_acc_len = item_ptr_48u (item.ptr);
    cerr << "xeng_acc_len=" << xeng_acc_len << endl;
  }
  else if (item.id == 0x1050)
  {
    beng_out_bits_per_sample = item_ptr_48u (item.ptr);
    cerr << "beng_out_bits_per_sample=" << beng_out_bits_per_sample << endl;
  }
  // input labelling
  else if (item.id == 0x100e)
  {
  }
  else if (item.id == 0x1027)
  {
    sync_time = item_ptr_48u (item.ptr);
    cerr << "sync_time=" << sync_time << endl;
  }
  else if (item.id == 0x1046)
  {
    scale_factor_timestamp = item_ptr_64f (item.ptr);
    cerr << "scale_factor_timestamp=" << scale_factor_timestamp << endl;
  }
  // per-channel digital scaling factors
  else if (item.id >= 0x1400 && item.id <= 0x1500)
  {

  }
  // beam weight
  else if (item.id == 0x2000)
  {
    beam_weights.resize(item.length / sizeof(int));
    memcpy (&beam_weights[0], item.ptr, item.length);
  }
  else
  {
    cerr << "spip::SPEADReceiver::parse_metadata unparsed item with ID 0x" << std::hex << item.id << std::dec << endl;
  }
}

double spip::SPEADReceiver::item_ptr_64f (const unsigned char * ptr)
{
  uint64_t value = 0;
  for (unsigned i=0; i<8; i++)
  {
    uint64_t tmp = (uint64_t) ptr[i];
    value |= tmp << ((7-i)*8);
  }
  double * dvalue = reinterpret_cast<double *>(&value);
  return *dvalue;
}

uint64_t spip::SPEADReceiver::item_ptr_48u (const unsigned char * ptr)
{
  uint64_t value = 0;
  for (unsigned i=0; i<6; i++)
  {
    uint64_t tmp = (uint64_t) ptr[i];
    value |= tmp << ((5-i)*8);
  }
  return value;
}


void spip::SPEADReceiver::show_heap(const spead2::recv::heap &fheap)
{
    std::cout << "Received heap with CNT " << fheap.get_cnt() << '\n';
    const auto &items = fheap.get_items();
    std::cout << items.size() << " item(s)\n";
    for (const auto &item : items)
    {
        std::cout << "    ID: 0x" << std::hex << item.id << std::dec << ' ';
        std::cout << "[" << item.length << " bytes]";
        std::cout << '\n';
    }
    std::vector<spead2::descriptor> descriptors = fheap.get_descriptors();
    for (const auto &descriptor : descriptors)
    {
        std::cout
            << "    0x" << std::hex << descriptor.id << std::dec << ":\n"
            << "        NAME:  " << descriptor.name << "\n"
            << "        DESC:  " << descriptor.description << "\n";
        if (descriptor.numpy_header.empty())
        {
            std::cout << "        TYPE:  ";
            for (const auto &field : descriptor.format)
                std::cout << field.first << field.second << ",";
            std::cout << "\n";
            std::cout << "        SHAPE: ";
            for (const auto &size : descriptor.shape)
                if (size == -1)
                    std::cout << "?,";
                else
                    std::cout << size << ",";
            std::cout << "\n";
        }
        else
            std::cout << "        DTYPE: " << descriptor.numpy_header << "\n";
    }
    time_point now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start;
    std::cout << elapsed.count() << "\n";
    std::cout << std::flush;
}
