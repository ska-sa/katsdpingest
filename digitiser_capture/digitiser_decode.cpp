#include "recv_reader.h"
#include "recv_stream.h"
#include "recv_ring_stream.h"
#include "recv_heap.h"
#include "common_logging.h"
#include <string>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <vector>
#include <limits>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <net/ethernet.h>
#include <pcap/pcap.h>
#include <tbb/pipeline.h>
#include <tbb/task_scheduler_init.h>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#ifndef __BYTE_ORDER__
# warning "Unable to detect byte order"
#elif __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
# error "Only little endian is currently supported"
#endif

/***************************************************************************/

/* spead2 reader for pcap files (not suitable for live streams, because it
 * assumes non-blocking operation). This could later be made part of spead2.
 */

class pcap_file_reader : public spead2::recv::reader
{
private:
    pcap_t *handle;

    void handler(const struct pcap_pkthdr *h, const std::uint8_t *bytes);
    static void handler_wrapper(u_char *user, const struct pcap_pkthdr *h, const u_char *bytes);
    void run();

public:
    pcap_file_reader(spead2::recv::stream &owner, const std::string &filename);
    virtual ~pcap_file_reader();

    virtual void stop() override;
};

void pcap_file_reader::handler(const struct pcap_pkthdr *h, const std::uint8_t *bytes)
{
    /* pcap filter ensures unfragmented IP wiith no options, so headers are
     * 14 bytes for Ethernet
     * 20 bytes for IP
     * 8 bytes for UDP
     */
    const int hdr_len = 42;
    spead2::recv::stream_base &s = get_stream_base();
    if (s.is_stopped() || h->caplen < h->len || h->caplen <= hdr_len)
        return;
    spead2::recv::packet_header packet;
    std::size_t size = decode_packet(packet, bytes + hdr_len, h->caplen - hdr_len);
    if (size > 0)
        s.add_packet(packet);
}

void pcap_file_reader::handler_wrapper(u_char *user, const struct pcap_pkthdr *h, const u_char *bytes)
{
    pcap_file_reader *self = (pcap_file_reader *) user;
    self->handler(h, bytes);
}

void pcap_file_reader::run()
{
    int status = pcap_loop(handle, -1, pcap_file_reader::handler_wrapper,
                           reinterpret_cast<u_char *>(this));
    switch (status)
    {
    case 0:
        // End of file
        get_stream_base().stop_received();
        break;
    case -1:
        // An error occurred
        spead2::log_warning("pcap error occurred: %s", pcap_geterr(handle));
        break;
    case -2:
        // pcap_breakloop was called. No need to call stop_received(),
        // since we were externally stopped.
        break;
    }
    stopped();
}

pcap_file_reader::pcap_file_reader(spead2::recv::stream &owner, const std::string &filename)
    : spead2::recv::reader(owner)
{
    // Open the file
    char errbuf[PCAP_ERRBUF_SIZE];
    handle = pcap_open_offline(filename.c_str(), errbuf);
    if (!handle)
        throw std::runtime_error(errbuf);
    // Set a filter to ensure that we only get UDP4 packets with no IP options or fragmentation
    bpf_program filter;
    if (pcap_compile(handle, &filter,
                     "ip proto \\udp and ip[0] & 0xf = 5 and ip[6:2] & 0x3fff = 0",
                     1, PCAP_NETMASK_UNKNOWN) != 0)
        throw std::runtime_error(pcap_geterr(handle));
    if (pcap_setfilter(handle, &filter) != 0)
    {
        // TODO: free the filter
        throw std::runtime_error(pcap_geterr(handle));
    }
    pcap_freecode(&filter);

    // Process the file
    get_stream().get_strand().post([this] { run(); });
}

pcap_file_reader::~pcap_file_reader()
{
    if (handle)
        pcap_close(handle);
}

void pcap_file_reader::stop()
{
    pcap_breakloop(handle);
}

/***************************************************************************/

/* Take buffer of packed 10-bit signed values (big-endian) and return them as 16-bit
 * values.
 */
static std::vector<std::int16_t> decode_10bit(const std::uint8_t *data, std::size_t length, bool non_icd)
{
    std::size_t out_length = length * 8 / 10;
    std::vector<std::int16_t> out;
    out.reserve(out_length);
    std::vector<std::uint8_t> data2;
    if (non_icd)
    {
        /* Non-compliant bit packing. To fix it up:
         * - take 320 bits (40 bytes)
         * - split it into 64-bit values, and reverse them
         * - split it into 80-bit values, and reverse them
         */
        data2.resize(length);
        for (std::size_t i = 0; i < length; i += 40)
        {
            char shuffle[40];
            for (int j = 0; j < 40; j += 8)
                std::memcpy(&shuffle[32 - j], &data[i + j], 8);
            for (int j = 0; j < 40; j += 10)
                memcpy(&data2[i + j], &shuffle[30 - j], 10);
        }
        data = data2.data();
    }
    std::uint64_t buffer = 0;
    int buffer_bits = 0;
    for (std::size_t i = 0; i < length; i += 4)
    {
        std::uint32_t chunk;
        std::memcpy(&chunk, &data[i], 4);
        chunk = ntohl(chunk);
        buffer = (buffer << 32) | chunk;
        buffer_bits += 32;
        while (buffer_bits >= 10)
        {
            buffer_bits -= 10;
            std::int64_t value = (buffer >> buffer_bits) & 1023;
            // Convert to signed
            if (value & 512)
                value -= 1024;
            out.push_back(value);
        }
    }
    return out;
}

/***************************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>

struct options
{
    bool non_icd = false;
    std::uint64_t max_heaps = std::numeric_limits<std::uint64_t>::max();
    std::string input_files[2];
    std::string output_file;
};

class heap_info
{
public:
    spead2::recv::heap heap;
    std::uint64_t timestamp = 0;
    const std::uint8_t *data = nullptr;
    std::size_t length = 0;

    explicit heap_info(spead2::recv::heap &&heap);
    heap_info &operator=(spead2::recv::heap &&heap);

private:
    void update();
};

heap_info::heap_info(spead2::recv::heap &&heap) : heap(std::move(heap))
{
    update();
}

heap_info &heap_info::operator=(spead2::recv::heap &&heap)
{
    this->heap = std::move(heap);
    update();
    return *this;
}

void heap_info::update()
{
    timestamp = 0;
    data = nullptr;
    length = 0;
    for (const auto &item : heap.get_items())
    {
        if (item.id == 0x1600 && item.is_immediate)
            timestamp = item.immediate_value;
        else if (item.id == 0x3300)
        {
            data = item.ptr;
            length = item.length;
        }
    }
}

typedef std::vector<heap_info> heap_info_batch;
typedef std::vector<std::vector<std::int16_t>> decoded_batch;

class interleaver
{
private:
    spead2::thread_pool thread_pool;
    std::unique_ptr<spead2::recv::ring_stream<>> streams[2];
    std::unique_ptr<heap_info> info[2];
    std::uint64_t next_timestamp;
    std::uint64_t max_heaps;

public:
    std::uint64_t n_heaps = 0;
    std::uint64_t first_timestamp = 0;

    explicit interleaver(const options &opts)
        : thread_pool(2), max_heaps(opts.max_heaps)
    {
        for (int i = 0; i < 2; i++)
        {
            streams[i].reset(new spead2::recv::ring_stream<>(thread_pool, 2, 128));
            streams[i]->emplace_reader<pcap_file_reader>(opts.input_files[i]);
        }

        try
        {
            // Skip ahead until first packet with timestamp
            for (int i = 0; i < 2; i++)
            {
                info[i].reset(new heap_info(streams[i]->pop()));
                while (info[i]->timestamp == 0)
                    *info[i] = streams[i]->pop();
                std::cout << "First timestamp in stream " << i << " is " << info[i]->timestamp << '\n';
            }
            // Align by timestamps
            for (int i = 0; i < 2; i++)
            {
                while (info[i]->timestamp + info[i]->length < info[!i]->timestamp)
                    *info[i] = streams[i]->pop();
            }
            next_timestamp = std::min(info[0]->timestamp, info[1]->timestamp);
            first_timestamp = next_timestamp;
            std::cout << "First synchronised timestamp is " << first_timestamp << '\n';
        }
        catch (spead2::ringbuffer_stopped)
        {
            throw std::runtime_error("End of stream reached before stream synchronisation");
        }
    }

    // Returns empty batch on reaching the end
    heap_info_batch next_batch()
    {
        constexpr int batch_size = 32;
        heap_info_batch batch;
        if (streams[0])
        {
            for (int i = 0; i < batch_size; i++)
            {
                if (n_heaps >= max_heaps)
                {
                    std::cout << "Stopping after " << max_heaps << " heaps\n";
                    streams[0].reset();
                    streams[1].reset();
                    break;
                }
                int idx;
                for (idx = 0; idx < 2; idx++)
                    if (info[idx] && info[idx]->timestamp == next_timestamp)
                        break;
                if (idx == 2)
                {
                    if (!info[0] || !info[1])
                        std::cout << "One or both streams has ended after " << n_heaps << " heaps\n";
                    else
                        std::cerr << "Timestamps do not match up, aborting\n"
                            << "Expected " << next_timestamp << ", have "
                            << info[0]->timestamp << ", " << info[1]->timestamp << '\n';
                    streams[0].reset();
                    streams[1].reset();
                    break;
                }
                n_heaps++;
                next_timestamp += info[idx]->length * 8 / 10;
                batch.push_back(std::move(*info[idx]));
                try
                {
                    *info[idx] = streams[idx]->pop();
                }
                catch (spead2::ringbuffer_stopped)
                {
                    info[idx].reset();
                }
            }
        }
        return batch;
    }
};

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

static po::typed_value<bool> *make_opt(bool &var)
{
    return po::bool_switch(&var)->default_value(var);
}

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: interleave [opts] <input1.pcap> <input2.pcap> <output.npy>\n";
    o << desc;
}

static options parse_options(int argc, char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("non-icd", make_opt(opts.non_icd), "Assume digitiser is not ICD compliant")
        ("heaps", make_opt(opts.max_heaps), "Number of heaps to process [all]")
    ;
    hidden.add_options()
        ("input1", make_opt(opts.input_files[0]), "input1")
        ("input2", make_opt(opts.input_files[1]), "input2")
        ("output", make_opt(opts.output_file), "output")
    ;
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("input1", 1);
    positional.add("input2", 1);
    positional.add("output", 1);
    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
            .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
            .options(all)
            .positional(positional)
            .run(), vm);
        po::notify(vm);
        if (vm.count("help"))
        {
            usage(std::cout, desc);
            std::exit(0);
        }
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc);
        std::exit(2);
    }
}

int main(int argc, char **argv)
{
    options opts = parse_options(argc, argv);
    // Leave 2 cores free for decoding the SPEAD stream
    int n_threads = tbb::task_scheduler_init::default_num_threads() - 2;
    if (n_threads < 1)
        n_threads = 1;
    tbb::task_scheduler_init init_tbb(n_threads);

    interleaver inter(opts);

    const int header_size = 96;
    std::ofstream out(opts.output_file, std::ios::out | std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    // Make space for the header
    out.seekp(header_size);
    std::uint64_t n_elements = 0;

    auto read_filter = [&] (tbb::flow_control &fc) -> std::shared_ptr<heap_info_batch>
    {
        std::shared_ptr<heap_info_batch> batch = std::make_shared<heap_info_batch>(inter.next_batch());
        if (batch->empty())
            fc.stop();
        return batch;
    };

    auto decode_filter = [&](std::shared_ptr<heap_info_batch> batch) -> std::shared_ptr<decoded_batch>
    {
        std::shared_ptr<decoded_batch> out = std::make_shared<decoded_batch>();
        for (const heap_info &info : *batch)
            out->push_back(decode_10bit(info.data, info.length, opts.non_icd));
        return out;
    };

    auto write_filter = [&](std::shared_ptr<decoded_batch> batch)
    {
        for (const std::vector<int16_t> &decoded : *batch)
        {
            out.write(reinterpret_cast<const char *>(decoded.data()),
                      decoded.size() * sizeof(decoded[0]));
            n_elements += decoded.size();
        }
    };

    tbb::parallel_pipeline(16,
        tbb::make_filter<void, std::shared_ptr<heap_info_batch>>(
            tbb::filter::serial_in_order, read_filter)
        & tbb::make_filter<std::shared_ptr<heap_info_batch>, std::shared_ptr<decoded_batch>>(
            tbb::filter::parallel, decode_filter)
        & tbb::make_filter<std::shared_ptr<decoded_batch>, void>(
            tbb::filter::serial_in_order, write_filter));

    // Write in the header
    out.seekp(0);
    char header_start[10] = "\x93NUMPY\x01\x00";
    header_start[8] = header_size - 10;
    header_start[9] = 0;
    out.write(header_start, 10);
    out << "{'descr': '<i2', 'fortran_order': False, 'shape': ("
        << n_elements << ",) }";
    if (out.tellp() >= header_size)
    {
        std::cerr << "Oops, header was too big for reserved space! File is corrupted!\n";
        return 1;
    }
    while (out.tellp() < header_size - 1)
        out << ' ';
    out << '\n';
    out.close();
    std::cout << "Header successfully written\n";

    // Write the timestamp file
    std::ofstream timestamp_file(opts.output_file + ".timestamp");
    timestamp_file.exceptions(std::ios::failbit | std::ios::badbit);
    timestamp_file << inter.first_timestamp << '\n';
    timestamp_file.close();
    std::cout << "Write timestamp file\n\n";
    std::cout << "Completed capture+conversion of " << inter.n_heaps
        << " heaps from timestamp " << inter.first_timestamp << '\n';
    return 0;
}
