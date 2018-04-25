<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

${wg_reduce.define_scratch('unsigned int', wgs, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('unsigned int', wgs, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

#define WGS ${wgs}
#define BITS 8

/* This implementation is far from optimal, but it's also not massively
 * performance-critical. Some ideas for future optimisation if necessary:
 *
 * - have each workitem load 32 bits at a time instead of 8 (has some
 *   complications if the number of channels is odd).
 * - have each workgroup handle several baselines and fewer channels (lower
 *   reduction costs, but could also reduce parallelism).
 * - have each workgroup handle only some of the channels, and do a final
 *   CPU-side reduction (more parallelism).
 * - do the reduction on all the counts jointly, instead of one at a time.
 */
KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void count_flags(
    GLOBAL unsigned int * RESTRICT counts,
    const GLOBAL unsigned char * RESTRICT flags,
    int flags_stride,
    int channels,
    int channel_start)
{
    LOCAL_DECL scratch_t scratch;

    int lid = get_local_id(0);
    int baseline = get_global_id(1);
    // Adjust pointer to start of current baseline
    flags += baseline * flags_stride + channel_start;
    unsigned int sums[BITS] = {};
    for (int i = lid; i < channels; i += WGS)
    {
        unsigned char flag = flags[i];
        for (int j = 0; j < 8; j++)
        {
            sums[j] += (flag & 1);
            flag >>= 1;
        }
    }

    // Accumulate across workitems
    for (int i = 0; i < BITS; i++)
        sums[i] = reduce(sums[i], lid, &scratch);

    // Write results
    if (lid == 0)
    {
        counts += baseline * BITS;
        for (int i = 0; i < BITS; i++)
            counts[i] = sums[i];
    }
}
