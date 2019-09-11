<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

${wg_reduce.define_scratch('unsigned int', wgs, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('unsigned int', wgs, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

#define WGS ${wgs}
#define BITS 8

/**
 * Count number of visibilities with each flag bit, per baseline.
 *
 * This implementation is far from optimal, but it's also not massively
 * performance-critical. Some ideas for future optimisation if necessary:
 *
 * - Have each workitem load 32 bits at a time instead of 8 (has some
 *   complications if the number of channels is odd).
 * - Have each workgroup handle several baselines and fewer channels. It would
 *   lower reduction costs, but could also reduce parallelism.
 * - Have each workgroup handle only some of the channels, and do a final
 *   CPU-side reduction (more parallelism).
 * - Do the reduction on all the counts jointly, instead of one at a time.
 *
 * @param[out] counts         Output counts, shape (baselines, 8), contiguous
 * @param[out] any_counts     Count of visibilities with any flag, per baseline
 * @param      flags          Per-visibility input flags, shape (baselines, flags_stride)
 * @param      flags_stride   Stride for @a flags
 * @param      channels       Number of channels over which to do count
 * @param      channel_start  Offset to first channel to count in @a flags
 * @param      mask           Mask ANDed with the flags before counting (used to eliminate the
 *                            pseudo-flag used to mark unflagged data).
 */
KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void count_flags(
    GLOBAL unsigned int * RESTRICT counts,
    GLOBAL unsigned int * RESTRICT any_counts,
    const GLOBAL unsigned char * RESTRICT flags,
    int flags_stride,
    int channels,
    int channel_start,
    unsigned char mask)
{
    LOCAL_DECL scratch_t scratch;

    int lid = get_local_id(0);
    int baseline = get_global_id(1);
    // Adjust pointer to start of current baseline
    flags += baseline * flags_stride + channel_start;
    unsigned int sums[BITS] = {};
    unsigned int any = 0;
    for (int i = lid; i < channels; i += WGS)
    {
        unsigned char flag = flags[i] & mask;
        any += (flag != 0);
        for (int j = 0; j < 8; j++)
        {
            sums[j] += (flag & 1);
            flag >>= 1;
        }
    }

    // Accumulate across workitems
    for (int i = 0; i < BITS; i++)
        sums[i] = reduce(sums[i], lid, &scratch);
    any = reduce(any, lid, &scratch);

    // Write results
    if (lid == 0)
    {
        counts += baseline * BITS;
        for (int i = 0; i < BITS; i++)
            counts[i] += sums[i];
        any_counts[baseline] += any;
    }
}
