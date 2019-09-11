<%include file="/port.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

${wg_reduce.define_scratch('float', wgsx, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('float', wgsx, 'reduce_max', 'scratch_t', wg_reduce.op_fmax, allow_shuffle=True, broadcast=True)}

/**
 * Produce more compact (approximate) representation of weights. On output,
 * each weight is represented as a product of a per-channel float32 and a
 * per-channel, per-baseline uint8.
 *
 * This kernel is modelled on hreduce.mako from katsdpsigproc. A workgroup is
 * 2D, with each row of a workgroup handling a complete channel.
 */
KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void compress_weights(
    GLOBAL unsigned char * RESTRICT weights_out,
    GLOBAL float * RESTRICT weights_channel,
    GLOBAL const float * RESTRICT weights_in,
    int weights_out_stride,
    int weights_in_stride,
    int baselines)
{
    LOCAL_DECL scratch_t scratch[${wgsy}];
    /* Find the largest value for each channel */
    int channel = get_global_id(1);
    int lid = get_local_id(0);
    int in_offset = weights_in_stride * channel;
    // Set a small lower bound (2^-96), to avoid divide-by-zero issues if all
    // weights are zero.
    float max_weight = 1.2621774e-29f;
    // Compute a per-workitem value
    for (int i = lid; i < baselines; i += ${wgsx})
        max_weight = fmax(max_weight, weights_in[in_offset + i]);
    // Reduce the per-workitem values
    max_weight = reduce_max(max_weight, lid, &scratch[get_local_id(1)]);
    float cweight = max_weight * (1.0f / 255.0f);
    if (lid == 0)
        weights_channel[channel] = cweight;

    /* Scale weights relative to cweight and convert to int */
    float scale = 1.0f / cweight;
    int out_offset = weights_out_stride * channel;
    for (int i = lid; i < baselines; i += ${wgsx})
    {
        float weight = weights_in[in_offset + i];
        weight = weight * scale + 0.5f;
        weights_out[out_offset + i] = (unsigned char) weight;
    }
}
