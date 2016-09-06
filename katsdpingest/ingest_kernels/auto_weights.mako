<%include file="/port.mako"/>

/* vis is listed as type float rather than float2, because we are only
 * interested in the real part of autocorrelations (the imaginary part should
 * be zero).
 */
KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void auto_weights(
    GLOBAL float * RESTRICT weights,
    const GLOBAL float * RESTRICT vis,
    const GLOBAL unsigned short * RESTRICT input_auto_baseline,
    int weights_stride,
    int vis_stride,
    int inputs,
    int channel_start,
    float scale)
{
    int channel = get_global_id(0);
    int input = get_global_id(1);
    if (input >= inputs)
        return;
    int baseline = input_auto_baseline[input];
    int out_idx = input * weights_stride + channel;
    int in_idx = (baseline * vis_stride + channel + channel_start) * 2;
    weights[out_idx] = scale / vis[in_idx];
}
