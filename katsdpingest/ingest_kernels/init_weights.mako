<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void init_weights(
    GLOBAL float * RESTRICT weights,
    const GLOBAL float * RESTRICT auto_weights,
    const GLOBAL ushort2 * RESTRICT baseline_inputs,
    int weights_stride,
    int auto_weights_stride,
    int baselines)
{
    int channel = get_global_id(0);
    int baseline = get_global_id(1);
    if (baseline >= baselines)
        return;
    ushort2 idx = baseline_inputs[baseline];
    float a = auto_weights[idx.x * auto_weights_stride + channel];
    float b = auto_weights[idx.y * auto_weights_stride + channel];
    float w = a * b;
    if (!isfinite(w))
    {
        /* Most likely cause is that one (or both) of the autocorrelations
         * had zero power. This is assumed to be something wrong in the
         * system, so we set the weight to something very close to 0
         * (2^-64).
         */
        w = 5.42101086e-20f;
    }
    weights[baseline * weights_stride + channel] = w;
}
