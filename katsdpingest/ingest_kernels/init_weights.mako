<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void init_weights(
    GLOBAL float * RESTRICT weights,
    const GLOBAL float * RESTRICT auto_weights,
    const GLOBAL ushort2 * RESTRICT input_map,
    int weights_stride,
    int auto_weights_stride,
    int baselines)
{
    int channel = get_global_id(0);
    int baseline = get_global_id(1);
    if (baseline >= baselines)
        return;
    ushort2 idx = input_map[baseline];
    float a = auto_weights[idx.x * auto_weights_stride + channel];
    float b = auto_weights[idx.y * auto_weights_stride + channel];
    weights[baseline * weights_stride + channel] = a * b;
}
