<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="transpose_vis" type="float2" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_data_class class_name="transpose_weights" type="float" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_data_class class_name="transpose_flags" type="unsigned char" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

DEVICE_FN void accum_vis(GLOBAL float2 *out, float2 value, float weight)
{
    float2 sum = *out;
    sum.x += value.x * weight;
    sum.y += value.y * weight;
    *out = sum;
}

KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void accum(
% for i in range(outputs):
    GLOBAL float2 * RESTRICT out_vis${i},
    GLOBAL float * RESTRICT out_weights${i},
    GLOBAL unsigned char * RESTRICT out_flags${i},
% endfor
    const GLOBAL float2 * RESTRICT in_vis,
    const GLOBAL float * RESTRICT in_weights,
    const GLOBAL unsigned char * RESTRICT in_flags,
% for i in range(outputs):
    int out_vis_stride${i},
    int out_weights_stride${i},
    int out_flags_stride${i},
% endfor
    int in_vis_stride,
    int in_weights_stride,
    int in_flags_stride,
    int channel_start)
{
    LOCAL_DECL transpose_vis local_vis;
    LOCAL_DECL transpose_weights local_weights;
    LOCAL_DECL transpose_flags local_flags;
    transpose_coords coords;

    transpose_coords_init_simple(&coords);

    // Load a block of data, for all channels
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        local_vis.arr[${lr}][${lc}] = in_vis[${r} * in_vis_stride + ${c} + channel_start];
        local_weights.arr[${lr}][${lc}] = in_weights[${r} * in_weights_stride + ${c}];
        local_flags.arr[${lr}][${lc}] = in_flags[${r} * in_flags_stride + ${c} + channel_start];
    </%transpose:transpose_load>

    BARRIER();

    // Apply flags to weights, and do weighted accumulation
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        float2 vis = local_vis.arr[${lr}][${lc}];
        float weight = local_weights.arr[${lr}][${lc}];
        unsigned int flag = local_flags.arr[${lr}][${lc}];
        if (flag != 0)
            weight *= 5.42101086e-20f;  // 2^-64
% for i in range(outputs):
        accum_vis(&out_vis${i}[${r} * out_vis_stride${i} + ${c}], vis, weight);
        out_weights${i}[${r} * out_weights_stride${i} + ${c}] += weight;
        out_flags${i}[${r} * out_flags_stride${i} + ${c}] &= flag;
% endfor
    </%transpose:transpose_store>
}
