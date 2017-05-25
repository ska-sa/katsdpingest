<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="transpose_vis" type="float2" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_data_class class_name="transpose_weights" type="float" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_data_class class_name="transpose_flags" type="unsigned char" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

DEVICE_FN void accum_vis(GLOBAL float2 *out, float2 value, float weight)
{
    float2 sum = *out;
    sum.x = fma(value.x, weight, sum.x);
    sum.y = fma(value.y, weight, sum.y);
    *out = sum;
}

/*
 * in_full_stride is for in_vis and in_flags (which are indexed from channel_start),
 * while in_kept_stride is for in_weights (which is indexed from 0).
 */
KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void accum(
% for i in range(outputs):
    GLOBAL float2 * RESTRICT out_vis${i},
    GLOBAL float * RESTRICT out_weights${i},
    GLOBAL unsigned char * RESTRICT out_flags${i},
% endfor
    const GLOBAL float2 * RESTRICT in_vis,
    const GLOBAL float * RESTRICT in_weights,
    const GLOBAL unsigned char * RESTRICT in_flags,
    const GLOBAL unsigned char * RESTRICT channel_flags,
    const GLOBAL unsigned char * RESTRICT baseline_flags,
    int out_stride,
    int in_full_stride,
    int in_kept_stride,
    int channel_start)
{
    LOCAL_DECL transpose_vis local_vis;
    LOCAL_DECL transpose_weights local_weights;
    LOCAL_DECL transpose_flags local_flags;
    transpose_coords coords;

    transpose_coords_init_simple(&coords);

    // Load a block of data, for all channels
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int full_addr = ${r} * in_full_stride + ${c} + channel_start;
        int kept_addr = ${r} * in_kept_stride + ${c};
        local_vis.arr[${lr}][${lc}] = in_vis[full_addr];
        local_weights.arr[${lr}][${lc}] = in_weights[kept_addr];
        local_flags.arr[${lr}][${lc}] = in_flags[full_addr] | baseline_flags[${r}];
    </%transpose:transpose_load>

    BARRIER();

    // Apply flags to weights, and do weighted accumulation
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        float2 vis = local_vis.arr[${lr}][${lc}];
        float weight = local_weights.arr[${lr}][${lc}];
        unsigned int flag = local_flags.arr[${lr}][${lc}];
        flag |= channel_flags[${r}];
        if (flag != 0)
            weight *= 5.42101086e-20f;  // 2^-64
        else
            flag = ${unflagged_bit};
        int addr = ${r} * out_stride + ${c};
% for i in range(outputs):
        accum_vis(&out_vis${i}[addr], vis, weight);
        out_weights${i}[addr] += weight;
        out_flags${i}[addr] |= flag;
% endfor
    </%transpose:transpose_store>
}
