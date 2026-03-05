<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="transpose_flags" type="uchar" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void prepare_flags(
    GLOBAL uchar * RESTRICT flags,
    GLOBAL float2 * RESTRICT vis,
    const GLOBAL uchar * RESTRICT channel_mask,
    const GLOBAL uint * RESTRICT channel_mask_idx,
    int flags_stride,
    int vis_stride,
    int channel_mask_stride,
    uint max_mask,
    uchar zero_flag,
    uchar missing_flag)
{
    LOCAL_DECL transpose_flags local_flags;
    transpose_coords coords;
    transpose_coords_init_simple(&coords);

    // Compute flags into shared memory
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int idx = min(max_mask, channel_mask_idx[${r}]);
        local_flags.arr[${lr}][${lc}] = channel_mask[idx * channel_mask_stride + ${c}];
    </%transpose:transpose_load>

    BARRIER();

    // Write flags back to global memory in channel-major order
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int flags_addr = ${r} * flags_stride + ${c};
        int vis_addr = ${r} * vis_stride + ${c};
        uchar f = local_flags.arr[${lr}][${lc}];
        float2 v = vis[vis_addr];
        // Flag zero visibilities
        if (v.x == 0.0f && v.y == 0.0f)
            f |= zero_flag;
        // Flag missing visibilities and zero them afterwards
        else if (isnan(v.x) || isnan(v.y))
        {
            f |= missing_flag;
            float2 zero_vis;
            zero_vis.x = 0.0f;
            zero_vis.y = 0.0f;
            vis[vis_addr] = zero_vis;
        }
        flags[flags_addr] = f;
    </%transpose:transpose_store>
}
