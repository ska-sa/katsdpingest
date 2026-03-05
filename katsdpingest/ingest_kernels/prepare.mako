<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="transpose_values" type="float2" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void prepare(
    GLOBAL float2 * RESTRICT vis_out,
    const GLOBAL int2 * RESTRICT vis_in,
    const GLOBAL short * RESTRICT permutation,
    int vis_out_stride,
    int vis_in_stride,
    int baselines,
    float scale)
{
    LOCAL_DECL transpose_values values;
    transpose_coords coords;
    transpose_coords_init_simple(&coords);

    /* Load values into shared memory, applying the type conversion and
     * scaling. The input array is padded, so no range checks are needed.
     * CBF indicates missing data by setting the real int32 value to -2**31.
     * Convert these to NaNs instead in order to flag them in prepare_flags.
     */
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int2 in_value = vis_in[${r} * vis_in_stride + ${c}];
        float2 scaled_value;
        if (in_value.x == -1 << 31)
        {
            scaled_value.x = NAN;
            scaled_value.y = NAN;
        } else {
            scaled_value.x = (float) in_value.x * scale;
            scaled_value.y = (float) in_value.y * scale;
        }
        values.arr[${lr}][${lc}] = scaled_value;
    </%transpose:transpose_load>

    BARRIER();

    /* Write value back to memory, applying baseline permutation.
     * Due to the permutation, we now have to check for out-of-range
     * baseline, but not channel.
     */
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        if (${r} < baselines)
        {
            int baseline = permutation[${r}];
            if (baseline >= 0)
            {
                float2 vis = values.arr[${lr}][${lc}];
                vis_out[baseline * vis_out_stride + ${c}] = vis;
            }
        }
    </%transpose:transpose_store>
}
