<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="transpose_values" type="float2" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void prepare(
    GLOBAL float2 * RESTRICT vis_out,
    GLOBAL float * RESTRICT weights,
    const GLOBAL int2 * RESTRICT vis_in,
    const GLOBAL short * RESTRICT permutation,
    int vis_out_stride,
    int weights_stride,
    int vis_in_stride,
    int channel_start,
    int channel_end,
    int baselines,
    float scale)
{
    LOCAL_DECL transpose_values values;
    transpose_coords coords;
    transpose_coords_init_simple(&coords);

    /* Load values into shared memory, applying the type conversion and
     * scaling. The input array is padded, so no range checks are needed.
     */
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int2 in_value = vis_in[${r} * vis_in_stride + ${c}];
        float2 scaled_value;
        scaled_value.x = (float) in_value.x * scale;
        scaled_value.y = (float) in_value.y * scale;
        values.arr[${lr}][${lc}] = scaled_value;
    </%transpose:transpose_load>

    BARRIER();

    /* Write value back to memory, applying baseline permutation.
     * Due to the permutation, we now have to check for out-of-range
     * baseline, but not channel.
     *
     * For now the weights are all set to 1, but this may change with
     * Van Vleck correction.
     */
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        if (${r} < baselines)
        {
            int baseline = permutation[${r}];
            if (baseline >= 0)
            {
                float2 vis = values.arr[${lr}][${lc}];
                vis_out[baseline * vis_out_stride + ${c}] = vis;
                if (${c} >= channel_start && ${c} < channel_end)
                {
                    int weight_addr = baseline * weights_stride + ${c} - channel_start;
                    weights[weight_addr] = 1.0f;
                }
            }
        }
    </%transpose:transpose_store>
}
