<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="transpose_flags" type="uchar" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void merge_flags(
    GLOBAL uchar * RESTRICT out_flags,
    const GLOBAL uchar * RESTRICT in_flags,
    const GLOBAL uchar * RESTRICT baseline_flags,
    int out_flags_stride,
    int in_flags_stride)
{
    LOCAL_DECL transpose_flags local_flags;
    transpose_coords coords;
    transpose_coords_init_simple(&coords);

    // Load input flags into shared memory
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int addr = ${r} * in_flags_stride + ${c};
        local_flags.arr[${lr}][${lc}] = in_flags[addr];
    </%transpose:transpose_load>

    BARRIER();

    // Combine with output and baseline flags
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int addr = ${r} * out_flags_stride + ${c};
        out_flags[addr] |= local_flags.arr[${lr}][${lc}] | baseline_flags[${r}];
    </%transpose:transpose_store>
}
