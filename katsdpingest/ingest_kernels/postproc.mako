<%include file="/port.mako"/>

/* If there was partial excision and the sum of the non-excised visibilities is
 * zero, the visibility will come out infinitesimal rather than zero due to the
 * 2^-64 scaling trick. If it's an auto-correlation, it may get inverted later
 * to compute statistical weights, which leads to numerical issues, so we flush
 * it to zero.
 *
 * The minimum non-zero absolute value for the weighted sum is 1/n_accs, while
 * the largest possible spurious value is 2^-33 * (m-1)/n_accs, where m is the
 * number of input dumps added together (potentially somewhat large for
 * continuum) and n_accs is the number of accumulations in the correlator.
 * If one needs to support a very large range of n_accs then it should be
 * an extra parameter (or scaling by n_accs should be delayed until this
 * stage), but 2e-9 should be safe for all reasonable cases for now.
 */
DEVICE_FN float flush_zero(float x)
{
    return fabsf(x) < 2e-9f ? 0.0f : x;
}

DEVICE_FN float2 flush_zero2(float2 vis)
{
    return make_float2(flush_zero(vis.x), flush_zero(vis.y));
}

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void postproc(
    GLOBAL float2 * RESTRICT vis,
    GLOBAL float * RESTRICT weights,
    GLOBAL unsigned char * RESTRICT flags,
% if continuum:
    GLOBAL float2 * RESTRICT cont_vis,
    GLOBAL float * RESTRICT cont_weights,
    GLOBAL unsigned char * RESTRICT cont_flags,
    int cont_factor,
% endif
    int stride)
{
% if not continuum:
    const int cont_factor = 1;
% endif
    int baseline = get_global_id(0);
    int cont_channel = get_global_id(1);
    int channel0 = cont_channel * cont_factor;

% if continuum:
    float2 cv;
    cv.x = 0.0f;
    cv.y = 0.0f;
    float cw = 0.0f;
    unsigned char cf = 0;
% endif
#pragma unroll 4
    for (int i = 0; i < cont_factor; i++)
    {
        int channel = channel0 + i;
        int addr = channel * stride + baseline;
        GLOBAL float2 *vptr = &vis[addr];
        float2 v = *vptr;
        GLOBAL float *wptr = &weights[addr];
        float w = *wptr;
        GLOBAL unsigned char *fptr = &flags[addr];
        unsigned char f = *fptr;
% if continuum:
        cv.x += v.x;
        cv.y += v.y;
        cw += w;
        cf |= f;
% endif
        float scale = 1.0f / w;
% if excise:
        if (!(f & ${unflagged_bit}))
            *wptr = 1.8446744e19f * w;  // scale by 2^64, to compensate for previous 2^-64
        else
        {
            *fptr = 0;
            v = flush_zero2(v);
        }
% endif
        v.x *= scale;
        v.y *= scale;
        *vptr = v;
    }

% if continuum:
    float scale = 1.0 / cw;
    cv.x *= scale;
    cv.y *= scale;
% if excise:
    if (!(cf & ${unflagged_bit}))
        cw *= 1.8446744e19;     // scale by 2^64, to compensate for previous 2^-64
    else
    {
        cf = 0;
        cv = flush_zero2(cv);
    }
% endif
    int cont_addr = cont_channel * stride + baseline;
    cont_vis[cont_addr] = cv;
    cont_weights[cont_addr] = cw;
    cont_flags[cont_addr] = cf;
% endif
}
