<%include file="/port.mako"/>

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
            *fptr = 0;
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
        cf = 0;
% endif
    int cont_addr = cont_channel * stride + baseline;
    cont_vis[cont_addr] = cv;
    cont_weights[cont_addr] = cw;
    cont_flags[cont_addr] = cf;
% endif
}
