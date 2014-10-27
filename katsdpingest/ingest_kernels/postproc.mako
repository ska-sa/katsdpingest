<%include file="/port.mako"/>

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1) void postproc(
    GLOBAL float2 * RESTRICT vis,
    GLOBAL float * RESTRICT weights,
    const GLOBAL unsigned char * RESTRICT flags,
    GLOBAL float2 * RESTRICT cont_vis,
    GLOBAL float * RESTRICT cont_weights,
    GLOBAL unsigned char * RESTRICT cont_flags,
    int vis_stride,
    int weights_stride,
    int flags_stride,
    int cont_vis_stride,
    int cont_weights_stride,
    int cont_flags_stride)
{
    const int C = ${cont_factor};
    int baseline = get_global_id(0);
    int cont_channel = get_global_id(1);
    int channel0 = cont_channel * C;

    float2 cv = make_float2(0.0f, 0.0f);
    float cw = 0.0f;
    unsigned char cf = 0xff;
    for (int i = 0; i < C; i++)
    {
        int channel = channel0 + i;
        GLOBAL float2 *vptr = &vis[channel * vis_stride + baseline];
        float2 v = *vptr;
        GLOBAL float *wptr = &weights[channel * weights_stride + baseline];
        float w = *wptr;
        unsigned char f = flags[channel * flags_stride + baseline];
        cv.x += v.x;
        cv.y += v.y;
        cw += w;
        cf &= f;
        float scale = 1.0f / w;
        if (f)
            *wptr = 0.0f;
        v.x *= scale;
        v.y *= scale;
        *vptr = v;
    }

    float scale = 1.0 / cw;
    cv.x *= scale;
    cv.y *= scale;
    if (cf)
        cw = 0.0f;
    cont_vis[cont_channel * cont_vis_stride + baseline] = cv;
    cont_weights[cont_channel * cont_weights_stride + baseline] = cw;
    cont_flags[cont_channel * cont_flags_stride + baseline] = cf;
}
