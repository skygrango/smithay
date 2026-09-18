#version 450

layout(location = 0) in vec2 v_pos;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D tex;
layout(binding = 1) uniform sampler3D lut3d;

// Specialization constants:
// 0: Generic (runtime checks via push constants)
// 1: Passthrough (skipColorTransform = 1)
// 2: SdrToHdr (SDR sRGB/Rec709 -> HDR BT.2020)
// 3: PqToHdr (PQ BT.2020 -> HDR BT.2020 with tone mapping)
// 4: Lut3d (Hardware 3D LUT sampling)
layout(constant_id = 0) const uint SPEC_MODE = 0u;

layout(push_constant, std140) uniform PushConstants {
    vec4 dstRect;
    vec2 screenSize;
    float depth;
    float _pad0;
    vec4 srcRect;
    uint srcTransform;
    float alpha;
    uint hasAlpha;
    float referenceWhite;
    float sdrGamma;
    float gamutStretch;
    float maxContentLuminance;
    float maxDestinationLuminance;
    float contentReference;
    uint flags;
    vec2 _pad1;
    vec4 clipRect;
    vec4 cornerRadius;
} params;

bool flag_hardware_offload()     { return (params.flags & (1u << 0)) != 0u; }
bool flag_target_is_sdr()        { return (params.flags & (1u << 1)) != 0u; }
bool flag_input_is_pq()          { return (params.flags & (1u << 2)) != 0u; }
bool flag_input_is_hlg()         { return (params.flags & (1u << 3)) != 0u; }
uint flag_input_primaries()      { return (params.flags >> 4u) & 3u; }
bool flag_skip_color_transform() { return (params.flags & (1u << 6)) != 0u; }

float get_clip_alpha() {
    if (params.clipRect.z <= 0.0 && params.cornerRadius == vec4(0.0)) {
        return 1.0;
    }
    vec2 pixel = params.dstRect.xy + v_pos * params.dstRect.zw;
    vec2 coords = pixel - params.clipRect.xy;
    vec2 size = params.clipRect.zw;

    if (coords.x < 0.0 || coords.x > size.x || coords.y < 0.0 || coords.y > size.y) {
        return 0.0;
    }

    if (params.cornerRadius == vec4(0.0)) {
        return 1.0;
    }

    vec2 center;
    float radius;

    if (coords.x < params.cornerRadius.x && coords.y < params.cornerRadius.x) {
        radius = params.cornerRadius.x;
        center = vec2(radius, radius);
    } else if (size.x - params.cornerRadius.y < coords.x && coords.y < params.cornerRadius.y) {
        radius = params.cornerRadius.y;
        center = vec2(size.x - radius, radius);
    } else if (size.x - params.cornerRadius.z < coords.x && size.y - params.cornerRadius.z < coords.y) {
        radius = params.cornerRadius.z;
        center = vec2(size.x - radius, size.y - radius);
    } else if (coords.x < params.cornerRadius.w && size.y - params.cornerRadius.w < coords.y) {
        radius = params.cornerRadius.w;
        center = vec2(radius, size.y - radius);
    } else {
        return 1.0;
    }

    float dist = distance(coords, center);
    return 1.0 - smoothstep(radius - 0.5, radius + 0.5, dist);
}

// Linear Rec.709 to linear BT.2020. Column-major GLSL constructor.
const mat3 rec709_to_bt2020 = mat3(
    0.6274040, 0.0690970, 0.0163916,
    0.3292820, 0.9195400, 0.0880132,
    0.0433136, 0.0113612, 0.8955950
);

// Linear BT.2020 to linear Rec.709. Column-major GLSL constructor.
const mat3 bt2020_to_rec709 = mat3(
     1.6604903, -0.1245500, -0.0181511,
    -0.5876391,  1.1328999, -0.1005787,
    -0.0728516, -0.0083480,  1.1187299
);

// Display P3 linear to linear BT.2020. Column-major GLSL constructor.
const mat3 p3_to_bt2020 = mat3(
    0.7538330,  0.0457438, -0.0012103,
    0.1985974,  0.9417772,  0.0176017,
    0.0475696,  0.0124789,  0.9836086
);

// ITU-R BT.2100 / SMPTE RP 2092 ICtCp color space matrices.
const mat3 to_ictcp = mat3(
    0.5,  1.613769531250,   4.378173828125,
    0.5, -3.323486328125, -4.245605468750,
    0.0,  1.709716796875, -0.132568359375
);

const mat3 from_ictcp = mat3(
    1.0,               1.0,               1.0,
    0.00860903703793, -0.00860903703793,  0.56003133571068,
    0.11102962500303, -0.11102962500303, -0.32062717498732
);

// BT.2020 linear RGB to Dolby LMS (for ICtCp).
const mat3 bt2020_to_lms = mat3(
    0.412109375, 0.166748046875, 0.024169921875,
    0.52392578125, 0.720458984375, 0.075439453125,
    0.06396484375, 0.11279296875, 0.900390625
);

const mat3 lms_to_bt2020 = mat3(
    3.43660669, -0.79132956, -0.02594990,
    -2.50645212, 1.98360045, -0.09891371,
    0.06984542, -0.19227090, 1.12486361
);

const float pq_m1 = 0.1593017578125;
const float pq_m2 = 78.84375;
const float pq_c1 = 0.8359375;
const float pq_c2 = 18.8515625;
const float pq_c3 = 18.6875;

float encode_pq(float value) {
    float p = pow(clamp(value, 0.0, 1.0), pq_m1);
    return pow((pq_c1 + pq_c2 * p) / (1.0 + pq_c3 * p), pq_m2);
}

vec3 encode_pq_v(vec3 value) {
    vec3 p = pow(clamp(value, vec3(0.0), vec3(1.0)), vec3(pq_m1));
    return pow((vec3(pq_c1) + pq_c2 * p) / (vec3(1.0) + pq_c3 * p), vec3(pq_m2));
}

float pq_to_linear(float code) {
    float p = pow(clamp(code, 0.0, 1.0), 1.0 / pq_m2);
    return pow(max(p - pq_c1, 0.0) / (pq_c2 - pq_c3 * p), 1.0 / pq_m1);
}

vec3 pq_to_linear_v(vec3 code) {
    vec3 p = pow(clamp(code, vec3(0.0), vec3(1.0)), vec3(1.0 / pq_m2));
    return pow(max(p - vec3(pq_c1), vec3(0.0)) / (vec3(pq_c2) - pq_c3 * p), vec3(1.0 / pq_m1));
}

float hlg_to_scene(float e) {
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    e = clamp(e, 0.0, 1.0);
    if (e <= 0.5) {
        return (e * e) / 3.0;
    } else {
        return (exp((e - c) / a) + b) / 12.0;
    }
}

vec3 decode_sdr_v(vec3 value, float gamma) {
    if (gamma == 1.0) {
        return value;
    }
    if (gamma > 0.0) {
        return pow(max(value, vec3(0.0)), vec3(gamma));
    }
    vec3 low = value / 12.92;
    vec3 high = pow((value + 0.055) / 1.055, vec3(2.4));
    vec3 cutoff = step(vec3(0.04045), value);
    return mix(low, high, cutoff);
}

vec3 encode_sdr_v(vec3 value, float gamma) {
    if (gamma == 1.0) {
        return value;
    }
    if (gamma > 0.0) {
        return pow(max(value, vec3(0.0)), vec3(1.0 / gamma));
    }
    vec3 low = value * 12.92;
    vec3 high = 1.055 * pow(max(value, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    vec3 cutoff = step(vec3(0.0031308), value);
    return mix(low, high, cutoff);
}

// Tone-mapping in ICtCp space following KWin's color management pipeline
vec3 tonemap_ictcp(vec3 linear_10k) {
    if (params.maxContentLuminance <= params.maxDestinationLuminance * 1.01) {
        return clamp(linear_10k, vec3(0.0), vec3(params.maxDestinationLuminance / 10000.0));
    }

    vec3 lms = bt2020_to_lms * linear_10k;
    vec3 lms_pq = encode_pq_v(lms);
    vec3 ictcp = to_ictcp * lms_pq;

    // Luminance in nits
    float lum = pq_to_linear(ictcp.r) * 10000.0;

    // Modified Reinhard roll-off matching KWin
    float ref_white = clamp(params.referenceWhite, 80.0, 10000.0);
    float rel_lum = max(lum / ref_white, 0.0);
    float in_range = params.maxContentLuminance / ref_white;
    float out_range = params.maxDestinationLuminance / ref_white;
    float v = (out_range * (1.0 + in_range) - in_range) / (in_range * in_range);
    rel_lum = rel_lum * (1.0 + rel_lum * v) / (1.0 + rel_lum);
    lum = rel_lum * ref_white;

    ictcp.r = encode_pq(lum / 10000.0);
    vec3 mapped_lms_pq = from_ictcp * ictcp;
    vec3 mapped_lms = pq_to_linear_v(mapped_lms_pq);
    vec3 mapped_rgb = lms_to_bt2020 * mapped_lms;
    return clamp(mapped_rgb, vec3(0.0), vec3(params.maxDestinationLuminance / 10000.0));
}

vec3 source_to_linear_10k(vec3 raw_rgb) {
    if (SPEC_MODE == 3u || (SPEC_MODE == 0u && flag_input_is_pq())) {
        vec3 linear_10k = pq_to_linear_v(raw_rgb);
        float ref_scale = clamp(params.referenceWhite, 80.0, 10000.0) / max(params.contentReference, 80.0);
        linear_10k *= ref_scale;
        return tonemap_ictcp(linear_10k);
    } else if (SPEC_MODE == 0u && flag_input_is_hlg()) {
        vec3 scene = vec3(
            hlg_to_scene(raw_rgb.r),
            hlg_to_scene(raw_rgb.g),
            hlg_to_scene(raw_rgb.b)
        );
        float ys = dot(scene, vec3(0.2627, 0.6780, 0.0593));
        float gain = pow(max(ys, 1e-6), 0.2) * 0.10;
        float ref_scale = clamp(params.referenceWhite, 80.0, 10000.0) / max(params.contentReference, 80.0);
        vec3 display = scene * (gain * ref_scale);
        return tonemap_ictcp(display);
    } else {
        vec3 linear_input = decode_sdr_v(raw_rgb, params.sdrGamma);
        vec3 linear_bt2020;
        if (SPEC_MODE == 0u && flag_input_primaries() == 1u) {
            linear_bt2020 = p3_to_bt2020 * linear_input;
        } else if (SPEC_MODE == 0u && flag_input_primaries() == 2u) {
            linear_bt2020 = linear_input;
        } else {
            linear_bt2020 = mix(rec709_to_bt2020 * linear_input, linear_input, clamp(params.gamutStretch, 0.0, 1.0));
        }
        linear_bt2020 = max(linear_bt2020, vec3(0.0));
        float ref_white = clamp(params.referenceWhite, 80.0, 10000.0);
        vec3 linear_10k = linear_bt2020 * (ref_white / 10000.0);
        return linear_10k;
    }
}

vec2 applyTransform(vec2 uv, uint transform) {
    switch (transform) {
        case 0:
            return uv;
        case 1: // 90
            return vec2(1.0 - uv.y, uv.x);
        case 2: // 180
            return vec2(1.0 - uv.x, 1.0 - uv.y);
        case 3: // 270
            return vec2(uv.y, 1.0 - uv.x);
        case 4: // Flipped
            return vec2(1.0 - uv.x, uv.y);
        case 5: // Flipped 90
            return vec2(1.0 - uv.y, 1.0 - uv.x);
        case 6: // Flipped 180
            return vec2(uv.x, 1.0 - uv.y);
        case 7: // Flipped 270
            return vec2(uv.y, uv.x);
    }
    return uv;
}

float optical_alpha_pq(float a, float ref_white, vec3 rgb) {
    if (flag_target_is_sdr() || flag_hardware_offload()) {
        return a;
    }
    if (a <= 0.0001) return 0.0;
    if (a >= 0.9999) return 1.0;
    float white_norm = clamp(ref_white, 80.0, 10000.0) / 10000.0;
    float pq_white = encode_pq(white_norm);

    // Drop shadows and absorbing occluders (rgb == 0) attenuate ambient background
    // light rather than emitting optical light. In non-linear PQ framebuffers,
    // hardware blending with ONE_MINUS_SRC_ALPHA performs:
    //   E_dst_new = E_dst * (1.0 - alpha_blend)
    // To match the physical luminance attenuation (1 - a)^gamma of SDR blending:
    //   E_dst * (1.0 - alpha_pq) = PQ(L_bg * (1 - a)^gamma)
    // Solving for alpha_pq gives:
    //   alpha_pq = 1.0 - PQ(white_norm * (1 - a)^gamma) / pq_white
    // This perfectly preserves the smooth Gaussian gradient falloff to zero and
    // prevents shadow darkening distortion on HDR displays.
    if (max(rgb.r, max(rgb.g, rgb.b)) < 0.001) {
        float gamma = params.sdrGamma > 0.0 ? params.sdrGamma : 2.2;
        float atten = pow(clamp(1.0 - a, 0.0, 1.0), gamma);
        float pq_atten = encode_pq(white_norm * atten);
        return clamp(1.0 - (pq_atten / max(pq_white, 0.001)), 0.0, 1.0);
    }
    float pq_lum = encode_pq(white_norm * a);
    return clamp(pq_lum / max(pq_white, 0.001), 0.0, 1.0);
}

// Tetrahedral interpolation decomposes each cube cell into 6 tetrahedra along the diagonal (R=G=B).
// This guarantees that the neutral axis (gray ramp) is strictly preserved without hue shifts or
// trilinear interpolation cubic artifacts in wide-gamut HDR spaces.
vec3 tetrahedral_sample(sampler3D lut, vec3 color, float lut_size) {
    vec3 p = clamp(color, 0.0, 1.0) * (lut_size - 1.0);
    ivec3 p0 = ivec3(floor(p));
    vec3 f = p - vec3(p0);
    int max_coord = int(lut_size) - 1;
    ivec3 p1 = min(p0 + ivec3(1), ivec3(max_coord));
    p0 = min(p0, ivec3(max_coord));

    vec3 c000 = texelFetch(lut, p0, 0).rgb;
    vec3 c111 = texelFetch(lut, p1, 0).rgb;
    vec3 c1, c2;

    if (f.r >= f.g) {
        if (f.g >= f.b) {
            // f.r >= f.g >= f.b (Tetrahedron 1)
            c1 = texelFetch(lut, ivec3(p1.x, p0.y, p0.z), 0).rgb;
            c2 = texelFetch(lut, ivec3(p1.x, p1.y, p0.z), 0).rgb;
            return c000 * (1.0 - f.r) + c1 * (f.r - f.g) + c2 * (f.g - f.b) + c111 * f.b;
        } else if (f.r >= f.b) {
            // f.r >= f.b > f.g (Tetrahedron 2)
            c1 = texelFetch(lut, ivec3(p1.x, p0.y, p0.z), 0).rgb;
            c2 = texelFetch(lut, ivec3(p1.x, p0.y, p1.z), 0).rgb;
            return c000 * (1.0 - f.r) + c1 * (f.r - f.b) + c2 * (f.b - f.g) + c111 * f.g;
        } else {
            // f.b > f.r >= f.g (Tetrahedron 5)
            c1 = texelFetch(lut, ivec3(p0.x, p0.y, p1.z), 0).rgb;
            c2 = texelFetch(lut, ivec3(p1.x, p0.y, p1.z), 0).rgb;
            return c000 * (1.0 - f.b) + c1 * (f.b - f.r) + c2 * (f.r - f.g) + c111 * f.g;
        }
    } else {
        if (f.b >= f.g) {
            // f.b >= f.g > f.r (Tetrahedron 6)
            c1 = texelFetch(lut, ivec3(p0.x, p0.y, p1.z), 0).rgb;
            c2 = texelFetch(lut, ivec3(p0.x, p1.y, p1.z), 0).rgb;
            return c000 * (1.0 - f.b) + c1 * (f.b - f.g) + c2 * (f.g - f.r) + c111 * f.r;
        } else if (f.b >= f.r) {
            // f.g > f.b >= f.r (Tetrahedron 4)
            c1 = texelFetch(lut, ivec3(p0.x, p1.y, p0.z), 0).rgb;
            c2 = texelFetch(lut, ivec3(p0.x, p1.y, p1.z), 0).rgb;
            return c000 * (1.0 - f.g) + c1 * (f.g - f.b) + c2 * (f.b - f.r) + c111 * f.r;
        } else {
            // f.g > f.r > f.b (Tetrahedron 3)
            c1 = texelFetch(lut, ivec3(p0.x, p1.y, p0.z), 0).rgb;
            c2 = texelFetch(lut, ivec3(p1.x, p1.y, p0.z), 0).rgb;
            return c000 * (1.0 - f.g) + c1 * (f.g - f.r) + c2 * (f.r - f.b) + c111 * f.b;
        }
    }
}

void main() {
    float clip_a = get_clip_alpha();
    if (clip_a <= 0.0) {
        discard;
    }

    uvec2 texSize = textureSize(tex, 0);
    vec2 srcUV = ((v_pos * params.srcRect.zw) + params.srcRect.xy) / vec2(texSize);
    vec2 uv = applyTransform(srcUV, params.srcTransform);
    vec4 raw = texture(tex, uv);
    if (params.hasAlpha == 0) {
        raw.a = 1.0;
    }

    float src_a = raw.a;
    vec3 raw_rgb = src_a > 0.00001 ? raw.rgb / src_a : vec3(0.0);
    float eff_alpha = clamp(src_a * params.alpha * clip_a, 0.0, 1.0);
    if (eff_alpha <= 0.0001) {
        discard;
    }

    bool skip_transform = (SPEC_MODE == 1u) || (SPEC_MODE == 0u && flag_skip_color_transform());
    if (skip_transform) {
        if (flag_target_is_sdr()) {
            outColor = vec4(raw_rgb * eff_alpha, eff_alpha);
            return;
        }
        float opt_a = optical_alpha_pq(eff_alpha, params.referenceWhite, raw_rgb);
        outColor = vec4(raw_rgb * opt_a, opt_a);
        return;
    }

    if (SPEC_MODE == 4u) {
        vec3 mapped_rgb = tetrahedral_sample(lut3d, raw_rgb, 33.0);
        if (flag_target_is_sdr()) {
            outColor = vec4(mapped_rgb * eff_alpha, eff_alpha);
        } else {
            float opt_a = optical_alpha_pq(eff_alpha, params.referenceWhite, mapped_rgb);
            outColor = vec4(mapped_rgb * opt_a, opt_a);
        }
        return;
    }

    vec3 src_linear_10k = source_to_linear_10k(raw_rgb);

    if (flag_target_is_sdr()) {
        float inv_white = 10000.0 / clamp(params.referenceWhite, 80.0, 10000.0);
        vec3 linear_sdr_bt2020 = src_linear_10k * inv_white;
        vec3 rec709 = clamp(bt2020_to_rec709 * linear_sdr_bt2020, vec3(0.0), vec3(1.0));
        vec3 out_sdr = encode_sdr_v(rec709, params.sdrGamma);
        outColor = vec4(out_sdr * eff_alpha, eff_alpha);
        return;
    }

    if (flag_hardware_offload()) {
        float ref_white = clamp(params.referenceWhite, 80.0, 10000.0);
        vec3 src_linear = src_linear_10k * (10000.0 / ref_white);
        outColor = vec4(src_linear * eff_alpha, eff_alpha);
        return;
    }

    vec3 out_pq = encode_pq_v(src_linear_10k);
    float opt_a = optical_alpha_pq(eff_alpha, params.referenceWhite, raw_rgb);
    outColor = vec4(out_pq * opt_a, opt_a);
}
