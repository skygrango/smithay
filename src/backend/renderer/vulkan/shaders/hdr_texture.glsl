#version 450
#extension GL_EXT_shader_image_load_formatted : enable

layout(
    local_size_x = 8,
    local_size_y = 8,
    local_size_z = 1
) in;

layout(binding = 0) uniform image2D dst;
layout(binding = 1) uniform sampler2D tex;

layout(push_constant, std140) uniform PushConstants {
    vec4 srcRect;
    vec4 dstRect;
    uint srcTransform;
    float alpha;
    uint damageSize;
    uint isBgr;

    uint hasAlpha;
    float referenceWhite;
    float sdrGamma;
    float gamutStretch;

    float maxContentLuminance;
    float maxDestinationLuminance;
    uint hardwareOffload;
    uint targetIsSdr;

    uint inputIsPq;
    uint inputIsHlg;
    uint inputPrimaries;
    uint skipColorTransform;

    float contentReference;
    uint _pad0;
    uint _pad1;
    uint _pad2;

    ivec4 damage[4];
} params;

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
    if (params.inputIsPq != 0) {
        vec3 linear_10k = pq_to_linear_v(raw_rgb);
        float ref_scale = clamp(params.referenceWhite, 80.0, 10000.0) / max(params.contentReference, 80.0);
        linear_10k *= ref_scale;
        return tonemap_ictcp(linear_10k);
    } else if (params.inputIsHlg != 0) {
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
        if (params.inputPrimaries == 1) {
            linear_bt2020 = p3_to_bt2020 * linear_input;
        } else if (params.inputPrimaries == 2) {
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
}

void main() {
    uvec2 coord = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

    if (coord.x < params.dstRect.x || coord.x >= (params.dstRect.x + params.dstRect.z) ||
        coord.y < params.dstRect.y || coord.y >= (params.dstRect.y + params.dstRect.w))
        return;

    uvec2 outSize = imageSize(dst);
    uvec2 texSize = textureSize(tex, 0);
    vec2 dstUV = (vec2(coord) - params.dstRect.xy) / params.dstRect.zw;
    vec2 srcUV = ((dstUV * params.srcRect.zw) + params.srcRect.xy) / texSize;
    vec2 uv = applyTransform(srcUV, params.srcTransform);

    for (int i = 0; i < params.damageSize; i++) {
        ivec4 rect = params.damage[i];

        if (coord.x >= rect.x && coord.x < (rect.x + rect.z) &&
            coord.y >= rect.y && coord.y < (rect.y + rect.w))
        {
            vec4 raw = texture(tex, uv);
            if (params.hasAlpha == 0) {
                raw.a = 1.0;
            }

            float src_a = raw.a;
            vec3 raw_rgb = src_a > 0.00001 ? raw.rgb / src_a : vec3(0.0);
            float eff_alpha = clamp(src_a * params.alpha, 0.0, 1.0);

            // Conditional conversion: If source color matches destination target color space,
            // avoid unnecessary transforms!
            if (params.skipColorTransform != 0) {
                if (params.targetIsSdr != 0) {
                    vec4 color = vec4(raw_rgb * eff_alpha, eff_alpha);
                    if (color.a < 1.0) {
                        vec4 dstColor = imageLoad(dst, ivec2(coord));
                        if (params.isBgr != 0) {
                            dstColor = dstColor.bgra;
                        }
                        color = color + dstColor * (1.0 - color.a);
                    }
                    if (params.isBgr != 0) {
                        color = color.bgra;
                    }
                    imageStore(dst, ivec2(coord), color);
                    break;
                }

                // Native HDR: buffer is already BT.2020 PQ
                vec3 out_pq;
                if (eff_alpha >= 0.9999) {
                    out_pq = raw_rgb;
                } else {
                    vec4 dstColor = imageLoad(dst, ivec2(coord));
                    vec3 dst_pq = (params.isBgr != 0) ? dstColor.bgr : dstColor.rgb;
                    vec3 dst_linear_10k = pq_to_linear_v(dst_pq);
                    vec3 src_linear_10k = pq_to_linear_v(raw_rgb);
                    vec3 blended_linear = src_linear_10k * eff_alpha + dst_linear_10k * (1.0 - eff_alpha);
                    out_pq = encode_pq_v(blended_linear);
                }

                vec4 final_color = vec4(params.isBgr != 0 ? out_pq.bgr : out_pq, 1.0);
                imageStore(dst, ivec2(coord), final_color);
                break;
            }

            // Colorspace transform required: Convert to linear light in target primaries
            vec3 src_linear_10k = source_to_linear_10k(raw_rgb);

            if (params.targetIsSdr != 0) {
                float inv_white = 10000.0 / clamp(params.referenceWhite, 80.0, 10000.0);
                vec3 linear_sdr_bt2020 = src_linear_10k * inv_white;
                vec3 rec709 = clamp(bt2020_to_rec709 * linear_sdr_bt2020, vec3(0.0), vec3(1.0));
                vec3 out_sdr = encode_sdr_v(rec709, params.sdrGamma);

                vec4 color = vec4(out_sdr * eff_alpha, eff_alpha);
                if (color.a < 1.0) {
                    vec4 dstColor = imageLoad(dst, ivec2(coord));
                    if (params.isBgr != 0) {
                        dstColor = dstColor.bgra;
                    }
                    color = color + dstColor * (1.0 - color.a);
                }
                if (params.isBgr != 0) {
                    color = color.bgra;
                }
                imageStore(dst, ivec2(coord), color);
                break;
            }

            if (params.hardwareOffload != 0) {
                // In hardware offload mode, destination buffer stores normalized linear light
                // where 1.0 = reference_white (CRTC GAMMA_LUT will encode to PQ).
                float ref_white = clamp(params.referenceWhite, 80.0, 10000.0);
                vec3 src_linear = src_linear_10k * (10000.0 / ref_white);

                vec3 out_rgb;
                if (eff_alpha >= 0.9999) {
                    out_rgb = src_linear;
                } else {
                    vec4 dstColor = imageLoad(dst, ivec2(coord));
                    vec3 dst_rgb = (params.isBgr != 0) ? dstColor.bgr : dstColor.rgb;
                    out_rgb = src_linear * eff_alpha + dst_rgb * (1.0 - eff_alpha);
                }

                vec4 final_color = vec4(params.isBgr != 0 ? out_rgb.bgr : out_rgb, 1.0);
                imageStore(dst, ivec2(coord), final_color);
                break;
            }

            // Native HDR mode: destination buffer stores ST 2084 PQ encoded code values.
            vec3 out_pq;
            if (eff_alpha >= 0.9999) {
                out_pq = encode_pq_v(src_linear_10k);
            } else {
                // Linear light blending: decode dst PQ to linear light, blend in linear space, encode back to PQ.
                vec4 dstColor = imageLoad(dst, ivec2(coord));
                vec3 dst_pq = (params.isBgr != 0) ? dstColor.bgr : dstColor.rgb;
                vec3 dst_linear_10k = pq_to_linear_v(dst_pq);
                vec3 blended_linear = src_linear_10k * eff_alpha + dst_linear_10k * (1.0 - eff_alpha);
                out_pq = encode_pq_v(blended_linear);
            }

            vec4 final_color = vec4(params.isBgr != 0 ? out_pq.bgr : out_pq, 1.0);
            imageStore(dst, ivec2(coord), final_color);
            break;
        }
    }
}
