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
    uint pad0;
    ivec2 offset;
    ivec4 damage[4];
} params;

vec2 applyTransform(vec2 uv, uint transform) {
    switch (transform) {
        case 0:
        return uv;
        case 1: // 90
        return vec2(1.0-uv.y, uv.x);
        case 2: // 180
        return vec2(1.0-uv.x, 1.0-uv.y);
        case 3: // 270
        return vec2(uv.y, 1.0-uv.x);
        case 4: // Flipped
        return vec2(1.0-uv.x, uv.y);
        case 5: // Flipped 90
        return vec2(1.0-uv.y, 1.0-uv.x);
        case 6: // Flipped 180
        return vec2(uv.x, 1.0-uv.y);
        case 7: // Flipped 270
        return vec2(uv.y, uv.x);
    }
}

void main() {
    ivec2 signedCoord = ivec2(gl_GlobalInvocationID.xy) + params.offset;
    if (signedCoord.x < 0 || signedCoord.y < 0)
        return;
    uvec2 coord = uvec2(signedCoord);
    uvec2 outSize = imageSize(dst);

    if (coord.x >= outSize.x || coord.y >= outSize.y)
        return;

    if (coord.x < params.dstRect.x || coord.x >= (params.dstRect.x + params.dstRect.z) || coord.y < params.dstRect.y || coord.y >= (params.dstRect.y + params.dstRect.w))
        return;

    bool in_damage = false;
    for (int i = 0; i < params.damageSize; i++) {
        ivec4 rect = params.damage[i];
        if (coord.x >= rect.x && coord.x < (rect.x + rect.z) &&
            coord.y >= rect.y && coord.y < (rect.y + rect.w))
        {
            in_damage = true;
            break;
        }
    }
    if (!in_damage)
        return;

    uvec2 texSize = textureSize(tex, 0);
    vec4 raw;
    bool is1to1 = (params.srcTransform == 0) &&
                  (abs(params.srcRect.z - params.dstRect.z) < 0.001) &&
                  (abs(params.srcRect.w - params.dstRect.w) < 0.001) &&
                  (abs(params.srcRect.x - floor(params.srcRect.x)) < 0.001) &&
                  (abs(params.srcRect.y - floor(params.srcRect.y)) < 0.001);
    if (is1to1) {
        ivec2 srcCoord = ivec2(coord) - ivec2(round(params.dstRect.xy)) + ivec2(round(params.srcRect.xy));
        srcCoord = clamp(srcCoord, ivec2(0), ivec2(texSize) - ivec2(1));
        raw = texelFetch(tex, srcCoord, 0);
    } else {
        vec2 dstUV = (vec2(coord) + vec2(0.5) - params.dstRect.xy) / params.dstRect.zw;
        vec2 srcUV = ((dstUV * params.srcRect.zw) + params.srcRect.xy) / vec2(texSize);
        vec2 uv = applyTransform(srcUV, params.srcTransform);
        raw = texture(tex, uv);
    }
    if (params.hasAlpha == 0) {
        raw.a = 1.0;
    }
    vec4 color = raw * params.alpha;

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
}
