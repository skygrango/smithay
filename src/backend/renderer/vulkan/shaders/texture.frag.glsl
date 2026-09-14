#version 450

layout(location = 0) in vec2 v_pos;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D tex;

layout(push_constant, std140) uniform PushConstants {
    vec4 dstRect;
    vec2 screenSize;
    vec2 _pad0;
    vec4 srcRect;
    uint srcTransform;
    float alpha;
    uint hasAlpha;
    uint _pad1;
} params;

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

void main() {
    uvec2 texSize = textureSize(tex, 0);
    vec4 raw;
    bool is1to1 = (params.srcTransform == 0) &&
                  (abs(params.srcRect.z - params.dstRect.z) < 0.001) &&
                  (abs(params.srcRect.w - params.dstRect.w) < 0.001) &&
                  (abs(params.srcRect.x - floor(params.srcRect.x)) < 0.001) &&
                  (abs(params.srcRect.y - floor(params.srcRect.y)) < 0.001);
    if (is1to1) {
        ivec2 coord = ivec2(gl_FragCoord.xy);
        ivec2 srcCoord = coord - ivec2(round(params.dstRect.xy)) + ivec2(round(params.srcRect.xy));
        srcCoord = clamp(srcCoord, ivec2(0), ivec2(texSize) - ivec2(1));
        raw = texelFetch(tex, srcCoord, 0);
    } else {
        vec2 srcUV = ((v_pos * params.srcRect.zw) + params.srcRect.xy) / vec2(texSize);
        vec2 uv = applyTransform(srcUV, params.srcTransform);
        raw = texture(tex, uv);
    }
    if (params.hasAlpha == 0) {
        raw.a = 1.0;
    }
    outColor = raw * params.alpha;
}
