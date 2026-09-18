#version 450

layout(location = 0) in vec2 v_pos;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D tex;

layout(push_constant, std140) uniform PushConstants {
    vec4 dstRect;
    vec2 screenSize;
    float depth;
    float _pad0;
    vec4 srcRect;
    uint srcTransform;
    float alpha;
    uint hasAlpha;
    uint _pad1;
    vec4 clipRect;
    vec4 cornerRadius;
} params;

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
    outColor = raw * (params.alpha * clip_a);
}
