#version 450

layout(location = 0) in vec2 v_pos;
layout(location = 0) out vec4 outColor;

layout(push_constant, std140) uniform PushConstants {
    vec4 dstRect;
    vec2 screenSize;
    float depth;
    float thickness;
    vec4 color;
    vec4 cornerRadius;
} params;

float rounding_alpha(vec2 coords, vec2 size, vec4 radius) {
    vec2 center;
    float r;

    if (coords.x < radius.x && coords.y < radius.x) {
        r = radius.x;
        center = vec2(r, r);
    } else if (size.x - radius.y < coords.x && coords.y < radius.y) {
        r = radius.y;
        center = vec2(size.x - r, r);
    } else if (size.x - radius.z < coords.x && size.y - radius.z < coords.y) {
        r = radius.z;
        center = vec2(size.x - r, size.y - r);
    } else if (coords.x < radius.w && size.y - radius.w < coords.y) {
        r = radius.w;
        center = vec2(r, size.y - r);
    } else {
        return 1.0;
    }

    if (r <= 0.0) {
        return 1.0;
    }

    float dist = distance(coords, center);
    return 1.0 - smoothstep(r - 0.5, r + 0.5, dist);
}

void main() {
    vec2 size = params.dstRect.zw;
    vec2 coords = v_pos * size;

    float outer_alpha = rounding_alpha(coords, size, params.cornerRadius);
    float inner_alpha = 1.0;

    if (params.thickness > 0.0) {
        vec2 inner_coords = coords - vec2(params.thickness);
        vec2 inner_size = size - vec2(params.thickness * 2.0);
        if (inner_size.x > 0.0 && inner_size.y > 0.0
            && inner_coords.x >= 0.0 && inner_coords.x <= inner_size.x
            && inner_coords.y >= 0.0 && inner_coords.y <= inner_size.y)
        {
            vec4 inner_radius = max(params.cornerRadius - vec4(params.thickness), vec4(0.0));
            inner_alpha = 1.0 - rounding_alpha(inner_coords, inner_size, inner_radius);
        }
    }

    float eff_alpha = outer_alpha * inner_alpha;
    if (eff_alpha <= 0.0001) {
        discard;
    }

    outColor = params.color * eff_alpha;
}
