#version 450

layout(location = 0) out vec2 v_pos;

layout(push_constant, std140) uniform PushConstants {
    vec4 dstRect;
    vec2 screenSize;
} params;

const vec2 positions[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    v_pos = pos;
    vec2 pixelPos = params.dstRect.xy + pos * params.dstRect.zw;
    vec2 ndc = (pixelPos / params.screenSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
