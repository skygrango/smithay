#version 450

layout(location = 0) in vec2 v_pos;
layout(location = 0) out vec4 outColor;

layout(push_constant, std140) uniform PushConstants {
    vec4 dstRect;
    vec2 screenSize;
    vec2 _pad;
    vec4 color;
} params;

void main() {
    outColor = params.color;
}
