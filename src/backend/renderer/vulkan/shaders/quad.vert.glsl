#version 450
#extension GL_ARB_shader_draw_parameters : enable

layout(location = 0) out vec2 v_pos;

layout(push_constant, std140) uniform PushConstants {
    vec4 dstRect;
    vec2 screenSize;
    float depth;
    float _pad;
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
    float depthOffset = 0.0;
#ifdef GL_ARB_shader_draw_parameters
    depthOffset = float(gl_BaseInstanceARB) * 0.0 + float(gl_DrawIDARB) * 0.0;
#endif
    gl_Position = vec4(ndc, params.depth + depthOffset, 1.0);
}
