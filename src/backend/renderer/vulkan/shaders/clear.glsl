#version 450
#extension GL_EXT_shader_image_load_formatted : enable

layout(
local_size_x = 8,
local_size_y = 8,
local_size_z = 1
) in;

layout(binding = 0) uniform image2D dst;
layout(push_constant, std140) uniform PushConstants {
    vec4 color;
    uint blend;
    uint rectSize;
    uint isBgr;
    uint _padding0;
    ivec2 offset;
    uvec2 _padding1;
    ivec4 rects[5];
} params;

void main() {
    ivec2 signedCoord = ivec2(gl_GlobalInvocationID.xy) + params.offset;
    if (signedCoord.x < 0 || signedCoord.y < 0)
        return;
    uvec2 coord = uvec2(signedCoord);
    uvec2 outSize = imageSize(dst);

    if (coord.x >= outSize.x || coord.y >= outSize.y)
        return;

    for (int i = 0; i < params.rectSize; i++) {
        ivec4 rect = params.rects[i];

        if (coord.x >= rect.x && coord.x < (rect.x + rect.z)
                && coord.y >= rect.y && coord.y < (rect.y + rect.w))
        {
            vec4 color = params.color;
            if (params.blend != 0 && color.a < 1.0) {
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
    }
}
