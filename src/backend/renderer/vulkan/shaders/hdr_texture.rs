use ash::vk::{DescriptorPoolSize, DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags};
use bytemuck::NoUninit;
use include_bytes_aligned::include_bytes_aligned;
use std::sync::LazyLock;

use crate::utils::{Buffer, Physical, Rectangle};

pub const HDR_TEX_SHADER: &[u8] =
    include_bytes_aligned!(32, concat!(env!("OUT_DIR"), "/vk/hdr_texture.glsl"));
pub static HDR_TEX_BINDINGS: LazyLock<[DescriptorSetLayoutBinding<'static>; 2]> = LazyLock::new(|| {
    [
        DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(DescriptorType::STORAGE_IMAGE)
            .stage_flags(ShaderStageFlags::COMPUTE)
            .descriptor_count(1),
        DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(ShaderStageFlags::COMPUTE)
            .descriptor_count(1),
    ]
});
pub static HDR_TEX_SIZES: LazyLock<[DescriptorPoolSize; 2]> = LazyLock::new(|| {
    [
        DescriptorPoolSize::default().ty(DescriptorType::STORAGE_IMAGE),
        DescriptorPoolSize::default().ty(DescriptorType::COMBINED_IMAGE_SAMPLER),
    ]
});

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, NoUninit)]
pub struct HdrTexPushConstants {
    pub src_rect: Rectangle<f32, Buffer>,
    pub dst_rect: Rectangle<f32, Physical>,
    pub src_transform: u32,
    pub alpha: f32,
    pub damage_size: u32,
    pub is_bgr: u32,
    pub has_alpha: u32,
    pub reference_white: f32,
    pub sdr_gamma: f32,
    pub gamut_stretch: f32,
    pub max_content_luminance: f32,
    pub max_destination_luminance: f32,
    pub hardware_offload: u32,
    pub target_is_sdr: u32,
    pub input_is_pq: u32,
    pub input_is_hlg: u32,
    pub input_primaries: u32,
    pub skip_color_transform: u32,
    pub content_reference: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub damage: [Rectangle<i32, Physical>; 4],
}
