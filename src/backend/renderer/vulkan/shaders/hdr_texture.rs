use ash::vk::{DescriptorPoolSize, DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags};
use bytemuck::NoUninit;
use include_bytes_aligned::include_bytes_aligned;
use std::sync::LazyLock;

use crate::utils::{Buffer, Physical, Rectangle};

pub const HDR_TEX_SHADER: &[u8] =
    include_bytes_aligned!(32, concat!(env!("OUT_DIR"), "/vk/hdr_texture.frag.glsl"));
pub static HDR_TEX_BINDINGS: LazyLock<[DescriptorSetLayoutBinding<'static>; 1]> = LazyLock::new(|| {
    [DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(ShaderStageFlags::FRAGMENT)
        .descriptor_count(1)]
});
pub static HDR_TEX_SIZES: LazyLock<[DescriptorPoolSize; 1]> = LazyLock::new(|| {
    [DescriptorPoolSize::default()
        .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)]
});

pub const SPEC_MODE_GENERIC: u32 = 0;
pub const SPEC_MODE_PASSTHROUGH: u32 = 1;
pub const SPEC_MODE_SDR: u32 = 2;
pub const SPEC_MODE_PQ: u32 = 3;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, NoUninit)]
pub struct HdrTexPushConstants {
    pub dst_rect: Rectangle<f32, Physical>,
    pub screen_size: [f32; 2],
    pub _pad0: [f32; 2],
    pub src_rect: Rectangle<f32, Buffer>,
    pub src_transform: u32,
    pub alpha: f32,
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
    pub _pad1: u32,
}
