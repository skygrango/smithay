use ash::vk::{DescriptorPoolSize, DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags};
use bytemuck::NoUninit;
use include_bytes_aligned::include_bytes_aligned;
use std::sync::LazyLock;

use crate::utils::{Buffer, Physical, Rectangle};

pub const TEX_SHADER: &[u8] = include_bytes_aligned!(32, concat!(env!("OUT_DIR"), "/vk/texture.frag.glsl"));
pub static TEX_BINDINGS: LazyLock<[DescriptorSetLayoutBinding<'static>; 1]> = LazyLock::new(|| {
    [DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(ShaderStageFlags::FRAGMENT)
        .descriptor_count(1)]
});
pub static TEX_SIZES: LazyLock<[DescriptorPoolSize; 1]> = LazyLock::new(|| {
    [DescriptorPoolSize::default()
        .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)]
});

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, NoUninit)]
pub struct TexPushConstants {
    pub dst_rect: Rectangle<f32, Physical>,
    pub screen_size: [f32; 2],
    pub _pad0: [f32; 2],
    pub src_rect: Rectangle<f32, Buffer>,
    pub src_transform: u32,
    pub alpha: f32,
    pub has_alpha: u32,
    pub _pad1: u32,
}
