use ash::vk::{DescriptorPoolSize, DescriptorSetLayoutBinding};
use bytemuck::NoUninit;
use include_bytes_aligned::include_bytes_aligned;
use std::sync::LazyLock;

use crate::utils::{Physical, Rectangle};

pub const OUTLINE_SHADER: &[u8] =
    include_bytes_aligned!(32, concat!(env!("OUT_DIR"), "/vk/outline.frag.glsl"));

pub static OUTLINE_BINDINGS: LazyLock<[DescriptorSetLayoutBinding<'static>; 0]> = LazyLock::new(|| []);
pub static OUTLINE_SIZES: LazyLock<[DescriptorPoolSize; 0]> = LazyLock::new(|| []);

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, NoUninit)]
pub struct OutlinePushConstants {
    pub dst_rect: Rectangle<f32, Physical>,
    pub screen_size: [f32; 2],
    pub depth: f32,
    pub thickness: f32,
    pub color: [f32; 4],
    pub corner_radius: [f32; 4],
}
