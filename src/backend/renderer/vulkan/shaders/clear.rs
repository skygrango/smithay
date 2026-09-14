use ash::vk::{DescriptorPoolSize, DescriptorSetLayoutBinding};
use bytemuck::NoUninit;
use include_bytes_aligned::include_bytes_aligned;
use std::sync::LazyLock;

use crate::utils::{Physical, Rectangle};

pub const CLEAR_SHADER: &[u8] = include_bytes_aligned!(32, concat!(env!("OUT_DIR"), "/vk/clear.frag.glsl"));

pub static CLEAR_BINDINGS: LazyLock<[DescriptorSetLayoutBinding<'static>; 0]> = LazyLock::new(|| []);
pub static CLEAR_SIZES: LazyLock<[DescriptorPoolSize; 0]> = LazyLock::new(|| []);

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, NoUninit)]
pub struct ClearPushConstants {
    pub dst_rect: Rectangle<f32, Physical>,
    pub screen_size: [f32; 2],
    pub _pad: [f32; 2],
    pub color: [f32; 4],
}
