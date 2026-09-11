use crate::backend::{
    allocator::Fourcc,
    renderer::Texture,
    vulkan::{format::get_drm_format, image::VulkanImage},
};

impl Texture for VulkanImage {
    fn width(&self) -> u32 {
        VulkanImage::width(self)
    }

    fn height(&self) -> u32 {
        VulkanImage::height(self)
    }

    fn format(&self) -> Option<Fourcc> {
        self.drm
            .map(|format| format.code)
            .or_else(|| get_drm_format(self.format()))
    }
}
