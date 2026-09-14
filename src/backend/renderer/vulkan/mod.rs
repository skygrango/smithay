//! Implementation of the rendering traits using Vulkan

use crate::{
    backend::{
        allocator::{
            Format, Fourcc,
            dmabuf::{Dmabuf, WeakDmabuf},
            format::FormatSet,
        },
        drm::{DrmDeviceFd, sync::DrmSyncPoint},
        renderer::{
            Bind, ContextId, ExportMem, Frame, ImportDma, ImportMem, Renderer, RendererSuper, Texture,
            TextureMapping,
            vulkan::shaders::{ClearPushConstants, HdrTexPushConstants, TexPushConstants},
        },
        vulkan::{
            PhysicalDevice, UnsupportedProperty,
            device::{Device, DeviceError, QueueType, WeakDevice},
            format::{get_drm_format, get_vk_format, known_formats},
            image::{Error as ImageError, ImageInner, ImageUsageFlags, VulkanImage},
            version::Version,
        },
    },
    reexports::drm::node::DrmNode,
    utils::{Buffer as BufferCoords, Physical, Point, Rectangle, Size, Transform},
};

use ash::vk::{
    self, AccessFlags2, BorderColor, CommandBufferSubmitInfo, CompareOp, DependencyInfo, DescriptorImageInfo,
    DescriptorType, Extent3D, Fence, Filter, FormatFeatureFlags, HostImageCopyFlagsEXT, ImageAspectFlags,
    ImageLayout, ImageMemoryBarrier2, ImageSubresourceLayers, ImageSubresourceRange, ImageToMemoryCopyEXT,
    MemoryMapFlags, MemoryPropertyFlags, MemoryToImageCopyEXT, Offset3D, PipelineBindPoint,
    PipelineStageFlags2, QUEUE_FAMILY_IGNORED, Result as VkResult, SamplerAddressMode, SamplerCreateFlags,
    SamplerCreateInfo, SamplerMipmapMode, SemaphoreSubmitInfo, SemaphoreWaitInfo, ShaderStageFlags,
    SubmitInfo2,
};
use gbm::Modifier;
use indexmap::IndexSet;

use std::{collections::HashMap, ffi::CStr, fmt, ptr::NonNull, sync::Arc};

use super::{Blit, BlitFrame, Color32F, HdrOutputConfig, TextureFilter, sdr_color_to_hdr, sync::SyncPoint};
use tracing::trace;

//mod buffer;
mod capabilities;
mod cmds;
mod device;
mod image;
mod shaders;
mod sync;

pub use self::capabilities::*;
pub use self::shaders::Error as PipelineError;
use self::shaders::Pipelines;
pub use self::sync::*;

#[derive(Debug)]
pub struct VulkanRenderer {
    capabilities: Vec<Capability>,
    dmabuf_cache: HashMap<WeakDmabuf, VulkanImage>,

    pipelines: Pipelines,
    cmd_pool: cmds::CommandPool,
    texture_sampler: vk::Sampler,

    seq_no: u64,
    timeline: sync::VulkanTimeline,
    pub(crate) node: Option<DrmNode>,

    debug_flags: super::DebugFlags,
    downscale_filter: super::TextureFilter,
    upscale_filter: super::TextureFilter,
    pub(crate) hdr_config: Option<HdrOutputConfig>,
    pub(crate) supports_optimal_host_copy: bool,

    // A bunch of the previous structs contain Weak-device references.
    // So we want to drop this last for proper cleanup and avoiding accidental
    // resource leaks.
    pub(crate) device: Device,
    pub(crate) phd: PhysicalDevice,
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.vk().device_wait_idle();
            self.cmd_pool.clean_old_buffers(u64::MAX);
            self.dmabuf_cache.clear();
            self.device.vk().destroy_sampler(self.texture_sampler, None);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unsupported version")]
    UnsupportedVersion,
    #[error("Provided drm device doesn't match PhysicalDevice")]
    MismatchedDrmDevice,
    #[error("Missing device feature: `{0}`")]
    MissingFeature(&'static str),
    #[error("Missing extension: `{0:?}`")]
    MissingExtension(&'static CStr),
    #[error(transparent)]
    DeviceError(#[from] DeviceError),
    #[error("Failed host image transition")]
    HostImageTransitionError(#[source] VkResult),
    #[error("Failed to copy from host memory to an image")]
    HostImageCopyError(#[source] VkResult),
    #[error(transparent)]
    ImageError(#[from] ImageError),
    #[error(transparent)]
    PipelineError(#[from] PipelineError),
    #[error("Failed to create vulkan command pool")]
    CommandPoolError(#[source] VkResult),
    #[error("Failed to create vulkan command buffer")]
    CommandBufferError(#[source] VkResult),
    #[error("Failed to create an image sampler")]
    SamplerError(#[source] VkResult),
    #[error("Failed to create vulkan semaphore")]
    SemaphoreError(#[source] VkResult),
    #[error("Failed to query semaphore counter value: `{0:?}`")]
    SemaphoreCounterError(#[source] VkResult),
    #[error("Failed to export semaphore fd: `{0:?}`")]
    SemaphoreExportError(#[source] VkResult),
    #[error("Failed to submit command buffer")]
    SubmitError(#[source] VkResult),
    #[error("Error intefacing with the underlying drm device")]
    DrmError(#[source] std::io::Error),
    #[error("Underlying Vulkan Device was destroyed")]
    DeadDevice,
    #[cfg(feature = "wayland_frontend")]
    #[error("Unsupported wl_shm pixel format: `{0:?}`")]
    UnsupportedWlPixelFormat(wayland_server::protocol::wl_shm::Format),
    #[error("Buffer access error")]
    BufferAccessError,
    #[error("Unsupported pixel format")]
    UnsupportedPixelFormat,
    #[error("GLES error: {0}")]
    GlesError(String),
}

impl VulkanRenderer {
    /// Maximum supported version instance version that may be used with the renderer.
    pub const MIN_DEVICE_VERSION: Version = Version::VERSION_1_3;
    pub const MAX_INSTANCE_VERSION: Version = Version::VERSION_1_3;

    //#[instrument(err, skip(phd), fields(physical_device = phd.name()))]
    pub fn new(phd: &PhysicalDevice, drm: Option<DrmDeviceFd>) -> Result<VulkanRenderer, Error> {
        // Check matching drm descriptor, if provided
        let node = if let Some(fd) = drm.as_ref() {
            let node = DrmNode::from_file(fd).map_err(|_| Error::MismatchedDrmDevice)?;
            let dev_render_node = node
                .node_with_type(crate::backend::drm::NodeType::Render)
                .and_then(Result::ok);
            let dev_primary_node = node
                .node_with_type(crate::backend::drm::NodeType::Primary)
                .and_then(Result::ok);
            let matches = phd
                .render_node()
                .ok()
                .flatten()
                .is_some_and(|n| n == node || Some(n) == dev_render_node)
                || phd
                    .primary_node()
                    .ok()
                    .flatten()
                    .is_some_and(|n| n == node || Some(n) == dev_primary_node);
            if !matches {
                return Err(Error::MismatchedDrmDevice);
            }
            Some(node)
        } else {
            phd.render_node().ok().flatten()
        };

        // Check instance version
        if phd.api_version() < Self::MIN_DEVICE_VERSION
            || phd.instance().api_version() > Self::MAX_INSTANCE_VERSION
        {
            return Err(Error::UnsupportedVersion);
        }

        // Check features
        let supported_features = Features::supported_features(phd);
        if let Err(feat) = supported_features.has_required_features() {
            return Err(Error::MissingFeature(feat));
        }
        let mut required_features = Features::required_features();

        // Check optional extensions
        let mut capabilities = Vec::new();
        capabilities.extend(Capability::supports_dmabuf_memory(phd));
        capabilities.extend(Capability::supports_export_timeline(phd));
        capabilities.extend(Capability::supports_host_image_copy(phd));
        capabilities.extend(Capability::supports_queue_family_foreign(phd));

        // Get extensions
        let extensions = Capability::as_extensions(&capabilities);

        // Create device
        let device = Device::new(
            phd,
            &extensions,
            unsafe { required_features.vk() },
            QueueType::Compute,
            true,
        )?;

        // Create pipelines
        let pipelines = Pipelines::new(&device)?;

        // Create command pool
        let cmd_pool = device.create_command_pool()?;

        // Create timeline semaphore for our queue
        let timeline =
            device.create_timeline_semaphore(if capabilities.contains(&Capability::ExportTimeline) {
                drm
            } else {
                None
            })?;

        let sampler = unsafe {
            device
                .vk()
                .create_sampler(
                    &SamplerCreateInfo::default()
                        .flags(SamplerCreateFlags::empty())
                        .mag_filter(Filter::LINEAR) // TODO
                        .min_filter(Filter::LINEAR)
                        .mipmap_mode(SamplerMipmapMode::NEAREST)
                        .address_mode_u(SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_w(SamplerAddressMode::CLAMP_TO_BORDER)
                        .mip_lod_bias(0.0)
                        .anisotropy_enable(false)
                        .max_anisotropy(0.0)
                        .compare_enable(false)
                        .compare_op(CompareOp::NEVER)
                        .min_lod(0.0)
                        .max_lod(0.0)
                        .border_color(BorderColor::FLOAT_TRANSPARENT_BLACK)
                        .unnormalized_coordinates(false),
                    None,
                )
                .map_err(Error::SamplerError)?
        };

        let supports_optimal_host_copy = if capabilities.contains(&Capability::HostImageCopy) {
            let mut hic_props = vk::PhysicalDeviceHostImageCopyPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut hic_props);
            unsafe {
                phd.instance()
                    .handle()
                    .get_physical_device_properties2(phd.handle(), &mut props2)
            };
            if hic_props.identical_memory_type_requirements != 0 {
                true
            } else {
                VulkanImage::new_with_fourcc(
                    &device,
                    1,
                    1,
                    Fourcc::Abgr8888,
                    vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                    false,
                )
                .is_ok()
            }
        } else {
            false
        };
        tracing::debug!(
            supports_optimal_host_copy,
            "Vulkan host image copy capability evaluated"
        );

        Ok(VulkanRenderer {
            phd: phd.clone(),
            device,
            capabilities,
            dmabuf_cache: HashMap::new(),
            pipelines,
            cmd_pool,
            texture_sampler: sampler,
            seq_no: 0,
            node,
            timeline,
            debug_flags: super::DebugFlags::empty(),
            downscale_filter: super::TextureFilter::Linear,
            upscale_filter: super::TextureFilter::Linear,
            hdr_config: None,
            supports_optimal_host_copy,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn external_queue_family(&self) -> u32 {
        if self.capabilities.contains(&Capability::QueueFamilyForeign) {
            vk::QUEUE_FAMILY_FOREIGN_EXT
        } else {
            vk::QUEUE_FAMILY_EXTERNAL
        }
    }

    pub fn set_hdr_output(&mut self, config: Option<HdrOutputConfig>) {
        self.hdr_config = config;
    }

    pub fn hdr_output(&self) -> Option<HdrOutputConfig> {
        self.hdr_config
    }

    pub fn node(&self) -> Option<DrmNode> {
        self.node
    }

    pub fn cleanup(&mut self) -> Result<(), Error> {
        let val = match unsafe { self.device.vk().get_semaphore_counter_value(self.timeline.vk) } {
            Ok(val) => val,
            Err(vk::Result::ERROR_DEVICE_LOST) => return Err(Error::DeadDevice),
            Err(err) => return Err(Error::SemaphoreCounterError(err)),
        };
        self.cmd_pool.clean_old_buffers(val);
        Ok(())
    }

    fn upload_host_memory_to_image(
        &self,
        image: &VulkanImage,
        src_ptr: *const u8,
        src_stride: usize,
        bpp: usize,
        region: Rectangle<i32, BufferCoords>,
    ) -> Result<(), Error> {
        if bpp == 0 {
            return Err(Error::UnsupportedPixelFormat);
        }

        let image_rect = Rectangle::from_size((image.width as i32, image.height as i32).into());
        let Some(region) = region.intersection(image_rect) else {
            return Ok(());
        };
        if region.size.w <= 0 || region.size.h <= 0 {
            return Ok(());
        }

        let device_copy = self
            .device
            .vk_ext_host_image_copy()
            .ok_or(Error::MissingExtension(ash::ext::host_image_copy::NAME))?;

        if image.current_layout() != ImageLayout::GENERAL {
            unsafe {
                device_copy
                    .transition_image_layout(&[vk::HostImageLayoutTransitionInfoEXT::default()
                        .old_layout(image.current_layout())
                        .new_layout(ImageLayout::GENERAL)
                        .image(*image.vk())
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        )])
                    .map_err(Error::HostImageTransitionError)?;
            }
            image.set_current_layout(ImageLayout::GENERAL);
        }

        if image.is_linear() && image.mem_bits().contains(MemoryPropertyFlags::HOST_VISIBLE) {
            let subresource = vk::ImageSubresource::default()
                .aspect_mask(if image.tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT {
                    ImageAspectFlags::MEMORY_PLANE_0_EXT
                } else {
                    ImageAspectFlags::COLOR
                })
                .mip_level(0)
                .array_layer(0);

            let layout = unsafe {
                self.device
                    .vk()
                    .get_image_subresource_layout(*image.vk(), subresource)
            };

            let map_ptr = unsafe {
                self.device
                    .vk()
                    .map_memory(image.inner.memory, 0, vk::WHOLE_SIZE, MemoryMapFlags::empty())
                    .map_err(Error::HostImageCopyError)?
            };

            let dst_base = unsafe { (map_ptr as *mut u8).add(layout.offset as usize) };
            let dst_pitch = layout.row_pitch as usize;
            let copy_bytes_per_row = region.size.w as usize * bpp;

            unsafe {
                for r in 0..region.size.h as usize {
                    let y = region.loc.y as usize + r;
                    let x = region.loc.x as usize;
                    let src_row = src_ptr.add(y * src_stride + x * bpp);
                    let dst_row = dst_base.add(y * dst_pitch + x * bpp);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, copy_bytes_per_row);
                }

                if !image.mem_bits().contains(MemoryPropertyFlags::HOST_COHERENT) {
                    let range = vk::MappedMemoryRange::default()
                        .memory(image.inner.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    let _ = self.device.vk().flush_mapped_memory_ranges(&[range]);
                }

                self.device.vk().unmap_memory(image.inner.memory);
            }
        } else {
            let host_offset = (region.loc.y as usize * src_stride) + (region.loc.x as usize * bpp);
            let host_ptr = unsafe { src_ptr.add(host_offset) };

            unsafe {
                device_copy
                    .copy_memory_to_image(
                        &vk::CopyMemoryToImageInfoEXT::default()
                            .flags(HostImageCopyFlagsEXT::empty())
                            .dst_image(*image.vk())
                            .dst_image_layout(ImageLayout::GENERAL)
                            .regions(&[MemoryToImageCopyEXT::default()
                                .host_pointer(host_ptr as *const _)
                                .memory_row_length((src_stride / bpp) as u32)
                                .memory_image_height(0)
                                .image_subresource(
                                    ImageSubresourceLayers::default()
                                        .aspect_mask(ImageAspectFlags::COLOR)
                                        .mip_level(0)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                )
                                .image_offset(Offset3D {
                                    x: region.loc.x,
                                    y: region.loc.y,
                                    z: 0,
                                })
                                .image_extent(
                                    Extent3D::default()
                                        .depth(1)
                                        .width(region.size.w as u32)
                                        .height(region.size.h as u32),
                                )]),
                    )
                    .map_err(Error::HostImageCopyError)?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "wayland_frontend")]
    fn upload_shm_memory_to_image(
        &self,
        image: &VulkanImage,
        ptr: *const u8,
        stride: i32,
        region: Rectangle<i32, BufferCoords>,
    ) -> Result<(), Error> {
        let bpp = image
            .drm
            .map(|f| f.code)
            .or_else(|| get_drm_format(image.format()))
            .and_then(crate::backend::allocator::format::get_bpp)
            .unwrap_or(32)
            / 8;

        self.upload_host_memory_to_image(image, ptr, stride as usize, bpp, region)
    }
}

impl RendererSuper for VulkanRenderer {
    type Error = Error;
    type TextureId = VulkanImage;
    type Framebuffer<'buffer> = VulkanFramebuffer;

    type Frame<'frame, 'buffer>
        = VulkanFrame<'frame, 'buffer>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for VulkanRenderer {
    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.device.context()
    }

    fn downscale_filter(&mut self, filter: super::TextureFilter) -> Result<(), Self::Error> {
        self.downscale_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: super::TextureFilter) -> Result<(), Self::Error> {
        self.upscale_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: super::DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> super::DebugFlags {
        self.debug_flags
    }

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        Ok(VulkanFrame {
            renderer: self,
            fb: framebuffer,
            transform: dst_transform,
            size: output_size,
            cmd_buffer: None,
            descriptors: Vec::with_capacity(16),
            images: Vec::with_capacity(16),
            has_draws: false,
            _marker: std::marker::PhantomData,
            #[cfg(feature = "wayland_frontend")]
            active_color_description: None,
            is_blit: false,
        })
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        if let Some(drm_sync) = sync.get::<crate::backend::drm::sync::DrmSyncPoint>() {
            if let Some(timeline) = self.timeline.drm.as_ref() {
                if timeline == drm_sync.timeline() {
                    // On Vulkan, all submissions on this renderer queue already wait on the
                    // previous sequence number via self.timeline.vk.
                    // Since drm_sync.point() <= self.seq_no, GPU queue ordering is guaranteed
                    // without any CPU-side stall.
                    return Ok(());
                }
            }
        }
        while let Err(super::sync::Interrupted) = sync.wait() {}
        Ok(())
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        let _ = self.cleanup();
        self.dmabuf_cache.retain(|weak, _| weak.upgrade().is_some());
        Ok(())
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoords>,
        _flipped: bool, // TODO
    ) -> Result<Self::TextureId, Self::Error> {
        let bpp = crate::backend::allocator::format::get_bpp(format).unwrap_or(32) / 8;
        let stride = size.w as usize * bpp;
        if data.len() < stride * size.h as usize {
            return Err(Error::BufferAccessError);
        }

        let linear = !self.supports_optimal_host_copy;
        let image = VulkanImage::new_with_fourcc(
            &self.device,
            size.w as u32,
            size.h as u32,
            format,
            vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
            linear,
        )
        .or_else(|_| {
            if !linear {
                VulkanImage::new_with_fourcc(
                    &self.device,
                    size.w as u32,
                    size.h as u32,
                    format,
                    vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                    true,
                )
            } else {
                Err(ImageError::NoMemoryAvailable)
            }
        })
        .map_err(Error::ImageError)?;

        self.upload_host_memory_to_image(&image, data.as_ptr(), stride, bpp, Rectangle::from_size(size))?;

        Ok(image)
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, BufferCoords>,
    ) -> Result<(), Self::Error> {
        let bpp = texture
            .drm
            .map(|f| f.code)
            .or_else(|| get_drm_format(texture.format()))
            .and_then(crate::backend::allocator::format::get_bpp)
            .unwrap_or(32)
            / 8;

        let stride = texture.width() as usize * bpp;
        if region.loc.x < 0 || region.loc.y < 0 {
            return Err(Error::BufferAccessError);
        }
        let max_y = (region.loc.y + region.size.h) as usize;
        let max_x = (region.loc.x + region.size.w) as usize;
        if max_x > texture.width() as usize || max_y > texture.height() as usize {
            return Err(Error::BufferAccessError);
        }
        if (max_y - 1) * stride + max_x * bpp > data.len() {
            return Err(Error::BufferAccessError);
        }

        self.upload_host_memory_to_image(texture, data.as_ptr(), stride, bpp, region)
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new(
            vec![
                Fourcc::Abgr8888,
                Fourcc::Xbgr8888,
                Fourcc::Argb8888,
                Fourcc::Xrgb8888,
                // TODO
                /*
                Fourcc::Abgr2101010,
                Fourcc::Xbgr2101010,
                Fourcc::Abgr16161616f,
                Fourcc::Xbgr16161616f,
                */
            ]
            .into_iter(),
        )
    }
}

impl ImportDma for VulkanRenderer {
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoords>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        VulkanImage::new_from_dmabuf(
            &self.device,
            dmabuf,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_SRC,
        )
        .map_err(Error::ImageError)
    }

    fn dmabuf_formats(&self) -> FormatSet {
        self.device.formats().map(|entry| entry.format.clone()).collect()
    }

    fn has_dmabuf_format(&self, format: Format) -> bool {
        self.device.formats().any(|entry| entry.format == format)
    }
}

#[cfg(feature = "wayland_frontend")]
impl crate::backend::renderer::ImportMemWl for VulkanRenderer {
    #[profiling::function]
    fn import_shm_buffer(
        &mut self,
        buffer: &wayland_server::protocol::wl_buffer::WlBuffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoords>],
    ) -> Result<Self::TextureId, Self::Error> {
        use crate::wayland::shm::{shm_format_to_fourcc, with_buffer_contents};

        type CacheMap = HashMap<ContextId<VulkanImage>, VulkanImage>;

        let mut surface_lock = surface.as_ref().map(|surface_data| {
            surface_data
                .data_map
                .get_or_insert_threadsafe(|| std::sync::Arc::new(std::sync::Mutex::new(CacheMap::new())))
                .lock()
                .unwrap()
        });

        with_buffer_contents(buffer, |ptr, len, data| {
            let offset = data.offset;
            let width = data.width;
            let height = data.height;
            let stride = data.stride;
            let fourcc =
                shm_format_to_fourcc(data.format).ok_or(Error::UnsupportedWlPixelFormat(data.format))?;

            if !self.mem_formats().any(|f| f == fourcc) {
                return Err(Error::UnsupportedWlPixelFormat(data.format));
            }

            let expected_len = (offset + stride * height) as usize;
            if len < expected_len {
                return Err(Error::BufferAccessError);
            }

            let size = (width, height).into();
            let id = self.context_id();
            let expected_vk_format = get_vk_format(fourcc).unwrap_or(vk::Format::UNDEFINED);
            let cached_texture = surface_lock
                .as_ref()
                .and_then(|cache| cache.get(&id).cloned())
                .filter(|texture| texture.size() == size && texture.format() == expected_vk_format);

            let base_ptr = unsafe { ptr.offset(offset as isize) };

            let texture = if let Some(texture) = cached_texture {
                if damage.is_empty() {
                    self.upload_shm_memory_to_image(&texture, base_ptr, stride, Rectangle::from_size(size))?;
                } else {
                    let buffer_rect = Rectangle::from_size(size);
                    for region in damage.iter().filter_map(|r| r.intersection(buffer_rect)) {
                        self.upload_shm_memory_to_image(&texture, base_ptr, stride, region)?;
                    }
                }
                texture
            } else {
                let linear = !self.supports_optimal_host_copy;
                let image = VulkanImage::new_with_fourcc(
                    &self.device,
                    width as u32,
                    height as u32,
                    fourcc,
                    vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                    linear,
                )
                .or_else(|_| {
                    if !linear {
                        VulkanImage::new_with_fourcc(
                            &self.device,
                            width as u32,
                            height as u32,
                            fourcc,
                            vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                            true,
                        )
                    } else {
                        Err(ImageError::NoMemoryAvailable)
                    }
                })
                .map_err(Error::ImageError)?;

                self.upload_shm_memory_to_image(&image, base_ptr, stride, Rectangle::from_size(size))?;

                if let Some(cache) = surface_lock.as_mut() {
                    cache.insert(id, image.clone());
                }
                image
            };
            Ok(texture)
        })
        .map_err(|_| Error::BufferAccessError)?
    }

    fn shm_formats(&self) -> Box<dyn Iterator<Item = wayland_server::protocol::wl_shm::Format>> {
        Box::new(
            self.mem_formats()
                .filter_map(crate::wayland::shm::fourcc_to_shm_format),
        )
    }
}

#[cfg(feature = "wayland_frontend")]
impl crate::backend::renderer::ImportDmaWl for VulkanRenderer {}

pub enum VulkanMapping {
    Mapped(NonNull<u8>, usize, VulkanImage, WeakDevice),
    Copied(Vec<u8>, VulkanImage),
}

impl fmt::Debug for VulkanMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mapped(_, _, image, _) => f
                .debug_struct("VulkanMapping::Mapped")
                .field("image", image)
                .finish_non_exhaustive(),
            Self::Copied(_, image) => f
                .debug_struct("VulkanMapping::Copied")
                .field("image", image)
                .finish_non_exhaustive(),
        }
    }
}

impl Drop for VulkanMapping {
    fn drop(&mut self) {
        if let VulkanMapping::Mapped(_, _, image, dev) = self {
            if let Some(device) = dev.upgrade() {
                unsafe { device.vk().unmap_memory(image.inner.memory) };
            }
        }
    }
}

impl TextureMapping for VulkanMapping {
    fn flipped(&self) -> bool {
        false
    }
}

impl Texture for VulkanMapping {
    fn width(&self) -> u32 {
        let (Self::Mapped(_, _, image, _) | Self::Copied(_, image)) = self;
        image.width()
    }

    fn height(&self) -> u32 {
        let (Self::Mapped(_, _, image, _) | Self::Copied(_, image)) = self;
        image.height()
    }

    fn format(&self) -> Option<Fourcc> {
        let (Self::Mapped(_, _, image, _) | Self::Copied(_, image)) = self;
        get_drm_format(image.format())
    }
}

impl VulkanRenderer {
    fn copy_from_image(
        &mut self,
        image: &VulkanImage,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<VulkanMapping, Error> {
        if image.mem_bits().contains(MemoryPropertyFlags::HOST_VISIBLE) && image.is_linear() {
            let layout = unsafe {
                self.device.vk().get_image_subresource_layout(
                    *image.vk(),
                    vk::ImageSubresource {
                        aspect_mask: if image.tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT {
                            ImageAspectFlags::MEMORY_PLANE_0_EXT
                        } else {
                            ImageAspectFlags::COLOR
                        },
                        mip_level: 0,
                        array_layer: 0,
                    },
                )
            };

            let ptr = unsafe {
                self.device
                    .vk()
                    .map_memory(
                        image.inner.memory,
                        layout.offset,
                        layout.size,
                        MemoryMapFlags::empty(),
                    )
                    .map_err(Error::HostImageCopyError)? // TODO
            };

            Ok(VulkanMapping::Mapped(
                unsafe { NonNull::new_unchecked(ptr as *mut _) },
                layout.size as usize,
                image.clone(),
                self.device.downgrade(),
            ))
        } else {
            use ash::ext::host_image_copy;

            let device_copy = self
                .device
                .vk_ext_host_image_copy()
                .ok_or(Error::MissingExtension(host_image_copy::NAME))?;
            unsafe {
                if self.seq_no > 0 {
                    let _ = self.device.vk().wait_semaphores(
                        &ash::vk::SemaphoreWaitInfo::default()
                            .semaphores(&[self.timeline.vk])
                            .values(&[self.seq_no]),
                        10_000_000_000,
                    );
                }
                device_copy
                    .transition_image_layout(&[vk::HostImageLayoutTransitionInfoEXT::default()
                        .old_layout(image.current_layout())
                        .new_layout(ImageLayout::GENERAL)
                        .image(*image.vk())
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        )])
                    .map_err(Error::HostImageTransitionError)?;

                let bpp = 4usize;
                let copy_width = (region.size.w as u32).min(image.width()).max(1);
                let copy_height = (region.size.h as u32).min(image.height()).max(1);
                let offset_x = region.loc.x.max(0);
                let offset_y = region.loc.y.max(0);
                let stride = copy_width as usize * bpp;
                let size = stride * copy_height as usize;
                let mut data = vec![0u8; size];
                device_copy
                    .copy_image_to_memory(
                        &vk::CopyImageToMemoryInfoEXT::default()
                            .flags(HostImageCopyFlagsEXT::empty())
                            .src_image(*image.vk())
                            .src_image_layout(ImageLayout::GENERAL)
                            .regions(&[ImageToMemoryCopyEXT::default()
                                .host_pointer(data.as_mut_ptr() as *mut _)
                                .memory_image_height(copy_height)
                                .memory_row_length(copy_width)
                                .image_subresource(
                                    ImageSubresourceLayers::default()
                                        .aspect_mask(ImageAspectFlags::COLOR)
                                        .mip_level(0)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                )
                                .image_offset(Offset3D {
                                    x: offset_x,
                                    y: offset_y,
                                    z: 0,
                                })
                                .image_extent(
                                    Extent3D::default().depth(1).width(copy_width).height(copy_height),
                                )]),
                    )
                    .map_err(Error::HostImageCopyError)?;

                Ok(VulkanMapping::Copied(data, image.clone()))
            }
        }
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanMapping;

    fn copy_framebuffer(
        &mut self,
        target: &Self::Framebuffer<'_>,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        self.copy_from_image(&target.0, region, format)
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        self.copy_from_image(texture, region, format)
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        Ok(
            (texture.mem_bits().contains(MemoryPropertyFlags::HOST_VISIBLE) && texture.is_linear())
                || self.device.vk_ext_host_image_copy().is_some(),
        )
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        match texture_mapping {
            VulkanMapping::Mapped(ptr, len, _, _) => {
                Ok(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) })
            }
            VulkanMapping::Copied(data, _) => Ok(&**data),
        }
    }
}

#[derive(Debug)]
pub struct VulkanFramebuffer(VulkanImage);

impl Texture for VulkanFramebuffer {
    fn width(&self) -> u32 {
        Texture::width(&self.0)
    }

    fn height(&self) -> u32 {
        Texture::height(&self.0)
    }

    fn format(&self) -> Option<gbm::Format> {
        Texture::format(&self.0)
    }

    fn size(&self) -> Size<i32, BufferCoords> {
        Texture::size(&self.0)
    }
}

fn is_bgr_format(_format: vk::Format) -> bool {
    false
}

pub struct VulkanFrame<'frame, 'buffer> {
    renderer: &'frame mut VulkanRenderer,
    fb: &'frame mut VulkanFramebuffer,
    _marker: std::marker::PhantomData<&'buffer ()>,

    transform: Transform,
    size: Size<i32, Physical>,
    cmd_buffer: Option<ash::vk::CommandBuffer>,
    descriptors: Vec<shaders::DescriptorSet>,
    images: Vec<Arc<ImageInner>>,
    has_draws: bool,
    #[cfg(feature = "wayland_frontend")]
    active_color_description: Option<crate::wayland::color::management::ImageDescription>,
    is_blit: bool,
}

impl VulkanFrame<'_, '_> {
    fn get_or_create_cmd_buffer(&mut self) -> Result<ash::vk::CommandBuffer, Error> {
        if let Some(buf) = self.cmd_buffer {
            Ok(buf)
        } else {
            self.renderer.cleanup()?;
            let buf = self.renderer.cmd_pool.create_and_begin_buffer()?;
            self.cmd_buffer = Some(buf);
            Ok(buf)
        }
    }
}

impl Frame for VulkanFrame<'_, '_> {
    type Error = Error;
    type TextureId = VulkanImage;

    #[cfg(feature = "wayland_frontend")]
    fn set_surface_color_description(
        &mut self,
        desc: Option<&crate::wayland::color::management::ImageDescription>,
    ) {
        self.active_color_description = desc.cloned();
    }

    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.renderer.device.context()
    }

    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        self.draw_color(Rectangle::from_size(self.size), at, color, false)
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        self.draw_color(dst, damage, color, true)
    }

    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        trace!(?src, ?dst, damage_len = damage.len(), alpha, fb_size = ?self.size, "VulkanFrame::render_texture_from_to executing");
        if damage.is_empty() {
            return Ok(());
        }
        let untransformed_dst = dst;
        let dst = self.transform.transform_rect_in(dst, &self.size);

        let is_hdr = !self.is_blit && self.renderer.hdr_config.is_some_and(|c| !c.is_sdr);
        let buf = self.get_or_create_cmd_buffer()?;
        self.has_draws = true;
        if !self.images.iter().any(|img| Arc::ptr_eq(img, &texture.inner)) {
            self.images.push(texture.inner.clone());
        }
        let descriptor = if is_hdr {
            self.renderer
                .pipelines
                .alloc_descriptor_set(shaders::BuiltinShader::HdrTexture)?
        } else {
            self.renderer
                .pipelines
                .alloc_descriptor_set(shaders::BuiltinShader::Texture)?
        };

        // SAFETY: If we were able to bind it, it has a view
        let view = self.fb.0.vk_view().unwrap();
        let fb_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::GENERAL)
            .image_view(*view)];

        let view = texture
            .vk_view()
            .ok_or(Error::ImageError(ImageError::MissingOrInvalidUsage))?;
        let tex_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(*view)
            .sampler(self.renderer.texture_sampler)];

        let descriptor_update = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor.vk())
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(DescriptorType::STORAGE_IMAGE)
                .image_info(&fb_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor.vk())
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&tex_image_info),
        ];
        let (layout, pipeline) = if is_hdr {
            (
                *self.renderer.pipelines.hdr_tex_pipeline_layout(),
                *self.renderer.pipelines.hdr_tex_pipeline(),
            )
        } else {
            (
                *self.renderer.pipelines.tex_pipeline_layout(),
                *self.renderer.pipelines.tex_pipeline(),
            )
        };

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let fb_needs_acquire = self.fb.0.needs_acquire();
        let (fb_src_queue, fb_dst_queue) = if fb_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let fb_old_layout = self.fb.0.current_layout();
        let (fb_src_stage, fb_src_access) = if fb_needs_acquire {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if fb_old_layout == ImageLayout::UNDEFINED {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else {
            (
                PipelineStageFlags2::COMPUTE_SHADER,
                AccessFlags2::SHADER_STORAGE_WRITE,
            )
        };

        let tex_needs_acquire = texture.needs_acquire() && texture.dmabuf_exportable();
        let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let tex_old_layout = texture.current_layout();
        let (tex_src_stage, tex_src_access) = if tex_needs_acquire {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if tex_old_layout == ImageLayout::UNDEFINED {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if tex_old_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            (
                PipelineStageFlags2::COMPUTE_SHADER,
                AccessFlags2::SHADER_SAMPLED_READ,
            )
        } else {
            (
                PipelineStageFlags2::COMPUTE_SHADER
                    | PipelineStageFlags2::ALL_TRANSFER
                    | PipelineStageFlags2::HOST,
                AccessFlags2::SHADER_STORAGE_WRITE | AccessFlags2::TRANSFER_WRITE | AccessFlags2::HOST_WRITE,
            )
        };

        let fb_barrier = ImageMemoryBarrier2::default()
            .image(*self.fb.0.vk())
            .old_layout(fb_old_layout)
            .new_layout(ImageLayout::GENERAL)
            .src_queue_family_index(fb_src_queue)
            .dst_queue_family_index(fb_dst_queue)
            .src_stage_mask(fb_src_stage)
            .src_access_mask(fb_src_access)
            .dst_stage_mask(PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(AccessFlags2::SHADER_STORAGE_READ | AccessFlags2::SHADER_STORAGE_WRITE)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        let needs_tex_barrier = tex_needs_acquire || tex_old_layout != ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let tex_barrier = ImageMemoryBarrier2::default()
            .image(*texture.vk())
            .old_layout(tex_old_layout)
            .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(tex_src_queue)
            .dst_queue_family_index(tex_dst_queue)
            .src_stage_mask(tex_src_stage)
            .src_access_mask(tex_src_access)
            .dst_stage_mask(PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(AccessFlags2::SHADER_SAMPLED_READ)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.renderer
                .device
                .vk()
                .update_descriptor_sets(&descriptor_update, &[]);
            self.renderer
                .device
                .vk()
                .cmd_bind_pipeline(buf, PipelineBindPoint::COMPUTE, pipeline);
            self.renderer.device.vk().cmd_bind_descriptor_sets(
                buf,
                PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[descriptor.vk()],
                &[],
            );
            if needs_tex_barrier {
                self.renderer.device.vk().cmd_pipeline_barrier2(
                    buf,
                    &DependencyInfo::default().image_memory_barriers(&[fb_barrier, tex_barrier]),
                );
            } else {
                self.renderer.device.vk().cmd_pipeline_barrier2(
                    buf,
                    &DependencyInfo::default().image_memory_barriers(&[fb_barrier]),
                );
            }
        }

        self.fb.0.set_current_layout(ImageLayout::GENERAL);
        self.fb.0.set_needs_acquire(false);
        texture.set_current_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        texture.set_needs_acquire(false);

        let src_rect = Rectangle::new(
            Point::new(src.loc.x as f32, src.loc.y as f32),
            Size::new(src.size.w as f32, src.size.h as f32),
        );
        let dst_rect = Rectangle::new(
            Point::new(dst.loc.x as f32, dst.loc.y as f32),
            Size::new(dst.size.w as f32, dst.size.h as f32),
        );
        let src_transform_val = match src_transform {
            Transform::Normal => 0,
            Transform::_90 => 1,
            Transform::_180 => 2,
            Transform::_270 => 3,
            Transform::Flipped => 4,
            Transform::Flipped90 => 5,
            Transform::Flipped180 => 6,
            Transform::Flipped270 => 7,
        };

        let is_bgr = (is_bgr_format(self.fb.0.format())
            || matches!(
                self.fb.0.format(),
                vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::A2R10G10B10_UNORM_PACK32
            )
            || self
                .fb
                .0
                .drm
                .as_ref()
                .map(|f| {
                    matches!(
                        f.code,
                        Fourcc::Argb8888 | Fourcc::Xrgb8888 | Fourcc::Argb2101010 | Fourcc::Xrgb2101010
                    )
                })
                .unwrap_or(false)) as u32;
        let has_alpha = texture.has_alpha() as u32;

        for chunk in damage.chunks(4) {
            let damage_rects: [Rectangle<i32, Physical>; 4] = chunk
                .iter()
                .flat_map(|rect| {
                    let mut rect = *rect;
                    rect.loc += untransformed_dst.loc;
                    rect.intersection(untransformed_dst)
                })
                .map(|rect| {
                    let rect = self.transform.transform_rect_in(rect, &self.size);
                    let dest_size = Size::new(self.fb.width() as i32, self.fb.height() as i32);
                    let rect_constrained_loc = rect.loc.constrain(Rectangle::from_size(dest_size));
                    let rect_clamped_size = rect
                        .size
                        .clamp((0, 0), (dest_size.to_point() - rect_constrained_loc).to_size());

                    Rectangle::new(rect_constrained_loc, rect_clamped_size)
                })
                .chain(std::iter::repeat_with(Rectangle::zero))
                .take(4)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            if is_hdr {
                let config =
                    self.renderer
                        .hdr_config
                        .unwrap_or(crate::backend::renderer::gles::HdrOutputConfig {
                            reference_white: 203.0,
                            max_luminance: 1000.0,
                            sdr_gamma: 2.2,
                            gamut_stretch: 0.0,
                            hardware_offload: false,
                            is_sdr: false,
                        });
                let mut reference_white = config.reference_white;
                let mut sdr_gamma = config.sdr_gamma;
                let mut gamut_stretch = config.gamut_stretch;
                let mut max_content_luminance = config.reference_white;
                let max_destination_luminance = config.max_luminance;
                let mut hardware_offload = config.hardware_offload as u32;
                let target_is_sdr = config.is_sdr as u32;

                let mut input_is_pq = 0u32;
                let mut input_is_hlg = 0u32;
                let mut input_primaries = 0u32;
                let mut skip_color_transform = 0u32;
                let mut content_reference = 203.0f32;

                #[cfg(feature = "wayland_frontend")]
                if let Some(desc) = self.active_color_description.as_ref() {
                    use crate::wayland::color::management::{Primaries, TransferFunction};
                    if desc.is_pq_bt2020() {
                        input_is_pq = 1;
                        input_primaries = 2;
                        content_reference =
                            desc.luminances.map(|(_, _, r)| r.max(80) as f32).unwrap_or(203.0);
                        max_content_luminance = desc
                            .luminances
                            .map(|(_, m, _)| m as f32)
                            .or_else(|| desc.max_cll.map(|v| v as f32))
                            .or_else(|| desc.mastering_luminance.map(|(_, m)| m as f32))
                            .unwrap_or(1000.0);

                        // If target is HDR (not SDR) and not hardware offload, and content is display-referred or fits destination:
                        // Passthrough directly without unnecessary color transforms!
                        if !config.is_sdr
                            && !config.hardware_offload
                            && (desc.windows_bt2100 || max_content_luminance <= config.max_luminance * 1.01)
                            && (content_reference - config.reference_white).abs() < 1.0
                        {
                            skip_color_transform = 1;
                        }
                    } else if desc.transfer == TransferFunction::Hlg {
                        input_is_hlg = 1;
                        input_primaries = 2;
                        content_reference =
                            desc.luminances.map(|(_, _, r)| r.max(80) as f32).unwrap_or(203.0);
                        max_content_luminance = desc
                            .luminances
                            .map(|(_, m, _)| m as f32)
                            .or_else(|| desc.max_cll.map(|v| v as f32))
                            .or_else(|| desc.mastering_luminance.map(|(_, m)| m as f32))
                            .unwrap_or(1000.0);
                    } else if desc.windows_scrgb || desc.transfer == TransferFunction::ExtLinear {
                        reference_white = 80.0;
                        sdr_gamma = 1.0;
                        gamut_stretch = 0.0;
                        max_content_luminance = config.max_luminance;
                        content_reference = 203.0;
                    } else {
                        sdr_gamma = match desc.transfer {
                            TransferFunction::Bt1886 => 2.4,
                            TransferFunction::Gamma22 => 2.2,
                            TransferFunction::CompoundPower24 | TransferFunction::Srgb => 0.0,
                            _ => config.sdr_gamma,
                        };
                        input_primaries = match desc.primaries.named {
                            Some(Primaries::DisplayP3) => 1,
                            Some(Primaries::Bt2020) => 2,
                            _ => 0,
                        };
                        if input_primaries > 0 {
                            gamut_stretch = 0.0;
                        }
                        if config.is_sdr && input_primaries == 0 && sdr_gamma == 0.0 {
                            skip_color_transform = 1;
                        }
                    }
                } else {
                    // Untagged SDR surface
                    if config.is_sdr && sdr_gamma == 0.0 {
                        skip_color_transform = 1;
                    }
                }

                let push_constants = HdrTexPushConstants {
                    src_rect,
                    dst_rect,
                    src_transform: src_transform_val,
                    alpha,
                    damage_size: chunk.len() as u32,
                    is_bgr,
                    has_alpha,
                    reference_white,
                    sdr_gamma,
                    gamut_stretch,
                    max_content_luminance,
                    max_destination_luminance,
                    hardware_offload,
                    target_is_sdr,
                    input_is_pq,
                    input_is_hlg,
                    input_primaries,
                    skip_color_transform,
                    content_reference,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    damage: damage_rects,
                };

                unsafe {
                    self.renderer.device.vk().cmd_push_constants(
                        buf,
                        layout,
                        ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&push_constants),
                    );
                    self.renderer.device.vk().cmd_dispatch(
                        buf,
                        (self.fb.width() as f64 / 8.).ceil() as u32,
                        (self.fb.height() as f64 / 8.).ceil() as u32,
                        1,
                    );
                }
            } else {
                let push_constants = TexPushConstants {
                    src_rect,
                    dst_rect,
                    src_transform: src_transform_val,
                    alpha,
                    damage_size: chunk.len() as u32,
                    is_bgr,
                    has_alpha,
                    _padding0: [0; 3],
                    damage: damage_rects,
                };

                unsafe {
                    self.renderer.device.vk().cmd_push_constants(
                        buf,
                        layout,
                        ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&push_constants),
                    );
                    self.renderer.device.vk().cmd_dispatch(
                        buf,
                        (self.fb.width() as f64 / 8.).ceil() as u32,
                        (self.fb.height() as f64 / 8.).ceil() as u32,
                        1,
                    );
                }
            }
        }

        self.descriptors.push(descriptor);
        Ok(())
    }

    fn transformation(&self) -> Transform {
        self.transform
    }

    fn output_size(&self) -> Size<i32, Physical> {
        self.size
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.wait(sync)
    }

    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        trace!(
            has_draws = self.has_draws,
            has_drm = self.renderer.timeline.drm.is_some(),
            "VulkanFrame::finish"
        );
        if !self.has_draws {
            return Ok(SyncPoint::signaled());
        }

        let Some(buf) = self.cmd_buffer.take() else {
            return Ok(SyncPoint::signaled());
        };

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let barrier = ImageMemoryBarrier2::default()
            .image(*self.fb.0.vk())
            .old_layout(ImageLayout::GENERAL)
            .new_layout(ImageLayout::GENERAL)
            .src_queue_family_index(qfam)
            .dst_queue_family_index(ext_queue)
            .src_stage_mask(PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(AccessFlags2::NONE)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.renderer
                .device
                .vk()
                .cmd_pipeline_barrier2(buf, &DependencyInfo::default().image_memory_barriers(&[barrier]));
            self.renderer
                .device
                .vk()
                .end_command_buffer(buf)
                .map_err(Error::CommandBufferError)?;
        }

        self.fb.0.set_needs_acquire(true);

        let prev_seq_no = self.renderer.seq_no;
        let next_seq_no = prev_seq_no + 1;

        let cmd_buffer_info = [CommandBufferSubmitInfo::default().command_buffer(buf)];
        let signal_semaphore_info = [SemaphoreSubmitInfo::default()
            .semaphore(self.renderer.timeline.vk)
            .value(next_seq_no)
            .stage_mask(PipelineStageFlags2::ALL_COMMANDS)];

        let wait_semaphore_info = if prev_seq_no > 0 {
            vec![
                SemaphoreSubmitInfo::default()
                    .semaphore(self.renderer.timeline.vk)
                    .value(prev_seq_no)
                    .stage_mask(PipelineStageFlags2::ALL_COMMANDS),
            ]
        } else {
            Vec::new()
        };

        let submit_info = SubmitInfo2::default()
            .command_buffer_infos(&cmd_buffer_info)
            .signal_semaphore_infos(&signal_semaphore_info)
            .wait_semaphore_infos(&wait_semaphore_info);

        let submit_res = unsafe {
            self.renderer.device.vk().queue_submit2(
                *self.renderer.device.queue(),
                &[submit_info],
                Fence::null(),
            )
        };

        if let Err(err) = submit_res {
            self.cmd_buffer = Some(buf);
            if err == vk::Result::ERROR_DEVICE_LOST {
                return Err(Error::DeadDevice);
            } else {
                return Err(Error::SubmitError(err));
            }
        }

        self.renderer.seq_no = next_seq_no;

        let point = next_seq_no;
        let descs = std::mem::take(&mut self.descriptors);
        let mut images = std::mem::take(&mut self.images);
        if !images.iter().any(|img| Arc::ptr_eq(img, &self.fb.0.inner)) {
            images.push(self.fb.0.inner.clone());
        }
        self.renderer
            .cmd_pool
            .store_pending_buffer(buf, point, descs, images);

        trace!(point, "VulkanFrame::finish single command buffer submitted");

        if let Some(timeline) = self.renderer.timeline.drm.as_ref() {
            Ok(DrmSyncPoint {
                timeline: timeline.clone(),
                point,
            }
            .into())
        } else {
            // TODO: vulkan syncpoint
            while let Err(VkResult::TIMEOUT) = unsafe {
                self.renderer.device.vk().wait_semaphores(
                    &SemaphoreWaitInfo::default()
                        .semaphores(&[self.renderer.timeline.vk])
                        .values(&[point]),
                    u64::MAX,
                )
            } {}
            Ok(SyncPoint::signaled())
        }
    }
}

impl Drop for VulkanFrame<'_, '_> {
    fn drop(&mut self) {
        if let Some(buf) = self.cmd_buffer.take() {
            unsafe {
                let _ = self.renderer.device.vk().device_wait_idle();
                self.renderer
                    .device
                    .vk()
                    .free_command_buffers(self.renderer.cmd_pool.vk(), &[buf]);
            }
        }
    }
}

impl Blit for VulkanRenderer {
    fn blit(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        _filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        let size = Size::from((Texture::width(&to.0) as i32, Texture::height(&to.0) as i32));
        let mut frame = VulkanFrame {
            renderer: self,
            fb: to,
            _marker: std::marker::PhantomData,
            transform: Transform::Normal,
            size,
            cmd_buffer: None,
            descriptors: Vec::with_capacity(4),
            images: Vec::with_capacity(4),
            has_draws: false,
            #[cfg(feature = "wayland_frontend")]
            active_color_description: None,
            is_blit: true,
        };
        let src_rect = Rectangle::from_loc_and_size(
            (src.loc.x as f64, src.loc.y as f64),
            (src.size.w as f64, src.size.h as f64),
        );
        frame.render_texture_from_to(&from.0, src_rect, dst, &[dst], &[], Transform::Normal, 1.0)?;
        frame.finish()
    }
}

impl BlitFrame<VulkanFramebuffer> for VulkanFrame<'_, '_> {
    fn blit_to(
        &mut self,
        to: &mut VulkanFramebuffer,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        self.renderer.blit(self.fb, to, src, dst, filter)
    }

    fn blit_from(
        &mut self,
        from: &VulkanFramebuffer,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        self.renderer.blit(from, self.fb, src, dst, filter)
    }
}

impl VulkanFrame<'_, '_> {
    fn draw_color(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
        should_blend: bool,
    ) -> Result<(), Error> {
        trace!(?dst, damage_len = damage.len(), ?color, should_blend, fb_size = ?self.size, "VulkanFrame::draw_color executing");
        if damage.is_empty() {
            return Ok(());
        }
        let untransformed_dst = dst;
        let _dst = self.transform.transform_rect_in(dst, &self.size);
        self.renderer.cleanup()?;

        let color = if let Some(config) = self.renderer.hdr_config {
            sdr_color_to_hdr(
                color,
                config.reference_white,
                config.sdr_gamma,
                config.gamut_stretch,
                config.hardware_offload,
                config.is_sdr,
            )
        } else {
            color
        };

        let buf = self.get_or_create_cmd_buffer()?;
        self.has_draws = true;
        let descriptor = self
            .renderer
            .pipelines
            .alloc_descriptor_set(shaders::BuiltinShader::Clear)?;
        // SAFETY: If we were able to bind it, it has a view
        let view = self.fb.0.vk_view().unwrap();

        let image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::GENERAL)
            .image_view(*view)];
        let descriptor_update = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor.vk())
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_count(1)
            .descriptor_type(DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info)];
        let layout = *self.renderer.pipelines.clear_pipeline_layout();

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let fb_needs_acquire = self.fb.0.needs_acquire();
        let (fb_src_queue, fb_dst_queue) = if fb_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let fb_old_layout = self.fb.0.current_layout();
        let (fb_src_stage, fb_src_access) = if fb_needs_acquire {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if fb_old_layout == ImageLayout::UNDEFINED {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else {
            (
                PipelineStageFlags2::COMPUTE_SHADER,
                AccessFlags2::SHADER_STORAGE_WRITE,
            )
        };

        let fb_barrier = ImageMemoryBarrier2::default()
            .image(*self.fb.0.vk())
            .old_layout(fb_old_layout)
            .new_layout(ImageLayout::GENERAL)
            .src_queue_family_index(fb_src_queue)
            .dst_queue_family_index(fb_dst_queue)
            .src_stage_mask(fb_src_stage)
            .src_access_mask(fb_src_access)
            .dst_stage_mask(PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(AccessFlags2::SHADER_STORAGE_READ | AccessFlags2::SHADER_STORAGE_WRITE)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.renderer
                .device
                .vk()
                .update_descriptor_sets(&descriptor_update, &[]);
            self.renderer.device.vk().cmd_bind_pipeline(
                buf,
                PipelineBindPoint::COMPUTE,
                *self.renderer.pipelines.clear_pipeline(),
            );
            self.renderer.device.vk().cmd_bind_descriptor_sets(
                buf,
                PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[descriptor.vk()],
                &[],
            );
            self.renderer.device.vk().cmd_pipeline_barrier2(
                buf,
                &DependencyInfo::default().image_memory_barriers(&[fb_barrier]),
            );
        }

        self.fb.0.set_current_layout(ImageLayout::GENERAL);
        self.fb.0.set_needs_acquire(false);

        for chunk in damage.chunks(6) {
            let push_constants = ClearPushConstants {
                color: color.components(),
                blend: should_blend as u32,
                size: chunk.len() as u32,
                is_bgr: is_bgr_format(self.fb.0.format()) as u32,
                _padding0: [0; 1],
                rects: chunk
                    .iter()
                    .flat_map(|rect| {
                        let mut rect = *rect;
                        rect.loc += untransformed_dst.loc;
                        rect.intersection(untransformed_dst)
                    })
                    .map(|rect| {
                        let rect = self.transform.transform_rect_in(rect, &self.size);
                        let dest_size = Size::new(self.fb.width() as i32, self.fb.height() as i32);
                        let rect_constrained_loc = rect.loc.constrain(Rectangle::from_size(dest_size));
                        let rect_clamped_size = rect
                            .size
                            .clamp((0, 0), (dest_size.to_point() - rect_constrained_loc).to_size());

                        Rectangle::new(rect_constrained_loc, rect_clamped_size)
                    })
                    .chain(std::iter::repeat_with(Rectangle::zero))
                    .take(6)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            };

            unsafe {
                self.renderer.device.vk().cmd_push_constants(
                    buf,
                    layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
                self.renderer.device.vk().cmd_dispatch(
                    buf,
                    (self.fb.width() as f64 / 8.).ceil() as u32,
                    (self.fb.height() as f64 / 8.).ceil() as u32,
                    1,
                );
            }
        }

        self.descriptors.push(descriptor);
        Ok(())
    }
}

impl Bind<VulkanImage> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut VulkanImage) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if target.vk_usage().contains(ImageUsageFlags::STORAGE) {
            Ok(VulkanFramebuffer(target.clone()))
        } else {
            Err(Error::ImageError(ImageError::MissingOrInvalidUsage))
        }
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        known_formats()
            .iter()
            .try_fold(IndexSet::new(), |mut set, fourcc| {
                let format = get_vk_format(*fourcc).unwrap();
                let props = self.phd.get_format_modifier_properties(format)?;
                // TODO: else use get_format_properties and map to LINEAR and INVALID
                for prop in props {
                    if prop
                        .drm_format_modifier_tiling_features
                        .contains(FormatFeatureFlags::STORAGE_IMAGE)
                    {
                        set.insert(Format {
                            code: *fourcc,
                            modifier: Modifier::from(prop.drm_format_modifier),
                        });
                        if let Some(opaque) = crate::backend::allocator::format::get_opaque(*fourcc) {
                            set.insert(Format {
                                code: opaque,
                                modifier: Modifier::from(prop.drm_format_modifier),
                            });
                        }
                    }
                }
                Result::<_, UnsupportedProperty>::Ok(set)
            })
            .ok()
            .map(FormatSet::from_formats)
    }
}

impl super::Offscreen<VulkanImage> for VulkanRenderer {
    fn create_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoords>,
    ) -> Result<VulkanImage, Self::Error> {
        let mut usage = ImageUsageFlags::STORAGE
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST
            | ImageUsageFlags::SAMPLED;
        if self.device.vk_ext_host_image_copy().is_some() {
            usage |= ImageUsageFlags::HOST_TRANSFER_EXT;
        }
        VulkanImage::new_with_fourcc(&self.device, size.w as u32, size.h as u32, format, usage, false)
            .map_err(Error::ImageError)
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        use crate::backend::allocator::Buffer as AllocBuffer;
        trace!(format = ?AllocBuffer::format(target), width = AllocBuffer::width(target), height = AllocBuffer::height(target), "VulkanRenderer::bind dmabuf");
        let image = match self.dmabuf_cache.get(&target.weak()) {
            Some(image) if image.vk_usage().contains(ImageUsageFlags::STORAGE) => image.clone(),
            _ => {
                let image = VulkanImage::new_from_dmabuf(
                    &self.device,
                    target,
                    ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::SAMPLED,
                )
                .map_err(Error::ImageError)?;
                self.dmabuf_cache.insert(target.weak(), image.clone());
                image
            }
        };
        Ok(VulkanFramebuffer(image))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        // TODO: Can external_memory_dma_buf also provide formats?
        known_formats()
            .iter()
            .try_fold(IndexSet::new(), |mut set, fourcc| {
                let format = get_vk_format(*fourcc).unwrap();
                let props = self.phd.get_format_modifier_properties(format)?;
                // TODO: else use get_format_properties and map to LINEAR and INVALID
                for prop in props {
                    if prop
                        .drm_format_modifier_tiling_features
                        .contains(FormatFeatureFlags::STORAGE_IMAGE)
                    {
                        set.insert(Format {
                            code: *fourcc,
                            modifier: Modifier::from(prop.drm_format_modifier),
                        });
                        if let Some(opaque) = crate::backend::allocator::format::get_opaque(*fourcc) {
                            set.insert(Format {
                                code: opaque,
                                modifier: Modifier::from(prop.drm_format_modifier),
                            });
                        }
                    }
                }
                Result::<_, UnsupportedProperty>::Ok(set)
            })
            .ok()
            .map(FormatSet::from_formats)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_vulkan_renderer_version_check() {
        // Instance 1.2: should fail with UnsupportedVersion
        if let Ok(instance_1_2) = crate::backend::vulkan::Instance::new(Version::VERSION_1_2, None) {
            if let Ok(phds) = PhysicalDevice::enumerate(&instance_1_2) {
                for phd in phds {
                    assert!(
                        matches!(VulkanRenderer::new(&phd, None), Err(Error::UnsupportedVersion)),
                        "VulkanRenderer::new with Vulkan 1.2 instance must fail with UnsupportedVersion"
                    );
                }
            }
        }

        // Instance 1.3: should succeed for Vulkan 1.3 capable physical devices
        if let Ok(instance_1_3) = crate::backend::vulkan::Instance::new(Version::VERSION_1_3, None) {
            if let Ok(phds) = PhysicalDevice::enumerate(&instance_1_3) {
                for phd in phds {
                    if phd.api_version() >= VulkanRenderer::MIN_DEVICE_VERSION {
                        assert!(
                            VulkanRenderer::new(&phd, None).is_ok(),
                            "VulkanRenderer::new with Vulkan 1.3 instance must succeed"
                        );
                    }
                }
            }
        }
    }
}
