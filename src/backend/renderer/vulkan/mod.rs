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
            vulkan::shaders::{ClearPushConstants, TexPushConstants},
        },
        vulkan::{
            PhysicalDevice, UnsupportedProperty,
            device::{Device, DeviceError, QueueType, WeakDevice},
            format::{get_drm_format, get_vk_format, known_formats},
            image::{Error as ImageError, ImageUsageFlags, VulkanImage},
            version::Version,
        },
    },
    reexports::drm::node::DrmNode,
    utils::{Buffer as BufferCoords, Physical, Point, Rectangle, Size, Transform},
};

use ash::vk::{
    self, AccessFlags, BorderColor, CompareOp, DependencyFlags, DescriptorImageInfo, DescriptorType,
    Extent3D, Fence, Filter, FormatFeatureFlags, HostImageCopyFlagsEXT, ImageAspectFlags, ImageLayout,
    ImageMemoryBarrier, ImageSubresourceLayers, ImageSubresourceRange, ImageToMemoryCopyEXT, MemoryMapFlags,
    MemoryPropertyFlags, MemoryToImageCopyEXT, Offset3D, PipelineBindPoint, PipelineStageFlags,
    QUEUE_FAMILY_IGNORED, Result as VkResult, SamplerAddressMode, SamplerCreateFlags, SamplerCreateInfo,
    SamplerMipmapMode, SemaphoreWaitInfo, ShaderStageFlags, SubmitInfo, TimelineSemaphoreSubmitInfo,
};
use gbm::Modifier;
use indexmap::IndexSet;

use std::{collections::HashMap, ffi::CStr, fmt, ptr::NonNull};

use super::{Blit, BlitFrame, Color32F, TextureFilter, sync::SyncPoint};

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
    pub(crate) phd: PhysicalDevice,
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

    // A bunch of the previous structs contain Weak-device references.
    // So we want to drop this last for proper cleanup and avoiding accidental
    // resource leaks.
    pub(crate) device: Device,
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe { self.device.vk().destroy_sampler(self.texture_sampler, None) };
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
            if !(phd.render_node().ok().flatten().is_some_and(|n| n == node)
                || phd.primary_node().ok().flatten().is_some_and(|n| n == node))
            {
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
        })
    }

    pub fn cleanup(&mut self) -> Result<(), Error> {
        let val = unsafe {
            self.device
                .vk()
                .get_semaphore_counter_value(self.timeline.vk)
                .map_err(Error::SemaphoreError)?
        };
        self.cmd_pool.clean_old_buffers(val);
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
        use ash::ext::host_image_copy;

        if region.size.w <= 0 || region.size.h <= 0 {
            return Ok(());
        }

        let device_copy = self
            .device
            .vk_ext_host_image_copy()
            .ok_or(Error::MissingExtension(host_image_copy::NAME))?;

        let host_ptr =
            unsafe { ptr.offset((region.loc.y as isize * stride as isize) + (region.loc.x as isize * 4)) };

        unsafe {
            device_copy
                .transition_image_layout(&[vk::HostImageLayoutTransitionInfoEXT::default()
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::GENERAL)
                    .image(*image.vk())
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )])
                .map_err(Error::HostImageTransitionError)?;
            device_copy
                .copy_memory_to_image(
                    &vk::CopyMemoryToImageInfoEXT::default()
                        .flags(HostImageCopyFlagsEXT::MEMCPY)
                        .dst_image(*image.vk())
                        .dst_image_layout(ImageLayout::GENERAL)
                        .regions(&[MemoryToImageCopyEXT::default()
                            .host_pointer(host_ptr as *const _)
                            .memory_row_length((stride / 4) as u32)
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

        Ok(())
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
            last_sequence: None,
            _marker: std::marker::PhantomData,
        })
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        while let Err(super::sync::Interrupted) = sync.wait() {}
        Ok(())
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
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
        use ash::ext::host_image_copy;

        let device_copy = self
            .device
            .vk_ext_host_image_copy()
            .ok_or(Error::MissingExtension(host_image_copy::NAME))?;

        //let vk_format = get_vk_format(format).ok_or(Error::ImageError(ImageError::UnsupportedFormat))?;
        let image = VulkanImage::new_exportable(
            // TODO: Non-exportable with explicit drm format
            &self.device,
            size.w as u32,
            size.h as u32,
            format,
            std::iter::once(Modifier::Linear),
            vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
        )
        .map_err(Error::ImageError)?;

        unsafe {
            device_copy
                .transition_image_layout(&[vk::HostImageLayoutTransitionInfoEXT::default()
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::GENERAL)
                    .image(*image.vk())
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )])
                .map_err(Error::HostImageTransitionError)?;
            device_copy
                .copy_memory_to_image(
                    &vk::CopyMemoryToImageInfoEXT::default()
                        .flags(HostImageCopyFlagsEXT::MEMCPY)
                        .dst_image(*image.vk())
                        .dst_image_layout(ImageLayout::GENERAL)
                        .regions(&[MemoryToImageCopyEXT::default()
                            .host_pointer(data.as_ptr() as *const _)
                            .memory_row_length(0)
                            .memory_image_height(0)
                            .image_subresource(
                                ImageSubresourceLayers::default()
                                    .aspect_mask(ImageAspectFlags::COLOR)
                                    .mip_level(0)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            )
                            .image_offset(Offset3D::default())
                            .image_extent(
                                Extent3D::default()
                                    .depth(1)
                                    .width(size.w as u32)
                                    .height(size.h as u32),
                            )]),
                )
                .map_err(Error::HostImageCopyError)?;
        }

        Ok(image)
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, BufferCoords>,
    ) -> Result<(), Self::Error> {
        if region.size.w <= 0 || region.size.h <= 0 {
            return Ok(());
        }

        use ash::ext::host_image_copy;

        let device_copy = self
            .device
            .vk_ext_host_image_copy()
            .ok_or(Error::MissingExtension(host_image_copy::NAME))?;

        unsafe {
            device_copy
                .transition_image_layout(&[vk::HostImageLayoutTransitionInfoEXT::default()
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::GENERAL)
                    .image(*texture.vk())
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )])
                .map_err(Error::HostImageTransitionError)?;
            device_copy
                .copy_memory_to_image(
                    &vk::CopyMemoryToImageInfoEXT::default()
                        .flags(HostImageCopyFlagsEXT::MEMCPY)
                        .dst_image(*texture.vk())
                        .dst_image_layout(ImageLayout::GENERAL)
                        .regions(&[MemoryToImageCopyEXT::default()
                            .host_pointer(data.as_ptr() as *const _)
                            .memory_row_length(0)
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

        Ok(())
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
                let image = VulkanImage::new_exportable(
                    &self.device,
                    width as u32,
                    height as u32,
                    fourcc,
                    std::iter::once(Modifier::Linear),
                    vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                )
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
        let layout = unsafe {
            self.device.vk().get_image_subresource_layout(
                *image.vk(),
                vk::ImageSubresource {
                    aspect_mask: if image.drm.is_some() {
                        ImageAspectFlags::MEMORY_PLANE_0_EXT
                    } else {
                        ImageAspectFlags::COLOR
                    },
                    mip_level: 0,
                    array_layer: 0,
                },
            )
        };

        if image.mem_bits().contains(MemoryPropertyFlags::HOST_VISIBLE) && image.is_linear() {
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
                device_copy
                    .transition_image_layout(&[vk::HostImageLayoutTransitionInfoEXT::default()
                        .old_layout(ImageLayout::UNDEFINED)
                        .new_layout(ImageLayout::GENERAL)
                        .image(*image.vk())
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        )])
                    .map_err(Error::HostImageTransitionError)?;

                let mut data = Vec::with_capacity(layout.size as usize);
                device_copy
                    .copy_image_to_memory(
                        &vk::CopyImageToMemoryInfoEXT::default()
                            .flags(HostImageCopyFlagsEXT::empty())
                            .src_image(*image.vk())
                            .src_image_layout(ImageLayout::GENERAL)
                            .regions(&[ImageToMemoryCopyEXT::default()
                                .host_pointer(data.as_mut_ptr() as *mut _)
                                .memory_image_height(image.height())
                                .memory_row_length(layout.row_pitch as u32)
                                .image_subresource(
                                    ImageSubresourceLayers::default()
                                        .aspect_mask(ImageAspectFlags::COLOR)
                                        .mip_level(0)
                                        .base_array_layer(0)
                                        .layer_count(1),
                                )
                                .image_offset(Offset3D::default())
                                .image_extent(
                                    Extent3D::default()
                                        .depth(1)
                                        .width(image.width())
                                        .height(image.height()),
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

pub struct VulkanFrame<'frame, 'buffer> {
    renderer: &'frame mut VulkanRenderer,
    fb: &'frame mut VulkanFramebuffer,
    _marker: std::marker::PhantomData<&'buffer ()>,

    transform: Transform,
    size: Size<i32, Physical>,
    last_sequence: Option<u64>,
}

impl Frame for VulkanFrame<'_, '_> {
    type Error = Error;
    type TextureId = VulkanImage;

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
        if damage.is_empty() {
            return Ok(());
        }
        let dst = self.transform.transform_rect_in(dst, &self.size);
        self.renderer.cleanup()?;

        let buf = self.renderer.cmd_pool.create_and_begin_buffer()?;
        let descriptor = self
            .renderer
            .pipelines
            .alloc_descriptor_set(shaders::BuiltinShader::Texture)?;

        // SAFETY: If we were able to bind it, it has a view
        let view = self.fb.0.vk_view().unwrap();
        let fb_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::GENERAL)
            .image_view(*view)];

        let view = texture
            .vk_view()
            .ok_or(Error::ImageError(ImageError::MissingOrInvalidUsage))?;
        let tex_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::GENERAL)
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
        let layout = *self.renderer.pipelines.tex_pipeline_layout();

        unsafe {
            self.renderer
                .device
                .vk()
                .update_descriptor_sets(&descriptor_update, &[]);
            self.renderer.device.vk().cmd_bind_pipeline(
                buf,
                PipelineBindPoint::COMPUTE,
                *self.renderer.pipelines.tex_pipeline(),
            );
            self.renderer.device.vk().cmd_bind_descriptor_sets(
                buf,
                PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[descriptor.vk()],
                &[],
            );
            self.renderer.device.vk().cmd_pipeline_barrier(
                buf,
                PipelineStageFlags::COMPUTE_SHADER,
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[
                    ImageMemoryBarrier::default()
                        .image(*self.fb.0.vk())
                        .old_layout(ImageLayout::UNDEFINED)
                        .new_layout(ImageLayout::GENERAL)
                        .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                        .src_access_mask(AccessFlags::empty())
                        .dst_access_mask(AccessFlags::empty())
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        ),
                    ImageMemoryBarrier::default()
                        .image(*texture.vk())
                        .old_layout(ImageLayout::UNDEFINED)
                        .new_layout(ImageLayout::GENERAL)
                        .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                        .src_access_mask(AccessFlags::empty())
                        .dst_access_mask(AccessFlags::empty())
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        ),
                ],
            );
        }

        for chunk in damage.chunks(4) {
            let push_constants = TexPushConstants {
                src_rect: Rectangle::new(
                    Point::new(src.loc.x as f32, src.loc.y as f32),
                    Size::new(src.size.w as f32, src.size.h as f32),
                ),
                dst_rect: Rectangle::new(
                    Point::new(dst.loc.x as f32, dst.loc.y as f32),
                    Size::new(dst.size.w as f32, dst.size.h as f32),
                ),
                src_transform: match src_transform {
                    Transform::Normal => 0,
                    Transform::_90 => 1,
                    Transform::_180 => 2,
                    Transform::_270 => 3,
                    Transform::Flipped => 4,
                    Transform::Flipped90 => 5,
                    Transform::Flipped180 => 6,
                    Transform::Flipped270 => 7,
                },
                alpha,
                damage_size: chunk.len() as u32,
                damage: chunk
                    .iter()
                    .flat_map(|rect| {
                        let mut rect = *rect;
                        rect.loc += dst.loc;
                        rect.intersection(dst)
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
                    .unwrap(),
                _padding0: [0; 1],
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

        unsafe {
            /*
            self.renderer.device.vk().cmd_pipeline_barrier(
                buf,
                PipelineStageFlags::COMPUTE_SHADER,
                PipelineStageFlags::HOST,
                DependencyFlags::empty(),
                &[],
                &[],
                &[ImageMemoryBarrier::default()
                    .image(*texture.vk())
                    .old_layout(ImageLayout::GENERAL)
                    .new_layout(ImageLayout::GENERAL)
                    .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                    .src_access_mask(AccessFlags::empty())
                    .dst_access_mask(AccessFlags::HOST_READ)
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )],
            );
            */
            self.renderer
                .device
                .vk()
                .end_command_buffer(buf)
                .map_err(Error::CommandBufferError)?;
        }

        self.renderer.seq_no += 1;
        let next_seq_no = [self.renderer.seq_no];
        let mut timeline_info = TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&next_seq_no);

        unsafe {
            self.renderer
                .device
                .vk()
                .queue_submit(
                    *self.renderer.device.queue(),
                    &[SubmitInfo::default()
                        .command_buffers(&[buf])
                        .signal_semaphores(&[self.renderer.timeline.vk])
                        .push_next(&mut timeline_info)],
                    Fence::null(),
                )
                .map_err(Error::SubmitError)?;
        }

        self.renderer
            .cmd_pool
            .store_pending_buffer(buf, next_seq_no[0], descriptor);
        self.last_sequence = Some(next_seq_no[0]);

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

    fn finish(self) -> Result<SyncPoint, Self::Error> {
        if let Some(point) = self.last_sequence {
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
        } else {
            Ok(SyncPoint::signaled())
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
            last_sequence: None,
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
        if damage.is_empty() {
            return Ok(());
        }
        let dst = self.transform.transform_rect_in(dst, &self.size);
        self.renderer.cleanup()?;

        let buf = self.renderer.cmd_pool.create_and_begin_buffer()?;
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
            self.renderer.device.vk().cmd_pipeline_barrier(
                buf,
                PipelineStageFlags::COMPUTE_SHADER,
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[ImageMemoryBarrier::default()
                    .image(*self.fb.0.vk())
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::GENERAL)
                    .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                    .src_access_mask(AccessFlags::empty())
                    .dst_access_mask(AccessFlags::empty())
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    )],
            );
        }

        for chunk in damage.chunks(6) {
            let push_constants = ClearPushConstants {
                color: color.components(),
                blend: should_blend as u32,
                size: chunk.len() as u32,
                rects: chunk
                    .iter()
                    .flat_map(|rect| {
                        let mut rect = *rect;
                        rect.loc += dst.loc;
                        rect.intersection(dst)
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
                _padding0: [0; 2],
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
                self.renderer
                    .device
                    .vk()
                    .end_command_buffer(buf)
                    .map_err(Error::CommandBufferError)?;
            }
        }

        self.renderer.seq_no += 1;
        let next_seq_no = [self.renderer.seq_no];
        let mut timeline_info = TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&next_seq_no);

        unsafe {
            self.renderer
                .device
                .vk()
                .queue_submit(
                    *self.renderer.device.queue(),
                    &[SubmitInfo::default()
                        .command_buffers(&[buf])
                        .signal_semaphores(&[self.renderer.timeline.vk])
                        .push_next(&mut timeline_info)],
                    Fence::null(),
                )
                .map_err(Error::SubmitError)?;
        }

        self.renderer
            .cmd_pool
            .store_pending_buffer(buf, next_seq_no[0], descriptor);
        self.last_sequence = Some(next_seq_no[0]);

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
                set.extend(
                    props
                        .into_iter()
                        .filter(|prop| {
                            prop.drm_format_modifier_tiling_features
                                .contains(FormatFeatureFlags::STORAGE_IMAGE)
                        })
                        .map(|prop| Format {
                            code: *fourcc,
                            modifier: Modifier::from(prop.drm_format_modifier),
                        }),
                );
                Result::<_, UnsupportedProperty>::Ok(set)
            })
            .ok()
            .map(FormatSet::from_formats)
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
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
                set.extend(
                    props
                        .into_iter()
                        .filter(|prop| {
                            prop.drm_format_modifier_tiling_features
                                .contains(FormatFeatureFlags::STORAGE_IMAGE)
                        })
                        .map(|prop| Format {
                            code: *fourcc,
                            modifier: Modifier::from(prop.drm_format_modifier),
                        }),
                );
                Result::<_, UnsupportedProperty>::Ok(set)
            })
            .ok()
            .map(FormatSet::from_formats)
    }
}
