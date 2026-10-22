//! Implementation of the rendering traits using Vulkan

use crate::{
    backend::{
        allocator::{
            Buffer, Format, Fourcc,
            dmabuf::{Dmabuf, WeakDmabuf},
            format::{FormatSet, get_bpp, get_opaque},
        },
        drm::{
            DrmDeviceFd, NodeType,
            sync::{DrmSyncPoint, DrmTimeline, WeakDrmTimeline},
        },
        renderer::{
            Bind, ContextId, ExportMem, Frame, ImportDma, ImportMem, Renderer, RendererSuper, Texture,
            TextureMapping,
            vulkan::shaders::{ClearPushConstants, HdrTexPushConstants, TexPushConstants},
        },
        vulkan::{
            Instance, PhysicalDevice, UnsupportedProperty,
            device::{Device, DeviceError, QueueType},
            format::{get_drm_format, get_vk_format, is_ycbcr_fourcc, known_formats},
            image::{Error as ImageError, ImageInner, ImageUsageFlags, VulkanImage},
            memory::MemoryUsagePreference,
            version::Version,
            ycbcr::YcbcrKey,
        },
    },
    reexports::drm::node::DrmNode,
    utils::{Buffer as BufferCoords, Physical, Point, Rectangle, Size, Transform},
};

#[cfg(feature = "wayland_frontend")]
use crate::{
    backend::renderer::{ImportDmaWl, ImportMemWl},
    wayland::{
        color::{
            management::{ImageDescription, Primaries, TransferFunction},
            representation::get_color_representation_from_states,
        },
        compositor::SurfaceData,
        dmabuf::get_dmabuf,
        shm::{fourcc_to_shm_format, shm_format_to_fourcc, with_buffer_contents},
    },
};
#[cfg(feature = "wayland_frontend")]
use wayland_server::protocol::wl_buffer::WlBuffer;

use ash::vk::{
    self, AccessFlags2, CommandBufferSubmitInfo, DependencyInfo, DescriptorImageInfo, DescriptorType,
    DrawIndirectCommand, Extent3D, Fence, FormatFeatureFlags, HostImageCopyFlagsEXT, ImageAspectFlags,
    ImageLayout, ImageMemoryBarrier2, ImageSubresourceLayers, ImageSubresourceRange, MemoryMapFlags,
    MemoryPropertyFlags, MemoryToImageCopyEXT, Offset3D, PipelineBindPoint, PipelineStageFlags2,
    QUEUE_FAMILY_IGNORED, Result as VkResult, SemaphoreSubmitInfo, SemaphoreWaitInfo, ShaderStageFlags,
    SubmitInfo2,
};
use gbm::Modifier;
use indexmap::IndexSet;

use std::{
    collections::{HashMap, VecDeque},
    ffi::CStr,
    fmt,
    ptr::NonNull,
    sync::Arc,
};

use super::{Blit, BlitFrame, Color32F, HdrOutputConfig, TextureFilter, sdr_color_to_hdr, sync::SyncPoint};
use tracing::trace;

//mod buffer;
mod capabilities;
mod cmds;
mod device;
mod image;
pub mod indirect;
mod shaders;
mod sync;

pub use self::capabilities::*;
pub use self::indirect::*;
use self::shaders::Pipelines;
pub use self::shaders::{Error as PipelineError, SharedPipelines};
pub use self::sync::*;

pub use crate::backend::vulkan::memory::MemoryBudgetInfo;

mod lut3d;
use lut3d::{Lut3dTexture, generate_ictcp_tonemap_lut};

#[derive(Debug)]
pub struct QueuedSubmit {
    pub cmd_buffer: vk::CommandBuffer,
    pub cmd_buffer_infos: Vec<CommandBufferSubmitInfo<'static>>,
    pub signal_semaphore_infos: Vec<SemaphoreSubmitInfo<'static>>,
    pub wait_semaphore_infos: Vec<SemaphoreSubmitInfo<'static>>,
    pub point: u64,
    pub descs: Vec<self::shaders::DescriptorSet>,
    pub images: Vec<Arc<ImageInner>>,
    pub sampled_images: Vec<VulkanImage>,
}

pub(crate) struct DeferredDestruction {
    pub point: u64,
    pub callback: Box<dyn FnOnce(&Device) + Send>,
}

impl fmt::Debug for DeferredDestruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeferredDestruction")
            .field("point", &self.point)
            .finish()
    }
}

pub struct PendingVulkanShmCopy {
    pub device: Device,
    pub staging_buffer: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub buffer_size: vk::DeviceSize,
    pub is_coherent: bool,
    pub timeline: sync::VulkanTimeline,
    pub point: u64,
    pub width: u32,
    pub height: u32,
    pub bpp: usize,
}

impl fmt::Debug for PendingVulkanShmCopy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingVulkanShmCopy")
            .field("staging_buffer", &self.staging_buffer)
            .field("buffer_size", &self.buffer_size)
            .field("point", &self.point)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

unsafe impl Send for PendingVulkanShmCopy {}
unsafe impl Sync for PendingVulkanShmCopy {}

impl PendingVulkanShmCopy {
    pub fn wait_and_copy(
        mut self,
        dst_ptr: *mut u8,
        dst_len: usize,
        dst_offset: i32,
        dst_stride: i32,
    ) -> Result<(), vk::Result> {
        let vk = self.device.vk();
        let sems = [self.timeline.vk];
        let vals = [self.point];
        let wait_info = SemaphoreWaitInfo::default().semaphores(&sems).values(&vals);
        loop {
            let res = unsafe { vk.wait_semaphores(&wait_info, 100_000_000) };
            match res {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                Err(err) => return Err(err),
            }
        }

        let src_ptr =
            unsafe { vk.map_memory(self.staging_memory, 0, self.buffer_size, MemoryMapFlags::empty())? };

        if !self.is_coherent {
            let mapped_range = vk::MappedMemoryRange::default()
                .memory(self.staging_memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                let _ = vk.invalidate_mapped_memory_ranges(&[mapped_range]);
            }
        }

        let row_bytes = (self.width as usize * self.bpp).min(dst_stride as usize);
        for i in 0..(self.height as usize) {
            let src_row = unsafe { (src_ptr as *const u8).add(i * self.width as usize * self.bpp) };
            let dst_row_offset = dst_offset as usize + i * dst_stride as usize;
            if dst_row_offset + row_bytes <= dst_len {
                unsafe {
                    std::ptr::copy_nonoverlapping::<u8>(src_row, dst_ptr.add(dst_row_offset), row_bytes);
                }
            }
        }

        unsafe {
            vk.unmap_memory(self.staging_memory);
            vk.destroy_buffer(self.staging_buffer, None);
            vk.free_memory(self.staging_memory, None);
        }
        self.staging_buffer = vk::Buffer::null();
        self.staging_memory = vk::DeviceMemory::null();

        Ok(())
    }
}

impl Drop for PendingVulkanShmCopy {
    fn drop(&mut self) {
        let vk = self.device.vk();
        if self.staging_buffer != vk::Buffer::null() {
            unsafe {
                let sems = [self.timeline.vk];
                let vals = [self.point];
                let wait_info = SemaphoreWaitInfo::default().semaphores(&sems).values(&vals);
                let _ = vk.wait_semaphores(&wait_info, 100_000_000);
                vk.destroy_buffer(self.staging_buffer, None);
                self.staging_buffer = vk::Buffer::null();
            }
        }
        if self.staging_memory != vk::DeviceMemory::null() {
            unsafe {
                vk.free_memory(self.staging_memory, None);
                self.staging_memory = vk::DeviceMemory::null();
            }
        }
    }
}

#[derive(Debug)]
pub struct VulkanRenderer {
    capabilities: Vec<Capability>,
    dmabuf_cache: Arc<std::sync::Mutex<HashMap<WeakDmabuf, VulkanImage>>>,

    pipelines: Pipelines,
    cmd_pool: cmds::CommandPool,
    texture_sampler: vk::Sampler,
    texture_sampler_nearest: vk::Sampler,
    pub(crate) lut3d: Option<Lut3dTexture>,

    seq_no: u64,
    timeline: sync::VulkanTimeline,
    pub(crate) node: Option<DrmNode>,

    debug_flags: super::DebugFlags,
    downscale_filter: super::TextureFilter,
    upscale_filter: super::TextureFilter,
    pub(crate) hdr_config: Option<HdrOutputConfig>,
    pub(crate) supports_optimal_host_copy: bool,
    pub(crate) supports_descriptor_indexing: bool,
    pub(crate) supports_multi_draw_indirect: bool,
    pub(crate) supports_shader_draw_parameters: bool,
    pub(crate) indirect_buffer: Option<DrawIndirectBuffer>,
    pub(crate) bindless_pool: Option<BindlessDescriptorPool>,
    pub(crate) batch_submits: bool,
    pub(crate) queued_submits: Vec<QueuedSubmit>,

    imported_timelines: HashMap<WeakDrmTimeline, vk::Semaphore>,
    pending_waits: Vec<DrmSyncPoint>,
    pub(crate) depth_image: Option<VulkanImage>,
    deferred_destructions: VecDeque<DeferredDestruction>,

    // A bunch of the previous structs contain Weak-device references.
    // So we want to drop this last for proper cleanup and avoiding accidental
    // resource leaks.
    pub(crate) device: Device,
    pub(crate) phd: PhysicalDevice,
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            if let Some(last_seq) = self.cmd_pool.last_pending_seq() {
                while let Err(VkResult::TIMEOUT) = self.device.vk().wait_semaphores(
                    &SemaphoreWaitInfo::default()
                        .semaphores(&[self.timeline.vk])
                        .values(&[last_seq]),
                    u64::MAX,
                ) {}
            }
            for q in self.queued_submits.drain(..) {
                self.device
                    .vk()
                    .free_command_buffers(self.cmd_pool.vk(), &[q.cmd_buffer]);
            }
            self.cmd_pool.clean_old_buffers(u64::MAX);
            self.depth_image.take();
            for destruction in self.deferred_destructions.drain(..) {
                (destruction.callback)(&self.device);
            }
            if Arc::strong_count(&self.dmabuf_cache) == 1 {
                self.dmabuf_cache.lock().unwrap().clear();
            }
            if let Some(mut lut) = self.lut3d.take() {
                lut.destroy(&self.device);
            }
            for (_, sem) in self.imported_timelines.drain() {
                self.device.vk().destroy_semaphore(sem, None);
            }
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
    #[error("Failed to allocate indirect buffer: `{0:?}`")]
    IndirectBufferError(#[source] VkResult),
    #[error("Failed to create descriptor pool: `{0:?}`")]
    DescriptorPoolError(#[source] VkResult),
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
    #[error("Failed to import semaphore fd: `{0:?}`")]
    SemaphoreImportError(#[source] VkResult),
    #[error("Failed to submit command buffer")]
    SubmitError(#[source] VkResult),
    #[error("Error intefacing with the underlying drm device")]
    DrmError(#[source] std::io::Error),
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

    /// Validate that a [`PhysicalDevice`] meets the version, feature, and optional DRM node
    /// requirements of [`VulkanRenderer`] without creating a logical `VkDevice`.
    pub fn check_physical_device(phd: &PhysicalDevice, drm: Option<&DrmDeviceFd>) -> Result<(), Error> {
        // Check matching drm descriptor, if provided
        if let Some(fd) = drm {
            let node = DrmNode::from_file(fd).map_err(|_| Error::MismatchedDrmDevice)?;
            let dev_render_node = node.node_with_type(NodeType::Render).and_then(Result::ok);
            let dev_primary_node = node.node_with_type(NodeType::Primary).and_then(Result::ok);
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
        }

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

        Ok(())
    }

    //#[instrument(err, skip(phd), fields(physical_device = phd.name()))]
    pub fn new(phd: &PhysicalDevice, drm: Option<DrmDeviceFd>) -> Result<VulkanRenderer, Error> {
        Self::check_physical_device(phd, drm.as_ref())?;

        let supported_features = Features::supported_features(phd);
        let mut required_features = Features::required_features();

        // Check optional extensions
        let mut capabilities = Vec::new();
        capabilities.extend(Capability::supports_dmabuf_memory(phd));
        capabilities.extend(Capability::supports_export_timeline(phd));
        capabilities.extend(Capability::supports_host_image_copy(phd));
        capabilities.extend(Capability::supports_queue_family_foreign(phd));
        capabilities.extend(Capability::supports_push_descriptor(phd));
        capabilities.extend(Capability::supports_memory_budget(phd));
        capabilities.extend(Capability::supports_dynamic_rendering(phd));
        capabilities.extend(Capability::supports_descriptor_indexing(phd));
        capabilities.extend(Capability::supports_multi_draw_indirect(phd));
        capabilities.extend(Capability::supports_shader_draw_parameters(phd));
        capabilities.extend(Capability::supports_memory_priority(phd));
        capabilities.extend(Capability::supports_global_priority(phd));

        if capabilities.contains(&Capability::MemoryPriority) {
            required_features.enable_memory_priority();
        }

        if supported_features.sampler_ycbcr_conversion() {
            required_features.enable_sampler_ycbcr_conversion();
        }

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

        Self::with_device(
            &device,
            drm,
            Arc::new(std::sync::Mutex::new(HashMap::new())),
            None,
        )
    }

    /// Create a [`VulkanRenderer`] sharing an existing [`Device`], `dmabuf_cache`, and optional [`SharedPipelines`].
    pub fn with_device(
        device: &Device,
        drm: Option<DrmDeviceFd>,
        dmabuf_cache: Arc<std::sync::Mutex<HashMap<WeakDmabuf, VulkanImage>>>,
        shared_pipelines: Option<Arc<SharedPipelines>>,
    ) -> Result<VulkanRenderer, Error> {
        let phd = device.physical_device();

        // Check matching drm descriptor, if provided
        let node = if let Some(fd) = drm.as_ref() {
            let node = DrmNode::from_file(fd).map_err(|_| Error::MismatchedDrmDevice)?;
            let dev_render_node = node.node_with_type(NodeType::Render).and_then(Result::ok);
            let dev_primary_node = node.node_with_type(NodeType::Primary).and_then(Result::ok);
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

        let mut capabilities = Vec::new();
        capabilities.extend(Capability::supports_dmabuf_memory(phd));
        capabilities.extend(Capability::supports_export_timeline(phd));
        capabilities.extend(Capability::supports_host_image_copy(phd));
        capabilities.extend(Capability::supports_queue_family_foreign(phd));
        capabilities.extend(Capability::supports_push_descriptor(phd));
        capabilities.extend(Capability::supports_memory_budget(phd));
        capabilities.extend(Capability::supports_dynamic_rendering(phd));
        capabilities.extend(Capability::supports_descriptor_indexing(phd));
        capabilities.extend(Capability::supports_multi_draw_indirect(phd));
        capabilities.extend(Capability::supports_shader_draw_parameters(phd));
        capabilities.extend(Capability::supports_memory_priority(phd));
        capabilities.extend(Capability::supports_global_priority(phd));

        // Create or reuse shared pipelines
        let shared_pipelines = match shared_pipelines {
            Some(shared) => shared,
            None => SharedPipelines::new(device)?,
        };
        let sampler = shared_pipelines.texture_sampler();
        let sampler_nearest = shared_pipelines.texture_sampler_nearest();
        let pipelines = Pipelines::from_shared(shared_pipelines);

        // Create command pool
        let cmd_pool = device.create_command_pool()?;

        // Create timeline semaphore for our queue
        let timeline =
            device.create_timeline_semaphore(if capabilities.contains(&Capability::ExportTimeline) {
                drm
            } else {
                None
            })?;

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
                    device,
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

        // Trilinear 3D LUT is disabled in favor of analytical HDR precision / tetrahedral interpolation
        let lut3d = None;

        let supports_descriptor_indexing = capabilities.contains(&Capability::DescriptorIndexing);
        let supports_multi_draw_indirect = capabilities.contains(&Capability::MultiDrawIndirect);
        let supports_shader_draw_parameters = capabilities.contains(&Capability::ShaderDrawParameters);

        let indirect_buffer = if supports_multi_draw_indirect {
            match DrawIndirectBuffer::new(device, DrawIndirectBuffer::DEFAULT_CAPACITY) {
                Ok(buf) => {
                    tracing::debug!(
                        "Vulkan DrawIndirectBuffer initialized for multi-draw indirect dispatches"
                    );
                    Some(buf)
                }
                Err(err) => {
                    tracing::warn!(
                        ?err,
                        "Failed to initialize Vulkan DrawIndirectBuffer, falling back to direct draw"
                    );
                    None
                }
            }
        } else {
            None
        };

        let bindless_pool = if supports_descriptor_indexing {
            match BindlessDescriptorPool::new(device, BindlessDescriptorPool::DEFAULT_MAX_TEXTURES) {
                Ok(pool) => {
                    tracing::debug!("Vulkan BindlessDescriptorPool initialized for descriptor indexing");
                    Some(pool)
                }
                Err(err) => {
                    tracing::warn!(
                        ?err,
                        "Failed to initialize Vulkan BindlessDescriptorPool, falling back to standard descriptors"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(VulkanRenderer {
            phd: phd.clone(),
            device: device.clone(),
            capabilities,
            dmabuf_cache,
            pipelines,
            cmd_pool,
            texture_sampler: sampler,
            texture_sampler_nearest: sampler_nearest,
            imported_timelines: HashMap::new(),
            pending_waits: Vec::new(),
            lut3d,
            seq_no: 0,
            node,
            timeline,
            debug_flags: super::DebugFlags::empty(),
            downscale_filter: super::TextureFilter::Linear,
            upscale_filter: super::TextureFilter::Linear,
            hdr_config: None,
            supports_optimal_host_copy,
            supports_descriptor_indexing,
            supports_multi_draw_indirect,
            supports_shader_draw_parameters,
            indirect_buffer,
            bindless_pool,
            batch_submits: false,
            queued_submits: Vec::new(),
            depth_image: None,
            deferred_destructions: VecDeque::new(),
        })
    }

    /// Query the supported DMA-BUF texture formats for a [`PhysicalDevice`] without creating a `VkDevice`.
    pub fn supported_dmabuf_formats(phd: &PhysicalDevice) -> FormatSet {
        phd.drm_formats().into_iter().map(|entry| entry.format).collect()
    }

    /// Query the supported render target formats for a [`PhysicalDevice`] without creating a `VkDevice`.
    pub fn supported_render_formats(phd: &PhysicalDevice) -> Option<FormatSet> {
        known_formats()
            .iter()
            .try_fold(IndexSet::new(), |mut set, fourcc| {
                let format = get_vk_format(*fourcc).unwrap();
                let props = phd.get_format_modifier_properties(format)?;
                for prop in props {
                    if prop
                        .drm_format_modifier_tiling_features
                        .contains(FormatFeatureFlags::COLOR_ATTACHMENT)
                        || prop
                            .drm_format_modifier_tiling_features
                            .contains(FormatFeatureFlags::STORAGE_IMAGE)
                    {
                        set.insert(Format {
                            code: *fourcc,
                            modifier: Modifier::from(prop.drm_format_modifier),
                        });
                        if let Some(opaque) = get_opaque(*fourcc) {
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

    pub fn dmabuf_cache(&self) -> &Arc<std::sync::Mutex<HashMap<WeakDmabuf, VulkanImage>>> {
        &self.dmabuf_cache
    }

    pub fn shared_pipelines(&self) -> &Arc<SharedPipelines> {
        self.pipelines.shared()
    }

    /// Whether this renderer supports descriptor indexing (bindless texture arrays).
    pub fn supports_descriptor_indexing(&self) -> bool {
        self.supports_descriptor_indexing
    }

    /// Whether this renderer supports multi-draw indirect (`vkCmdDrawIndirect` with count > 1).
    pub fn supports_multi_draw_indirect(&self) -> bool {
        self.supports_multi_draw_indirect
    }

    /// Whether this renderer supports shader draw parameters (`gl_DrawID` in vertex shaders).
    pub fn supports_shader_draw_parameters(&self) -> bool {
        self.supports_shader_draw_parameters
    }

    /// Underlying draw indirect buffer, if supported and initialized.
    pub fn indirect_buffer(&self) -> Option<&DrawIndirectBuffer> {
        self.indirect_buffer.as_ref()
    }

    /// Underlying mutable draw indirect buffer, if supported and initialized.
    pub fn indirect_buffer_mut(&mut self) -> Option<&mut DrawIndirectBuffer> {
        self.indirect_buffer.as_mut()
    }

    /// Underlying bindless descriptor pool, if supported and initialized.
    pub fn bindless_pool(&self) -> Option<&BindlessDescriptorPool> {
        self.bindless_pool.as_ref()
    }

    /// Underlying mutable bindless descriptor pool, if supported and initialized.
    pub fn bindless_pool_mut(&mut self) -> Option<&mut BindlessDescriptorPool> {
        self.bindless_pool.as_mut()
    }

    pub fn begin_batch(&mut self) {
        self.cancel_batch();
        self.batch_submits = true;
    }

    pub fn cancel_batch(&mut self) {
        self.batch_submits = false;
        let queued = std::mem::take(&mut self.queued_submits);
        for q in queued {
            unsafe {
                self.device
                    .vk()
                    .free_command_buffers(self.cmd_pool.vk(), &[q.cmd_buffer]);
            }
        }
    }

    pub fn flush_batch(&mut self) -> Result<(), Error> {
        self.batch_submits = false;
        if self.queued_submits.is_empty() {
            return Ok(());
        }

        let mut queued = std::mem::take(&mut self.queued_submits);
        let device = self.device.clone();

        let submit_res = device.with_queue_lock(|_vk, queue| {
            let mut acquire_bufs: Vec<Option<vk::CommandBuffer>> = Vec::with_capacity(queued.len());
            for q in &mut queued {
                match self.record_acquire_cmd_buffer(&q.sampled_images) {
                    Ok(Some(abuf)) => {
                        q.cmd_buffer_infos
                            .insert(0, CommandBufferSubmitInfo::default().command_buffer(abuf));
                        acquire_bufs.push(Some(abuf));
                    }
                    Ok(None) => {
                        acquire_bufs.push(None);
                    }
                    Err(err) => {
                        for abuf in acquire_bufs.into_iter().flatten() {
                            unsafe {
                                device.vk().free_command_buffers(self.cmd_pool.vk(), &[abuf]);
                            }
                        }
                        return Err(err);
                    }
                }
            }

            let mut submit_infos = Vec::with_capacity(queued.len());
            for q in &queued {
                submit_infos.push(
                    SubmitInfo2::default()
                        .command_buffer_infos(&q.cmd_buffer_infos)
                        .signal_semaphore_infos(&q.signal_semaphore_infos)
                        .wait_semaphore_infos(&q.wait_semaphore_infos),
                );
            }

            let res = unsafe { device.vk().queue_submit2(queue, &submit_infos, Fence::null()) };
            if let Err(err) = res {
                for abuf in acquire_bufs.into_iter().flatten() {
                    unsafe {
                        device.vk().free_command_buffers(self.cmd_pool.vk(), &[abuf]);
                    }
                }
                return Err(Error::SubmitError(err));
            }

            Ok(acquire_bufs)
        });

        let acquire_bufs = match submit_res {
            Ok(abufs) => abufs,
            Err(err) => {
                for q in queued {
                    unsafe {
                        self.device
                            .vk()
                            .free_command_buffers(self.cmd_pool.vk(), &[q.cmd_buffer]);
                    }
                }
                return Err(err);
            }
        };

        for (q, abuf) in queued.into_iter().zip(acquire_bufs) {
            if let Some(abuf) = abuf {
                self.cmd_pool
                    .store_pending_buffer(abuf, q.point, Vec::new(), Vec::new());
            }
            self.cmd_pool
                .store_pending_buffer(q.cmd_buffer, q.point, q.descs, q.images);
        }

        Ok(())
    }

    fn record_acquire_cmd_buffer(
        &mut self,
        sampled_images: &[VulkanImage],
    ) -> Result<Option<vk::CommandBuffer>, Error> {
        if sampled_images.is_empty() {
            return Ok(None);
        }
        let qfam = self.device.queue_family_idx();
        let ext_queue = self.external_queue_family();
        let mut barriers = Vec::new();
        let mut transitioned: Vec<&VulkanImage> = Vec::new();

        for texture in sampled_images {
            let tex_needs_acquire = texture.needs_acquire() && texture.dmabuf_exportable();
            let tex_old_layout = texture.current_layout();
            let needs_tex_barrier =
                tex_needs_acquire || tex_old_layout != ImageLayout::SHADER_READ_ONLY_OPTIMAL;

            if needs_tex_barrier {
                if transitioned.iter().any(|t| Arc::ptr_eq(&t.inner, &texture.inner)) {
                    continue;
                }
                let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
                    (ext_queue, qfam)
                } else {
                    (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
                };
                let (tex_src_stage, tex_src_access) =
                    src_stage_and_access_for_layout(tex_needs_acquire, tex_old_layout);

                barriers.push(
                    ImageMemoryBarrier2::default()
                        .image(*texture.vk())
                        .old_layout(tex_old_layout)
                        .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(tex_src_queue)
                        .dst_queue_family_index(tex_dst_queue)
                        .src_stage_mask(tex_src_stage)
                        .src_access_mask(tex_src_access)
                        .dst_stage_mask(PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(AccessFlags2::SHADER_SAMPLED_READ)
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        ),
                );
                transitioned.push(texture);
            }
        }

        if barriers.is_empty() {
            return Ok(None);
        }

        let buf = self.cmd_pool.create_and_begin_buffer()?;
        unsafe {
            self.device
                .vk()
                .cmd_pipeline_barrier2(buf, &DependencyInfo::default().image_memory_barriers(&barriers));
            self.device
                .vk()
                .end_command_buffer(buf)
                .map_err(Error::CommandBufferError)?;
        }
        for tex in transitioned {
            tex.set_current_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            tex.set_needs_acquire(false);
        }
        Ok(Some(buf))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn supports_optimal_host_copy(&self) -> bool {
        self.supports_optimal_host_copy
    }

    pub fn supports_push_descriptor(&self) -> bool {
        self.capabilities.contains(&Capability::PushDescriptor)
    }

    pub fn supports_memory_budget(&self) -> bool {
        self.capabilities.contains(&Capability::MemoryBudget)
    }

    pub fn supports_dynamic_rendering(&self) -> bool {
        self.capabilities.contains(&Capability::DynamicRendering)
    }

    pub fn memory_budget(&self) -> Option<MemoryBudgetInfo> {
        if !self.supports_memory_budget() {
            return None;
        }

        let mut budget_prop = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut mem_prop2 = vk::PhysicalDeviceMemoryProperties2::default();
        mem_prop2.p_next = &mut budget_prop as *mut _ as *mut _;

        unsafe {
            self.phd
                .instance()
                .handle()
                .get_physical_device_memory_properties2(self.phd.handle(), &mut mem_prop2);
        }

        Some(MemoryBudgetInfo {
            heap_budget: budget_prop.heap_budget,
            heap_usage: budget_prop.heap_usage,
        })
    }

    pub fn external_queue_family(&self) -> u32 {
        if self.capabilities.contains(&Capability::QueueFamilyForeign) {
            vk::QUEUE_FAMILY_FOREIGN_EXT
        } else {
            vk::QUEUE_FAMILY_EXTERNAL
        }
    }

    /// Defers execution of a destruction closure until the GPU timeline semaphore reaches `point`.
    pub(crate) fn defer_cleanup(&mut self, point: u64, f: impl FnOnce(&Device) + Send + 'static) {
        self.deferred_destructions.push_back(DeferredDestruction {
            point,
            callback: Box::new(f),
        });
    }

    pub fn update_3d_lut(&mut self, ref_white: f32, max_content: f32, max_dest: f32) {
        let params = (ref_white as u32, max_content as u32, max_dest as u32);
        if let Some(lut) = self.lut3d.as_ref() {
            if lut.params == params {
                return;
            }
        }
        let data = generate_ictcp_tonemap_lut(33, ref_white, 203.0, max_content, max_dest);
        if let Ok(new_lut) = Lut3dTexture::new(&self.device, &data, 33, params) {
            if let Some(mut old) = self.lut3d.take() {
                let seq = self.seq_no;
                self.defer_cleanup(seq, move |device| unsafe {
                    old.destroy(device);
                });
            }
            self.lut3d = Some(new_lut);
        }
    }

    pub fn set_hdr_output(&mut self, config: Option<HdrOutputConfig>) {
        if let Some(c) = config {
            self.update_3d_lut(c.reference_white, c.max_luminance, c.max_luminance);
        }
        self.hdr_config = config;
    }

    pub fn hdr_output(&self) -> Option<HdrOutputConfig> {
        self.hdr_config
    }

    pub fn blit_hdr_to_sdr(
        &mut self,
        from: &<Self as RendererSuper>::Framebuffer<'_>,
        to: &mut <Self as RendererSuper>::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
        config: &HdrOutputConfig,
    ) -> Result<SyncPoint, Error> {
        Blit::blit_hdr_to_sdr(self, from, to, src, dst, filter, config)
    }

    pub fn blit_sdr_to_hdr(
        &mut self,
        from: &<Self as RendererSuper>::Framebuffer<'_>,
        to: &mut <Self as RendererSuper>::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
        config: &HdrOutputConfig,
    ) -> Result<SyncPoint, Error> {
        Blit::blit_sdr_to_hdr(self, from, to, src, dst, filter, config)
    }

    pub fn node(&self) -> Option<DrmNode> {
        self.node
    }

    pub fn drm_timeline(&self) -> Option<&DrmTimeline> {
        self.timeline.drm.as_ref()
    }

    pub fn timeline_semaphore(&self) -> vk::Semaphore {
        self.timeline.vk
    }

    pub fn cleanup(&mut self) -> Result<(), Error> {
        let val = unsafe {
            self.device
                .vk()
                .get_semaphore_counter_value(self.timeline.vk)
                .map_err(Error::SemaphoreCounterError)?
        };
        self.cmd_pool.clean_old_buffers(val);
        if self.cmd_pool.is_empty() {
            if let Some(ref mut ib) = self.indirect_buffer {
                ib.reset();
            }
        }

        // Retire deferred destructions whose timeline point <= val
        while let Some(front) = self.deferred_destructions.front() {
            if front.point <= val {
                let d = self.deferred_destructions.pop_front().unwrap();
                (d.callback)(&self.device);
            } else {
                break;
            }
        }

        self.imported_timelines.retain(|weak, sem| {
            if weak.upgrade().is_none() {
                unsafe { self.device.vk().destroy_semaphore(*sem, None) };
                false
            } else {
                true
            }
        });
        Ok(())
    }

    pub fn active_sampler(&self, is_downscaling: bool) -> vk::Sampler {
        let filter = if is_downscaling {
            self.downscale_filter
        } else {
            self.upscale_filter
        };
        match filter {
            super::TextureFilter::Linear => self.texture_sampler,
            super::TextureFilter::Nearest => self.texture_sampler_nearest,
        }
    }

    pub(super) fn get_or_import_timeline_semaphore(
        &mut self,
        timeline: &DrmTimeline,
    ) -> Result<vk::Semaphore, Error> {
        let weak = timeline.downgrade();
        if let Some(&sem) = self.imported_timelines.get(&weak) {
            return Ok(sem);
        }
        let sem = self.device.import_timeline_semaphore(timeline.timeline_fd())?;
        self.imported_timelines.insert(weak, sem);
        Ok(sem)
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

            let map_ptr = image.map().map_err(Error::HostImageCopyError)?;

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
                    let atom = self.device.non_coherent_atom_size().max(1);
                    let flush_size = (layout.size + atom - 1) & !(atom - 1);
                    let range = vk::MappedMemoryRange::default()
                        .memory(image.inner.memory)
                        .offset(image.inner.memory_offset)
                        .size(
                            if !image.inner.suballocated && flush_size >= image.inner.allocation_size {
                                vk::WHOLE_SIZE
                            } else {
                                flush_size
                            },
                        );
                    let _ = self.device.vk().flush_mapped_memory_ranges(&[range]);
                }
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

        // Transition host-uploaded sampled textures (SHM, import_memory, NinePatch) immediately
        // to SHADER_READ_ONLY_OPTIMAL on the host so multiple output threads can sample them
        // concurrently without needing any GPU pipeline barrier or mutating current_layout.
        if image.vk_usage().contains(ImageUsageFlags::SAMPLED)
            && !image.dmabuf_exportable()
            && !image.vk_usage().contains(ImageUsageFlags::COLOR_ATTACHMENT)
        {
            unsafe {
                device_copy
                    .transition_image_layout(&[vk::HostImageLayoutTransitionInfoEXT::default()
                        .old_layout(ImageLayout::GENERAL)
                        .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(*image.vk())
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        )])
                    .map_err(Error::HostImageTransitionError)?;
            }
            image.set_current_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL);
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
            .and_then(get_bpp)
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
        let pending_waits = std::mem::take(&mut self.pending_waits);
        Ok(VulkanFrame {
            renderer: self,
            fb: framebuffer,
            transform: dst_transform,
            size: output_size,
            cmd_buffer: None,
            descriptors: Vec::with_capacity(16),
            images: Vec::with_capacity(16),
            sampled_images: Vec::with_capacity(16),
            has_draws: false,
            _marker: std::marker::PhantomData,
            #[cfg(feature = "wayland_frontend")]
            active_color_description: None,
            active_clip: None,
            is_blit: false,
            rendering: false,
            depth_enabled: false,
            current_depth: 0.0,
            pending_clears: Vec::new(),
            pending_waits,
            hdr_to_sdr: None,
            sdr_to_hdr: None,
            release_source: None,
        })
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        if let Some(drm_sync) = sync.get::<DrmSyncPoint>() {
            if let Some(timeline) = self.timeline.drm.as_ref() {
                if timeline == drm_sync.timeline() {
                    // On Vulkan, all submissions on this renderer queue already wait on the
                    // previous sequence number via self.timeline.vk.
                    // Since drm_sync.point() <= self.seq_no, GPU queue ordering is guaranteed
                    // without any CPU-side stall.
                    return Ok(());
                }
            }
            self.pending_waits.push(drm_sync.clone());
            return Ok(());
        }
        if let Some(vk_sync) = sync.get::<VulkanSyncPoint>() {
            if self.timeline == vk_sync.timeline {
                return Ok(());
            }
        }
        while let Err(super::sync::Interrupted) = sync.wait() {}
        Ok(())
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        let _ = self.cleanup();
        self.dmabuf_cache
            .lock()
            .unwrap()
            .retain(|weak, _| weak.upgrade().is_some());
        Ok(())
    }

    fn invalidate_caches(&mut self) -> Result<(), Self::Error> {
        self.dmabuf_cache.lock().unwrap().clear();
        let _ = self.cleanup();
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
        let bpp = get_bpp(format).unwrap_or(32) / 8;
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
        .or_else(|err| {
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
                Err(err)
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
            .and_then(get_bpp)
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
    #[profiling::function]
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoords>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        let fourcc = Buffer::format(dmabuf).code;
        let ycbcr_key = if is_ycbcr_fourcc(fourcc) {
            YcbcrKey::default_for_fourcc(fourcc)
        } else {
            None
        };
        let required_usage = ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_SRC;
        if let Some(image) = self.dmabuf_cache.lock().unwrap().get(&dmabuf.weak()).cloned() {
            if image.vk_usage().contains(required_usage) && image.ycbcr() == ycbcr_key {
                trace!("Re-using VulkanImage {:?} for {:?}", image.vk(), dmabuf);
                image.set_current_layout(ImageLayout::GENERAL);
                image.set_needs_acquire(true);
                return Ok(image);
            }
        }

        let image = VulkanImage::new_from_dmabuf_with_ycbcr(&self.device, dmabuf, required_usage, ycbcr_key)
            .map_err(Error::ImageError)?;

        self.dmabuf_cache
            .lock()
            .unwrap()
            .insert(dmabuf.weak(), image.clone());
        Ok(image)
    }

    fn dmabuf_formats(&self) -> FormatSet {
        self.device.formats().map(|entry| entry.format.clone()).collect()
    }

    fn has_dmabuf_format(&self, format: Format) -> bool {
        self.device.formats().any(|entry| entry.format == format)
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportMemWl for VulkanRenderer {
    #[profiling::function]
    fn import_shm_buffer(
        &mut self,
        buffer: &WlBuffer,
        surface: Option<&SurfaceData>,
        damage: &[Rectangle<i32, BufferCoords>],
    ) -> Result<Self::TextureId, Self::Error> {
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
                .filter(|texture| Texture::size(texture) == size && texture.format() == expected_vk_format);

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
        Box::new(self.mem_formats().filter_map(fourcc_to_shm_format))
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportDmaWl for VulkanRenderer {
    #[profiling::function]
    fn import_dma_buffer(
        &mut self,
        buffer: &WlBuffer,
        surface: Option<&SurfaceData>,
        _damage: &[Rectangle<i32, BufferCoords>],
    ) -> Result<Self::TextureId, Self::Error> {
        let dmabuf = get_dmabuf(buffer).expect("import_dma_buffer without checking buffer type?");

        let fourcc = Buffer::format(dmabuf).code;
        let ycbcr_key = if is_ycbcr_fourcc(fourcc) {
            let color_rep = surface.map(get_color_representation_from_states);
            color_rep
                .as_ref()
                .and_then(|rep| YcbcrKey::from_color_representation(fourcc, rep))
                .or_else(|| YcbcrKey::default_for_fourcc(fourcc))
        } else {
            None
        };

        let required_usage = ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_SRC;
        if let Some(image) = self.dmabuf_cache.lock().unwrap().get(&dmabuf.weak()).cloned() {
            if image.vk_usage().contains(required_usage) && image.ycbcr() == ycbcr_key {
                trace!("Re-using VulkanImage {:?} for {:?}", image.vk(), dmabuf);
                image.set_current_layout(ImageLayout::GENERAL);
                image.set_needs_acquire(true);
                return Ok(image);
            }
        }

        let image = VulkanImage::new_from_dmabuf_with_ycbcr(&self.device, dmabuf, required_usage, ycbcr_key)
            .map_err(Error::ImageError)?;

        self.dmabuf_cache
            .lock()
            .unwrap()
            .insert(dmabuf.weak(), image.clone());
        Ok(image)
    }
}

pub enum VulkanMapping {
    Mapped(NonNull<u8>, usize, VulkanImage),
    Copied(Vec<u8>, VulkanImage),
}

impl fmt::Debug for VulkanMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mapped(_, _, image) => f
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
        // Mapped memory is persistently mapped and will be unmapped when the underlying VulkanImage is dropped.
    }
}

impl TextureMapping for VulkanMapping {
    fn flipped(&self) -> bool {
        false
    }
}

impl Texture for VulkanMapping {
    fn width(&self) -> u32 {
        let (Self::Mapped(_, _, image) | Self::Copied(_, image)) = self;
        image.width()
    }

    fn height(&self) -> u32 {
        let (Self::Mapped(_, _, image) | Self::Copied(_, image)) = self;
        image.height()
    }

    fn format(&self) -> Option<Fourcc> {
        let (Self::Mapped(_, _, image) | Self::Copied(_, image)) = self;
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

            let base_ptr = image.map().map_err(Error::HostImageCopyError)?;
            let ptr = unsafe { base_ptr.add(layout.offset as usize) };

            if !image.mem_bits().contains(MemoryPropertyFlags::HOST_COHERENT) {
                let atom = self.device.non_coherent_atom_size().max(1);
                let inv_size = (layout.size + atom - 1) & !(atom - 1);
                let range = vk::MappedMemoryRange::default()
                    .memory(image.inner.memory)
                    .offset(image.inner.memory_offset)
                    .size(
                        if !image.inner.suballocated && inv_size >= image.inner.allocation_size {
                            vk::WHOLE_SIZE
                        } else {
                            inv_size
                        },
                    );
                unsafe {
                    let _ = self.device.vk().invalidate_mapped_memory_ranges(&[range]);
                }
            }

            Ok(VulkanMapping::Mapped(
                unsafe { NonNull::new_unchecked(ptr as *mut _) },
                layout.size as usize,
                image.clone(),
            ))
        } else {
            self.copy_image_via_staging_buffer(image, region, format)
        }
    }

    fn copy_image_via_staging_buffer(
        &mut self,
        image: &VulkanImage,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<VulkanMapping, Error> {
        if !self.queued_submits.is_empty() {
            let was_batch = self.batch_submits;
            self.flush_batch()?;
            self.batch_submits = was_batch;
        }

        let bpp = match format {
            Fourcc::Abgr2101010 | Fourcc::Xbgr2101010 | Fourcc::Argb2101010 | Fourcc::Xrgb2101010 => 4usize,
            Fourcc::Abgr16161616f | Fourcc::Xbgr16161616f => 8usize,
            _ => 4usize,
        };

        let copy_width = (region.size.w as u32).min(image.width()).max(1);
        let copy_height = (region.size.h as u32).min(image.height()).max(1);
        let offset_x = region.loc.x.max(0);
        let offset_y = region.loc.y.max(0);
        let stride = copy_width as usize * bpp;
        let buffer_size = (stride * copy_height as usize) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe {
            self.device
                .vk()
                .create_buffer(&buffer_info, None)
                .map_err(Error::HostImageCopyError)?
        };

        let mem_reqs = unsafe { self.device.vk().get_buffer_memory_requirements(staging_buffer) };

        let mem_type = self
            .device
            .find_memory_type(mem_reqs.memory_type_bits, MemoryUsagePreference::HostVisible);

        let Some(mem_type) = mem_type else {
            unsafe {
                self.device.vk().destroy_buffer(staging_buffer, None);
            }
            return Err(Error::ImageError(ImageError::NoMemoryAvailable));
        };

        let mem_type_index = mem_type.type_index;
        let is_coherent = mem_type
            .property_flags
            .contains(MemoryPropertyFlags::HOST_COHERENT);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let staging_memory = match unsafe { self.device.vk().allocate_memory(&alloc_info, None) } {
            Ok(mem) => mem,
            Err(err) => {
                unsafe {
                    self.device.vk().destroy_buffer(staging_buffer, None);
                }
                return Err(Error::ImageError(ImageError::VulkanAllocate(err)));
            }
        };

        if let Err(err) = unsafe {
            self.device
                .vk()
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
        } {
            unsafe {
                self.device.vk().destroy_buffer(staging_buffer, None);
                self.device.vk().free_memory(staging_memory, None);
            }
            return Err(Error::ImageError(ImageError::VulkanBind(err)));
        }

        struct StagingCleanup {
            vk: ash::Device,
            buffer: vk::Buffer,
            memory: vk::DeviceMemory,
        }
        impl Drop for StagingCleanup {
            fn drop(&mut self) {
                unsafe {
                    if self.buffer != vk::Buffer::null() {
                        self.vk.destroy_buffer(self.buffer, None);
                    }
                    if self.memory != vk::DeviceMemory::null() {
                        self.vk.free_memory(self.memory, None);
                    }
                }
            }
        }
        let cleanup_guard = StagingCleanup {
            vk: self.device.vk().clone(),
            buffer: staging_buffer,
            memory: staging_memory,
        };

        self.cleanup()?;
        let buf = self.cmd_pool.create_and_begin_buffer()?;

        let qfam = self.device.queue_family_idx();
        let ext_queue = self.external_queue_family();
        let tex_needs_acquire = image.needs_acquire() && image.dmabuf_exportable();
        let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let old_layout = image.current_layout();
        let (tex_src_stage, tex_src_access) = src_stage_and_access_for_layout(tex_needs_acquire, old_layout);

        let img_barrier = ImageMemoryBarrier2::default()
            .image(*image.vk())
            .old_layout(old_layout)
            .new_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(tex_src_queue)
            .dst_queue_family_index(tex_dst_queue)
            .src_stage_mask(tex_src_stage)
            .src_access_mask(tex_src_access)
            .dst_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(AccessFlags2::TRANSFER_READ)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.device.vk().cmd_pipeline_barrier2(
                buf,
                &DependencyInfo::default().image_memory_barriers(&[img_barrier]),
            );
        }

        image.set_needs_acquire(false);

        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(copy_width)
            .buffer_image_height(copy_height)
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
            .image_extent(Extent3D {
                width: copy_width,
                height: copy_height,
                depth: 1,
            });

        unsafe {
            self.device.vk().cmd_copy_image_to_buffer(
                buf,
                *image.vk(),
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buffer,
                &[copy_region],
            );
        }

        let (restore_layout, restore_src_q, restore_dst_q, restore_dst_stage, restore_dst_access) =
            if tex_needs_acquire {
                (
                    ImageLayout::GENERAL,
                    qfam,
                    ext_queue,
                    PipelineStageFlags2::NONE,
                    AccessFlags2::NONE,
                )
            } else {
                let layout = if old_layout == ImageLayout::UNDEFINED {
                    ImageLayout::GENERAL
                } else {
                    old_layout
                };
                (
                    layout,
                    QUEUE_FAMILY_IGNORED,
                    QUEUE_FAMILY_IGNORED,
                    PipelineStageFlags2::ALL_COMMANDS,
                    AccessFlags2::MEMORY_READ | AccessFlags2::MEMORY_WRITE,
                )
            };
        let restore_barrier = ImageMemoryBarrier2::default()
            .image(*image.vk())
            .old_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(restore_layout)
            .src_queue_family_index(restore_src_q)
            .dst_queue_family_index(restore_dst_q)
            .src_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(restore_dst_stage)
            .dst_access_mask(restore_dst_access)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.device.vk().cmd_pipeline_barrier2(
                buf,
                &DependencyInfo::default().image_memory_barriers(&[restore_barrier]),
            );
            self.device
                .vk()
                .end_command_buffer(buf)
                .map_err(Error::CommandBufferError)?;
        }
        image.set_current_layout(restore_layout);
        if tex_needs_acquire {
            image.set_needs_acquire(true);
        }

        let cmd_buffer_info = [CommandBufferSubmitInfo::default().command_buffer(buf)];
        let next_seq_no = self.seq_no + 1;
        let prev_seq_no = self.seq_no;

        let signal_semaphore_info = [SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.vk)
            .value(next_seq_no)
            .stage_mask(PipelineStageFlags2::ALL_TRANSFER)];

        let wait_semaphore_info = if prev_seq_no > 0 {
            vec![
                SemaphoreSubmitInfo::default()
                    .semaphore(self.timeline.vk)
                    .value(prev_seq_no)
                    .stage_mask(PipelineStageFlags2::ALL_TRANSFER),
            ]
        } else {
            Vec::new()
        };

        let submit_info = SubmitInfo2::default()
            .command_buffer_infos(&cmd_buffer_info)
            .signal_semaphore_infos(&signal_semaphore_info)
            .wait_semaphore_infos(&wait_semaphore_info);

        unsafe {
            self.device
                .queue_submit2(&[submit_info], Fence::null())
                .map_err(Error::SubmitError)?;
        }

        self.seq_no = next_seq_no;
        let point = next_seq_no;
        self.cmd_pool
            .store_pending_buffer(buf, point, vec![], vec![image.inner.clone()]);

        while let Err(VkResult::TIMEOUT) = unsafe {
            self.device.vk().wait_semaphores(
                &SemaphoreWaitInfo::default()
                    .semaphores(&[self.timeline.vk])
                    .values(&[point]),
                u64::MAX,
            )
        } {}

        let ptr = unsafe {
            self.device
                .vk()
                .map_memory(staging_memory, 0, buffer_size, MemoryMapFlags::empty())
                .map_err(Error::HostImageCopyError)?
        };

        if !is_coherent {
            let mapped_range = vk::MappedMemoryRange::default()
                .memory(staging_memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                let _ = self.device.vk().invalidate_mapped_memory_ranges(&[mapped_range]);
            }
        }

        let mut data = vec![0u8; buffer_size as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, data.as_mut_ptr(), buffer_size as usize);
            self.device.vk().unmap_memory(staging_memory);
        }

        drop(cleanup_guard);
        self.cleanup()?;

        Ok(VulkanMapping::Copied(data, image.clone()))
    }

    pub fn record_copy_image_to_shm(
        &mut self,
        image: &VulkanImage,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<PendingVulkanShmCopy, Error> {
        if !self.batch_submits && !self.queued_submits.is_empty() {
            self.flush_batch()?;
        }

        let bpp = match format {
            Fourcc::Abgr2101010 | Fourcc::Xbgr2101010 | Fourcc::Argb2101010 | Fourcc::Xrgb2101010 => 4usize,
            Fourcc::Abgr16161616f | Fourcc::Xbgr16161616f => 8usize,
            _ => 4usize,
        };

        let copy_width = (region.size.w as u32).min(image.width()).max(1);
        let copy_height = (region.size.h as u32).min(image.height()).max(1);
        let offset_x = region.loc.x.max(0);
        let offset_y = region.loc.y.max(0);
        let stride = copy_width as usize * bpp;
        let buffer_size = (stride * copy_height as usize) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe {
            self.device
                .vk()
                .create_buffer(&buffer_info, None)
                .map_err(Error::HostImageCopyError)?
        };

        let mem_reqs = unsafe { self.device.vk().get_buffer_memory_requirements(staging_buffer) };

        let mem_type = self
            .device
            .find_memory_type(mem_reqs.memory_type_bits, MemoryUsagePreference::HostVisible);

        let Some(mem_type) = mem_type else {
            unsafe {
                self.device.vk().destroy_buffer(staging_buffer, None);
            }
            return Err(Error::ImageError(ImageError::NoMemoryAvailable));
        };

        let mem_type_index = mem_type.type_index;
        let is_coherent = mem_type
            .property_flags
            .contains(MemoryPropertyFlags::HOST_COHERENT);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let staging_memory = match unsafe { self.device.vk().allocate_memory(&alloc_info, None) } {
            Ok(mem) => mem,
            Err(err) => {
                unsafe {
                    self.device.vk().destroy_buffer(staging_buffer, None);
                }
                return Err(Error::ImageError(ImageError::VulkanAllocate(err)));
            }
        };

        if let Err(err) = unsafe {
            self.device
                .vk()
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
        } {
            unsafe {
                self.device.vk().destroy_buffer(staging_buffer, None);
                self.device.vk().free_memory(staging_memory, None);
            }
            return Err(Error::ImageError(ImageError::VulkanBind(err)));
        }

        self.cleanup()?;
        let buf = self.cmd_pool.create_and_begin_buffer()?;

        let qfam = self.device.queue_family_idx();
        let ext_queue = self.external_queue_family();
        let tex_needs_acquire = image.needs_acquire() && image.dmabuf_exportable();
        let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let old_layout = image.current_layout();
        let (tex_src_stage, tex_src_access) = src_stage_and_access_for_layout(tex_needs_acquire, old_layout);

        let img_barrier = ImageMemoryBarrier2::default()
            .image(*image.vk())
            .old_layout(old_layout)
            .new_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(tex_src_queue)
            .dst_queue_family_index(tex_dst_queue)
            .src_stage_mask(tex_src_stage)
            .src_access_mask(tex_src_access)
            .dst_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(AccessFlags2::TRANSFER_READ)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.device.vk().cmd_pipeline_barrier2(
                buf,
                &DependencyInfo::default().image_memory_barriers(&[img_barrier]),
            );
        }

        image.set_needs_acquire(false);

        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(copy_width)
            .buffer_image_height(copy_height)
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
            .image_extent(Extent3D {
                width: copy_width,
                height: copy_height,
                depth: 1,
            });

        unsafe {
            self.device.vk().cmd_copy_image_to_buffer(
                buf,
                *image.vk(),
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buffer,
                &[copy_region],
            );
        }

        let (restore_layout, restore_src_q, restore_dst_q, restore_dst_stage, restore_dst_access) =
            if tex_needs_acquire {
                (
                    ImageLayout::GENERAL,
                    qfam,
                    ext_queue,
                    PipelineStageFlags2::NONE,
                    AccessFlags2::NONE,
                )
            } else {
                let layout = if old_layout == ImageLayout::UNDEFINED {
                    ImageLayout::GENERAL
                } else {
                    old_layout
                };
                (
                    layout,
                    QUEUE_FAMILY_IGNORED,
                    QUEUE_FAMILY_IGNORED,
                    PipelineStageFlags2::ALL_COMMANDS,
                    AccessFlags2::MEMORY_READ | AccessFlags2::MEMORY_WRITE,
                )
            };
        let restore_barrier = ImageMemoryBarrier2::default()
            .image(*image.vk())
            .old_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(restore_layout)
            .src_queue_family_index(restore_src_q)
            .dst_queue_family_index(restore_dst_q)
            .src_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(restore_dst_stage)
            .dst_access_mask(restore_dst_access)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.device.vk().cmd_pipeline_barrier2(
                buf,
                &DependencyInfo::default().image_memory_barriers(&[restore_barrier]),
            );
            self.device
                .vk()
                .end_command_buffer(buf)
                .map_err(Error::CommandBufferError)?;
        }
        image.set_current_layout(restore_layout);
        if tex_needs_acquire {
            image.set_needs_acquire(true);
        }

        let cmd_buffer_info = [CommandBufferSubmitInfo::default().command_buffer(buf)];
        let next_seq_no = self.seq_no + 1;
        let prev_seq_no = self.seq_no;

        let signal_semaphore_info = [SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.vk)
            .value(next_seq_no)
            .stage_mask(PipelineStageFlags2::ALL_TRANSFER)];

        let wait_semaphore_info = if prev_seq_no > 0 {
            vec![
                SemaphoreSubmitInfo::default()
                    .semaphore(self.timeline.vk)
                    .value(prev_seq_no)
                    .stage_mask(PipelineStageFlags2::ALL_TRANSFER),
            ]
        } else {
            Vec::new()
        };

        if self.batch_submits {
            self.queued_submits.push(QueuedSubmit {
                cmd_buffer: buf,
                cmd_buffer_infos: cmd_buffer_info.to_vec(),
                signal_semaphore_infos: signal_semaphore_info.to_vec(),
                wait_semaphore_infos: wait_semaphore_info,
                point: next_seq_no,
                descs: Vec::new(),
                images: vec![image.inner.clone()],
                sampled_images: Vec::new(),
            });
        } else {
            let submit_info = SubmitInfo2::default()
                .command_buffer_infos(&cmd_buffer_info)
                .signal_semaphore_infos(&signal_semaphore_info)
                .wait_semaphore_infos(&wait_semaphore_info);

            unsafe {
                self.device
                    .queue_submit2(&[submit_info], Fence::null())
                    .map_err(Error::SubmitError)?;
            }

            self.cmd_pool
                .store_pending_buffer(buf, next_seq_no, vec![], vec![image.inner.clone()]);
        }

        self.seq_no = next_seq_no;

        Ok(PendingVulkanShmCopy {
            device: self.device.clone(),
            staging_buffer,
            staging_memory,
            buffer_size,
            is_coherent,
            timeline: self.timeline.clone(),
            point: next_seq_no,
            width: copy_width,
            height: copy_height,
            bpp,
        })
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
                || (texture.vk_usage().contains(ImageUsageFlags::HOST_TRANSFER_EXT)
                    && self.device.vk_ext_host_image_copy().is_some())
                || texture.vk_usage().contains(ImageUsageFlags::TRANSFER_SRC),
        )
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        match texture_mapping {
            VulkanMapping::Mapped(ptr, len, _) => {
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

pub(crate) fn calculate_damage_scissors(
    damage: &[Rectangle<i32, Physical>],
    untransformed_dst: Rectangle<i32, Physical>,
    transform: Transform,
    screen_size: &Size<i32, Physical>,
    fb_width: u32,
    fb_height: u32,
) -> Vec<vk::Rect2D> {
    if damage.is_empty() {
        return Vec::new();
    }

    let dest_size: Size<i32, Physical> = Size::new(fb_width as i32, fb_height as i32);
    let mut raw_rects: Vec<Rectangle<i32, Physical>> = Vec::with_capacity(damage.len());

    for &rect in damage {
        let mut r = rect;
        r.loc += untransformed_dst.loc;
        let Some(intersection) = r.intersection(untransformed_dst) else {
            continue;
        };
        let r = transform.transform_rect_in(intersection, screen_size);
        let Some(clipped) = r.intersection(Rectangle::from_size(dest_size)) else {
            continue;
        };

        if clipped.size.w > 0 && clipped.size.h > 0 {
            raw_rects.push(clipped);
        }
    }

    if raw_rects.len() <= 1 {
        return raw_rects
            .into_iter()
            .map(|r| vk::Rect2D {
                offset: vk::Offset2D {
                    x: r.loc.x,
                    y: r.loc.y,
                },
                extent: vk::Extent2D {
                    width: r.size.w as u32,
                    height: r.size.h as u32,
                },
            })
            .collect();
    }

    // Coalesce / merge rectangles (lossless exact merges only):
    let mut merged = raw_rects;
    loop {
        let mut changed = false;
        let mut i = 0;
        while i < merged.len() {
            let mut j = i + 1;
            while j < merged.len() {
                let a = merged[i];
                let b = merged[j];

                // 1. If A contains B, drop B
                if a.contains_rect(b) {
                    merged.swap_remove(j);
                    changed = true;
                    continue;
                }
                // 2. If B contains A, replace A with B, drop B
                if b.contains_rect(a) {
                    merged[i] = b;
                    merged.swap_remove(j);
                    changed = true;
                    continue;
                }
                // 3. Exact horizontal strip merge: same x & w, touching or overlapping in y
                let can_merge_vertically = a.loc.x == b.loc.x
                    && a.size.w == b.size.w
                    && ((a.loc.y + a.size.h >= b.loc.y && b.loc.y + b.size.h >= a.loc.y)
                        || a.loc.y + a.size.h == b.loc.y
                        || b.loc.y + b.size.h == a.loc.y);

                if can_merge_vertically {
                    let y1 = a.loc.y.min(b.loc.y);
                    let y2 = (a.loc.y + a.size.h).max(b.loc.y + b.size.h);
                    merged[i].loc.y = y1;
                    merged[i].size.h = y2 - y1;
                    merged.swap_remove(j);
                    changed = true;
                    continue;
                }

                // 4. Exact vertical strip merge: same y & h, touching or overlapping in x
                let can_merge_horizontally = a.loc.y == b.loc.y
                    && a.size.h == b.size.h
                    && ((a.loc.x + a.size.w >= b.loc.x && b.loc.x + b.size.w >= a.loc.x)
                        || a.loc.x + a.size.w == b.loc.x
                        || b.loc.x + b.size.w == a.loc.x);

                if can_merge_horizontally {
                    let x1 = a.loc.x.min(b.loc.x);
                    let x2 = (a.loc.x + a.size.w).max(b.loc.x + b.size.w);
                    merged[i].loc.x = x1;
                    merged[i].size.w = x2 - x1;
                    merged.swap_remove(j);
                    changed = true;
                    continue;
                }

                j += 1;
            }
            i += 1;
        }
        if !changed {
            break;
        }
    }

    merged
        .into_iter()
        .map(|r| vk::Rect2D {
            offset: vk::Offset2D {
                x: r.loc.x,
                y: r.loc.y,
            },
            extent: vk::Extent2D {
                width: r.size.w as u32,
                height: r.size.h as u32,
            },
        })
        .collect()
}

/// Helper function to check if a Vulkan format represents an HDR format (10-bit or FP16).
pub fn is_hdr_vk_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::A2B10G10R10_UNORM_PACK32
            | vk::Format::A2R10G10B10_UNORM_PACK32
            | vk::Format::R16G16B16A16_SFLOAT
            | vk::Format::R16G16B16A16_UNORM
    )
}

fn src_stage_and_access_for_layout(
    needs_acquire: bool,
    old_layout: ImageLayout,
) -> (PipelineStageFlags2, AccessFlags2) {
    if needs_acquire || old_layout == ImageLayout::UNDEFINED {
        (PipelineStageFlags2::NONE, AccessFlags2::NONE)
    } else if old_layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        (
            PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            AccessFlags2::COLOR_ATTACHMENT_READ | AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )
    } else if old_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL {
        (
            PipelineStageFlags2::FRAGMENT_SHADER,
            AccessFlags2::SHADER_SAMPLED_READ,
        )
    } else if old_layout == ImageLayout::TRANSFER_SRC_OPTIMAL {
        (PipelineStageFlags2::ALL_TRANSFER, AccessFlags2::TRANSFER_READ)
    } else if old_layout == ImageLayout::TRANSFER_DST_OPTIMAL {
        (PipelineStageFlags2::ALL_TRANSFER, AccessFlags2::TRANSFER_WRITE)
    } else {
        (
            PipelineStageFlags2::COMPUTE_SHADER
                | PipelineStageFlags2::ALL_TRANSFER
                | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                | PipelineStageFlags2::FRAGMENT_SHADER
                | PipelineStageFlags2::HOST,
            AccessFlags2::SHADER_STORAGE_READ
                | AccessFlags2::SHADER_STORAGE_WRITE
                | AccessFlags2::TRANSFER_READ
                | AccessFlags2::TRANSFER_WRITE
                | AccessFlags2::COLOR_ATTACHMENT_READ
                | AccessFlags2::COLOR_ATTACHMENT_WRITE
                | AccessFlags2::SHADER_SAMPLED_READ
                | AccessFlags2::HOST_WRITE,
        )
    }
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
    sampled_images: Vec<VulkanImage>,
    has_draws: bool,
    #[cfg(feature = "wayland_frontend")]
    active_color_description: Option<ImageDescription>,
    active_clip: Option<(Rectangle<i32, Physical>, [f32; 4])>,
    is_blit: bool,
    rendering: bool,
    pub(crate) depth_enabled: bool,
    pub(crate) current_depth: f32,
    pub(crate) pending_clears: Vec<(Color32F, Vec<vk::ClearRect>)>,
    pub(crate) pending_waits: Vec<DrmSyncPoint>,
    pub(crate) hdr_to_sdr: Option<HdrOutputConfig>,
    pub(crate) sdr_to_hdr: Option<HdrOutputConfig>,
    pub(crate) release_source: Option<VulkanImage>,
}

impl VulkanFrame<'_, '_> {
    pub fn set_depth_enabled(&mut self, enabled: bool) {
        if self.depth_enabled != enabled && self.rendering {
            self.end_rendering();
        }
        self.depth_enabled = enabled;
    }

    pub fn depth_enabled(&self) -> bool {
        self.depth_enabled
    }

    pub fn set_current_depth(&mut self, depth: f32) {
        self.current_depth = depth.clamp(0.0, 1.0);
    }

    pub fn current_depth(&self) -> f32 {
        self.current_depth
    }

    /// Issues an indirect multi-draw command using `vkCmdDrawIndirect`.
    pub fn cmd_draw_indirect(
        &mut self,
        indirect_buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Error> {
        if draw_count == 0 {
            return Ok(());
        }
        let buf = self.ensure_rendering()?;
        self.has_draws = true;
        unsafe {
            self.renderer
                .device
                .vk()
                .cmd_draw_indirect(buf, indirect_buffer, offset, draw_count, stride);
        }
        Ok(())
    }

    /// Renders quads through multi-draw indirect buffer when supported, falling back to direct draw.
    pub(crate) fn draw_quads(&mut self, buf: vk::CommandBuffer, scissors: &[vk::Rect2D]) {
        if scissors.is_empty() {
            return;
        }

        if self.renderer.supports_multi_draw_indirect {
            if let Some(indirect_buf) = self.renderer.indirect_buffer.as_mut() {
                if scissors.len() == 1 {
                    unsafe {
                        self.renderer.device.vk().cmd_set_scissor(buf, 0, &scissors[..1]);
                    }
                    let cmd = DrawIndirectCommand {
                        vertex_count: 6,
                        instance_count: 1,
                        first_vertex: 0,
                        first_instance: 0,
                    };
                    if let Ok(offset) = indirect_buf.push(cmd) {
                        unsafe {
                            self.renderer.device.vk().cmd_draw_indirect(
                                buf,
                                indirect_buf.buffer(),
                                offset,
                                1,
                                indirect_buf.stride(),
                            );
                        }
                        return;
                    }
                } else {
                    let cmds: Vec<DrawIndirectCommand> = (0..scissors.len())
                        .map(|idx| DrawIndirectCommand {
                            vertex_count: 6,
                            instance_count: 1,
                            first_vertex: 0,
                            first_instance: idx as u32,
                        })
                        .collect();

                    if let Ok(base_offset) = indirect_buf.write_commands(&cmds) {
                        let stride = indirect_buf.stride() as vk::DeviceSize;
                        for (i, scissor) in scissors.iter().enumerate() {
                            unsafe {
                                self.renderer.device.vk().cmd_set_scissor(buf, 0, &[*scissor]);
                                self.renderer.device.vk().cmd_draw_indirect(
                                    buf,
                                    indirect_buf.buffer(),
                                    base_offset + (i as vk::DeviceSize * stride),
                                    1,
                                    indirect_buf.stride(),
                                );
                            }
                        }
                        return;
                    }
                }
            }
        }

        for scissor in scissors {
            unsafe {
                self.renderer.device.vk().cmd_set_scissor(buf, 0, &[*scissor]);
                self.renderer.device.vk().cmd_draw(buf, 6, 1, 0, 0);
            }
        }
    }

    /// Batches layout transitions and queue family acquires for a collection of textures into a single
    /// pipeline barrier, avoiding breaking dynamic rendering passes mid-frame.
    pub fn prepare_textures<'a, I>(&mut self, textures: I) -> Result<(), Error>
    where
        I: IntoIterator<Item = &'a VulkanImage>,
    {
        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let mut barriers = Vec::new();
        let mut transitioned: Vec<&'a VulkanImage> = Vec::new();

        for texture in textures {
            if !self.images.iter().any(|img| Arc::ptr_eq(img, &texture.inner)) {
                self.images.push(texture.inner.clone());
            }

            if texture.dmabuf_exportable() && !self.is_blit {
                if !self
                    .sampled_images
                    .iter()
                    .any(|img| Arc::ptr_eq(&img.inner, &texture.inner))
                {
                    self.sampled_images.push(texture.clone());
                }
                continue;
            }

            let tex_needs_acquire = texture.needs_acquire() && texture.dmabuf_exportable();
            let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
                (ext_queue, qfam)
            } else {
                (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
            };

            let tex_old_layout = texture.current_layout();
            let needs_tex_barrier =
                tex_needs_acquire || tex_old_layout != ImageLayout::SHADER_READ_ONLY_OPTIMAL;

            if needs_tex_barrier {
                if transitioned.iter().any(|t| Arc::ptr_eq(&t.inner, &texture.inner)) {
                    continue;
                }

                let (tex_src_stage, tex_src_access) =
                    src_stage_and_access_for_layout(tex_needs_acquire, tex_old_layout);

                let barrier = ImageMemoryBarrier2::default()
                    .image(*texture.vk())
                    .old_layout(tex_old_layout)
                    .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(tex_src_queue)
                    .dst_queue_family_index(tex_dst_queue)
                    .src_stage_mask(tex_src_stage)
                    .src_access_mask(tex_src_access)
                    .dst_stage_mask(PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(AccessFlags2::SHADER_SAMPLED_READ)
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    );

                barriers.push(barrier);
                transitioned.push(texture);
            }
        }

        if !barriers.is_empty() {
            if self.rendering {
                self.end_rendering();
            }
            let buf = self.get_or_create_cmd_buffer()?;
            unsafe {
                self.renderer
                    .device
                    .vk()
                    .cmd_pipeline_barrier2(buf, &DependencyInfo::default().image_memory_barriers(&barriers));
            }
            for tex in transitioned {
                tex.set_current_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                tex.set_needs_acquire(false);
            }
        }

        Ok(())
    }

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

    fn ensure_rendering(&mut self) -> Result<ash::vk::CommandBuffer, Error> {
        let buf = self.get_or_create_cmd_buffer()?;
        if self.rendering {
            return Ok(buf);
        }

        if self.renderer.queued_submits.iter().any(|q| {
            q.sampled_images
                .iter()
                .any(|img| Arc::ptr_eq(&img.inner, &self.fb.0.inner))
        }) {
            let was_batch = self.renderer.batch_submits;
            self.renderer.flush_batch()?;
            self.renderer.batch_submits = was_batch;
        }

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let fb_needs_acquire = self.fb.0.needs_acquire() && self.fb.0.dmabuf_exportable();
        let (fb_src_queue, fb_dst_queue) = if fb_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let fb_old_layout = self.fb.0.current_layout();
        let (fb_src_stage, fb_src_access) = src_stage_and_access_for_layout(fb_needs_acquire, fb_old_layout);

        let fb_barrier = ImageMemoryBarrier2::default()
            .image(*self.fb.0.vk())
            .old_layout(fb_old_layout)
            .new_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(fb_src_queue)
            .dst_queue_family_index(fb_dst_queue)
            .src_stage_mask(fb_src_stage)
            .src_access_mask(fb_src_access)
            .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(AccessFlags2::COLOR_ATTACHMENT_READ | AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.renderer.device.vk().cmd_pipeline_barrier2(
                buf,
                &DependencyInfo::default().image_memory_barriers(&[fb_barrier]),
            );
        }

        self.fb
            .0
            .set_current_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        self.fb.0.set_needs_acquire(false);

        let fb_width = self.fb.0.width();
        let fb_height = self.fb.0.height();
        let render_width = (self.size.w as u32).min(fb_width);
        let render_height = (self.size.h as u32).min(fb_height);

        let mut load_op = if fb_old_layout == ImageLayout::UNDEFINED {
            vk::AttachmentLoadOp::DONT_CARE
        } else {
            vk::AttachmentLoadOp::LOAD
        };
        let mut clear_color_val = vk::ClearColorValue::default();

        if let Some(pos) = self.pending_clears.iter().position(|(_, rects)| {
            rects.len() == 1
                && rects[0].rect.offset.x == 0
                && rects[0].rect.offset.y == 0
                && rects[0].rect.extent.width == render_width
                && rects[0].rect.extent.height == render_height
        }) {
            let (color, _) = self.pending_clears.remove(pos);
            load_op = vk::AttachmentLoadOp::CLEAR;
            clear_color_val = vk::ClearColorValue {
                float32: color.components(),
            };
            self.has_draws = true;
        }

        let view = self.fb.0.vk_view().unwrap();
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(*view)
            .image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .clear_value(vk::ClearValue {
                color: clear_color_val,
            })
            .store_op(vk::AttachmentStoreOp::STORE);

        let color_attachments = [color_attachment];
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: render_width,
                    height: render_height,
                },
            })
            .layer_count(1)
            .color_attachments(&color_attachments);

        let depth_attachment;
        if self.depth_enabled {
            let width = render_width.max(1);
            let height = render_height.max(1);
            let needs_alloc = self
                .renderer
                .depth_image
                .as_ref()
                .map_or(true, |img| img.width() != width || img.height() != height);
            if needs_alloc {
                let img = VulkanImage::new(
                    &self.renderer.device,
                    width,
                    height,
                    vk::Format::D16_UNORM,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                    false,
                )?;
                let old = self.renderer.depth_image.replace(img);
                if let Some(old_depth) = old {
                    if !self.images.iter().any(|i| Arc::ptr_eq(i, &old_depth.inner)) {
                        self.images.push(old_depth.inner.clone());
                    }
                }
            }

            let depth_img = self.renderer.depth_image.as_ref().unwrap();
            let depth_old_layout = depth_img.current_layout();
            let depth_barrier = ImageMemoryBarrier2::default()
                .image(*depth_img.vk())
                .old_layout(depth_old_layout)
                .new_layout(ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                .src_stage_mask(PipelineStageFlags2::NONE)
                .src_access_mask(AccessFlags2::NONE)
                .dst_stage_mask(
                    PipelineStageFlags2::EARLY_FRAGMENT_TESTS | PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                )
                .subresource_range(
                    ImageSubresourceRange::default()
                        .aspect_mask(ImageAspectFlags::DEPTH)
                        .layer_count(1)
                        .level_count(1),
                );
            unsafe {
                self.renderer.device.vk().cmd_pipeline_barrier2(
                    buf,
                    &DependencyInfo::default().image_memory_barriers(&[depth_barrier]),
                );
            }
            depth_img.set_current_layout(ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);

            depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(*depth_img.vk_view().unwrap())
                .image_layout(ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                })
                .store_op(vk::AttachmentStoreOp::DONT_CARE);

            rendering_info = rendering_info.depth_attachment(&depth_attachment);
        }

        unsafe {
            self.renderer
                .device
                .vk()
                .cmd_begin_rendering(buf, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: render_width as f32,
                height: render_height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.renderer.device.vk().cmd_set_viewport(buf, 0, &[viewport]);
        }

        self.rendering = true;

        if !self.pending_clears.is_empty() {
            let pending = std::mem::take(&mut self.pending_clears);
            for (color, clear_rects) in pending {
                let mut attachments = Vec::with_capacity(2);
                attachments.push(vk::ClearAttachment {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    color_attachment: 0,
                    clear_value: vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: color.components(),
                        },
                    },
                });

                if self.depth_enabled {
                    attachments.push(vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    });
                }

                unsafe {
                    self.renderer
                        .device
                        .vk()
                        .cmd_clear_attachments(buf, &attachments, &clear_rects);
                }
                self.has_draws = true;
            }
        }

        Ok(buf)
    }

    fn end_rendering(&mut self) {
        if self.rendering {
            if let Some(buf) = self.cmd_buffer {
                unsafe {
                    self.renderer.device.vk().cmd_end_rendering(buf);
                }
            }
            self.rendering = false;
        }
    }

    pub fn can_copy_image(
        &self,
        texture: &VulkanImage,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
    ) -> bool {
        texture.format() == self.fb.0.format()
            && src.size == dst.size
            && texture.vk_usage().contains(ImageUsageFlags::TRANSFER_SRC)
            && self.fb.0.vk_usage().contains(ImageUsageFlags::TRANSFER_DST)
            && src.loc.x >= 0
            && src.loc.y >= 0
            && dst.loc.x >= 0
            && dst.loc.y >= 0
            && (src.loc.x + src.size.w) as u32 <= texture.width()
            && (src.loc.y + src.size.h) as u32 <= texture.height()
            && (dst.loc.x + dst.size.w) as u32 <= self.fb.0.width()
            && (dst.loc.y + dst.size.h) as u32 <= self.fb.0.height()
    }

    pub fn copy_image_from_to(
        &mut self,
        texture: &VulkanImage,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
    ) -> Result<(), Error> {
        self.end_rendering();
        if self.renderer.queued_submits.iter().any(|q| {
            q.sampled_images.iter().any(|img| {
                Arc::ptr_eq(&img.inner, &self.fb.0.inner) || Arc::ptr_eq(&img.inner, &texture.inner)
            })
        }) {
            let was_batch = self.renderer.batch_submits;
            self.renderer.flush_batch()?;
            self.renderer.batch_submits = was_batch;
        }
        let cmd_buffer = self.get_or_create_cmd_buffer()?;

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let fb_needs_acquire = self.fb.0.needs_acquire() && self.fb.0.dmabuf_exportable();
        let (fb_src_queue, fb_dst_queue) = if fb_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let fb_old_layout = self.fb.0.current_layout();
        let (fb_src_stage, fb_src_access) = src_stage_and_access_for_layout(fb_needs_acquire, fb_old_layout);

        let tex_needs_acquire = texture.needs_acquire() && texture.dmabuf_exportable();
        let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let tex_old_layout = texture.current_layout();
        let (tex_src_stage, tex_src_access) =
            src_stage_and_access_for_layout(tex_needs_acquire, tex_old_layout);

        let fb_barrier = ImageMemoryBarrier2::default()
            .image(*self.fb.0.vk())
            .old_layout(fb_old_layout)
            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(fb_src_queue)
            .dst_queue_family_index(fb_dst_queue)
            .src_stage_mask(fb_src_stage)
            .src_access_mask(fb_src_access)
            .dst_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(AccessFlags2::TRANSFER_WRITE)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        let tex_barrier = ImageMemoryBarrier2::default()
            .image(*texture.vk())
            .old_layout(tex_old_layout)
            .new_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(tex_src_queue)
            .dst_queue_family_index(tex_dst_queue)
            .src_stage_mask(tex_src_stage)
            .src_access_mask(tex_src_access)
            .dst_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .dst_access_mask(AccessFlags2::TRANSFER_READ)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.renderer.device.vk().cmd_pipeline_barrier2(
                cmd_buffer,
                &DependencyInfo::default().image_memory_barriers(&[fb_barrier, tex_barrier]),
            );
        }

        self.fb.0.set_needs_acquire(false);
        texture.set_needs_acquire(false);

        let copy_region = vk::ImageCopy::default()
            .src_subresource(
                ImageSubresourceLayers::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_offset(Offset3D {
                x: src.loc.x,
                y: src.loc.y,
                z: 0,
            })
            .dst_subresource(
                ImageSubresourceLayers::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_offset(Offset3D {
                x: dst.loc.x,
                y: dst.loc.y,
                z: 0,
            })
            .extent(Extent3D {
                width: src.size.w as u32,
                height: src.size.h as u32,
                depth: 1,
            });

        unsafe {
            self.renderer.device.vk().cmd_copy_image(
                cmd_buffer,
                *texture.vk(),
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                *self.fb.0.vk(),
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        }

        let fb_post_barrier = ImageMemoryBarrier2::default()
            .image(*self.fb.0.vk())
            .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
            .src_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(AccessFlags2::COLOR_ATTACHMENT_READ | AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        let tex_restore_layout = if tex_old_layout == ImageLayout::UNDEFINED {
            ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            tex_old_layout
        };
        let tex_post_barrier = ImageMemoryBarrier2::default()
            .image(*texture.vk())
            .old_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(tex_restore_layout)
            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
            .src_stage_mask(PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(AccessFlags2::MEMORY_READ | AccessFlags2::MEMORY_WRITE)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        unsafe {
            self.renderer.device.vk().cmd_pipeline_barrier2(
                cmd_buffer,
                &DependencyInfo::default().image_memory_barriers(&[fb_post_barrier, tex_post_barrier]),
            );
        }

        self.fb
            .0
            .set_current_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        texture.set_current_layout(tex_restore_layout);

        self.images.push(texture.inner.clone());
        self.has_draws = true;

        Ok(())
    }
}

pub fn transform_corner_radius(radii: [f32; 4], transform: Transform) -> [f32; 4] {
    let [tl, tr, br, bl] = radii;
    match transform {
        Transform::Normal => [tl, tr, br, bl],
        Transform::_90 => [bl, tl, tr, br],
        Transform::_180 => [br, bl, tl, tr],
        Transform::_270 => [tr, br, bl, tl],
        Transform::Flipped => [tr, tl, bl, br],
        Transform::Flipped90 => [tl, bl, br, tr],
        Transform::Flipped180 => [bl, br, tr, tl],
        Transform::Flipped270 => [br, tr, tl, bl],
    }
}

impl Frame for VulkanFrame<'_, '_> {
    type Error = Error;
    type TextureId = VulkanImage;

    #[cfg(feature = "wayland_frontend")]
    fn set_surface_color_description(&mut self, desc: Option<&ImageDescription>) {
        self.active_color_description = desc.cloned();
    }

    fn set_surface_clip(&mut self, clip: Option<(Rectangle<i32, Physical>, [f32; 4])>) {
        self.active_clip = clip;
    }

    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.renderer.device.context()
    }

    fn prepare_textures<'a, I>(&mut self, textures: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = &'a Self::TextureId>,
        Self::TextureId: 'a,
    {
        VulkanFrame::prepare_textures(self, textures)
    }

    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        trace!(?color, at_len = at.len(), fb_size = ?self.size, "VulkanFrame::clear executing");
        if at.is_empty() {
            return Ok(());
        }

        let scissors = calculate_damage_scissors(
            at,
            Rectangle::from_size(self.size),
            self.transform,
            &self.size,
            self.fb.width(),
            self.fb.height(),
        );

        if scissors.is_empty() {
            return Ok(());
        }

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

        let clear_rects: Vec<vk::ClearRect> = scissors
            .into_iter()
            .map(|rect| vk::ClearRect {
                rect,
                base_array_layer: 0,
                layer_count: 1,
            })
            .collect();

        if self.rendering {
            let buf = self.cmd_buffer.unwrap();
            let mut attachments = Vec::with_capacity(2);
            attachments.push(vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: color.components(),
                    },
                },
            });

            if self.depth_enabled {
                attachments.push(vk::ClearAttachment {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    color_attachment: 0,
                    clear_value: vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                });
            }

            unsafe {
                self.renderer
                    .device
                    .vk()
                    .cmd_clear_attachments(buf, &attachments, &clear_rects);
            }
            self.has_draws = true;
        } else {
            self.pending_clears.push((color, clear_rects));
        }

        Ok(())
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        self.draw_color(dst, damage, color, true)
    }

    fn draw_rounded_outline(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        thickness: f32,
        radius: [f32; 4],
        color: Color32F,
        outline_rect: Option<Rectangle<i32, Physical>>,
    ) -> Result<(), Self::Error> {
        trace!(
            ?dst,
            damage_len = damage.len(),
            thickness,
            ?radius,
            ?color,
            ?outline_rect,
            fb_size = ?self.size,
            "VulkanFrame::draw_rounded_outline executing"
        );
        if damage.is_empty() {
            return Ok(());
        }
        let untransformed_dst = dst;
        let transformed_dst = self.transform.transform_rect_in(dst, &self.size);
        let transformed_radii = transform_corner_radius(radius, self.transform);

        let uncropped_outline = outline_rect.unwrap_or(dst);
        let transformed_outline = self.transform.transform_rect_in(uncropped_outline, &self.size);

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

        let fb_format = self.fb.0.format();
        let format_pipelines = self
            .renderer
            .pipelines
            .get_or_create_format_pipelines(fb_format, self.depth_enabled)?;
        let pipeline = format_pipelines.outline_blend_pipeline;
        let layout = *self.renderer.pipelines.outline_pipeline_layout();

        let buf = self.ensure_rendering()?;
        self.has_draws = true;

        unsafe {
            self.renderer
                .device
                .vk()
                .cmd_bind_pipeline(buf, PipelineBindPoint::GRAPHICS, pipeline);
        }

        let dst_rect = Rectangle::new(
            Point::new(transformed_dst.loc.x as f32, transformed_dst.loc.y as f32),
            Size::new(transformed_dst.size.w as f32, transformed_dst.size.h as f32),
        );
        let outline_rect = Rectangle::new(
            Point::new(transformed_outline.loc.x as f32, transformed_outline.loc.y as f32),
            Size::new(
                transformed_outline.size.w as f32,
                transformed_outline.size.h as f32,
            ),
        );
        let screen_size = [self.size.w as f32, self.size.h as f32];
        let depth_val = if self.depth_enabled {
            self.current_depth
        } else {
            0.0
        };

        let push_constants = shaders::OutlinePushConstants {
            dst_rect,
            screen_size,
            depth: depth_val,
            thickness,
            color: color.components(),
            corner_radius: transformed_radii,
            outline_rect,
        };

        unsafe {
            self.renderer.device.vk().cmd_push_constants(
                buf,
                layout,
                ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );
        }

        let scissors = calculate_damage_scissors(
            damage,
            untransformed_dst,
            self.transform,
            &self.size,
            self.fb.width(),
            self.fb.height(),
        );

        self.draw_quads(buf, &scissors);

        Ok(())
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

        // Use the HDR shader pipeline whenever hdr_config is present (even with is_sdr=true),
        // or when explicitly performing an HDR-to-SDR or SDR-to-HDR blit.
        let is_hdr = if self.hdr_to_sdr.is_some() || self.sdr_to_hdr.is_some() {
            true
        } else {
            !self.is_blit && self.renderer.hdr_config.is_some()
        };
        self.has_draws = true;
        if !self.images.iter().any(|img| Arc::ptr_eq(img, &texture.inner)) {
            self.images.push(texture.inner.clone());
        }

        if texture.dmabuf_exportable() && !self.is_blit {
            if !self
                .sampled_images
                .iter()
                .any(|img| Arc::ptr_eq(&img.inner, &texture.inner))
            {
                self.sampled_images.push(texture.clone());
            }
        } else {
            let qfam = self.renderer.device.queue_family_idx();
            let ext_queue = self.renderer.external_queue_family();

            let tex_needs_acquire = texture.needs_acquire() && texture.dmabuf_exportable();
            let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
                (ext_queue, qfam)
            } else {
                (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
            };

            let tex_old_layout = texture.current_layout();
            let needs_tex_barrier =
                tex_needs_acquire || tex_old_layout != ImageLayout::SHADER_READ_ONLY_OPTIMAL;

            if needs_tex_barrier {
                if self.rendering {
                    self.end_rendering();
                }
                let buf = self.get_or_create_cmd_buffer()?;

                let (tex_src_stage, tex_src_access) =
                    src_stage_and_access_for_layout(tex_needs_acquire, tex_old_layout);

                let tex_barrier = ImageMemoryBarrier2::default()
                    .image(*texture.vk())
                    .old_layout(tex_old_layout)
                    .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(tex_src_queue)
                    .dst_queue_family_index(tex_dst_queue)
                    .src_stage_mask(tex_src_stage)
                    .src_access_mask(tex_src_access)
                    .dst_stage_mask(PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(AccessFlags2::SHADER_SAMPLED_READ)
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    );

                unsafe {
                    self.renderer.device.vk().cmd_pipeline_barrier2(
                        buf,
                        &DependencyInfo::default().image_memory_barriers(&[tex_barrier]),
                    );
                }

                texture.set_current_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                texture.set_needs_acquire(false);
            }
        }

        let buf = self.ensure_rendering()?;

        let ycbcr_info = texture
            .ycbcr()
            .and_then(|key| texture.ycbcr_conversion().map(|conv| (key, conv.sampler)));

        let push_descriptor = self.renderer.device.vk_khr_push_descriptor();
        let descriptor = if push_descriptor.is_none() {
            Some(if let Some((key, sampler)) = ycbcr_info {
                self.renderer
                    .pipelines
                    .alloc_ycbcr_descriptor_set(key, sampler, is_hdr)?
            } else if is_hdr {
                self.renderer
                    .pipelines
                    .alloc_descriptor_set(shaders::BuiltinShader::HdrTexture)?
            } else {
                self.renderer
                    .pipelines
                    .alloc_descriptor_set(shaders::BuiltinShader::Texture)?
            })
        } else {
            None
        };

        let view = texture
            .vk_view()
            .ok_or(Error::ImageError(ImageError::MissingOrInvalidUsage))?;
        let is_downscaling = (dst.size.w as f64) < src.size.w || (dst.size.h as f64) < src.size.h;
        let sampler = if let Some((_, ycbcr_sampler)) = ycbcr_info {
            ycbcr_sampler
        } else {
            self.renderer.active_sampler(is_downscaling)
        };

        if self.renderer.supports_descriptor_indexing && ycbcr_info.is_none() {
            if let Some(ref bindless) = self.renderer.bindless_pool {
                let _ = bindless.update_texture(0, *view, sampler);
            }
        }

        let tex_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(*view)
            .sampler(sampler)];

        let lut3d_view = self
            .renderer
            .lut3d
            .as_ref()
            .map(|l| l.view)
            .unwrap_or(vk::ImageView::null());
        let lut3d_sampler = self
            .renderer
            .lut3d
            .as_ref()
            .map(|l| l.sampler)
            .unwrap_or(self.renderer.texture_sampler);
        let lut3d_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(lut3d_view)
            .sampler(lut3d_sampler)];

        let dst_set = descriptor
            .as_ref()
            .map(|d| d.vk())
            .unwrap_or(vk::DescriptorSet::null());

        let descriptor_update_hdr = [
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&tex_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&lut3d_image_info),
        ];

        let descriptor_update_sdr = [vk::WriteDescriptorSet::default()
            .dst_set(dst_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_count(1)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&tex_image_info)];

        let descriptor_update: &[vk::WriteDescriptorSet<'_>] = if is_hdr {
            &descriptor_update_hdr
        } else {
            &descriptor_update_sdr
        };

        let src_rect = Rectangle::new(
            Point::new(src.loc.x as f32, src.loc.y as f32),
            Size::new(src.size.w as f32, src.size.h as f32),
        );
        let dst_rect = Rectangle::new(
            Point::new(dst.loc.x as f32, dst.loc.y as f32),
            Size::new(dst.size.w as f32, dst.size.h as f32),
        );
        let screen_size = [self.size.w as f32, self.size.h as f32];

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

        let has_alpha = texture.has_alpha() as u32;
        let should_blend = texture.has_alpha() || alpha < 1.0;

        let fb_format = self.fb.0.format();
        let format_pipelines = self
            .renderer
            .pipelines
            .get_or_create_format_pipelines(fb_format, self.depth_enabled)?;
        let ycbcr_format_pipelines = if let Some((key, sampler)) = ycbcr_info {
            Some(self.renderer.pipelines.get_or_create_ycbcr_format_pipelines(
                key,
                sampler,
                fb_format,
                self.depth_enabled,
            )?)
        } else {
            None
        };

        let depth_val = if self.depth_enabled {
            self.current_depth
        } else {
            0.0
        };

        let (clip_rect, corner_radius) = if let Some((geo, radii)) = self.active_clip {
            let transformed_geo = self.transform.transform_rect_in(geo, &self.size);
            let transformed_radii = transform_corner_radius(radii, self.transform);
            (
                Rectangle::new(
                    Point::new(transformed_geo.loc.x as f32, transformed_geo.loc.y as f32),
                    Size::new(transformed_geo.size.w as f32, transformed_geo.size.h as f32),
                ),
                transformed_radii,
            )
        } else {
            (
                Rectangle::new(Point::new(0.0, 0.0), Size::new(0.0, 0.0)),
                [0.0; 4],
            )
        };

        let (layout, pipeline, hdr_push_constants) = if is_hdr {
            let (config, is_hdr_to_sdr, is_sdr_to_hdr) = if let Some(ref blit_cfg) = self.hdr_to_sdr {
                (*blit_cfg, true, false)
            } else if let Some(ref blit_cfg) = self.sdr_to_hdr {
                (*blit_cfg, false, true)
            } else {
                (
                    self.renderer.hdr_config.unwrap_or(HdrOutputConfig {
                        reference_white: 203.0,
                        max_luminance: 1000.0,
                        sdr_gamma: 2.2,
                        gamut_stretch: 0.0,
                        hardware_offload: false,
                        is_sdr: false,
                    }),
                    false,
                    false,
                )
            };
            let mut reference_white = config.reference_white;
            let mut sdr_gamma = config.sdr_gamma;
            let mut gamut_stretch = config.gamut_stretch;
            let mut max_content_luminance = if is_hdr_to_sdr {
                config.max_luminance
            } else {
                config.reference_white
            };
            let max_destination_luminance = if is_hdr_to_sdr {
                config.reference_white
            } else {
                config.max_luminance
            };
            let hardware_offload = if is_hdr_to_sdr || is_sdr_to_hdr {
                0u32
            } else {
                config.hardware_offload as u32
            };
            let target_is_sdr = if is_hdr_to_sdr || (!is_sdr_to_hdr && !is_hdr_vk_format(self.fb.0.format()))
            {
                1u32
            } else {
                config.is_sdr as u32
            };

            let mut input_is_pq = if is_hdr_to_sdr { 1u32 } else { 0u32 };
            let mut input_is_hlg = 0u32;
            let mut input_primaries = if is_hdr_to_sdr { 2u32 } else { 0u32 };
            let mut skip_color_transform = 0u32;
            let mut content_reference = config.reference_white;

            #[cfg(feature = "wayland_frontend")]
            if !is_hdr_to_sdr
                && !is_sdr_to_hdr
                && let Some(desc) = self.active_color_description.as_ref()
            {
                if desc.is_pq_bt2020() {
                    input_is_pq = 1;
                    input_primaries = 2;
                    content_reference = desc.luminances.map(|(_, _, r)| r.max(80) as f32).unwrap_or(203.0);
                    max_content_luminance = desc
                        .luminances
                        .map(|(_, m, _)| m as f32)
                        .or_else(|| desc.max_cll.map(|v| v as f32))
                        .or_else(|| desc.mastering_luminance.map(|(_, m)| m as f32))
                        .unwrap_or(1000.0);

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
                    content_reference = desc.luminances.map(|(_, _, r)| r.max(80) as f32).unwrap_or(203.0);
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
            } else if !is_hdr_to_sdr && !is_sdr_to_hdr {
                if config.is_sdr && sdr_gamma == 0.0 {
                    skip_color_transform = 1;
                }
            }

            let mut flags = 0u32;
            if hardware_offload != 0 {
                flags |= shaders::HDR_FLAG_HARDWARE_OFFLOAD;
            }
            if target_is_sdr != 0 {
                flags |= shaders::HDR_FLAG_TARGET_IS_SDR;
            }
            if input_is_pq != 0 {
                flags |= shaders::HDR_FLAG_INPUT_IS_PQ;
            }
            if input_is_hlg != 0 {
                flags |= shaders::HDR_FLAG_INPUT_IS_HLG;
            }
            flags |= (input_primaries & 3) << shaders::HDR_FLAG_INPUT_PRIMARIES_SHIFT;
            if skip_color_transform != 0 {
                flags |= shaders::HDR_FLAG_SKIP_COLOR_TRANSFORM;
            }

            let push_constants = HdrTexPushConstants {
                dst_rect,
                screen_size,
                depth: depth_val,
                _pad0: 0.0,
                src_rect,
                src_transform: src_transform_val,
                alpha,
                has_alpha,
                reference_white,
                sdr_gamma,
                gamut_stretch,
                max_content_luminance,
                max_destination_luminance,
                content_reference,
                flags,
                _pad1: [0; 2],
                clip_rect,
                corner_radius,
            };

            let chosen_pipeline = if let Some(ycbcr_p) = ycbcr_format_pipelines {
                if should_blend {
                    if skip_color_transform != 0 {
                        ycbcr_p.hdr_passthrough_blend_pipeline
                    } else if input_is_pq != 0 && target_is_sdr == 0 {
                        ycbcr_p.hdr_pq_blend_pipeline
                    } else if input_is_hlg == 0 && input_primaries == 0 && target_is_sdr == 0 {
                        ycbcr_p.hdr_sdr_blend_pipeline
                    } else {
                        ycbcr_p.hdr_tex_blend_pipeline
                    }
                } else {
                    if skip_color_transform != 0 {
                        ycbcr_p.hdr_passthrough_pipeline
                    } else if input_is_pq != 0 && target_is_sdr == 0 {
                        ycbcr_p.hdr_pq_pipeline
                    } else if input_is_hlg == 0 && input_primaries == 0 && target_is_sdr == 0 {
                        ycbcr_p.hdr_sdr_pipeline
                    } else {
                        ycbcr_p.hdr_tex_pipeline
                    }
                }
            } else if should_blend {
                if skip_color_transform != 0 {
                    format_pipelines.hdr_passthrough_blend_pipeline
                } else if input_is_pq != 0 && target_is_sdr == 0 {
                    format_pipelines.hdr_pq_blend_pipeline
                } else if input_is_hlg == 0 && input_primaries == 0 && target_is_sdr == 0 {
                    format_pipelines.hdr_sdr_blend_pipeline
                } else {
                    format_pipelines.hdr_tex_blend_pipeline
                }
            } else {
                if skip_color_transform != 0 {
                    format_pipelines.hdr_passthrough_pipeline
                } else if input_is_pq != 0 && target_is_sdr == 0 {
                    format_pipelines.hdr_pq_pipeline
                } else if input_is_hlg == 0 && input_primaries == 0 && target_is_sdr == 0 {
                    format_pipelines.hdr_sdr_pipeline
                } else {
                    format_pipelines.hdr_tex_pipeline
                }
            };

            let layout = if let Some((key, _)) = ycbcr_info {
                self.renderer.pipelines.ycbcr_pipeline_layout(key, true).unwrap()
            } else {
                *self.renderer.pipelines.hdr_tex_pipeline_layout()
            };

            (layout, chosen_pipeline, Some(push_constants))
        } else {
            let chosen_pipeline = if let Some(ycbcr_p) = ycbcr_format_pipelines {
                if should_blend {
                    ycbcr_p.tex_blend_pipeline
                } else {
                    ycbcr_p.tex_pipeline
                }
            } else if should_blend {
                format_pipelines.tex_blend_pipeline
            } else {
                format_pipelines.tex_pipeline
            };

            let layout = if let Some((key, _)) = ycbcr_info {
                self.renderer.pipelines.ycbcr_pipeline_layout(key, false).unwrap()
            } else {
                *self.renderer.pipelines.tex_pipeline_layout()
            };

            (layout, chosen_pipeline, None)
        };

        unsafe {
            self.renderer
                .device
                .vk()
                .cmd_bind_pipeline(buf, PipelineBindPoint::GRAPHICS, pipeline);
            if let Some(push) = push_descriptor {
                push.cmd_push_descriptor_set(buf, PipelineBindPoint::GRAPHICS, layout, 0, descriptor_update);
            } else {
                self.renderer
                    .device
                    .vk()
                    .update_descriptor_sets(descriptor_update, &[]);
                self.renderer.device.vk().cmd_bind_descriptor_sets(
                    buf,
                    PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[descriptor.as_ref().unwrap().vk()],
                    &[],
                );
            }
        }

        if let Some(hdr_pc) = hdr_push_constants {
            unsafe {
                self.renderer.device.vk().cmd_push_constants(
                    buf,
                    layout,
                    ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&hdr_pc),
                );
            }
        } else {
            let push_constants = TexPushConstants {
                dst_rect,
                screen_size,
                depth: depth_val,
                _pad0: 0.0,
                src_rect,
                src_transform: src_transform_val,
                alpha,
                has_alpha,
                _pad1: 0,
                clip_rect,
                corner_radius,
            };

            unsafe {
                self.renderer.device.vk().cmd_push_constants(
                    buf,
                    layout,
                    ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
            }
        }

        let scissors = calculate_damage_scissors(
            damage,
            untransformed_dst,
            self.transform,
            &self.size,
            self.fb.width(),
            self.fb.height(),
        );

        self.draw_quads(buf, &scissors);

        if let Some(descriptor) = descriptor {
            self.descriptors.push(descriptor);
        }
        Ok(())
    }

    fn transformation(&self) -> Transform {
        self.transform
    }

    fn output_size(&self) -> Size<i32, Physical> {
        self.size
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        if let Some(drm_sync) = sync.get::<DrmSyncPoint>() {
            if let Some(timeline) = self.renderer.timeline.drm.as_ref() {
                if timeline == drm_sync.timeline() {
                    return Ok(());
                }
            }
            self.pending_waits.push(drm_sync.clone());
            return Ok(());
        }
        self.renderer.wait(sync)
    }

    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        trace!(
            has_draws = self.has_draws,
            has_drm = self.renderer.timeline.drm.is_some(),
            "VulkanFrame::finish"
        );
        if !self.pending_clears.is_empty() {
            self.ensure_rendering()?;
        }
        if !self.has_draws {
            return Ok(SyncPoint::signaled());
        }

        self.end_rendering();

        let Some(buf) = self.cmd_buffer.take() else {
            return Ok(SyncPoint::signaled());
        };

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let mut release_barriers = Vec::with_capacity(2);

        if self.fb.0.dmabuf_exportable() {
            let current_layout = self.fb.0.current_layout();
            let (src_stage, src_access) = src_stage_and_access_for_layout(false, current_layout);

            // Per Vulkan 1.3 spec (VK_EXT_image_drm_format_modifier & VK_EXT_queue_family_foreign),
            // any image exported to an external/foreign consumer (DRM/KMS scanout, Wayland client, etc.)
            // must be transitioned to VK_IMAGE_LAYOUT_GENERAL on queue family ownership release,
            // with dstStageMask = NONE and dstAccessMask = NONE.
            // Both Mesa RADV (radv_layout_dcc_compressed / radv_image_need_retile) and Intel ANV
            // preserve and prepare DCC/CCS compression metadata for scanout when releasing to
            // VK_QUEUE_FAMILY_FOREIGN_EXT with newLayout = VK_IMAGE_LAYOUT_GENERAL.
            let new_layout = ImageLayout::GENERAL;

            release_barriers.push(
                ImageMemoryBarrier2::default()
                    .image(*self.fb.0.vk())
                    .old_layout(current_layout)
                    .new_layout(new_layout)
                    .src_queue_family_index(qfam)
                    .dst_queue_family_index(ext_queue)
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(PipelineStageFlags2::NONE)
                    .dst_access_mask(AccessFlags2::NONE)
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    ),
            );
            self.fb.0.set_current_layout(new_layout);
            self.fb.0.set_needs_acquire(true);
        }

        if let Some(ref src_img) = self.release_source {
            if !Arc::ptr_eq(&src_img.inner, &self.fb.0.inner)
                && src_img.dmabuf_exportable()
                && !src_img.needs_acquire()
            {
                let src_layout = src_img.current_layout();
                let (src_stage, src_access) = src_stage_and_access_for_layout(false, src_layout);
                let new_layout = ImageLayout::GENERAL;

                release_barriers.push(
                    ImageMemoryBarrier2::default()
                        .image(*src_img.vk())
                        .old_layout(src_layout)
                        .new_layout(new_layout)
                        .src_queue_family_index(qfam)
                        .dst_queue_family_index(ext_queue)
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(PipelineStageFlags2::NONE)
                        .dst_access_mask(AccessFlags2::NONE)
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1),
                        ),
                );
                src_img.set_current_layout(new_layout);
                src_img.set_needs_acquire(true);
            }
        }

        unsafe {
            if !release_barriers.is_empty() {
                self.renderer.device.vk().cmd_pipeline_barrier2(
                    buf,
                    &DependencyInfo::default().image_memory_barriers(&release_barriers),
                );
            }
            self.renderer
                .device
                .vk()
                .end_command_buffer(buf)
                .map_err(Error::CommandBufferError)?;
        }

        let prev_seq_no = self.renderer.seq_no;
        let next_seq_no = prev_seq_no + 1;

        let cmd_buffer_info = [CommandBufferSubmitInfo::default().command_buffer(buf)];
        let signal_semaphore_info = [SemaphoreSubmitInfo::default()
            .semaphore(self.renderer.timeline.vk)
            .value(next_seq_no)
            .stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT | PipelineStageFlags2::ALL_TRANSFER)];

        let mut wait_semaphore_info = if prev_seq_no > 0 {
            vec![
                SemaphoreSubmitInfo::default()
                    .semaphore(self.renderer.timeline.vk)
                    .value(prev_seq_no)
                    .stage_mask(
                        PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                            | PipelineStageFlags2::FRAGMENT_SHADER
                            | PipelineStageFlags2::ALL_TRANSFER,
                    ),
            ]
        } else {
            Vec::new()
        };

        let mut max_points: HashMap<DrmTimeline, u64> = HashMap::new();
        for sync in self.pending_waits.drain(..) {
            max_points
                .entry(sync.timeline.clone())
                .and_modify(|p| *p = (*p).max(sync.point))
                .or_insert(sync.point);
        }

        for (timeline, point) in max_points {
            match self.renderer.get_or_import_timeline_semaphore(&timeline) {
                Ok(vk_sem) => {
                    wait_semaphore_info.push(
                        SemaphoreSubmitInfo::default()
                            .semaphore(vk_sem)
                            .value(point)
                            .stage_mask(
                                PipelineStageFlags2::TOP_OF_PIPE
                                    | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                                    | PipelineStageFlags2::FRAGMENT_SHADER
                                    | PipelineStageFlags2::ALL_TRANSFER,
                            ),
                    );
                }
                Err(err) => {
                    let is_signaled = timeline
                        .query_signalled_point()
                        .map(|p| p >= point)
                        .unwrap_or(false);
                    if !is_signaled {
                        tracing::warn!(
                            "Failed to import client timeline semaphore, falling back to CPU wait: {:?}",
                            err
                        );
                        // Wait up to 100ms instead of hanging the renderer thread indefinitely
                        let _ = DrmSyncPoint { timeline, point }.wait(100_000_000);
                    }
                }
            }
        }

        self.renderer.seq_no = next_seq_no;

        let point = next_seq_no;
        let descs = std::mem::take(&mut self.descriptors);
        let mut images = std::mem::take(&mut self.images);
        let sampled_images = std::mem::take(&mut self.sampled_images);
        if !images.iter().any(|img| Arc::ptr_eq(img, &self.fb.0.inner)) {
            images.push(self.fb.0.inner.clone());
        }
        if self.depth_enabled {
            if let Some(depth) = self.renderer.depth_image.as_ref() {
                if !images.iter().any(|img| Arc::ptr_eq(img, &depth.inner)) {
                    images.push(depth.inner.clone());
                }
            }
        }

        if self.renderer.batch_submits {
            self.renderer.queued_submits.push(QueuedSubmit {
                cmd_buffer: buf,
                cmd_buffer_infos: cmd_buffer_info.to_vec(),
                signal_semaphore_infos: signal_semaphore_info.to_vec(),
                wait_semaphore_infos: wait_semaphore_info,
                point,
                descs,
                images,
                sampled_images,
            });
            trace!(point, "VulkanFrame::finish command buffer queued in batch");
        } else {
            let device = self.renderer.device.clone();
            let submit_res = device.with_queue_lock(|_vk, queue| {
                let acquire_buf = self.renderer.record_acquire_cmd_buffer(&sampled_images)?;
                let mut cmd_infos = Vec::with_capacity(2);
                if let Some(abuf) = acquire_buf {
                    cmd_infos.push(CommandBufferSubmitInfo::default().command_buffer(abuf));
                }
                cmd_infos.extend_from_slice(&cmd_buffer_info);

                let submit_info = SubmitInfo2::default()
                    .command_buffer_infos(&cmd_infos)
                    .signal_semaphore_infos(&signal_semaphore_info)
                    .wait_semaphore_infos(&wait_semaphore_info);

                let res = unsafe { device.vk().queue_submit2(queue, &[submit_info], Fence::null()) };
                match res {
                    Ok(()) => Ok(acquire_buf),
                    Err(err) => {
                        if let Some(abuf) = acquire_buf {
                            unsafe {
                                device
                                    .vk()
                                    .free_command_buffers(self.renderer.cmd_pool.vk(), &[abuf]);
                            }
                        }
                        Err(Error::SubmitError(err))
                    }
                }
            });

            let acquire_buf = match submit_res {
                Ok(abuf) => abuf,
                Err(err) => {
                    self.cmd_buffer = Some(buf);
                    return Err(err);
                }
            };

            if let Some(abuf) = acquire_buf {
                self.renderer
                    .cmd_pool
                    .store_pending_buffer(abuf, point, Vec::new(), Vec::new());
            }
            self.renderer
                .cmd_pool
                .store_pending_buffer(buf, point, descs, images);

            trace!(point, "VulkanFrame::finish single command buffer submitted");
        }

        if let Some(timeline) = self.renderer.timeline.drm.as_ref() {
            Ok(DrmSyncPoint {
                timeline: timeline.clone(),
                point,
            }
            .into())
        } else {
            Ok(VulkanSyncPoint {
                timeline: self.renderer.timeline.clone(),
                point,
            }
            .into())
        }
    }
}

impl Drop for VulkanFrame<'_, '_> {
    fn drop(&mut self) {
        if let Some(buf) = self.cmd_buffer.take() {
            unsafe {
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
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        let is_to_hdr = is_hdr_vk_format(to.0.format());
        let is_from_hdr = is_hdr_vk_format(from.0.format());
        let is_hdr_mode = self.hdr_config.as_ref().map(|c| !c.is_sdr).unwrap_or(false);

        if !is_to_hdr && is_from_hdr && is_hdr_mode {
            let config = self.hdr_config.unwrap_or_else(HdrOutputConfig::default);
            return self.blit_hdr_to_sdr(from, to, src, dst, filter, &config);
        } else if is_to_hdr && !is_from_hdr && is_hdr_mode {
            let config = HdrOutputConfig::sdr_to_hdr();
            return self.blit_sdr_to_hdr(from, to, src, dst, filter, &config);
        }

        let size = Size::from((Texture::width(&to.0) as i32, Texture::height(&to.0) as i32));
        let saved_downscale = self.downscale_filter;
        let saved_upscale = self.upscale_filter;
        self.downscale_filter = filter;
        self.upscale_filter = filter;
        let pending_waits = std::mem::take(&mut self.pending_waits);
        let release_source = (from.0.needs_acquire() && from.0.dmabuf_exportable()).then(|| from.0.clone());
        let mut frame = VulkanFrame {
            renderer: self,
            fb: to,
            _marker: std::marker::PhantomData,
            transform: Transform::Normal,
            size,
            cmd_buffer: None,
            descriptors: Vec::with_capacity(4),
            images: Vec::with_capacity(4),
            sampled_images: Vec::new(),
            has_draws: false,
            #[cfg(feature = "wayland_frontend")]
            active_color_description: None,
            active_clip: None,
            is_blit: true,
            release_source,
            rendering: false,
            depth_enabled: false,
            current_depth: 0.0,
            pending_clears: Vec::new(),
            pending_waits,
            hdr_to_sdr: None,
            sdr_to_hdr: None,
        };
        let res = if frame.can_copy_image(&from.0, src, dst) {
            frame.copy_image_from_to(&from.0, src, dst)?;
            frame.finish()
        } else {
            let src_rect = Rectangle::new(
                Point::from((src.loc.x as f64, src.loc.y as f64)),
                Size::from((src.size.w as f64, src.size.h as f64)),
            );
            frame.render_texture_from_to(
                &from.0,
                src_rect,
                dst,
                &[Rectangle::from_size(dst.size)],
                &[],
                Transform::Normal,
                1.0,
            )?;
            frame.finish()
        };
        self.downscale_filter = saved_downscale;
        self.upscale_filter = saved_upscale;
        res
    }

    fn blit_hdr_to_sdr(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
        config: &HdrOutputConfig,
    ) -> Result<SyncPoint, Self::Error> {
        tracing::debug!(
            from_format = ?from.0.format(),
            to_format = ?to.0.format(),
            ref_white = config.reference_white,
            max_lum = config.max_luminance,
            "blit_hdr_to_sdr: performing PQ to SDR tonemapping with custom color parameters"
        );
        let size = Size::from((Texture::width(&to.0) as i32, Texture::height(&to.0) as i32));
        let saved_downscale = self.downscale_filter;
        let saved_upscale = self.upscale_filter;
        self.downscale_filter = filter;
        self.upscale_filter = filter;
        let pending_waits = std::mem::take(&mut self.pending_waits);
        let release_source = (from.0.needs_acquire() && from.0.dmabuf_exportable()).then(|| from.0.clone());
        let mut frame = VulkanFrame {
            renderer: self,
            fb: to,
            _marker: std::marker::PhantomData,
            transform: Transform::Normal,
            size,
            cmd_buffer: None,
            descriptors: Vec::with_capacity(4),
            images: Vec::with_capacity(4),
            sampled_images: Vec::new(),
            has_draws: false,
            #[cfg(feature = "wayland_frontend")]
            active_color_description: None,
            active_clip: None,
            is_blit: true,
            release_source,
            rendering: false,
            depth_enabled: false,
            current_depth: 0.0,
            pending_clears: Vec::new(),
            pending_waits,
            hdr_to_sdr: Some(*config),
            sdr_to_hdr: None,
        };
        let src_rect = Rectangle::new(
            Point::from((src.loc.x as f64, src.loc.y as f64)),
            Size::from((src.size.w as f64, src.size.h as f64)),
        );
        frame.render_texture_from_to(
            &from.0,
            src_rect,
            dst,
            &[Rectangle::from_size(dst.size)],
            &[],
            Transform::Normal,
            1.0,
        )?;
        let res = frame.finish();
        self.downscale_filter = saved_downscale;
        self.upscale_filter = saved_upscale;
        res
    }

    fn blit_sdr_to_hdr(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
        config: &HdrOutputConfig,
    ) -> Result<SyncPoint, Self::Error> {
        tracing::debug!(
            from_format = ?from.0.format(),
            to_format = ?to.0.format(),
            ref_white = config.reference_white,
            max_lum = config.max_luminance,
            "blit_sdr_to_hdr: performing SDR to PQ HDR conversion with custom color parameters"
        );
        let size = Size::from((Texture::width(&to.0) as i32, Texture::height(&to.0) as i32));
        let saved_downscale = self.downscale_filter;
        let saved_upscale = self.upscale_filter;
        self.downscale_filter = filter;
        self.upscale_filter = filter;
        let pending_waits = std::mem::take(&mut self.pending_waits);
        let release_source = (from.0.needs_acquire() && from.0.dmabuf_exportable()).then(|| from.0.clone());
        let mut frame = VulkanFrame {
            renderer: self,
            fb: to,
            _marker: std::marker::PhantomData,
            transform: Transform::Normal,
            size,
            cmd_buffer: None,
            descriptors: Vec::with_capacity(4),
            images: Vec::with_capacity(4),
            sampled_images: Vec::new(),
            has_draws: false,
            #[cfg(feature = "wayland_frontend")]
            active_color_description: None,
            active_clip: None,
            is_blit: true,
            release_source,
            rendering: false,
            depth_enabled: false,
            current_depth: 0.0,
            pending_clears: Vec::new(),
            pending_waits,
            hdr_to_sdr: None,
            sdr_to_hdr: Some(*config),
        };
        let src_rect = Rectangle::new(
            Point::from((src.loc.x as f64, src.loc.y as f64)),
            Size::from((src.size.w as f64, src.size.h as f64)),
        );
        frame.render_texture_from_to(
            &from.0,
            src_rect,
            dst,
            &[Rectangle::from_size(dst.size)],
            &[],
            Transform::Normal,
            1.0,
        )?;
        let res = frame.finish();
        self.downscale_filter = saved_downscale;
        self.upscale_filter = saved_upscale;
        res
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

        let fb_format = self.fb.0.format();
        let format_pipelines = self
            .renderer
            .pipelines
            .get_or_create_format_pipelines(fb_format, self.depth_enabled)?;
        let pipeline = if should_blend && color.a() < 1.0 {
            format_pipelines.clear_blend_pipeline
        } else {
            format_pipelines.clear_pipeline
        };
        let layout = *self.renderer.pipelines.clear_pipeline_layout();

        let buf = self.ensure_rendering()?;
        self.has_draws = true;

        unsafe {
            self.renderer
                .device
                .vk()
                .cmd_bind_pipeline(buf, PipelineBindPoint::GRAPHICS, pipeline);
        }

        let dst_rect = Rectangle::new(
            Point::new(_dst.loc.x as f32, _dst.loc.y as f32),
            Size::new(_dst.size.w as f32, _dst.size.h as f32),
        );
        let screen_size = [self.size.w as f32, self.size.h as f32];

        let depth_val = if self.depth_enabled {
            self.current_depth
        } else {
            0.0
        };

        let push_constants = ClearPushConstants {
            dst_rect,
            screen_size,
            depth: depth_val,
            _pad: 0.0,
            color: color.components(),
        };

        unsafe {
            self.renderer.device.vk().cmd_push_constants(
                buf,
                layout,
                ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );
        }

        let scissors = calculate_damage_scissors(
            damage,
            untransformed_dst,
            self.transform,
            &self.size,
            self.fb.width(),
            self.fb.height(),
        );

        self.draw_quads(buf, &scissors);

        Ok(())
    }
}

impl Bind<VulkanImage> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut VulkanImage) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if target.vk_usage().contains(ImageUsageFlags::COLOR_ATTACHMENT)
            || target.vk_usage().contains(ImageUsageFlags::STORAGE)
            || target.vk_usage().contains(ImageUsageFlags::TRANSFER_DST)
        {
            Ok(VulkanFramebuffer(target.clone()))
        } else {
            Err(Error::ImageError(ImageError::MissingOrInvalidUsage))
        }
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Self::supported_render_formats(&self.phd)
    }
}

impl super::Offscreen<VulkanImage> for VulkanRenderer {
    fn create_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoords>,
    ) -> Result<VulkanImage, Self::Error> {
        let usage = ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST
            | ImageUsageFlags::SAMPLED;
        tracing::debug!(
            "VulkanRenderer::create_buffer: format={:?}, size={:?}, usage={:?}",
            format,
            size,
            usage,
        );
        VulkanImage::new_with_fourcc(&self.device, size.w as u32, size.h as u32, format, usage, false)
            .or_else(|err| {
                tracing::warn!(
                    "VulkanRenderer::create_buffer optimal tiling failed ({:?}), falling back to linear",
                    err
                );
                let linear_usage = ImageUsageFlags::COLOR_ATTACHMENT
                    | ImageUsageFlags::TRANSFER_SRC
                    | ImageUsageFlags::TRANSFER_DST
                    | ImageUsageFlags::SAMPLED;
                VulkanImage::new_with_fourcc(
                    &self.device,
                    size.w as u32,
                    size.h as u32,
                    format,
                    linear_usage,
                    true,
                )
            })
            .map_err(|err| {
                tracing::error!(
                    "VulkanRenderer::create_buffer failed completely for format={:?}, size={:?}: {:?}",
                    format,
                    size,
                    err
                );
                Error::ImageError(err)
            })
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        trace!(
            width = Buffer::width(target),
            height = Buffer::height(target),
            modifier = ?target.format().modifier,
            "VulkanRenderer::bind dmabuf"
        );
        let cached = self.dmabuf_cache.lock().unwrap().get(&target.weak()).cloned();
        let image = match cached {
            Some(image) if image.vk_usage().contains(ImageUsageFlags::TRANSFER_DST) => image,
            _ => {
                let image = VulkanImage::new_from_dmabuf(
                    &self.device,
                    target,
                    ImageUsageFlags::COLOR_ATTACHMENT
                        | ImageUsageFlags::TRANSFER_SRC
                        | ImageUsageFlags::TRANSFER_DST
                        | ImageUsageFlags::SAMPLED,
                )
                .map_err(|err| {
                    tracing::error!("VulkanRenderer::bind dmabuf failed: {:?}", err);
                    Error::ImageError(err)
                })?;
                self.dmabuf_cache
                    .lock()
                    .unwrap()
                    .insert(target.weak(), image.clone());
                image
            }
        };
        Ok(VulkanFramebuffer(image))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Self::supported_render_formats(&self.phd)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_vulkan_renderer_version_check() {
        // Instance 1.2: should fail with UnsupportedVersion
        if let Ok(instance_1_2) = Instance::new(Version::VERSION_1_2, None) {
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
        if let Ok(instance_1_3) = Instance::new(Version::VERSION_1_3, None) {
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

    #[test]
    fn test_damage_scissors_coalescing() {
        let dst = Rectangle::new((0, 0).into(), (1920, 1080).into());
        let screen_size = Size::new(1920, 1080);

        // 1. Empty damage
        let scissors = calculate_damage_scissors(&[], dst, Transform::Normal, &screen_size, 1920, 1080);
        assert!(scissors.is_empty());

        // 2. Adjacent horizontal damage
        let d1 = Rectangle::new((0, 0).into(), (50, 100).into());
        let d2 = Rectangle::new((50, 0).into(), (50, 100).into());
        let scissors = calculate_damage_scissors(&[d1, d2], dst, Transform::Normal, &screen_size, 1920, 1080);
        assert_eq!(scissors.len(), 1);
        assert_eq!(scissors[0].offset.x, 0);
        assert_eq!(scissors[0].offset.y, 0);
        assert_eq!(scissors[0].extent.width, 100);
        assert_eq!(scissors[0].extent.height, 100);

        // 3. Adjacent vertical damage
        let d1 = Rectangle::new((10, 0).into(), (100, 50).into());
        let d2 = Rectangle::new((10, 50).into(), (100, 50).into());
        let scissors = calculate_damage_scissors(&[d1, d2], dst, Transform::Normal, &screen_size, 1920, 1080);
        assert_eq!(scissors.len(), 1);
        assert_eq!(scissors[0].offset.x, 10);
        assert_eq!(scissors[0].offset.y, 0);
        assert_eq!(scissors[0].extent.width, 100);
        assert_eq!(scissors[0].extent.height, 100);

        // 4. Contained damage
        let d1 = Rectangle::new((0, 0).into(), (200, 200).into());
        let d2 = Rectangle::new((10, 10).into(), (50, 50).into());
        let scissors = calculate_damage_scissors(&[d1, d2], dst, Transform::Normal, &screen_size, 1920, 1080);
        assert_eq!(scissors.len(), 1);
        assert_eq!(scissors[0].extent.width, 200);
        assert_eq!(scissors[0].extent.height, 200);

        // 5. Disjoint damage
        let d1 = Rectangle::new((0, 0).into(), (20, 20).into());
        let d2 = Rectangle::new((500, 500).into(), (20, 20).into());
        let scissors = calculate_damage_scissors(&[d1, d2], dst, Transform::Normal, &screen_size, 1920, 1080);
        assert_eq!(scissors.len(), 2);

        // 6. Negative coordinate clipping without displacement
        let offscreen_dst = Rectangle::new((-50, -50).into(), (100, 100).into());
        let d_local = Rectangle::new((0, 0).into(), (100, 100).into());
        let scissors = calculate_damage_scissors(
            &[d_local],
            offscreen_dst,
            Transform::Normal,
            &screen_size,
            1920,
            1080,
        );
        assert_eq!(scissors.len(), 1);
        assert_eq!(scissors[0].offset.x, 0);
        assert_eq!(scissors[0].offset.y, 0);
        assert_eq!(scissors[0].extent.width, 50);
        assert_eq!(scissors[0].extent.height, 50);

        // 7. Small gap between damaged areas should NOT be lossily merged into a bounding box
        let d1 = Rectangle::new((10, 10).into(), (100, 20).into());
        let d2 = Rectangle::new((10, 35).into(), (100, 20).into()); // 5px gap in y
        let scissors = calculate_damage_scissors(&[d1, d2], dst, Transform::Normal, &screen_size, 1920, 1080);
        assert_eq!(scissors.len(), 2);
    }

    #[test]
    fn test_push_constants_memory_layout() {
        use super::shaders::{
            ClearPushConstants, HdrTexPushConstants, OutlinePushConstants, TexPushConstants,
        };

        // Push constant blocks must have 16-byte alignment and fit in 128 bytes
        assert_eq!(align_of::<ClearPushConstants>(), 16);
        assert_eq!(size_of::<ClearPushConstants>(), 48);

        assert_eq!(align_of::<TexPushConstants>(), 16);
        assert_eq!(size_of::<TexPushConstants>(), 96);
        assert!(size_of::<TexPushConstants>() <= 128);

        assert_eq!(align_of::<HdrTexPushConstants>(), 16);
        assert_eq!(size_of::<HdrTexPushConstants>(), 128);
        assert!(size_of::<HdrTexPushConstants>() <= 128);

        assert_eq!(align_of::<OutlinePushConstants>(), 16);
        assert_eq!(size_of::<OutlinePushConstants>(), 80);
        assert!(size_of::<OutlinePushConstants>() <= 128);
    }

    #[test]
    fn test_transform_corner_radius() {
        let r = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            transform_corner_radius(r, Transform::Normal),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(transform_corner_radius(r, Transform::_90), [4.0, 1.0, 2.0, 3.0]);
        assert_eq!(transform_corner_radius(r, Transform::_180), [3.0, 4.0, 1.0, 2.0]);
        assert_eq!(transform_corner_radius(r, Transform::_270), [2.0, 3.0, 4.0, 1.0]);
        assert_eq!(
            transform_corner_radius(r, Transform::Flipped),
            [2.0, 1.0, 4.0, 3.0]
        );
    }

    #[test]
    fn test_is_hdr_vk_format() {
        assert!(is_hdr_vk_format(vk::Format::A2B10G10R10_UNORM_PACK32));
        assert!(is_hdr_vk_format(vk::Format::A2R10G10B10_UNORM_PACK32));
        assert!(is_hdr_vk_format(vk::Format::R16G16B16A16_SFLOAT));
        assert!(is_hdr_vk_format(vk::Format::R16G16B16A16_UNORM));
        assert!(!is_hdr_vk_format(vk::Format::B8G8R8A8_UNORM));
        assert!(!is_hdr_vk_format(vk::Format::R8G8B8A8_UNORM));
        assert!(!is_hdr_vk_format(vk::Format::B8G8R8A8_SRGB));
    }

    #[test]
    fn test_blit_hdr_to_sdr_condition() {
        let check_needs_hdr_to_sdr =
            |from_fmt: vk::Format, to_fmt: vk::Format, hdr_cfg: Option<HdrOutputConfig>| -> bool {
                let is_to_hdr = is_hdr_vk_format(to_fmt);
                let is_from_hdr = is_hdr_vk_format(from_fmt);
                let is_hdr_mode = hdr_cfg.as_ref().map(|c| !c.is_sdr).unwrap_or(false);
                !is_to_hdr && is_from_hdr && is_hdr_mode
            };

        let sdr_10bit = vk::Format::A2B10G10R10_UNORM_PACK32;
        let sdr_8bit = vk::Format::R8G8B8A8_UNORM;
        let hdr_fp16 = vk::Format::R16G16B16A16_SFLOAT;

        let sdr_config = Some(HdrOutputConfig::sdr_tonemapping());
        let hdr_config = Some(HdrOutputConfig::default());

        // 1. SDR screen with 10-bit swapchain copied to 8-bit OBS buffer -> must NOT trigger HDR-to-SDR
        assert!(!check_needs_hdr_to_sdr(sdr_10bit, sdr_8bit, sdr_config));

        // 2. SDR screen with no hdr_config -> must NOT trigger HDR-to-SDR
        assert!(!check_needs_hdr_to_sdr(sdr_10bit, sdr_8bit, None));

        // 3. HDR screen with 10-bit swapchain copied to 8-bit OBS buffer -> MUST trigger HDR-to-SDR
        assert!(check_needs_hdr_to_sdr(sdr_10bit, sdr_8bit, hdr_config));

        // 4. HDR screen with FP16 swapchain copied to 8-bit OBS buffer -> MUST trigger HDR-to-SDR
        assert!(check_needs_hdr_to_sdr(hdr_fp16, sdr_8bit, hdr_config));

        // 5. HDR screen copied to HDR 10-bit capture buffer -> must NOT trigger HDR-to-SDR (passthrough)
        assert!(!check_needs_hdr_to_sdr(sdr_10bit, sdr_10bit, hdr_config));
    }

    #[test]
    fn test_blit_sdr_to_hdr_condition() {
        let check_needs_sdr_to_hdr =
            |from_fmt: vk::Format, to_fmt: vk::Format, hdr_cfg: Option<HdrOutputConfig>| -> bool {
                let is_to_hdr = is_hdr_vk_format(to_fmt);
                let is_from_hdr = is_hdr_vk_format(from_fmt);
                let is_hdr_mode = hdr_cfg.as_ref().map(|c| !c.is_sdr).unwrap_or(false);
                is_to_hdr && !is_from_hdr && is_hdr_mode
            };

        let sdr_8bit = vk::Format::R8G8B8A8_UNORM;
        let hdr_10bit = vk::Format::A2B10G10R10_UNORM_PACK32;
        let hdr_fp16 = vk::Format::R16G16B16A16_SFLOAT;

        let sdr_config = Some(HdrOutputConfig::sdr_tonemapping());
        let hdr_config = Some(HdrOutputConfig::default());

        // 1. Direct scanout of SDR 8-bit game to HDR 10-bit portal buffer on HDR desktop -> MUST trigger SDR-to-HDR
        assert!(check_needs_sdr_to_hdr(sdr_8bit, hdr_10bit, hdr_config));

        // 2. Direct scanout of SDR 8-bit game to HDR FP16 portal buffer on HDR desktop -> MUST trigger SDR-to-HDR
        assert!(check_needs_sdr_to_hdr(sdr_8bit, hdr_fp16, hdr_config));

        // 3. SDR desktop (sdr_config) with 8-bit copied to 10-bit -> must NOT trigger SDR-to-HDR
        assert!(!check_needs_sdr_to_hdr(sdr_8bit, hdr_10bit, sdr_config));

        // 4. No hdr_config -> must NOT trigger SDR-to-HDR
        assert!(!check_needs_sdr_to_hdr(sdr_8bit, hdr_10bit, None));

        // 5. Native HDR game (10-bit) to HDR capture buffer (10-bit) -> must NOT trigger SDR-to-HDR
        assert!(!check_needs_sdr_to_hdr(hdr_10bit, hdr_10bit, hdr_config));
    }

    #[test]
    fn test_render_frame_with_indirect_draw_and_bindless() {
        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };
        let Ok(phds) = PhysicalDevice::enumerate(&instance) else {
            return;
        };

        for phd in phds {
            let Ok(mut renderer) = VulkanRenderer::new(&phd, None) else {
                continue;
            };

            assert!(renderer.supports_multi_draw_indirect());
            assert!(renderer.supports_shader_draw_parameters());
            assert!(renderer.supports_descriptor_indexing());
            assert!(renderer.indirect_buffer().is_some());
            assert!(renderer.bindless_pool().is_some());

            // 1. Create a color render target framebuffer image
            let mut target = VulkanImage::new(
                &renderer.device,
                128,
                128,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                false,
            )
            .expect("Failed to create target VulkanImage");

            // 2. Create a source texture image to render
            let src_tex = VulkanImage::new(
                &renderer.device,
                64,
                64,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
                false,
            )
            .expect("Failed to create src VulkanImage");

            // 3. Bind renderer to the target framebuffer and render
            let dst_rect = Rectangle::new((0, 0).into(), (128, 128).into());
            let damage = [Rectangle::new((0, 0).into(), (64, 64).into())];

            let mut fb = renderer.bind(&mut target).expect("Failed to bind framebuffer");
            let mut frame = renderer
                .render(&mut fb, (128, 128).into(), Transform::Normal)
                .expect("Failed to create frame");

            // Execute draw_solid (which calls draw_color -> draw_quads via DrawIndirectBuffer)
            frame
                .draw_solid(dst_rect, &damage, Color32F::new(0.2, 0.4, 0.8, 1.0))
                .expect("draw_solid failed");

            // Execute render_texture_from_to (which updates bindless descriptor pool and calls draw_quads)
            let src_rect = Rectangle::new(Point::new(0.0, 0.0), Size::new(64.0, 64.0));
            frame
                .render_texture_from_to(&src_tex, src_rect, dst_rect, &damage, &[], Transform::Normal, 1.0)
                .expect("render_texture_from_to failed");

            let sync = frame.finish().expect("VulkanFrame::finish failed");
            drop(sync);

            // Verify indirect commands were written into the DrawIndirectBuffer
            let indirect_buf = renderer.indirect_buffer().expect("indirect_buffer should exist");
            assert!(
                indirect_buf.len() > 0,
                "indirect_buffer should contain written DrawIndirectCommands"
            );

            // Wait idle and cleanup
            unsafe {
                let _ = renderer.device.wait_idle();
            }
            renderer.cleanup().expect("cleanup failed");
            assert_eq!(
                renderer.indirect_buffer().unwrap().len(),
                0,
                "indirect_buffer len should reset after cleanup"
            );
        }
    }

    #[test]
    fn test_src_stage_and_access_for_layout() {
        // Acquire from foreign/external queue or transition from UNDEFINED -> NONE / NONE
        assert_eq!(
            src_stage_and_access_for_layout(true, ImageLayout::GENERAL),
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        );
        assert_eq!(
            src_stage_and_access_for_layout(false, ImageLayout::UNDEFINED),
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        );
        // Same-queue transitions preserve exact producer stages/accesses
        assert_eq!(
            src_stage_and_access_for_layout(false, ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            (
                PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::COLOR_ATTACHMENT_READ | AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        );
        assert_eq!(
            src_stage_and_access_for_layout(false, ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            (
                PipelineStageFlags2::FRAGMENT_SHADER,
                AccessFlags2::SHADER_SAMPLED_READ,
            )
        );
        assert_eq!(
            src_stage_and_access_for_layout(false, ImageLayout::TRANSFER_SRC_OPTIMAL),
            (PipelineStageFlags2::ALL_TRANSFER, AccessFlags2::TRANSFER_READ)
        );
        assert_eq!(
            src_stage_and_access_for_layout(false, ImageLayout::TRANSFER_DST_OPTIMAL),
            (PipelineStageFlags2::ALL_TRANSFER, AccessFlags2::TRANSFER_WRITE)
        );
    }

    #[test]
    fn test_dmabuf_and_internal_image_layout_and_queue_ownership() {
        use crate::backend::allocator::{Allocator, Fourcc, Modifier, vulkan::VulkanAllocator};

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };
        let Ok(phds) = PhysicalDevice::enumerate(&instance) else {
            return;
        };

        for phd in phds {
            let Ok(mut renderer) = VulkanRenderer::new(&phd, None) else {
                continue;
            };

            // 1. Internal non-exportable image must start in UNDEFINED with needs_acquire = false
            let mut internal_target = VulkanImage::new(
                &renderer.device,
                64,
                64,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
                false,
            )
            .expect("Failed to create internal target VulkanImage");

            assert!(!internal_target.dmabuf_exportable());
            assert!(!internal_target.needs_acquire());
            assert_eq!(internal_target.current_layout(), ImageLayout::UNDEFINED);

            let rect = Rectangle::new((0, 0).into(), (64, 64).into());
            {
                let mut fb = renderer.bind(&mut internal_target).unwrap();
                let mut frame = renderer
                    .render(&mut fb, (64, 64).into(), Transform::Normal)
                    .unwrap();
                frame
                    .draw_solid(rect, &[rect], Color32F::new(1.0, 0.0, 0.0, 1.0))
                    .unwrap();
                let _sync = frame.finish().unwrap();
            }

            // Internal non-exportable image must NOT be released to foreign queue on finish
            assert!(!internal_target.needs_acquire());
            assert_eq!(
                internal_target.current_layout(),
                ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            );

            // 2. Test DMA-BUF exportable / imported lifecycle when supported by the device
            let mut allocator = VulkanAllocator::from_renderer(
                &renderer,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            );
            if let Ok(exportable_img) =
                allocator.create_buffer(64, 64, Fourcc::Abgr8888, &[Modifier::Linear, Modifier::Invalid])
            {
                // Newly allocated exportable image starts in UNDEFINED with needs_acquire = false
                assert!(exportable_img.dmabuf_exportable());
                assert!(!exportable_img.needs_acquire());
                assert_eq!(exportable_img.current_layout(), ImageLayout::UNDEFINED);

                if let Ok(mut dmabuf) = exportable_img.export() {
                    // Importing an external DMA-BUF must initialize current_layout = GENERAL and needs_acquire = true
                    let imported_tex = renderer
                        .import_dmabuf(&dmabuf, None)
                        .expect("Failed to import dmabuf");
                    assert!(imported_tex.dmabuf_exportable());
                    assert!(imported_tex.needs_acquire());
                    assert_eq!(imported_tex.current_layout(), ImageLayout::GENERAL);

                    // Binding and rendering into the DMA-BUF must release it to ext_queue in GENERAL on finish
                    {
                        let mut dmabuf_fb = renderer.bind(&mut dmabuf).expect("Failed to bind dmabuf");
                        let mut frame = renderer
                            .render(&mut dmabuf_fb, (64, 64).into(), Transform::Normal)
                            .unwrap();
                        frame
                            .draw_solid(rect, &[rect], Color32F::new(0.0, 1.0, 0.0, 1.0))
                            .unwrap();
                        let _sync = frame.finish().unwrap();
                        assert_eq!(dmabuf_fb.0.current_layout(), ImageLayout::GENERAL);
                        assert!(dmabuf_fb.0.needs_acquire());
                    }

                    // Blitting from the finished DMA-BUF framebuffer to an internal framebuffer must
                    // re-release the source DMA-BUF back to ext_queue in GENERAL
                    {
                        let dmabuf_fb = renderer.bind(&mut dmabuf).expect("Failed to bind dmabuf");
                        let mut dst_fb = renderer.bind(&mut internal_target).unwrap();
                        let _sync = renderer
                            .blit(&dmabuf_fb, &mut dst_fb, rect, rect, TextureFilter::Linear)
                            .expect("Blit from dmabuf_fb to internal_target failed");
                        assert_eq!(dmabuf_fb.0.current_layout(), ImageLayout::GENERAL);
                        assert!(dmabuf_fb.0.needs_acquire());
                        assert!(!internal_target.needs_acquire());
                    }

                    // Sampling the imported DMA-BUF acquires it into SHADER_READ_ONLY_OPTIMAL,
                    // and a subsequent import_dmabuf (next client buffer commit) resets it to GENERAL + needs_acquire
                    {
                        let tex = renderer.import_dmabuf(&dmabuf, None).unwrap();
                        assert_eq!(tex.current_layout(), ImageLayout::GENERAL);
                        assert!(tex.needs_acquire());

                        let mut dst_fb = renderer.bind(&mut internal_target).unwrap();
                        let mut frame = renderer
                            .render(&mut dst_fb, (64, 64).into(), Transform::Normal)
                            .unwrap();
                        let src_f64 = Rectangle::new(Point::new(0.0, 0.0), Size::new(64.0, 64.0));
                        frame
                            .render_texture_from_to(&tex, src_f64, rect, &[rect], &[], Transform::Normal, 1.0)
                            .unwrap();
                        let _sync = frame.finish().unwrap();
                        assert_eq!(tex.current_layout(), ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                        assert!(!tex.needs_acquire());

                        let reimported = renderer.import_dmabuf(&dmabuf, None).unwrap();
                        assert_eq!(reimported.current_layout(), ImageLayout::GENERAL);
                        assert!(reimported.needs_acquire());
                    }
                }
            }

            unsafe {
                let _ = renderer.device.wait_idle();
            }
            renderer.cleanup().expect("cleanup failed");
        }
    }

    #[test]
    fn test_shared_device_and_weak_image_cleanup() {
        use crate::backend::{
            allocator::{Allocator, vulkan::VulkanAllocator},
            renderer::Offscreen,
        };

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };
        let Ok(devices) = PhysicalDevice::enumerate(&instance) else {
            return;
        };

        for phd in devices {
            let static_dmabuf_formats = VulkanRenderer::supported_dmabuf_formats(&phd);
            let static_render_formats = VulkanRenderer::supported_render_formats(&phd);

            let Ok(mut renderer_a) = VulkanRenderer::new(&phd, None) else {
                continue;
            };
            assert_eq!(renderer_a.dmabuf_formats(), static_dmabuf_formats);
            assert_eq!(
                Bind::<Dmabuf>::supported_formats(&renderer_a),
                static_render_formats
            );

            let device = renderer_a.device().clone();
            let weak_device = device.downgrade();
            let shared_cache = renderer_a.dmabuf_cache().clone();
            let shared_pipelines = renderer_a.shared_pipelines().clone();

            let mut renderer_b = VulkanRenderer::with_device(
                &device,
                None,
                shared_cache.clone(),
                Some(shared_pipelines.clone()),
            )
            .expect("Failed to create shared VulkanRenderer::with_device");

            // Both renderers share the exact same VkDevice, ContextId, SharedPipelines, and samplers
            assert_eq!(renderer_a.context_id(), renderer_b.context_id());
            assert_eq!(
                renderer_a.device().vk().handle(),
                renderer_b.device().vk().handle()
            );
            assert!(Arc::ptr_eq(
                renderer_a.shared_pipelines(),
                renderer_b.shared_pipelines()
            ));
            assert_eq!(renderer_a.texture_sampler, renderer_b.texture_sampler);
            assert_eq!(
                renderer_a.texture_sampler_nearest,
                renderer_b.texture_sampler_nearest
            );

            // Texture allocated on renderer_a via host upload is immediately transitioned to SHADER_READ_ONLY_OPTIMAL
            // and can be safely rendered on renderer_b
            let shared_tex = renderer_a
                .import_memory(&[255u8; 32 * 32 * 4], Fourcc::Abgr8888, (32, 32).into(), false)
                .expect("Failed to import memory on renderer_a");
            assert_eq!(shared_tex.current_layout(), ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let mut target_a: VulkanImage = renderer_a
                .create_buffer(Fourcc::Abgr8888, (32, 32).into())
                .expect("Failed to create buffer on renderer_a");
            let mut target_b: VulkanImage = renderer_b
                .create_buffer(Fourcc::Abgr8888, (32, 32).into())
                .expect("Failed to create buffer on renderer_b");
            let rect = Rectangle::new(Point::new(0, 0), Size::new(32, 32));
            let src_f64 = Rectangle::new(Point::new(0.0, 0.0), Size::new(32.0, 32.0));
            {
                let mut fb_b = renderer_b.bind(&mut target_b).unwrap();
                let mut frame = renderer_b
                    .render(&mut fb_b, (32, 32).into(), Transform::Normal)
                    .unwrap();
                frame
                    .render_texture_from_to(&shared_tex, src_f64, rect, &[rect], &[], Transform::Normal, 1.0)
                    .unwrap();
                let _sync = frame.finish().unwrap();
            }

            // Test out-of-order batch submission across two renderers sharing a DMA-BUF texture:
            // renderer_a records first, renderer_b records second, renderer_b flushes first, renderer_a flushes second.
            let mut allocator = VulkanAllocator::from_renderer(
                &renderer_a,
                ImageUsageFlags::COLOR_ATTACHMENT
                    | ImageUsageFlags::TRANSFER_SRC
                    | ImageUsageFlags::TRANSFER_DST
                    | ImageUsageFlags::SAMPLED,
            );
            if let Ok(exportable) =
                allocator.create_buffer(32, 32, Fourcc::Abgr8888, &[Modifier::Linear, Modifier::Invalid])
            {
                if let Ok(dmabuf) = exportable.export() {
                    let shared_dmabuf_tex = renderer_a.import_dmabuf(&dmabuf, None).unwrap();
                    assert_eq!(shared_dmabuf_tex.current_layout(), ImageLayout::GENERAL);
                    assert!(shared_dmabuf_tex.needs_acquire());

                    renderer_a.begin_batch();
                    renderer_b.begin_batch();

                    let sync_a = {
                        let mut fb_a = renderer_a.bind(&mut target_a).unwrap();
                        let mut frame_a = renderer_a
                            .render(&mut fb_a, (32, 32).into(), Transform::Normal)
                            .unwrap();
                        frame_a
                            .render_texture_from_to(
                                &shared_dmabuf_tex,
                                src_f64,
                                rect,
                                &[rect],
                                &[],
                                Transform::Normal,
                                1.0,
                            )
                            .unwrap();
                        frame_a.finish().unwrap()
                    };

                    // Before flush_batch, layout transition has not yet been submitted to the queue
                    assert_eq!(shared_dmabuf_tex.current_layout(), ImageLayout::GENERAL);
                    assert!(shared_dmabuf_tex.needs_acquire());

                    let sync_b = {
                        let mut fb_b = renderer_b.bind(&mut target_b).unwrap();
                        let mut frame_b = renderer_b
                            .render(&mut fb_b, (32, 32).into(), Transform::Normal)
                            .unwrap();
                        frame_b
                            .render_texture_from_to(
                                &shared_dmabuf_tex,
                                src_f64,
                                rect,
                                &[rect],
                                &[],
                                Transform::Normal,
                                1.0,
                            )
                            .unwrap();
                        frame_b.finish().unwrap()
                    };

                    // Flush renderer_b FIRST (even though renderer_a recorded first)
                    renderer_b.flush_batch().unwrap();
                    assert_eq!(
                        shared_dmabuf_tex.current_layout(),
                        ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    );
                    assert!(!shared_dmabuf_tex.needs_acquire());

                    // Flush renderer_a SECOND (must not emit duplicate foreign-queue acquire barrier)
                    renderer_a.flush_batch().unwrap();
                    let _ = sync_b.wait();
                    let _ = sync_a.wait();

                    // Both renderers compiled/reused the exact same vk::Pipeline handles via SharedPipelines
                    let fmt = target_a.format();
                    let p_a = renderer_a
                        .pipelines
                        .get_or_create_format_pipelines(fmt, false)
                        .unwrap();
                    let p_b = renderer_b
                        .pipelines
                        .get_or_create_format_pipelines(fmt, false)
                        .unwrap();
                    assert_eq!(p_a.tex_pipeline, p_b.tex_pipeline);
                    assert_eq!(p_a.clear_pipeline, p_b.clear_pipeline);
                }
            }

            // Per VUID-vkDestroyDevice-device-05137, child VulkanImage handles keep VkDevice alive
            // until all child objects and caches are dropped.
            drop(allocator);
            drop(target_a);
            drop(renderer_a);
            drop(renderer_b);
            drop(shared_pipelines);
            drop(device);
            assert!(
                weak_device.upgrade().is_some(),
                "Live VulkanImage handles must keep VkDevice alive until destroyed (VUID-vkDestroyDevice-device-05137)"
            );

            // Deterministically evict caches and drop remaining images; VkDevice must be destroyed immediately
            shared_cache.lock().unwrap().clear();
            drop(shared_tex);
            drop(target_b);
            assert!(
                weak_device.upgrade().is_none(),
                "VkDevice must be destroyed immediately once all renderers, caches, and images are dropped"
            );
        }
    }
}
