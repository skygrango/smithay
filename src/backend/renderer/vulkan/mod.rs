//! Implementation of the rendering traits using Vulkan

use crate::{
    backend::{
        allocator::{
            Format, Fourcc,
            dmabuf::{Dmabuf, WeakDmabuf},
            format::FormatSet,
        },
        drm::{
            DrmDeviceFd,
            sync::{DrmSyncPoint, DrmTimeline, WeakDrmTimeline},
        },
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
    DescriptorType, DrawIndirectCommand, Extent3D, Fence, Filter, FormatFeatureFlags, HostImageCopyFlagsEXT,
    ImageAspectFlags, ImageLayout, ImageMemoryBarrier2, ImageSubresourceLayers, ImageSubresourceRange,
    MemoryMapFlags, MemoryPropertyFlags, MemoryToImageCopyEXT, Offset3D, PipelineBindPoint,
    PipelineStageFlags2, QUEUE_FAMILY_IGNORED, Result as VkResult, SamplerAddressMode, SamplerCreateFlags,
    SamplerCreateInfo, SamplerMipmapMode, SemaphoreSubmitInfo, SemaphoreWaitInfo, ShaderStageFlags,
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
pub use self::shaders::Error as PipelineError;
use self::shaders::Pipelines;
pub use self::sync::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudgetInfo {
    pub heap_budget: [vk::DeviceSize; vk::MAX_MEMORY_HEAPS],
    pub heap_usage: [vk::DeviceSize; vk::MAX_MEMORY_HEAPS],
}

impl MemoryBudgetInfo {
    pub fn total_budget(&self) -> vk::DeviceSize {
        self.heap_budget.iter().copied().sum()
    }

    pub fn total_usage(&self) -> vk::DeviceSize {
        self.heap_usage.iter().copied().sum()
    }

    pub fn usage_ratio(&self) -> f32 {
        let budget = self.total_budget();
        if budget == 0 {
            0.0
        } else {
            self.total_usage() as f32 / budget as f32
        }
    }
}

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
    pub device: ash::Device,
    pub staging_buffer: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub buffer_size: vk::DeviceSize,
    pub is_coherent: bool,
    pub timeline_sem: vk::Semaphore,
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
        let sems = [self.timeline_sem];
        let vals = [self.point];
        let wait_info = SemaphoreWaitInfo::default().semaphores(&sems).values(&vals);
        loop {
            let res = unsafe { self.device.wait_semaphores(&wait_info, 100_000_000) };
            match res {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                Err(err) => return Err(err),
            }
        }

        let src_ptr = unsafe {
            self.device
                .map_memory(self.staging_memory, 0, self.buffer_size, MemoryMapFlags::empty())?
        };

        if !self.is_coherent {
            let mapped_range = vk::MappedMemoryRange::default()
                .memory(self.staging_memory)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            unsafe {
                let _ = self.device.invalidate_mapped_memory_ranges(&[mapped_range]);
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
            self.device.unmap_memory(self.staging_memory);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);
        }
        self.staging_buffer = vk::Buffer::null();
        self.staging_memory = vk::DeviceMemory::null();

        Ok(())
    }
}

impl Drop for PendingVulkanShmCopy {
    fn drop(&mut self) {
        if self.staging_buffer != vk::Buffer::null() {
            unsafe {
                let sems = [self.timeline_sem];
                let vals = [self.point];
                let wait_info = SemaphoreWaitInfo::default().semaphores(&sems).values(&vals);
                let _ = self.device.wait_semaphores(&wait_info, 100_000_000);
                self.device.destroy_buffer(self.staging_buffer, None);
                self.staging_buffer = vk::Buffer::null();
            }
        }
        if self.staging_memory != vk::DeviceMemory::null() {
            unsafe {
                self.device.free_memory(self.staging_memory, None);
                self.staging_memory = vk::DeviceMemory::null();
            }
        }
    }
}

#[derive(Debug)]
pub struct VulkanRenderer {
    capabilities: Vec<Capability>,
    dmabuf_cache: HashMap<WeakDmabuf, VulkanImage>,

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
            let _ = self.device.vk().device_wait_idle();
            for q in self.queued_submits.drain(..) {
                self.device
                    .vk()
                    .free_command_buffers(self.cmd_pool.vk(), &[q.cmd_buffer]);
            }
            self.cmd_pool.clean_old_buffers(u64::MAX);
            for destruction in self.deferred_destructions.drain(..) {
                (destruction.callback)(&self.device);
            }
            self.dmabuf_cache.clear();
            if let Some(mut lut) = self.lut3d.take() {
                lut.destroy(&self.device);
            }
            self.device.vk().destroy_sampler(self.texture_sampler, None);
            self.device
                .vk()
                .destroy_sampler(self.texture_sampler_nearest, None);
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
                        .mag_filter(Filter::LINEAR)
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

        let sampler_nearest = unsafe {
            device
                .vk()
                .create_sampler(
                    &SamplerCreateInfo::default()
                        .flags(SamplerCreateFlags::empty())
                        .mag_filter(Filter::NEAREST)
                        .min_filter(Filter::NEAREST)
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

        // Trilinear 3D LUT is disabled in favor of analytical HDR precision / tetrahedral interpolation
        let lut3d = None;

        let supports_descriptor_indexing = capabilities.contains(&Capability::DescriptorIndexing);
        let supports_multi_draw_indirect = capabilities.contains(&Capability::MultiDrawIndirect);
        let supports_shader_draw_parameters = capabilities.contains(&Capability::ShaderDrawParameters);

        let indirect_buffer = if supports_multi_draw_indirect {
            match DrawIndirectBuffer::new(&device, DrawIndirectBuffer::DEFAULT_CAPACITY) {
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
            match BindlessDescriptorPool::new(&device, BindlessDescriptorPool::DEFAULT_MAX_TEXTURES) {
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
            device,
            capabilities,
            dmabuf_cache: HashMap::new(),
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
            deferred_destructions: VecDeque::new(),
        })
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

        let queued = std::mem::take(&mut self.queued_submits);
        let mut submit_infos = Vec::with_capacity(queued.len());
        for q in &queued {
            submit_infos.push(
                SubmitInfo2::default()
                    .command_buffer_infos(&q.cmd_buffer_infos)
                    .signal_semaphore_infos(&q.signal_semaphore_infos)
                    .wait_semaphore_infos(&q.wait_semaphore_infos),
            );
        }

        let res = unsafe {
            self.device
                .vk()
                .queue_submit2(*self.device.queue(), &submit_infos, Fence::null())
        };

        if let Err(err) = res {
            if err == vk::Result::ERROR_DEVICE_LOST {
                return Err(Error::DeadDevice);
            } else {
                return Err(Error::SubmitError(err));
            }
        }

        for q in queued {
            self.cmd_pool
                .store_pending_buffer(q.cmd_buffer, q.point, q.descs, q.images);
        }

        Ok(())
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
        let val = match unsafe { self.device.vk().get_semaphore_counter_value(self.timeline.vk) } {
            Ok(val) => val,
            Err(vk::Result::ERROR_DEVICE_LOST) => return Err(Error::DeadDevice),
            Err(err) => return Err(Error::SemaphoreCounterError(err)),
        };
        self.cmd_pool.clean_old_buffers(val);
        if self.cmd_pool.is_empty() {
            if let Some(ref mut ib) = self.indirect_buffer {
                ib.reset();
            }
        }

        // Retire deferred destructions whose timeline point <= val
        let idx = self.deferred_destructions.iter().position(|d| d.point > val);
        let ready = if let Some(idx) = idx {
            self.deferred_destructions.drain(..idx)
        } else {
            self.deferred_destructions.drain(..)
        };
        for destruction in ready {
            (destruction.callback)(&self.device);
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
                    let range = vk::MappedMemoryRange::default()
                        .memory(image.inner.memory)
                        .offset(image.inner.memory_offset)
                        .size(layout.size);
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
        let pending_waits = std::mem::take(&mut self.pending_waits);
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
            rendering: false,
            depth_enabled: false,
            current_depth: 0.0,
            depth_image: None,
            pending_clears: Vec::new(),
            pending_waits,
            hdr_to_sdr: None,
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

            let base_ptr = image.map().map_err(Error::HostImageCopyError)?;
            let ptr = unsafe { base_ptr.add(layout.offset as usize) };

            Ok(VulkanMapping::Mapped(
                unsafe { NonNull::new_unchecked(ptr as *mut _) },
                layout.size as usize,
                image.clone(),
                self.device.downgrade(),
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

        let mut mem_type_index = None;
        let mut is_coherent = false;
        for (i, mem_type) in self
            .device
            .memory_properties()
            .memory_types_as_slice()
            .iter()
            .enumerate()
        {
            if (mem_reqs.memory_type_bits & (1 << i)) != 0
                && mem_type
                    .property_flags
                    .contains(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT)
            {
                mem_type_index = Some(i as u32);
                is_coherent = true;
                break;
            }
        }
        if mem_type_index.is_none() {
            for (i, mem_type) in self
                .device
                .memory_properties()
                .memory_types_as_slice()
                .iter()
                .enumerate()
            {
                if (mem_reqs.memory_type_bits & (1 << i)) != 0
                    && mem_type
                        .property_flags
                        .contains(MemoryPropertyFlags::HOST_VISIBLE)
                {
                    mem_type_index = Some(i as u32);
                    is_coherent = false;
                    break;
                }
            }
        }

        let Some(mem_type_index) = mem_type_index else {
            unsafe {
                self.device.vk().destroy_buffer(staging_buffer, None);
            }
            return Err(Error::ImageError(ImageError::NoMemoryAvailable));
        };

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
        let (tex_src_stage, tex_src_access) = if tex_needs_acquire || old_layout == ImageLayout::UNDEFINED {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if old_layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            (
                PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        } else {
            (
                PipelineStageFlags2::ALL_TRANSFER
                    | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    | PipelineStageFlags2::COMPUTE_SHADER,
                AccessFlags2::TRANSFER_WRITE
                    | AccessFlags2::COLOR_ATTACHMENT_WRITE
                    | AccessFlags2::SHADER_STORAGE_WRITE,
            )
        };

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

        let restore_layout = if old_layout == ImageLayout::UNDEFINED {
            ImageLayout::GENERAL
        } else {
            old_layout
        };
        let restore_barrier = ImageMemoryBarrier2::default()
            .image(*image.vk())
            .old_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(restore_layout)
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

        let submit_res = unsafe {
            self.device
                .vk()
                .queue_submit2(*self.device.queue(), &[submit_info], Fence::null())
        };

        if let Err(err) = submit_res {
            if err == vk::Result::ERROR_DEVICE_LOST {
                return Err(Error::DeadDevice);
            } else {
                return Err(Error::SubmitError(err));
            }
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

        let mut mem_type_index = None;
        let mut is_coherent = false;
        for (i, mem_type) in self
            .device
            .memory_properties()
            .memory_types_as_slice()
            .iter()
            .enumerate()
        {
            if (mem_reqs.memory_type_bits & (1 << i)) != 0
                && mem_type
                    .property_flags
                    .contains(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT)
            {
                mem_type_index = Some(i as u32);
                is_coherent = true;
                break;
            }
        }
        if mem_type_index.is_none() {
            for (i, mem_type) in self
                .device
                .memory_properties()
                .memory_types_as_slice()
                .iter()
                .enumerate()
            {
                if (mem_reqs.memory_type_bits & (1 << i)) != 0
                    && mem_type
                        .property_flags
                        .contains(MemoryPropertyFlags::HOST_VISIBLE)
                {
                    mem_type_index = Some(i as u32);
                    is_coherent = false;
                    break;
                }
            }
        }

        let Some(mem_type_index) = mem_type_index else {
            unsafe {
                self.device.vk().destroy_buffer(staging_buffer, None);
            }
            return Err(Error::ImageError(ImageError::NoMemoryAvailable));
        };

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
        let (tex_src_stage, tex_src_access) = if tex_needs_acquire || old_layout == ImageLayout::UNDEFINED {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else {
            (
                PipelineStageFlags2::ALL_TRANSFER
                    | PipelineStageFlags2::COMPUTE_SHADER
                    | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::TRANSFER_WRITE
                    | AccessFlags2::SHADER_STORAGE_WRITE
                    | AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        };

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

        let restore_layout = if old_layout == ImageLayout::UNDEFINED {
            ImageLayout::GENERAL
        } else {
            old_layout
        };
        let restore_barrier = ImageMemoryBarrier2::default()
            .image(*image.vk())
            .old_layout(ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(restore_layout)
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
            });
        } else {
            let submit_info = SubmitInfo2::default()
                .command_buffer_infos(&cmd_buffer_info)
                .signal_semaphore_infos(&signal_semaphore_info)
                .wait_semaphore_infos(&wait_semaphore_info);

            let submit_res = unsafe {
                self.device
                    .vk()
                    .queue_submit2(*self.device.queue(), &[submit_info], Fence::null())
            };

            if let Err(err) = submit_res {
                if err == vk::Result::ERROR_DEVICE_LOST {
                    return Err(Error::DeadDevice);
                } else {
                    return Err(Error::SubmitError(err));
                }
            }

            self.cmd_pool
                .store_pending_buffer(buf, next_seq_no, vec![], vec![image.inner.clone()]);
        }

        self.seq_no = next_seq_no;

        Ok(PendingVulkanShmCopy {
            device: self.device.vk().clone(),
            staging_buffer,
            staging_memory,
            buffer_size,
            is_coherent,
            timeline_sem: self.timeline.vk,
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
    rendering: bool,
    pub(crate) depth_enabled: bool,
    pub(crate) current_depth: f32,
    pub(crate) depth_image: Option<VulkanImage>,
    pub(crate) pending_clears: Vec<(Color32F, Vec<vk::ClearRect>)>,
    pub(crate) pending_waits: Vec<DrmSyncPoint>,
    pub(crate) hdr_to_sdr: Option<HdrOutputConfig>,
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
        let mut transitioned = Vec::new();

        for texture in textures {
            if !self.images.iter().any(|img| Arc::ptr_eq(img, &texture.inner)) {
                self.images.push(texture.inner.clone());
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
                let (tex_src_stage, tex_src_access) =
                    if tex_needs_acquire || tex_old_layout == ImageLayout::UNDEFINED {
                        (PipelineStageFlags2::NONE, AccessFlags2::NONE)
                    } else if tex_old_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                        (
                            PipelineStageFlags2::FRAGMENT_SHADER,
                            AccessFlags2::SHADER_SAMPLED_READ,
                        )
                    } else {
                        (
                            PipelineStageFlags2::COMPUTE_SHADER
                                | PipelineStageFlags2::ALL_TRANSFER
                                | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                                | PipelineStageFlags2::HOST,
                            AccessFlags2::SHADER_STORAGE_WRITE
                                | AccessFlags2::TRANSFER_WRITE
                                | AccessFlags2::COLOR_ATTACHMENT_WRITE
                                | AccessFlags2::HOST_WRITE,
                        )
                    };

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

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let fb_needs_acquire = self.fb.0.needs_acquire();
        let (fb_src_queue, fb_dst_queue) = if fb_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let fb_old_layout = self.fb.0.current_layout();
        let (fb_src_stage, fb_src_access) = if fb_needs_acquire || fb_old_layout == ImageLayout::UNDEFINED {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if fb_old_layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            (
                PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        } else {
            (
                PipelineStageFlags2::ALL_TRANSFER | PipelineStageFlags2::COMPUTE_SHADER,
                AccessFlags2::TRANSFER_WRITE | AccessFlags2::SHADER_STORAGE_WRITE,
            )
        };

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
                let old = self.depth_image.replace(img);
                if let Some(old_depth) = old {
                    if !self.images.iter().any(|i| Arc::ptr_eq(i, &old_depth.inner)) {
                        self.images.push(old_depth.inner.clone());
                    }
                }
            }

            let depth_img = self.depth_image.as_ref().unwrap();
            let depth_old_layout = depth_img.current_layout();
            let depth_barrier = ImageMemoryBarrier2::default()
                .image(*depth_img.vk())
                .old_layout(depth_old_layout)
                .new_layout(ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
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
        let cmd_buffer = self.get_or_create_cmd_buffer()?;

        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let fb_needs_acquire = self.fb.0.needs_acquire();
        let (fb_src_queue, fb_dst_queue) = if fb_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let fb_old_layout = self.fb.0.current_layout();
        let (fb_src_stage, fb_src_access) = if fb_needs_acquire || fb_old_layout == ImageLayout::UNDEFINED {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if fb_old_layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            (
                PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        } else {
            (
                PipelineStageFlags2::ALL_TRANSFER | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::TRANSFER_WRITE | AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        };

        let tex_needs_acquire = texture.needs_acquire() && texture.dmabuf_exportable();
        let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let tex_old_layout = texture.current_layout();
        let (tex_src_stage, tex_src_access) = if tex_needs_acquire || tex_old_layout == ImageLayout::UNDEFINED
        {
            (PipelineStageFlags2::NONE, AccessFlags2::NONE)
        } else if tex_old_layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            (
                PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        } else if tex_old_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            (
                PipelineStageFlags2::FRAGMENT_SHADER,
                AccessFlags2::SHADER_SAMPLED_READ,
            )
        } else {
            (
                PipelineStageFlags2::ALL_TRANSFER | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::TRANSFER_READ
                    | AccessFlags2::TRANSFER_WRITE
                    | AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        };

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
        // or when explicitly performing an HDR-to-SDR blit.
        let is_hdr = if self.hdr_to_sdr.is_some() {
            true
        } else {
            !self.is_blit && self.renderer.hdr_config.is_some()
        };
        self.has_draws = true;
        if !self.images.iter().any(|img| Arc::ptr_eq(img, &texture.inner)) {
            self.images.push(texture.inner.clone());
        }
        let qfam = self.renderer.device.queue_family_idx();
        let ext_queue = self.renderer.external_queue_family();

        let tex_needs_acquire = texture.needs_acquire() && texture.dmabuf_exportable();
        let (tex_src_queue, tex_dst_queue) = if tex_needs_acquire {
            (ext_queue, qfam)
        } else {
            (QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
        };

        let tex_old_layout = texture.current_layout();
        let needs_tex_barrier = tex_needs_acquire || tex_old_layout != ImageLayout::SHADER_READ_ONLY_OPTIMAL;

        if needs_tex_barrier {
            if self.rendering {
                self.end_rendering();
            }
            let buf = self.get_or_create_cmd_buffer()?;

            let (tex_src_stage, tex_src_access) =
                if tex_needs_acquire || tex_old_layout == ImageLayout::UNDEFINED {
                    (PipelineStageFlags2::NONE, AccessFlags2::NONE)
                } else if tex_old_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                    (
                        PipelineStageFlags2::FRAGMENT_SHADER,
                        AccessFlags2::SHADER_SAMPLED_READ,
                    )
                } else {
                    (
                        PipelineStageFlags2::COMPUTE_SHADER
                            | PipelineStageFlags2::ALL_TRANSFER
                            | PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                            | PipelineStageFlags2::HOST,
                        AccessFlags2::SHADER_STORAGE_WRITE
                            | AccessFlags2::TRANSFER_WRITE
                            | AccessFlags2::COLOR_ATTACHMENT_WRITE
                            | AccessFlags2::HOST_WRITE,
                    )
                };

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

        let buf = self.ensure_rendering()?;

        let push_descriptor = self.renderer.device.vk_khr_push_descriptor();
        let descriptor = if push_descriptor.is_none() {
            Some(if is_hdr {
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
        let sampler = self.renderer.active_sampler(is_downscaling);

        if self.renderer.supports_descriptor_indexing {
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

        let depth_val = if self.depth_enabled {
            self.current_depth
        } else {
            0.0
        };

        let (layout, pipeline, hdr_push_constants) = if is_hdr {
            let (config, is_hdr_to_sdr) = if let Some(ref blit_cfg) = self.hdr_to_sdr {
                (*blit_cfg, true)
            } else {
                (
                    self.renderer
                        .hdr_config
                        .unwrap_or(crate::backend::renderer::gles::HdrOutputConfig {
                            reference_white: 203.0,
                            max_luminance: 1000.0,
                            sdr_gamma: 2.2,
                            gamut_stretch: 0.0,
                            hardware_offload: false,
                            is_sdr: false,
                        }),
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
            let hardware_offload = if is_hdr_to_sdr {
                0u32
            } else {
                config.hardware_offload as u32
            };
            let target_is_sdr = if is_hdr_to_sdr || !is_hdr_vk_format(self.fb.0.format()) {
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
            if !is_hdr_to_sdr && let Some(desc) = self.active_color_description.as_ref() {
                use crate::wayland::color::management::{Primaries, TransferFunction};
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
            } else if !is_hdr_to_sdr {
                if config.is_sdr && sdr_gamma == 0.0 {
                    skip_color_transform = 1;
                }
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
                hardware_offload,
                target_is_sdr,
                input_is_pq,
                input_is_hlg,
                input_primaries,
                skip_color_transform,
                content_reference,
                _pad1: 0,
            };

            let chosen_pipeline = if should_blend {
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

            (
                *self.renderer.pipelines.hdr_tex_pipeline_layout(),
                chosen_pipeline,
                Some(push_constants),
            )
        } else {
            let chosen_pipeline = if should_blend {
                format_pipelines.tex_blend_pipeline
            } else {
                format_pipelines.tex_pipeline
            };
            (
                *self.renderer.pipelines.tex_pipeline_layout(),
                chosen_pipeline,
                None,
            )
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

        let current_layout = self.fb.0.current_layout();
        let (src_stage, src_access) = if current_layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            (
                PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        } else {
            (PipelineStageFlags2::ALL_TRANSFER, AccessFlags2::TRANSFER_WRITE)
        };

        // For scanout release to ext_queue (VK_QUEUE_FAMILY_FOREIGN_EXT or VK_QUEUE_FAMILY_EXTERNAL),
        // keep current_layout (e.g. COLOR_ATTACHMENT_OPTIMAL) instead of GENERAL.
        // This avoids forcing an unnecessary DCC decompress pass in the driver and allows
        // direct compressed scanout by DCN/KMS.
        let new_layout = current_layout;

        let barrier = ImageMemoryBarrier2::default()
            .image(*self.fb.0.vk())
            .old_layout(current_layout)
            .new_layout(new_layout)
            .src_queue_family_index(qfam)
            .dst_queue_family_index(ext_queue)
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
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
                                PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
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
        if !images.iter().any(|img| Arc::ptr_eq(img, &self.fb.0.inner)) {
            images.push(self.fb.0.inner.clone());
        }
        if let Some(depth) = self.depth_image.take() {
            if !images.iter().any(|img| Arc::ptr_eq(img, &depth.inner)) {
                images.push(depth.inner.clone());
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
            });
            trace!(point, "VulkanFrame::finish command buffer queued in batch");
        } else {
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
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        let is_to_hdr = is_hdr_vk_format(to.0.format());
        let is_from_hdr = is_hdr_vk_format(from.0.format());
        let is_hdr_mode = self.hdr_config.as_ref().map(|c| !c.is_sdr).unwrap_or(false);

        if !is_to_hdr && is_from_hdr && is_hdr_mode {
            let config = self.hdr_config.unwrap_or_else(HdrOutputConfig::default);
            return self.blit_hdr_to_sdr(from, to, src, dst, filter, &config);
        }

        let size = Size::from((Texture::width(&to.0) as i32, Texture::height(&to.0) as i32));
        let saved_downscale = self.downscale_filter;
        let saved_upscale = self.upscale_filter;
        self.downscale_filter = filter;
        self.upscale_filter = filter;
        let pending_waits = std::mem::take(&mut self.pending_waits);
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
            rendering: false,
            depth_enabled: false,
            current_depth: 0.0,
            depth_image: None,
            pending_clears: Vec::new(),
            pending_waits,
            hdr_to_sdr: None,
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
            rendering: false,
            depth_enabled: false,
            current_depth: 0.0,
            depth_image: None,
            pending_clears: Vec::new(),
            pending_waits,
            hdr_to_sdr: Some(*config),
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
        known_formats()
            .iter()
            .try_fold(IndexSet::new(), |mut set, fourcc| {
                let format = get_vk_format(*fourcc).unwrap();
                let props = self.phd.get_format_modifier_properties(format)?;
                // TODO: else use get_format_properties and map to LINEAR and INVALID
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
        let mut usage = ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST
            | ImageUsageFlags::SAMPLED;
        if self.supports_optimal_host_copy {
            usage |= ImageUsageFlags::HOST_TRANSFER_EXT;
        }
        tracing::debug!(
            "VulkanRenderer::create_buffer: format={:?}, size={:?}, usage={:?}, supports_optimal_host_copy={}",
            format,
            size,
            usage,
            self.supports_optimal_host_copy
        );
        VulkanImage::new_with_fourcc(&self.device, size.w as u32, size.h as u32, format, usage, false)
            .or_else(|err| {
                tracing::warn!(
                    "VulkanRenderer::create_buffer optimal tiling failed ({:?}), falling back to linear",
                    err
                );
                let mut linear_usage = ImageUsageFlags::COLOR_ATTACHMENT
                    | ImageUsageFlags::TRANSFER_SRC
                    | ImageUsageFlags::TRANSFER_DST
                    | ImageUsageFlags::SAMPLED;
                if self.device.vk_ext_host_image_copy().is_some() {
                    linear_usage |= ImageUsageFlags::HOST_TRANSFER_EXT;
                }
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
        use crate::backend::allocator::Buffer as AllocBuffer;
        trace!(
            width = AllocBuffer::width(target),
            height = AllocBuffer::height(target),
            modifier = ?target.format().modifier,
            "VulkanRenderer::bind dmabuf"
        );
        let image = match self.dmabuf_cache.get(&target.weak()) {
            Some(image) if image.vk_usage().contains(ImageUsageFlags::TRANSFER_DST) => image.clone(),
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
                        .contains(FormatFeatureFlags::COLOR_ATTACHMENT)
                        || prop
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
        use super::shaders::{ClearPushConstants, HdrTexPushConstants, TexPushConstants};

        // Push constant blocks must have 16-byte alignment and fit in 128 bytes
        assert_eq!(align_of::<ClearPushConstants>(), 16);
        assert_eq!(size_of::<ClearPushConstants>(), 48);

        assert_eq!(align_of::<TexPushConstants>(), 16);
        assert_eq!(size_of::<TexPushConstants>(), 64);

        assert_eq!(align_of::<HdrTexPushConstants>(), 16);
        assert_eq!(size_of::<HdrTexPushConstants>(), 112);
        assert!(size_of::<HdrTexPushConstants>() <= 128);
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
        use crate::backend::renderer::color::HdrOutputConfig;

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
    fn test_render_frame_with_indirect_draw_and_bindless() {
        let Ok(instance) = crate::backend::vulkan::Instance::new(Version::VERSION_1_3, None) else {
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
            use crate::backend::renderer::{Frame, Renderer};
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
                let _ = renderer.device.vk().device_wait_idle();
            }
            renderer.cleanup().expect("cleanup failed");
            assert_eq!(
                renderer.indirect_buffer().unwrap().len(),
                0,
                "indirect_buffer len should reset after cleanup"
            );
        }
    }
}
