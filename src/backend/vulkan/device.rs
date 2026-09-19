use crate::backend::{
    renderer::ContextId,
    vulkan::{PhysicalDevice, format::FormatEntry, image::VulkanImage},
};
use ash::{
    Device as VkDevice, ext, khr,
    vk::{
        self, DeviceCreateInfo, DeviceQueueCreateInfo, ImageTiling, PhysicalDeviceFeatures2,
        PhysicalDeviceMemoryProperties, Queue, QueueFlags,
    },
};
use drm::node::DrmNode;
use std::{
    ffi::CStr,
    fmt,
    sync::{Arc, Weak},
};

#[derive(Debug, Clone)]
pub struct Device(Arc<InnerDevice>);
#[derive(Debug, Clone)]
pub struct WeakDevice(Weak<InnerDevice>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    Transfer,
    Compute,
    Graphics,
}

impl QueueType {
    pub fn vk_flags(&self) -> QueueFlags {
        match self {
            QueueType::Transfer => QueueFlags::TRANSFER,
            QueueType::Compute => QueueFlags::COMPUTE,
            QueueType::Graphics => QueueFlags::GRAPHICS,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("No matching queue")]
    NoUsableQueue,
    /// Vulkan API error.
    #[error(transparent)]
    Vk(#[from] vk::Result),
}

impl Device {
    pub fn new(
        phd: &PhysicalDevice,
        extensions: &[&CStr],
        required_features: &mut PhysicalDeviceFeatures2<'_>,
        queue_type: QueueType,
        fallback_to_graphics: bool,
    ) -> Result<Self, DeviceError> {
        let extension_pointers = extensions.iter().copied().map(CStr::as_ptr).collect::<Vec<_>>();
        for ptr in &extension_pointers {
            println!("{:x} {:?}", (*ptr) as usize, unsafe { CStr::from_ptr(*ptr) });
        }

        let queue_families = unsafe {
            phd.instance()
                .handle()
                .get_physical_device_queue_family_properties(phd.handle())
        };
        let flags = queue_type.vk_flags();
        let queue_index = queue_families
            .iter()
            // Find a queue with a matching type
            .position(|properties| properties.queue_flags.contains(flags))
            // Fallback
            .or_else(|| {
                if fallback_to_graphics {
                    queue_families
                        .iter()
                        .position(|properties| properties.queue_flags.contains(QueueFlags::GRAPHICS))
                } else {
                    None
                }
            })
            .ok_or(DeviceError::NoUsableQueue)?;

        let mem_properties = unsafe {
            phd.instance()
                .handle()
                .get_physical_device_memory_properties(phd.handle())
        };
        let memory_preferences = super::memory::MemoryPreferences::new(&mem_properties);

        let has_memory_priority = extensions
            .iter()
            .any(|&ext| ext == ash::vk::EXT_MEMORY_PRIORITY_NAME);
        let has_memory_budget = extensions
            .iter()
            .any(|&ext| ext == ash::vk::EXT_MEMORY_BUDGET_NAME);
        let has_global_priority = extensions
            .iter()
            .any(|&ext| ext == ash::vk::KHR_GLOBAL_PRIORITY_NAME || ext == ash::vk::EXT_GLOBAL_PRIORITY_NAME);

        let mut global_priority_info = vk::DeviceQueueGlobalPriorityCreateInfoKHR::default()
            .global_priority(vk::QueueGlobalPriorityKHR::HIGH);

        let mut queue_info = DeviceQueueCreateInfo::default()
            .queue_family_index(queue_index as u32)
            .queue_priorities(&[1.0]);

        if has_global_priority {
            queue_info = queue_info.push_next(&mut global_priority_info);
        }

        let queue_create_infos: &[DeviceQueueCreateInfo<'_>] = &[queue_info];

        let device_info = DeviceCreateInfo::default()
            .queue_create_infos(queue_create_infos)
            .enabled_extension_names(&extension_pointers)
            .push_next(required_features);

        let device = match unsafe {
            phd.instance()
                .handle()
                .create_device(phd.handle(), &device_info, None)
        } {
            Ok(dev) => dev,
            Err(err) if has_global_priority => {
                tracing::warn!(
                    "Vulkan Device creation with global priority failed: {:?}; falling back to default priority",
                    err
                );
                let fallback_queue_info = DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_index as u32)
                    .queue_priorities(&[1.0]);
                let fallback_queue_create_infos: &[DeviceQueueCreateInfo<'_>] = &[fallback_queue_info];
                let fallback_device_info = DeviceCreateInfo::default()
                    .queue_create_infos(fallback_queue_create_infos)
                    .enabled_extension_names(&extension_pointers)
                    .push_next(required_features);
                unsafe {
                    phd.instance()
                        .handle()
                        .create_device(phd.handle(), &fallback_device_info, None)
                        .map_err(DeviceError::Vk)?
                }
            }
            Err(err) => return Err(DeviceError::Vk(err)),
        };

        let queue = unsafe { device.get_device_queue(queue_index as u32, 0) };

        let khr_external_semaphore_fd = if extensions
            .iter()
            .any(|ext| ext == &khr::external_semaphore_fd::NAME)
        {
            Some(khr::external_semaphore_fd::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let khr_external_memory_fd = if extensions.iter().any(|ext| ext == &khr::external_memory_fd::NAME) {
            Some(khr::external_memory_fd::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let ext_image_drm_format_modifier = if extensions
            .iter()
            .any(|ext| ext == &ext::image_drm_format_modifier::NAME)
        {
            Some(ext::image_drm_format_modifier::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let ext_host_image_copy = if extensions.iter().any(|ext| ext == &ext::host_image_copy::NAME) {
            Some(ext::host_image_copy::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let khr_push_descriptor = if extensions.iter().any(|ext| ext == &khr::push_descriptor::NAME) {
            Some(khr::push_descriptor::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };

        let mut props = vk::PhysicalDeviceProperties2::default();
        unsafe { phd.get_properties(&mut props) };
        let pipeline_cache_uuid = props.properties.pipeline_cache_uuid;
        let buffer_image_granularity = props.properties.limits.buffer_image_granularity;
        let non_coherent_atom_size = props.properties.limits.non_coherent_atom_size;

        Ok(Device(Arc::new(InnerDevice {
            vk: device,
            phd: phd.clone(),
            khr_external_semaphore_fd,
            khr_external_memory_fd,
            ext_image_drm_format_modifier,
            ext_host_image_copy,
            khr_push_descriptor,

            mem_properties,
            memory_preferences,
            formats: phd.drm_formats(),
            #[cfg(feature = "backend_drm")]
            node: phd
                .render_node()
                .ok()
                .flatten()
                .or_else(|| phd.primary_node().ok().flatten()),

            queue,
            queue_idx: queue_index as u32,

            context: ContextId::new(),
            allocator: std::sync::Mutex::new(super::allocator::VulkanSuballocator::default()),
            pipeline_cache_uuid,
            buffer_image_granularity,
            non_coherent_atom_size,
            has_memory_priority,
            has_memory_budget,
        })))
    }

    pub fn vk(&self) -> &VkDevice {
        &self.0.vk
    }

    pub fn vk_khr_external_semaphore_fd(&self) -> Option<&khr::external_semaphore_fd::Device> {
        self.0.khr_external_semaphore_fd.as_ref()
    }

    pub fn vk_khr_external_memory_fd(&self) -> Option<&khr::external_memory_fd::Device> {
        self.0.khr_external_memory_fd.as_ref()
    }

    pub fn vk_ext_image_drm_format_modifier(&self) -> Option<&ext::image_drm_format_modifier::Device> {
        self.0.ext_image_drm_format_modifier.as_ref()
    }

    pub fn vk_ext_host_image_copy(&self) -> Option<&ext::host_image_copy::Device> {
        self.0.ext_host_image_copy.as_ref()
    }

    pub fn vk_khr_push_descriptor(&self) -> Option<&khr::push_descriptor::Device> {
        self.0.khr_push_descriptor.as_ref()
    }

    pub fn memory_properties(&self) -> &PhysicalDeviceMemoryProperties {
        &self.0.mem_properties
    }

    pub fn queue(&self) -> &Queue {
        &self.0.queue
    }

    pub fn queue_family_idx(&self) -> u32 {
        self.0.queue_idx
    }

    pub fn formats(&self) -> impl Iterator<Item = &FormatEntry> {
        self.0.formats.iter()
    }

    pub fn node(&self) -> Option<DrmNode> {
        self.0.node.clone()
    }

    pub fn downgrade(&self) -> WeakDevice {
        WeakDevice(Arc::downgrade(&self.0))
    }

    pub fn context(&self) -> ContextId<VulkanImage> {
        self.0.context.clone()
    }

    pub(crate) fn suballocate_memory(
        &self,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        memory_type_index: u32,
        tiling: ImageTiling,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize, Option<*mut u8>), vk::Result> {
        let flags = self.0.mem_properties.memory_types[memory_type_index as usize].property_flags;
        let host_visible = flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let host_coherent = flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);

        // Vulkan spec §6.6.1: Suballocating non-coherent host-visible memory requires
        // offset and size to be aligned to nonCoherentAtomSize to avoid cache line tearing.
        let (size, alignment) = if host_visible && !host_coherent {
            let atom = self.0.non_coherent_atom_size.max(1);
            let aligned_size = (size + atom - 1) & !(atom - 1);
            let aligned_req = alignment.max(atom);
            (aligned_size, aligned_req)
        } else {
            (size, alignment)
        };

        let mut alloc = self.0.allocator.lock().unwrap();
        let res = unsafe {
            alloc.allocate(
                &self.0.vk,
                size,
                alignment,
                memory_type_index,
                host_visible,
                tiling,
                self.0.buffer_image_granularity,
                self.0.has_memory_priority,
            )
        };

        match res {
            Ok(val) => Ok(val),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                // Out of memory: purge completely empty cached blocks to reclaim VRAM, then retry once!
                unsafe { alloc.trim(&self.0.vk) };
                unsafe {
                    alloc.allocate(
                        &self.0.vk,
                        size,
                        alignment,
                        memory_type_index,
                        host_visible,
                        tiling,
                        self.0.buffer_image_granularity,
                        self.0.has_memory_priority,
                    )
                }
            }
            Err(err) => Err(err),
        }
    }

    pub fn has_memory_priority(&self) -> bool {
        self.0.has_memory_priority
    }

    pub fn buffer_image_granularity(&self) -> vk::DeviceSize {
        self.0.buffer_image_granularity
    }

    pub fn non_coherent_atom_size(&self) -> vk::DeviceSize {
        self.0.non_coherent_atom_size
    }

    /// Purge all completely empty cached memory blocks across all pools.
    pub fn trim_memory(&self) {
        let mut alloc = self.0.allocator.lock().unwrap();
        unsafe { alloc.trim(&self.0.vk) };
    }

    /// Check if the physical device supports `VK_EXT_memory_budget`.
    pub fn supports_memory_budget(&self) -> bool {
        self.0.has_memory_budget
    }

    /// Query the current memory budget and usage for all heaps using `VK_EXT_memory_budget`.
    pub fn memory_budget(&self) -> Option<super::memory::MemoryBudgetInfo> {
        if !self.0.has_memory_budget {
            return None;
        }

        let mut budget_prop = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut mem_prop2 = vk::PhysicalDeviceMemoryProperties2::default();
        mem_prop2.p_next = &mut budget_prop as *mut _ as *mut _;

        unsafe {
            self.0
                .phd
                .instance()
                .handle()
                .get_physical_device_memory_properties2(self.0.phd.handle(), &mut mem_prop2);
        }

        Some(super::memory::MemoryBudgetInfo {
            heap_budget: budget_prop.heap_budget,
            heap_usage: budget_prop.heap_usage,
        })
    }

    pub(crate) fn free_suballocation(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        let mut alloc = self.0.allocator.lock().unwrap();
        unsafe { alloc.free(&self.0.vk, memory, offset, size) }
    }

    pub fn pipeline_cache_uuid(&self) -> [u8; 16] {
        self.0.pipeline_cache_uuid
    }

    /// Access the precomputed memory preferences.
    pub fn memory_preferences(&self) -> &super::memory::MemoryPreferences {
        &self.0.memory_preferences
    }

    /// Find the single best memory type matching `type_bits` for the specified `preference`.
    pub fn find_memory_type(
        &self,
        type_bits: u32,
        preference: super::memory::MemoryUsagePreference,
    ) -> Option<super::memory::MemoryTypeRank> {
        self.0.memory_preferences.find_best(type_bits, preference)
    }

    /// Find the index of the best memory type matching `type_bits` for the specified `preference`.
    pub fn find_memory_type_index(
        &self,
        type_bits: u32,
        preference: super::memory::MemoryUsagePreference,
    ) -> Option<u32> {
        self.find_memory_type(type_bits, preference).map(|r| r.type_index)
    }

    /// Return an iterator over memory type candidates that match `type_bits`, ordered from most to least preferred.
    pub fn ranked_memory_types(
        &self,
        type_bits: u32,
        preference: super::memory::MemoryUsagePreference,
    ) -> impl Iterator<Item = super::memory::MemoryTypeRank> + '_ {
        self.0.memory_preferences.filter_candidates(type_bits, preference)
    }
}

impl WeakDevice {
    pub fn upgrade(&self) -> Option<Device> {
        self.0.upgrade().map(Device)
    }
}

struct InnerDevice {
    vk: VkDevice,
    khr_external_semaphore_fd: Option<khr::external_semaphore_fd::Device>,
    khr_external_memory_fd: Option<khr::external_memory_fd::Device>,
    ext_image_drm_format_modifier: Option<ext::image_drm_format_modifier::Device>,
    ext_host_image_copy: Option<ext::host_image_copy::Device>,
    khr_push_descriptor: Option<khr::push_descriptor::Device>,

    mem_properties: PhysicalDeviceMemoryProperties,
    memory_preferences: super::memory::MemoryPreferences,
    formats: super::FormatList,
    #[cfg(feature = "backend_drm")]
    node: Option<DrmNode>,

    queue: Queue,
    queue_idx: u32,

    context: ContextId<VulkanImage>,
    allocator: std::sync::Mutex<super::allocator::VulkanSuballocator>,
    pipeline_cache_uuid: [u8; 16],
    /// `VkPhysicalDeviceLimits::bufferImageGranularity`.
    ///
    /// On some hardware, a Linear image and an Optimal image placed
    /// in the same `VkDeviceMemory` block must not share the same
    /// "granularity page" (typically 64 KiB).  The suballocator uses
    /// this value to align allocations and avoid the undefined behaviour
    /// described in Vulkan spec section 12.7.1.
    pub(super) buffer_image_granularity: vk::DeviceSize,
    pub(super) non_coherent_atom_size: vk::DeviceSize,
    pub(super) has_memory_priority: bool,
    pub(super) has_memory_budget: bool,
    pub(super) phd: PhysicalDevice,
}

impl fmt::Debug for InnerDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VkDevice").finish_non_exhaustive()
    }
}

impl Drop for InnerDevice {
    fn drop(&mut self) {
        unsafe {
            self.allocator.lock().unwrap().destroy(&self.vk);
            self.vk.destroy_device(None);
        }
    }
}
