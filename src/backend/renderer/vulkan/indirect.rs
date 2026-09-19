use ash::vk::{
    self, Buffer, BufferCreateInfo, BufferUsageFlags, DescriptorBindingFlags, DescriptorPool,
    DescriptorPoolCreateFlags, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutBindingFlagsCreateInfo, DescriptorSetLayoutCreateFlags, DescriptorType, DeviceMemory,
    DrawIndirectCommand, MemoryAllocateInfo, MemoryMapFlags, MemoryPropertyFlags,
    PhysicalDeviceMemoryProperties, ShaderStageFlags, SharingMode,
};

use crate::backend::{
    renderer::vulkan::Error,
    vulkan::{
        Device,
        device::{DeviceError, WeakDevice},
        memory::MemoryUsagePreference,
    },
};

/// Buffer for storing `VkDrawIndirectCommand` structures for multi-draw indirect calls.
#[derive(Debug)]
pub struct DrawIndirectBuffer {
    device: WeakDevice,
    buffer: Buffer,
    memory: DeviceMemory,
    capacity: usize,
    len: usize,
    mapped: *mut DrawIndirectCommand,
}

// SAFETY: DrawIndirectBuffer owns the mapped pointer exclusively, and writes are synchronized.
unsafe impl Send for DrawIndirectBuffer {}
unsafe impl Sync for DrawIndirectBuffer {}

impl DrawIndirectBuffer {
    /// Default initial capacity (number of DrawIndirectCommand structs).
    pub const DEFAULT_CAPACITY: usize = 256;

    /// Create a new host-visible indirect draw buffer.
    pub fn new(device: &Device, initial_capacity: usize) -> Result<Self, Error> {
        let cap = initial_capacity.max(16);
        let size = (cap * std::mem::size_of::<DrawIndirectCommand>()) as vk::DeviceSize;

        let create_info = BufferCreateInfo::default()
            .size(size)
            .usage(BufferUsageFlags::INDIRECT_BUFFER | BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .vk()
                .create_buffer(&create_info, None)
                .map_err(Error::IndirectBufferError)?
        };

        let mem_reqs = unsafe { device.vk().get_buffer_memory_requirements(buffer) };

        let mem_idx = device
            .find_memory_type_index(mem_reqs.memory_type_bits, MemoryUsagePreference::HostVisible)
            .ok_or(Error::DeviceError(DeviceError::NoUsableQueue))?;

        let mut priority_info = vk::MemoryPriorityAllocateInfoEXT::default().priority(1.0);
        let mut alloc_info = MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_idx);
        if device.has_memory_priority() {
            alloc_info = alloc_info.push_next(&mut priority_info);
        }

        let memory = unsafe {
            device
                .vk()
                .allocate_memory(&alloc_info, None)
                .map_err(Error::IndirectBufferError)?
        };

        unsafe {
            device
                .vk()
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(Error::IndirectBufferError)?;
        }

        let mapped = unsafe {
            device
                .vk()
                .map_memory(memory, 0, mem_reqs.size, MemoryMapFlags::empty())
                .map_err(Error::IndirectBufferError)? as *mut DrawIndirectCommand
        };

        Ok(DrawIndirectBuffer {
            device: device.downgrade(),
            buffer,
            memory,
            capacity: cap,
            len: 0,
            mapped,
        })
    }

    /// Ensure capacity for at least `required` commands, reallocating if necessary.
    pub fn ensure_capacity(&mut self, required: usize) -> Result<(), Error> {
        if required <= self.capacity {
            return Ok(());
        }

        let Some(device) = self.device.upgrade() else {
            return Err(Error::DeadDevice);
        };

        let new_cap = (self.capacity * 2).max(required);
        let new_buf = Self::new(&device, new_cap)?;

        let old = std::mem::replace(self, new_buf);
        if old.len > 0 && !old.mapped.is_null() && !self.mapped.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(old.mapped, self.mapped, old.len);
            }
            self.len = old.len;
        }

        Ok(())
    }

    /// Write an array of `DrawIndirectCommand` to the mapped buffer at current cursor, returning byte offset.
    pub fn write_commands(&mut self, commands: &[DrawIndirectCommand]) -> Result<vk::DeviceSize, Error> {
        let required = self.len + commands.len();
        self.ensure_capacity(required)?;
        let byte_offset = (self.len * std::mem::size_of::<DrawIndirectCommand>()) as vk::DeviceSize;
        if !commands.is_empty() && !self.mapped.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(commands.as_ptr(), self.mapped.add(self.len), commands.len());
            }
        }
        self.len += commands.len();
        Ok(byte_offset)
    }

    /// Clear the command count without freeing memory.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Reset write offset.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Push a single `DrawIndirectCommand`, returning the byte offset.
    pub fn push(&mut self, cmd: DrawIndirectCommand) -> Result<vk::DeviceSize, Error> {
        self.ensure_capacity(self.len + 1)?;
        let byte_offset = (self.len * std::mem::size_of::<DrawIndirectCommand>()) as vk::DeviceSize;
        unsafe {
            std::ptr::write(self.mapped.add(self.len), cmd);
        }
        self.len += 1;
        Ok(byte_offset)
    }

    /// Get the underlying `VkBuffer`.
    pub fn buffer(&self) -> Buffer {
        self.buffer
    }

    /// Number of commands currently written.
    pub fn len(&self) -> u32 {
        self.len as u32
    }

    /// True if no commands are currently written.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stride in bytes between consecutive commands (16 bytes).
    pub fn stride(&self) -> u32 {
        std::mem::size_of::<DrawIndirectCommand>() as u32
    }
}

impl Drop for DrawIndirectBuffer {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe {
                if !self.mapped.is_null() {
                    let _ = device.vk().unmap_memory(self.memory);
                    self.mapped = std::ptr::null_mut();
                }
                device.vk().destroy_buffer(self.buffer, None);
                device.vk().free_memory(self.memory, None);
            }
        }
    }
}

/// Manages a bindless descriptor indexing pool and descriptor set for texture arrays.
#[derive(Debug)]
pub struct BindlessDescriptorPool {
    device: WeakDevice,
    pool: DescriptorPool,
    layout: DescriptorSetLayout,
    descriptor_set: DescriptorSet,
    max_textures: u32,
}

impl BindlessDescriptorPool {
    /// Default maximum textures in a bindless array (1024 slots).
    pub const DEFAULT_MAX_TEXTURES: u32 = 1024;

    /// Create a new descriptor set layout and pool with `VK_EXT_descriptor_indexing` binding flags.
    pub fn new(device: &Device, max_textures: u32) -> Result<Self, Error> {
        let max_textures = max_textures.max(64);

        // 1. Descriptor binding: array of COMBINED_IMAGE_SAMPLER
        let binding = DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_textures)
            .stage_flags(ShaderStageFlags::FRAGMENT);

        let binding_flags =
            [DescriptorBindingFlags::PARTIALLY_BOUND | DescriptorBindingFlags::UPDATE_AFTER_BIND];

        let mut flags_info =
            DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let bindings = [binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut flags_info);

        let layout = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(Error::DescriptorPoolError)?
        };

        // 2. Descriptor pool with UPDATE_AFTER_BIND flag
        let pool_sizes = [DescriptorPoolSize::default()
            .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_textures)];

        let pool_info = DescriptorPoolCreateInfo::default()
            .flags(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let pool = unsafe {
            device
                .vk()
                .create_descriptor_pool(&pool_info, None)
                .map_err(Error::DescriptorPoolError)?
        };

        // 3. Allocate the descriptor set
        let layouts = [layout];
        let alloc_info = DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let set = unsafe {
            device
                .vk()
                .allocate_descriptor_sets(&alloc_info)
                .map_err(Error::DescriptorPoolError)?[0]
        };

        Ok(BindlessDescriptorPool {
            device: device.downgrade(),
            pool,
            layout,
            descriptor_set: set,
            max_textures,
        })
    }

    /// Update a texture slot in the bindless descriptor set.
    pub fn update_texture(&self, slot: u32, view: vk::ImageView, sampler: vk::Sampler) -> Result<(), Error> {
        if slot >= self.max_textures {
            return Err(Error::DescriptorPoolError(vk::Result::ERROR_OUT_OF_POOL_MEMORY));
        }

        let Some(device) = self.device.upgrade() else {
            return Err(Error::DeadDevice);
        };

        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)];

        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(slot)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .image_info(&image_info)];

        unsafe {
            device.vk().update_descriptor_sets(&write, &[]);
        }

        Ok(())
    }

    /// Underlying descriptor set.
    pub fn descriptor_set(&self) -> DescriptorSet {
        self.descriptor_set
    }

    /// Underlying descriptor set layout.
    pub fn layout(&self) -> DescriptorSetLayout {
        self.layout
    }

    /// Maximum number of textures supported in the bindless array.
    pub fn max_textures(&self) -> u32 {
        self.max_textures
    }
}

impl Drop for BindlessDescriptorPool {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe {
                device.vk().destroy_descriptor_pool(self.pool, None);
                device.vk().destroy_descriptor_set_layout(self.layout, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::renderer::vulkan::VulkanRenderer;
    use crate::backend::vulkan::{Instance, PhysicalDevice, version::Version};

    #[test]
    fn test_draw_indirect_command_memory_layout() {
        assert_eq!(std::mem::size_of::<DrawIndirectCommand>(), 16);
        assert_eq!(std::mem::align_of::<DrawIndirectCommand>(), 4);

        let cmd = DrawIndirectCommand {
            vertex_count: 6,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 42,
        };
        assert_eq!(cmd.vertex_count, 6);
        assert_eq!(cmd.instance_count, 1);
        assert_eq!(cmd.first_vertex, 0);
        assert_eq!(cmd.first_instance, 42);
    }

    #[test]
    fn test_draw_indirect_buffer_live_device() {
        if let Ok(instance) = Instance::new(Version::VERSION_1_3, None) {
            if let Ok(phds) = PhysicalDevice::enumerate(&instance) {
                for phd in phds {
                    let Ok(renderer) = VulkanRenderer::new(&phd, None) else {
                        continue;
                    };

                    let mut buf = DrawIndirectBuffer::new(&renderer.device, 32)
                        .expect("Failed to create DrawIndirectBuffer");

                    assert_eq!(buf.len(), 0);
                    assert!(buf.is_empty());
                    assert_eq!(buf.stride(), 16);

                    let commands = [
                        DrawIndirectCommand {
                            vertex_count: 6,
                            instance_count: 1,
                            first_vertex: 0,
                            first_instance: 0,
                        },
                        DrawIndirectCommand {
                            vertex_count: 6,
                            instance_count: 1,
                            first_vertex: 0,
                            first_instance: 1,
                        },
                    ];

                    buf.write_commands(&commands).expect("Failed to write commands");
                    assert_eq!(buf.len(), 2);
                    assert!(!buf.is_empty());

                    // Test pushing additional commands with capacity growth
                    buf.push(DrawIndirectCommand {
                        vertex_count: 6,
                        instance_count: 1,
                        first_vertex: 0,
                        first_instance: 2,
                    })
                    .expect("Failed to push command");
                    assert_eq!(buf.len(), 3);

                    buf.clear();
                    assert_eq!(buf.len(), 0);
                    assert!(buf.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_bindless_descriptor_pool_live_device() {
        if let Ok(instance) = Instance::new(Version::VERSION_1_3, None) {
            if let Ok(phds) = PhysicalDevice::enumerate(&instance) {
                for phd in phds {
                    let Ok(renderer) = VulkanRenderer::new(&phd, None) else {
                        continue;
                    };

                    if !renderer.supports_descriptor_indexing() {
                        continue;
                    }

                    let bindless_pool = BindlessDescriptorPool::new(&renderer.device, 128)
                        .expect("Failed to create BindlessDescriptorPool");

                    assert_ne!(bindless_pool.descriptor_set(), vk::DescriptorSet::null());
                    assert_ne!(bindless_pool.layout(), vk::DescriptorSetLayout::null());
                    assert_eq!(bindless_pool.max_textures(), 128);
                }
            }
        }
    }

    #[test]
    fn test_renderer_extended_features_enabled() {
        if let Ok(instance) = Instance::new(Version::VERSION_1_3, None) {
            if let Ok(phds) = PhysicalDevice::enumerate(&instance) {
                for phd in phds {
                    let Ok(renderer) = VulkanRenderer::new(&phd, None) else {
                        continue;
                    };

                    println!("VulkanRenderer on device: {}", phd.name());
                    println!(
                        "  supports_descriptor_indexing: {}",
                        renderer.supports_descriptor_indexing()
                    );
                    println!(
                        "  supports_multi_draw_indirect: {}",
                        renderer.supports_multi_draw_indirect()
                    );
                    println!(
                        "  supports_shader_draw_parameters: {}",
                        renderer.supports_shader_draw_parameters()
                    );

                    // Ensure our enabled features are correctly recognized
                    assert!(renderer.supports_multi_draw_indirect());
                    assert!(renderer.supports_shader_draw_parameters());
                    assert!(renderer.supports_descriptor_indexing());
                }
            }
        }
    }
}
