use ash::vk::{self, ImageTiling};

/// Tiling class used to enforce `bufferImageGranularity` isolation.
///
/// Vulkan spec §12.7.1: within a single `VkDeviceMemory` block, Linear and
/// Non-Linear (Optimal / DRM-modifier) resources must not overlap the same
/// "granularity page" boundary defined by
/// `VkPhysicalDeviceLimits::bufferImageGranularity`.
///
/// By dedicating distinct `VkDeviceMemory` blocks to distinct tiling classes,
/// granularity conflicts are eliminated by construction without expensive runtime collision checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilingClass {
    /// `LINEAR`
    Linear,
    /// `OPTIMAL` or `DRM_FORMAT_MODIFIER_EXT`
    Optimal,
}

impl TilingClass {
    pub fn from_tiling(t: ImageTiling) -> Self {
        match t {
            ImageTiling::LINEAR => TilingClass::Linear,
            _ => TilingClass::Optimal,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MemoryChunk {
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

#[derive(Debug)]
struct MemoryBlock {
    memory: vk::DeviceMemory,
    memory_type_index: u32,
    tiling_class: TilingClass,
    total_size: vk::DeviceSize,
    /// Free ranges available for new allocations.
    free_chunks: Vec<MemoryChunk>,
    mapped_base: Option<*mut u8>,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

#[derive(Debug, Default)]
pub struct VulkanSuballocator {
    blocks: Vec<MemoryBlock>,
}

unsafe impl Send for VulkanSuballocator {}
unsafe impl Sync for VulkanSuballocator {}

const TIER_1_SIZE: vk::DeviceSize = 8 * 1024 * 1024; // 8 MB
const TIER_2_SIZE: vk::DeviceSize = 32 * 1024 * 1024; // 32 MB
const TIER_3_SIZE: vk::DeviceSize = 64 * 1024 * 1024; // 64 MB

#[inline]
fn select_tier_block_size(size: vk::DeviceSize) -> vk::DeviceSize {
    if size <= TIER_1_SIZE {
        TIER_1_SIZE
    } else if size <= TIER_2_SIZE {
        TIER_2_SIZE
    } else if size <= TIER_3_SIZE {
        TIER_3_SIZE
    } else {
        (size + TIER_2_SIZE - 1) & !(TIER_2_SIZE - 1)
    }
}

#[inline]
fn is_tiered_size(size: vk::DeviceSize) -> bool {
    size == TIER_1_SIZE || size == TIER_2_SIZE || size == TIER_3_SIZE
}

impl VulkanSuballocator {
    pub unsafe fn allocate(
        &mut self,
        vk_device: &ash::Device,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        memory_type_index: u32,
        host_visible: bool,
        tiling: ImageTiling,
        _buffer_image_granularity: vk::DeviceSize,
        has_memory_priority: bool,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize, Option<*mut u8>), vk::Result> {
        let align = alignment.max(1);
        let new_class = TilingClass::from_tiling(tiling);

        // Try to fit inside an existing block of the same memory type and tiling class.
        // Partitioning blocks by (memory_type_index, TilingClass) guarantees no granularity conflict (Vulkan §12.7.1).
        for block in &mut self.blocks {
            if block.memory_type_index != memory_type_index || block.tiling_class != new_class {
                continue;
            }

            for i in 0..block.free_chunks.len() {
                let chunk = block.free_chunks[i];
                let aligned_offset = (chunk.offset + align - 1) & !(align - 1);
                let padding = aligned_offset - chunk.offset;
                if padding + size > chunk.size {
                    continue;
                }

                // Commit: split the free chunk.
                block.free_chunks.swap_remove(i);

                if padding > 0 {
                    block.free_chunks.push(MemoryChunk {
                        offset: chunk.offset,
                        size: padding,
                    });
                }

                let remaining = chunk.size - (padding + size);
                if remaining > 0 {
                    block.free_chunks.push(MemoryChunk {
                        offset: aligned_offset + size,
                        size: remaining,
                    });
                }

                let mapped_ptr = block
                    .mapped_base
                    .map(|base| unsafe { base.add(aligned_offset as usize) });
                return Ok((block.memory, aligned_offset, mapped_ptr));
            }
        }

        // No suitable existing block found – allocate a new one.
        // Try a tiered size first; fall back to exact size on OOM.
        let block_size = select_tier_block_size(size);
        let mut priority_info = vk::MemoryPriorityAllocateInfoEXT::default().priority(1.0);
        let mut alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(block_size)
            .memory_type_index(memory_type_index);
        if has_memory_priority {
            alloc_info = alloc_info.push_next(&mut priority_info);
        }

        let (memory, total_size) = match unsafe { vk_device.allocate_memory(&alloc_info, None) } {
            Ok(mem) => (mem, block_size),
            Err(err) if block_size > size => {
                let mut fallback_priority = vk::MemoryPriorityAllocateInfoEXT::default().priority(1.0);
                let mut fallback_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(memory_type_index);
                if has_memory_priority {
                    fallback_info = fallback_info.push_next(&mut fallback_priority);
                }
                let mem = unsafe { vk_device.allocate_memory(&fallback_info, None)? };
                (mem, size)
            }
            Err(err) => return Err(err),
        };

        let mapped_base = if host_visible {
            match unsafe { vk_device.map_memory(memory, 0, total_size, vk::MemoryMapFlags::empty()) } {
                Ok(ptr) => Some(ptr as *mut u8),
                Err(err) => {
                    tracing::warn!(
                        "Failed to persistently map host-visible suballocator block: {:?}",
                        err
                    );
                    None
                }
            }
        } else {
            None
        };

        let mut free_chunks = Vec::new();
        let remaining = total_size - size;
        if remaining > 0 {
            free_chunks.push(MemoryChunk {
                offset: size,
                size: remaining,
            });
        }

        self.blocks.push(MemoryBlock {
            memory,
            memory_type_index,
            tiling_class: new_class,
            total_size,
            free_chunks,
            mapped_base,
        });

        Ok((memory, 0, mapped_base))
    }

    pub unsafe fn free(
        &mut self,
        vk_device: &ash::Device,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        let mut empty_block_idx = None;

        for (idx, block) in self.blocks.iter_mut().enumerate() {
            if block.memory == memory {
                // Return region to the free list, then sort and coalesce in-place (zero heap allocations).
                block.free_chunks.push(MemoryChunk { offset, size });
                block.free_chunks.sort_unstable_by_key(|c| c.offset);

                if block.free_chunks.len() > 1 {
                    let mut write = 0;
                    for read in 1..block.free_chunks.len() {
                        let next = block.free_chunks[read];
                        if block.free_chunks[write].offset + block.free_chunks[write].size == next.offset {
                            block.free_chunks[write].size += next.size;
                        } else {
                            write += 1;
                            block.free_chunks[write] = next;
                        }
                    }
                    block.free_chunks.truncate(write + 1);
                }

                // Check if the entire block is now free.
                if block.free_chunks.len() == 1 && block.free_chunks[0].size == block.total_size {
                    empty_block_idx = Some(idx);
                }
                break;
            }
        }

        // Retain up to 2 empty tiered blocks as a hot cache per (memory_type_index, tiling_class); release the rest.
        if let Some(idx) = empty_block_idx {
            let block = &self.blocks[idx];
            let mem_type = block.memory_type_index;
            let tiling_class = block.tiling_class;
            let empty_count_same_type = self
                .blocks
                .iter()
                .filter(|b| {
                    b.memory_type_index == mem_type
                        && b.tiling_class == tiling_class
                        && b.free_chunks.len() == 1
                        && b.free_chunks[0].size == b.total_size
                })
                .count();

            let keep_as_cache = is_tiered_size(block.total_size) && empty_count_same_type <= 2;
            if !keep_as_cache {
                let removed = self.blocks.swap_remove(idx);
                unsafe {
                    if removed.mapped_base.is_some() {
                        vk_device.unmap_memory(removed.memory);
                    }
                    vk_device.free_memory(removed.memory, None);
                }
            }
        }
    }

    /// Purge all completely empty cached memory blocks and return their memory to the Vulkan driver.
    ///
    /// Useful when memory pressure is high or when reclaiming idle GPU resources.
    pub unsafe fn trim(&mut self, vk_device: &ash::Device) {
        let mut i = 0;
        while i < self.blocks.len() {
            let is_empty = self.blocks[i].free_chunks.len() == 1
                && self.blocks[i].free_chunks[0].size == self.blocks[i].total_size;

            if is_empty {
                let removed = self.blocks.swap_remove(i);
                if removed.mapped_base.is_some() {
                    vk_device.unmap_memory(removed.memory);
                }
                vk_device.free_memory(removed.memory, None);
            } else {
                i += 1;
            }
        }
    }

    pub unsafe fn destroy(&mut self, vk_device: &ash::Device) {
        for block in self.blocks.drain(..) {
            unsafe {
                if block.mapped_base.is_some() {
                    vk_device.unmap_memory(block.memory);
                }
                vk_device.free_memory(block.memory, None);
            }
        }
    }
}
