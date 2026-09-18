use ash::vk::{self, ImageTiling};

/// Tiling class used to enforce `bufferImageGranularity` isolation.
///
/// Vulkan spec §12.7.1: within a single `VkDeviceMemory` block, Linear and
/// Non-Linear (Optimal / DRM-modifier) resources must not overlap the same
/// "granularity page" boundary defined by
/// `VkPhysicalDeviceLimits::bufferImageGranularity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TilingClass {
    /// `LINEAR`
    Linear,
    /// `OPTIMAL` or `DRM_FORMAT_MODIFIER_EXT`
    Optimal,
}

impl TilingClass {
    fn from_tiling(t: ImageTiling) -> Self {
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
    /// Tiling class of the allocation that owns this range.
    /// `None` for free chunks.
    tiling_class: Option<TilingClass>,
}

#[derive(Debug)]
struct MemoryBlock {
    memory: vk::DeviceMemory,
    memory_type_index: u32,
    total_size: vk::DeviceSize,
    /// Free ranges available for new allocations.
    free_chunks: Vec<MemoryChunk>,
    /// Occupied ranges with tiling class, used for granularity conflict checks.
    used_chunks: Vec<MemoryChunk>,
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

/// Round `offset` up to the next multiple of `granularity`.
#[inline]
fn align_up(offset: vk::DeviceSize, granularity: vk::DeviceSize) -> vk::DeviceSize {
    if granularity <= 1 {
        return offset;
    }
    (offset + granularity - 1) & !(granularity - 1)
}

/// Check whether placing a resource of `new_class` at `[start, start+size)`
/// would conflict with any occupied chunk of a different tiling class that
/// shares a `bufferImageGranularity` page boundary.
///
/// Vulkan spec §12.7.1: two resources of different tiling classes must not
/// share a page if they are placed in the same `VkDeviceMemory`.
fn granularity_conflict(
    used_chunks: &[MemoryChunk],
    start: vk::DeviceSize,
    size: vk::DeviceSize,
    new_class: TilingClass,
    granularity: vk::DeviceSize,
) -> bool {
    if granularity <= 1 || size == 0 {
        return false;
    }
    let new_end = start + size - 1;
    let new_page_start = start / granularity;
    let new_page_end = new_end / granularity;

    for chunk in used_chunks {
        if let Some(existing_class) = chunk.tiling_class {
            if existing_class == new_class {
                continue;
            }
            if chunk.size == 0 {
                continue;
            }
            let ex_end = chunk.offset + chunk.size - 1;
            let ex_page_start = chunk.offset / granularity;
            let ex_page_end = ex_end / granularity;
            // Conflict when page ranges intersect
            if new_page_start <= ex_page_end && ex_page_start <= new_page_end {
                return true;
            }
        }
    }
    false
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
        buffer_image_granularity: vk::DeviceSize,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize, Option<*mut u8>), vk::Result> {
        let align = alignment.max(1);
        let new_class = TilingClass::from_tiling(tiling);

        // Try to fit inside an existing block of the same memory type.
        for block in &mut self.blocks {
            if block.memory_type_index != memory_type_index {
                continue;
            }

            for i in 0..block.free_chunks.len() {
                let chunk = block.free_chunks[i];
                let mut aligned_offset = (chunk.offset + align - 1) & !(align - 1);

                // If bufferImageGranularity > 1, we may need to push aligned_offset
                // to the next granularity page to avoid a tiling-class conflict.
                if buffer_image_granularity > 1
                    && granularity_conflict(
                        &block.used_chunks,
                        aligned_offset,
                        size,
                        new_class,
                        buffer_image_granularity,
                    )
                {
                    // Advance to the start of the next granularity page.
                    aligned_offset = align_up(aligned_offset + 1, buffer_image_granularity);
                }

                let padding = aligned_offset - chunk.offset;
                if padding + size > chunk.size {
                    continue;
                }

                // Final conflict check after the possible page-advance.
                if buffer_image_granularity > 1
                    && granularity_conflict(
                        &block.used_chunks,
                        aligned_offset,
                        size,
                        new_class,
                        buffer_image_granularity,
                    )
                {
                    continue;
                }

                // Commit: split the free chunk.
                block.free_chunks.swap_remove(i);

                if padding > 0 {
                    block.free_chunks.push(MemoryChunk {
                        offset: chunk.offset,
                        size: padding,
                        tiling_class: None,
                    });
                }

                let remaining = chunk.size - (padding + size);
                if remaining > 0 {
                    block.free_chunks.push(MemoryChunk {
                        offset: aligned_offset + size,
                        size: remaining,
                        tiling_class: None,
                    });
                }

                block.used_chunks.push(MemoryChunk {
                    offset: aligned_offset,
                    size,
                    tiling_class: Some(new_class),
                });

                let mapped_ptr = block
                    .mapped_base
                    .map(|base| unsafe { base.add(aligned_offset as usize) });
                return Ok((block.memory, aligned_offset, mapped_ptr));
            }
        }

        // No suitable existing block found – allocate a new one.
        // Try a tiered size first; fall back to exact size on OOM.
        let block_size = select_tier_block_size(size);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(block_size)
            .memory_type_index(memory_type_index);

        let (memory, total_size) = match unsafe { vk_device.allocate_memory(&alloc_info, None) } {
            Ok(mem) => (mem, block_size),
            Err(err) if block_size > size => {
                let fallback_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(memory_type_index);
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
                tiling_class: None,
            });
        }

        let used_chunks = vec![MemoryChunk {
            offset: 0,
            size,
            tiling_class: Some(new_class),
        }];

        self.blocks.push(MemoryBlock {
            memory,
            memory_type_index,
            total_size,
            free_chunks,
            used_chunks,
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
                // Remove the matching used chunk.
                block
                    .used_chunks
                    .retain(|c| !(c.offset == offset && c.size == size));

                // Return region to the free list, then sort and coalesce.
                block.free_chunks.push(MemoryChunk {
                    offset,
                    size,
                    tiling_class: None,
                });
                block.free_chunks.sort_unstable_by_key(|c| c.offset);

                let mut coalesced: Vec<MemoryChunk> = Vec::with_capacity(block.free_chunks.len());
                for chunk in block.free_chunks.drain(..) {
                    if let Some(last) = coalesced.last_mut() {
                        if last.offset + last.size == chunk.offset {
                            last.size += chunk.size;
                            continue;
                        }
                    }
                    coalesced.push(chunk);
                }
                block.free_chunks = coalesced;

                // Check if the entire block is now free.
                if block.free_chunks.len() == 1 && block.free_chunks[0].size == block.total_size {
                    empty_block_idx = Some(idx);
                }
                break;
            }
        }

        // Retain up to 2 empty tiered blocks as a hot cache; release the rest.
        if let Some(idx) = empty_block_idx {
            let block = &self.blocks[idx];
            let mem_type = block.memory_type_index;
            let empty_count_same_type = self
                .blocks
                .iter()
                .filter(|b| {
                    b.memory_type_index == mem_type
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
