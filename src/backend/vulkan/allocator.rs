use ash::vk;

#[derive(Debug, Clone, Copy)]
struct MemoryChunk {
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

#[derive(Debug)]
struct MemoryBlock {
    memory: vk::DeviceMemory,
    memory_type_index: u32,
    total_size: vk::DeviceSize,
    free_chunks: Vec<MemoryChunk>,
}

#[derive(Debug, Default)]
pub struct VulkanSuballocator {
    blocks: Vec<MemoryBlock>,
}

const DEFAULT_BLOCK_SIZE: vk::DeviceSize = 8 * 1024 * 1024; // 8 MB

impl VulkanSuballocator {
    pub unsafe fn allocate(
        &mut self,
        vk_device: &ash::Device,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        memory_type_index: u32,
    ) -> Result<(vk::DeviceMemory, vk::DeviceSize), vk::Result> {
        let align = alignment.max(1);

        // Try to allocate from an existing block matching memory_type_index
        for block in &mut self.blocks {
            if block.memory_type_index != memory_type_index {
                continue;
            }

            for i in 0..block.free_chunks.len() {
                let chunk = block.free_chunks[i];
                let aligned_offset = (chunk.offset + align - 1) & !(align - 1);
                let padding = aligned_offset - chunk.offset;
                if padding + size <= chunk.size {
                    // Split the chunk
                    block.free_chunks.swap_remove(i);

                    // Add leading free chunk if there was alignment padding
                    if padding > 0 {
                        block.free_chunks.push(MemoryChunk {
                            offset: chunk.offset,
                            size: padding,
                        });
                    }

                    // Add trailing free chunk if remaining size > 0
                    let remaining = chunk.size - (padding + size);
                    if remaining > 0 {
                        block.free_chunks.push(MemoryChunk {
                            offset: aligned_offset + size,
                            size: remaining,
                        });
                    }

                    return Ok((block.memory, aligned_offset));
                }
            }
        }

        // Need to allocate a new block. Try DEFAULT_BLOCK_SIZE first, fall back to exact size on OOM
        let block_size = DEFAULT_BLOCK_SIZE.max(size);
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
            total_size,
            free_chunks,
        });

        Ok((memory, 0))
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
                block.free_chunks.push(MemoryChunk { offset, size });
                // Sort and coalesce free chunks
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

                // Check if the entire block is now free
                if block.free_chunks.len() == 1 && block.free_chunks[0].size == block.total_size {
                    empty_block_idx = Some(idx);
                }
                break;
            }
        }

        // If an entire block is free, free it back to Vulkan if it's fallback/oversized or extra
        if let Some(idx) = empty_block_idx {
            let block = &self.blocks[idx];
            let mem_type = block.memory_type_index;
            let count_same_type = self
                .blocks
                .iter()
                .filter(|b| b.memory_type_index == mem_type)
                .count();
            if block.total_size != DEFAULT_BLOCK_SIZE || count_same_type > 1 {
                let removed = self.blocks.swap_remove(idx);
                unsafe {
                    vk_device.free_memory(removed.memory, None);
                }
            }
        }
    }

    pub unsafe fn destroy(&mut self, vk_device: &ash::Device) {
        for block in self.blocks.drain(..) {
            unsafe {
                vk_device.free_memory(block.memory, None);
            }
        }
    }
}
