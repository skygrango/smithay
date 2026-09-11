use ash::vk::{
    self, DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
    DescriptorSet as VkDescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, Result as VkError,
};

use std::sync::{Arc, Mutex, Weak};

use crate::backend::vulkan::{Device, device::WeakDevice};

#[derive(Debug)]
struct PoolInner {
    pool: DescriptorPool,
    free_sets: Mutex<Vec<VkDescriptorSet>>,
}

#[derive(Debug)]
pub struct DescriptorSet {
    device: WeakDevice,
    pool: Weak<PoolInner>,
    vk: VkDescriptorSet,
}

impl DescriptorSet {
    pub fn vk(&self) -> VkDescriptorSet {
        self.vk
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            if let Ok(mut free) = pool.free_sets.lock() {
                free.push(self.vk);
            }
        }
    }
}

#[derive(Debug)]
struct Pool {
    device: WeakDevice,
    inner: Arc<PoolInner>,
    allocated: usize,
    capacity: usize,
}

impl Drop for Pool {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe {
                device.vk().destroy_descriptor_pool(self.inner.pool, None);
            }
        }
    }
}

const START_DESCRIPTOR_COUNT: u32 = 256;

#[derive(Debug)]
pub struct DescriptorAllocator {
    device: WeakDevice,

    layout: DescriptorSetLayout,
    sizes: &'static [vk::DescriptorPoolSize],

    pools: Vec<Pool>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Device was destroyed")]
    LostDevice,
    #[error("Failed to allocate descriptor set")]
    AllocError(#[source] vk::Result),
    #[error("Failed to create descriptor pool")]
    DescriptorPool(#[source] vk::Result),
}

impl DescriptorAllocator {
    pub fn new(
        device: &Device,
        layout: DescriptorSetLayout,
        sizes: &'static [vk::DescriptorPoolSize],
    ) -> Self {
        DescriptorAllocator {
            device: device.downgrade(),
            layout,
            sizes,
            pools: Vec::new(),
        }
    }

    pub fn alloc_descriptor_set(&mut self) -> Result<DescriptorSet, Error> {
        let Some(device) = self.device.upgrade() else {
            return Err(Error::LostDevice);
        };

        // 1. Check if any pool has an already allocated and freed descriptor set
        for pool in self.pools.iter_mut() {
            if let Ok(mut free) = pool.inner.free_sets.lock() {
                if let Some(vk) = free.pop() {
                    return Ok(DescriptorSet {
                        device: self.device.clone(),
                        pool: Arc::downgrade(&pool.inner),
                        vk,
                    });
                }
            }
        }

        // 2. Check if any pool has remaining capacity to allocate a new set
        let layouts = &[self.layout];
        let mut alloc_info = DescriptorSetAllocateInfo::default().set_layouts(layouts);

        for pool in self.pools.iter_mut() {
            if pool.allocated < pool.capacity {
                alloc_info = alloc_info.descriptor_pool(pool.inner.pool);
                match unsafe { device.vk().allocate_descriptor_sets(&alloc_info) } {
                    Err(VkError::ERROR_FRAGMENTED_POOL) | Err(VkError::ERROR_OUT_OF_POOL_MEMORY) => continue,
                    Ok(set) => {
                        pool.allocated += 1;
                        return Ok(DescriptorSet {
                            device: self.device.clone(),
                            pool: Arc::downgrade(&pool.inner),
                            vk: set[0],
                        });
                    }
                    Err(err) => return Err(Error::AllocError(err)),
                }
            }
        }

        // 3. No (free) pool found, create a new one
        let size = self
            .pools
            .last()
            .map(|pool| (pool.capacity * 2) as u32)
            .unwrap_or(START_DESCRIPTOR_COUNT);
        let mut sizes = Vec::from_iter(self.sizes.iter().copied());
        for pool_size in &mut sizes {
            *pool_size = pool_size.descriptor_count(size);
        }
        let create_info = DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .flags(DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(size);
        let pool = unsafe {
            device
                .vk()
                .create_descriptor_pool(&create_info, None)
                .map_err(Error::DescriptorPool)?
        };
        let inner = Arc::new(PoolInner {
            pool,
            free_sets: Mutex::new(Vec::new()),
        });
        self.pools.push(Pool {
            device: self.device.clone(),
            inner: inner.clone(),
            allocated: 0,
            capacity: size as usize,
        });

        let pool = self.pools.last_mut().unwrap();
        alloc_info = alloc_info.descriptor_pool(pool.inner.pool);
        let set = unsafe {
            device
                .vk()
                .allocate_descriptor_sets(&alloc_info)
                .map_err(Error::AllocError)?
        };
        pool.allocated += 1;
        Ok(DescriptorSet {
            device: self.device.clone(),
            pool: Arc::downgrade(&pool.inner),
            vk: set[0],
        })
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe {
                device.vk().destroy_descriptor_set_layout(self.layout, None);
            }
        }
    }
}
