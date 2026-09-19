use ash::vk::{self, MemoryHeapFlags, MemoryPropertyFlags, PhysicalDeviceMemoryProperties};

/// Memory usage intent for selecting optimal Vulkan memory types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryUsagePreference {
    /// Regular GPU textures, render targets, compositor swapchain, framebuffer attachments.
    /// Prefers pure `DEVICE_LOCAL` without `HOST_VISIBLE` (GDDR, optimal tiling, DCC, zero BAR1 consumption).
    DeviceLocal,
    /// Host transfer images (e.g. wl_shm surfaces via `VK_EXT_host_image_copy`).
    /// Prefers System RAM (`HOST_VISIBLE | HOST_COHERENT`, non-`DEVICE_LOCAL`) to provide abundant memory,
    /// avoid BAR1 limits, and avoid PCIe bus contention during CPU writes.
    HostTransfer,
    /// CPU accessible / linear images or staging buffers (e.g. download/upload staging buffers, indirect buffers).
    /// Prefers System RAM (`HOST_VISIBLE | HOST_COHERENT`, non-`DEVICE_LOCAL`) to prevent BAR1 aperture exhaustion.
    HostVisible,
}

/// A ranked memory type candidate with its index and flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryTypeRank {
    /// The Vulkan memory type index.
    pub type_index: u32,
    /// Property flags of this memory type.
    pub property_flags: MemoryPropertyFlags,
}

/// Memory budget information returned by `VK_EXT_memory_budget`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudgetInfo {
    /// The memory budget for each heap (how much this process is allowed to allocate).
    pub heap_budget: [vk::DeviceSize; vk::MAX_MEMORY_HEAPS],
    /// The estimated current memory usage for each heap (by all processes and OS).
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

/// Precomputed memory type rankings for each usage intent.
///
/// Since `PhysicalDeviceMemoryProperties` is immutable throughout the device's lifetime,
/// the preference rankings are computed once during device initialization.
/// Selection during allocation is reduced to a zero-allocation bitmask filter over pre-sorted slices.
#[derive(Debug, Clone)]
pub struct MemoryPreferences {
    device_local: [MemoryTypeRank; 32],
    device_local_count: usize,
    host_transfer: [MemoryTypeRank; 32],
    host_transfer_count: usize,
    host_visible: [MemoryTypeRank; 32],
    host_visible_count: usize,
}

impl MemoryPreferences {
    /// Compute memory preference rankings from physical device memory properties.
    pub fn new(mem_props: &PhysicalDeviceMemoryProperties) -> Self {
        let memory_types = mem_props.memory_types_as_slice();
        let memory_heaps = mem_props.memory_heaps_as_slice();

        // Check if a memory type belongs to a small/constrained BAR1 aperture (<= 512 MB).
        // When Resizable BAR is disabled in BIOS, NVIDIA GPUs typically expose a 256 MiB BAR1 aperture.
        // Attempting to allocate large compositor images (e.g. 17MB) in a 256MB BAR1 heap
        // causes severe aperture exhaustion / GSP-RM failures under open kernel modules.
        let is_small_bar = |type_idx: usize| -> bool {
            if type_idx >= memory_types.len() {
                return false;
            }
            let heap_idx = memory_types[type_idx].heap_index as usize;
            if heap_idx < memory_heaps.len() {
                let heap = &memory_heaps[heap_idx];
                heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL) && heap.size <= 512 * 1024 * 1024
            } else {
                false
            }
        };

        // 1. Compute DeviceLocal rankings
        let mut dl_candidates: [(u32, MemoryPropertyFlags, i32); 32] =
            [(0, MemoryPropertyFlags::empty(), 0); 32];
        let mut dl_count = 0;
        for (i, mem_type) in memory_types.iter().enumerate() {
            let flags = mem_type.property_flags;
            let score = if flags.contains(MemoryPropertyFlags::DEVICE_LOCAL)
                && !flags.contains(MemoryPropertyFlags::HOST_VISIBLE)
            {
                1000
            } else if flags.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
                800
            } else {
                400
            };
            dl_candidates[dl_count] = (i as u32, flags, score);
            dl_count += 1;
        }
        dl_candidates[..dl_count].sort_by(|a, b| b.2.cmp(&a.2));

        let mut device_local = [MemoryTypeRank::default(); 32];
        for i in 0..dl_count {
            device_local[i] = MemoryTypeRank {
                type_index: dl_candidates[i].0,
                property_flags: dl_candidates[i].1,
            };
        }

        // 2. Compute HostTransfer rankings
        let mut ht_candidates: [(u32, MemoryPropertyFlags, i32); 32] =
            [(0, MemoryPropertyFlags::empty(), 0); 32];
        let mut ht_count = 0;
        for (i, mem_type) in memory_types.iter().enumerate() {
            let flags = mem_type.property_flags;
            let small_bar = is_small_bar(i);
            let score = if flags
                .contains(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT)
                && !flags.contains(MemoryPropertyFlags::DEVICE_LOCAL)
            {
                1000
            } else if flags.contains(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::DEVICE_LOCAL) {
                if !small_bar { 900 } else { 700 }
            } else if flags.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
                850
            } else if flags.contains(MemoryPropertyFlags::HOST_VISIBLE) {
                800
            } else {
                400
            };
            ht_candidates[ht_count] = (i as u32, flags, score);
            ht_count += 1;
        }
        ht_candidates[..ht_count].sort_by(|a, b| b.2.cmp(&a.2));

        let mut host_transfer = [MemoryTypeRank::default(); 32];
        for i in 0..ht_count {
            host_transfer[i] = MemoryTypeRank {
                type_index: ht_candidates[i].0,
                property_flags: ht_candidates[i].1,
            };
        }

        // 3. Compute HostVisible rankings (must contain HOST_VISIBLE)
        let mut hv_candidates: [(u32, MemoryPropertyFlags, i32); 32] =
            [(0, MemoryPropertyFlags::empty(), 0); 32];
        let mut hv_count = 0;
        for (i, mem_type) in memory_types.iter().enumerate() {
            let flags = mem_type.property_flags;
            if !flags.contains(MemoryPropertyFlags::HOST_VISIBLE) {
                continue;
            }
            let small_bar = is_small_bar(i);
            let score = if flags.contains(MemoryPropertyFlags::HOST_COHERENT)
                && !flags.contains(MemoryPropertyFlags::DEVICE_LOCAL)
            {
                1000
            } else if flags.contains(MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::DEVICE_LOCAL) {
                if !small_bar { 900 } else { 700 }
            } else if flags.contains(MemoryPropertyFlags::HOST_COHERENT) {
                850
            } else if flags.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
                if !small_bar { 800 } else { 600 }
            } else {
                500
            };
            hv_candidates[hv_count] = (i as u32, flags, score);
            hv_count += 1;
        }
        hv_candidates[..hv_count].sort_by(|a, b| b.2.cmp(&a.2));

        let mut host_visible = [MemoryTypeRank::default(); 32];
        for i in 0..hv_count {
            host_visible[i] = MemoryTypeRank {
                type_index: hv_candidates[i].0,
                property_flags: hv_candidates[i].1,
            };
        }

        Self {
            device_local,
            device_local_count: dl_count,
            host_transfer,
            host_transfer_count: ht_count,
            host_visible,
            host_visible_count: hv_count,
        }
    }

    /// Return the ranked slice of candidate memory types for the specified preference.
    #[inline]
    pub fn ranked_slice(&self, preference: MemoryUsagePreference) -> &[MemoryTypeRank] {
        match preference {
            MemoryUsagePreference::DeviceLocal => &self.device_local[..self.device_local_count],
            MemoryUsagePreference::HostTransfer => &self.host_transfer[..self.host_transfer_count],
            MemoryUsagePreference::HostVisible => &self.host_visible[..self.host_visible_count],
        }
    }

    /// Return an iterator over candidates that match `type_bits`, ordered from most to least preferred.
    #[inline]
    pub fn filter_candidates(
        &self,
        type_bits: u32,
        preference: MemoryUsagePreference,
    ) -> impl Iterator<Item = MemoryTypeRank> + '_ {
        self.ranked_slice(preference)
            .iter()
            .copied()
            .filter(move |r| (type_bits & (1 << r.type_index)) != 0)
    }

    /// Find the single best memory type matching `type_bits`.
    #[inline]
    pub fn find_best(&self, type_bits: u32, preference: MemoryUsagePreference) -> Option<MemoryTypeRank> {
        self.filter_candidates(type_bits, preference).next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::{MemoryHeap, MemoryHeapFlags, MemoryPropertyFlags, MemoryType};

    #[test]
    fn test_nvidia_small_bar_topology_ranking() {
        // Simulating an NVIDIA GPU with a 256 MiB BAR1 aperture and open kernel modules:
        // Heap 0: 8 GiB VRAM (DEVICE_LOCAL)
        // Heap 1: 16 GiB System RAM (non-DEVICE_LOCAL)
        // Heap 2: 256 MiB BAR1 aperture (DEVICE_LOCAL, small BAR)
        let mut heaps = [MemoryHeap::default(); 16];
        heaps[0] = MemoryHeap {
            size: 8 * 1024 * 1024 * 1024,
            flags: MemoryHeapFlags::DEVICE_LOCAL,
        };
        heaps[1] = MemoryHeap {
            size: 16 * 1024 * 1024 * 1024,
            flags: MemoryHeapFlags::empty(),
        };
        heaps[2] = MemoryHeap {
            size: 256 * 1024 * 1024,
            flags: MemoryHeapFlags::DEVICE_LOCAL,
        };

        // Type 0: Pure VRAM (Heap 0)
        // Type 1: Small BAR1 window into VRAM (Heap 2) - HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL
        // Type 2: System RAM (Heap 1) - HOST_VISIBLE | HOST_COHERENT
        let mut types = [MemoryType::default(); 32];
        types[0] = MemoryType {
            property_flags: MemoryPropertyFlags::DEVICE_LOCAL,
            heap_index: 0,
        };
        types[1] = MemoryType {
            property_flags: MemoryPropertyFlags::DEVICE_LOCAL
                | MemoryPropertyFlags::HOST_VISIBLE
                | MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 2,
        };
        types[2] = MemoryType {
            property_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 1,
        };

        let mem_props = PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            memory_types: types,
            memory_heap_count: 3,
            memory_heaps: heaps,
        };

        let prefs = MemoryPreferences::new(&mem_props);

        // 1. DeviceLocal (textures, render targets) must prefer pure VRAM (Type 0) over small BAR (Type 1)
        let dl_best = prefs.find_best(0b111, MemoryUsagePreference::DeviceLocal);
        assert_eq!(dl_best.map(|r| r.type_index), Some(0));

        // 2. HostVisible (staging buffers, linear) must prefer System RAM (Type 2) over small BAR (Type 1)
        // to prevent 256MB BAR1 exhaustion!
        let hv_best = prefs.find_best(0b111, MemoryUsagePreference::HostVisible);
        assert_eq!(hv_best.map(|r| r.type_index), Some(2));

        // 3. If System RAM (Type 2) is not compatible, it should fallback to Type 1
        let hv_fallback = prefs.find_best(0b011, MemoryUsagePreference::HostVisible);
        assert_eq!(hv_fallback.map(|r| r.type_index), Some(1));

        // 4. Type 0 has no HOST_VISIBLE, so for HostVisible it should return None
        let hv_none = prefs.find_best(0b001, MemoryUsagePreference::HostVisible);
        assert_eq!(hv_none, None);

        // 5. HostTransfer (VK_EXT_host_image_copy wl_shm) must prefer System RAM (Type 2) over small BAR
        let ht_best = prefs.find_best(0b111, MemoryUsagePreference::HostTransfer);
        assert_eq!(ht_best.map(|r| r.type_index), Some(2));
    }
}
