use std::sync::Arc;

use crate::backend::{drm::sync::DrmTimeline, vulkan::device::WeakDevice};
use ash::vk::Semaphore;

#[derive(Debug)]
pub struct VulkanTimelineInner {
    pub(crate) device: WeakDevice,
    pub(crate) vk: Semaphore,
    pub(crate) drm: Option<DrmTimeline>,
}

impl Drop for VulkanTimelineInner {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe { device.vk().destroy_semaphore(self.vk, None) };
        }
    }
}

/// Vulkan timeline semaphore
#[derive(Clone, Debug)]
pub struct VulkanTimeline(pub(crate) Arc<VulkanTimelineInner>);

impl PartialEq for VulkanTimeline {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for VulkanTimeline {}
impl std::hash::Hash for VulkanTimeline {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl std::ops::Deref for VulkanTimeline {
    type Target = VulkanTimelineInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl VulkanTimeline {
    pub(crate) fn new(device: WeakDevice, vk: Semaphore, drm: Option<DrmTimeline>) -> Self {
        Self(Arc::new(VulkanTimelineInner { device, vk, drm }))
    }

    pub fn semaphore(&self) -> Semaphore {
        self.0.vk
    }

    pub fn drm(&self) -> Option<&DrmTimeline> {
        self.0.drm.as_ref()
    }
}

/// Point on a Vulkan timeline semaphore
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VulkanSyncPoint {
    pub(crate) timeline: VulkanTimeline,
    pub(crate) point: u64,
}

impl VulkanSyncPoint {
    /// Create a new `VulkanSyncPoint`
    pub fn new(timeline: VulkanTimeline, point: u64) -> Self {
        Self { timeline, point }
    }

    /// Borrow the [`VulkanTimeline`] this point lives on.
    pub fn timeline(&self) -> &VulkanTimeline {
        &self.timeline
    }

    /// Numeric timeline value for this point.
    pub fn point(&self) -> u64 {
        self.point
    }
}
