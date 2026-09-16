use std::os::unix::io::OwnedFd;

use ash::vk::SemaphoreWaitInfo;

use super::{Fence, Interrupted};
use crate::backend::renderer::vulkan::VulkanSyncPoint;

impl Fence for VulkanSyncPoint {
    fn is_signaled(&self) -> bool {
        #[cfg(feature = "backend_drm")]
        if let Some(drm) = self.timeline.drm.as_ref() {
            if drm.query_signalled_point().ok().is_some_and(|p| p >= self.point) {
                return true;
            }
        }
        let Some(device) = self.timeline.device.upgrade() else {
            return false;
        };
        match unsafe { device.vk().get_semaphore_counter_value(self.timeline.vk) } {
            Ok(current) => current >= self.point,
            Err(_) => false,
        }
    }

    fn wait(&self) -> Result<(), Interrupted> {
        if self.is_signaled() {
            return Ok(());
        }
        let Some(device) = self.timeline.device.upgrade() else {
            return Ok(());
        };

        let sems = [self.timeline.vk];
        let vals = [self.point];
        let wait_info = SemaphoreWaitInfo::default().semaphores(&sems).values(&vals);

        // 10 seconds in nanoseconds, matching DrmSyncPoint timeout
        let timeout_ns = 10_000_000_000u64;

        match unsafe { device.vk().wait_semaphores(&wait_info, timeout_ns) } {
            Ok(()) => Ok(()),
            Err(ash::vk::Result::TIMEOUT) => {
                if self.is_signaled() {
                    Ok(())
                } else {
                    tracing::warn!(point = self.point, "VulkanSyncPoint wait timed out");
                    Ok(())
                }
            }
            Err(err) => {
                if self.is_signaled() {
                    Ok(())
                } else {
                    tracing::warn!(?err, point = self.point, "VulkanSyncPoint wait error");
                    Ok(())
                }
            }
        }
    }

    fn is_exportable(&self) -> bool {
        #[cfg(feature = "backend_drm")]
        {
            self.timeline.drm.is_some()
        }
        #[cfg(not(feature = "backend_drm"))]
        {
            false
        }
    }

    fn export(&self) -> Option<OwnedFd> {
        #[cfg(feature = "backend_drm")]
        {
            self.timeline.drm.as_ref().and_then(|drm| {
                crate::backend::drm::sync::DrmSyncPoint {
                    timeline: drm.clone(),
                    point: self.point,
                }
                .export_sync_file()
                .ok()
            })
        }
        #[cfg(not(feature = "backend_drm"))]
        {
            None
        }
    }
}
