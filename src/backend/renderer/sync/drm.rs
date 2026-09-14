use std::os::unix::io::OwnedFd;

use super::{Fence, Interrupted};
use crate::backend::drm::sync::DrmSyncPoint;

impl Fence for DrmSyncPoint {
    fn is_signaled(&self) -> bool {
        self.timeline
            .query_signalled_point()
            .ok()
            .is_some_and(|point| point >= self.point)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        if self.is_signaled() {
            return Ok(());
        }
        let ts = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
        let now_ns = ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64;
        let timeout_nsec = now_ns.saturating_add(10_000_000_000);

        match self.wait(timeout_nsec) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => Err(Interrupted),
            Err(err) => {
                if self.is_signaled() {
                    Ok(())
                } else {
                    tracing::warn!(?err, point = self.point, "DrmSyncPoint wait error");
                    Ok(())
                }
            }
        }
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        self.export_sync_file().ok()
    }
}
