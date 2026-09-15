use std::os::fd::{BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};

use crate::backend::{
    drm::{DrmDeviceFd, sync::DrmTimeline},
    renderer::vulkan::{Error, cmds::CommandPool, sync::VulkanTimeline},
    vulkan::Device,
};
use ash::{
    khr,
    vk::{
        self, CommandPoolCreateFlags, CommandPoolCreateInfo, ExportSemaphoreCreateInfo,
        ExternalSemaphoreHandleTypeFlags, SemaphoreCreateInfo, SemaphoreGetFdInfoKHR, SemaphoreType,
        SemaphoreTypeCreateInfo,
    },
};

impl Device {
    pub(super) fn create_timeline_semaphore(
        &self,
        export: Option<DrmDeviceFd>,
    ) -> Result<VulkanTimeline, Error> {
        let mut semaphore_type_info = SemaphoreTypeCreateInfo::default()
            .semaphore_type(SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut semaphore_export_info =
            ExportSemaphoreCreateInfo::default().handle_types(ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);

        let mut semaphore_create_info = SemaphoreCreateInfo::default().push_next(&mut semaphore_type_info);
        if export.is_some() {
            semaphore_create_info = semaphore_create_info.push_next(&mut semaphore_export_info);
        }

        let semaphore = unsafe {
            self.vk()
                .create_semaphore(&semaphore_create_info, None)
                .map_err(Error::SemaphoreError)?
        };

        let drm = if let Some(dev) = export {
            let Some(khr_external_semaphore_fd) = self.vk_khr_external_semaphore_fd() else {
                return Err(Error::MissingExtension(khr::external_semaphore_fd::NAME));
            };

            let semaphore_get_info = SemaphoreGetFdInfoKHR::default()
                .semaphore(semaphore.clone())
                .handle_type(ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);

            let fd = unsafe {
                OwnedFd::from_raw_fd(
                    khr_external_semaphore_fd
                        .get_semaphore_fd(&semaphore_get_info)
                        .map_err(Error::SemaphoreExportError)?,
                )
            };
            DrmTimeline::new(&dev, fd).ok()
        } else {
            None
        };

        Ok(VulkanTimeline {
            device: self.downgrade(),
            vk: semaphore,
            drm,
        })
    }

    pub(super) fn import_timeline_semaphore(
        &self,
        timeline_fd: BorrowedFd<'_>,
    ) -> Result<vk::Semaphore, Error> {
        let Some(khr_external_semaphore_fd) = self.vk_khr_external_semaphore_fd() else {
            return Err(Error::MissingExtension(khr::external_semaphore_fd::NAME));
        };

        let mut semaphore_type_info = SemaphoreTypeCreateInfo::default()
            .semaphore_type(SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut semaphore_export_info =
            ExportSemaphoreCreateInfo::default().handle_types(ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);

        let semaphore_create_info = SemaphoreCreateInfo::default()
            .push_next(&mut semaphore_type_info)
            .push_next(&mut semaphore_export_info);

        let semaphore = unsafe {
            self.vk()
                .create_semaphore(&semaphore_create_info, None)
                .map_err(Error::SemaphoreError)?
        };

        let dup_fd = match rustix::io::dup(timeline_fd) {
            Ok(fd) => fd,
            Err(_) => {
                unsafe { self.vk().destroy_semaphore(semaphore, None) };
                return Err(Error::SemaphoreImportError(
                    vk::Result::ERROR_INITIALIZATION_FAILED,
                ));
            }
        };

        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
            .fd(dup_fd.into_raw_fd());

        let res = unsafe { khr_external_semaphore_fd.import_semaphore_fd(&import_info) };

        if let Err(err) = res {
            unsafe { self.vk().destroy_semaphore(semaphore, None) };
            return Err(Error::SemaphoreImportError(err));
        }

        Ok(semaphore)
    }

    pub(super) fn create_command_pool(&self) -> Result<CommandPool, Error> {
        let pool_info = CommandPoolCreateInfo::default()
            .queue_family_index(self.queue_family_idx())
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe {
            self.vk()
                .create_command_pool(&pool_info, None)
                .map_err(Error::CommandPoolError)?
        };

        Ok(CommandPool::from_vk(self, cmd_pool))
    }
}
