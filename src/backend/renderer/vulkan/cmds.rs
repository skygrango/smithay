use std::{collections::VecDeque, sync::Arc};

use super::Error;
use crate::backend::{
    renderer::vulkan::shaders::DescriptorSet,
    vulkan::{device::Device, image::ImageInner},
};

use ash::vk::{
    CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
    CommandBufferUsageFlags, CommandPool as VkCommandPool,
};

#[derive(Debug)]
pub struct CommandPool {
    vk: VkCommandPool,
    pending_buffers: VecDeque<(u64, CommandBuffer, Vec<DescriptorSet>, Vec<Arc<ImageInner>>)>,
    recycled_buffers: Vec<CommandBuffer>,
    device: Device,
}

impl CommandPool {
    pub(super) fn from_vk(device: &Device, vk: VkCommandPool) -> Self {
        CommandPool {
            vk,
            pending_buffers: VecDeque::new(),
            recycled_buffers: Vec::new(),
            device: device.clone(),
        }
    }

    pub fn vk(&self) -> VkCommandPool {
        self.vk
    }

    pub fn create_and_begin_buffer(&mut self) -> Result<CommandBuffer, Error> {
        let device = &self.device;

        let buf = if let Some(buf) = self.recycled_buffers.pop() {
            buf
        } else {
            let allocate_info = CommandBufferAllocateInfo::default()
                .command_pool(self.vk)
                .level(CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            unsafe {
                device
                    .vk()
                    .allocate_command_buffers(&allocate_info)
                    .map_err(Error::CommandBufferError)?[0]
            }
        };

        let begin_info = CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .vk()
                .begin_command_buffer(buf, &begin_info)
                .map_err(Error::CommandBufferError)?;
        }

        Ok(buf)
    }

    pub fn store_pending_buffer(
        &mut self,
        buf: CommandBuffer,
        seq: u64,
        descs: Vec<DescriptorSet>,
        images: Vec<Arc<ImageInner>>,
    ) {
        self.pending_buffers.push_back((seq, buf, descs, images));
    }

    pub fn clean_old_buffers(&mut self, seq: u64) {
        let idx = self.pending_buffers.iter().position(|(s, _, _, _)| *s > seq);

        let completed = (if let Some(idx) = idx {
            self.pending_buffers.drain(..idx)
        } else {
            self.pending_buffers.drain(..)
        })
        .map(|(_, buf, _, _)| buf);

        self.recycled_buffers.extend(completed);
    }

    pub fn is_empty(&self) -> bool {
        self.pending_buffers.is_empty()
    }

    pub fn last_pending_seq(&self) -> Option<u64> {
        self.pending_buffers.back().map(|(seq, _, _, _)| *seq)
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        if !self.pending_buffers.is_empty() {
            unsafe {
                let _ = self.device.wait_idle();
            }
        }

        let mut bufs = self
            .pending_buffers
            .drain(..)
            .map(|(_, buf, _, _)| buf)
            .collect::<Vec<_>>();
        bufs.extend(self.recycled_buffers.drain(..));

        unsafe {
            if !bufs.is_empty() {
                self.device.vk().free_command_buffers(self.vk, &bufs);
            }
            self.device.vk().destroy_command_pool(self.vk, None);
        }
    }
}
