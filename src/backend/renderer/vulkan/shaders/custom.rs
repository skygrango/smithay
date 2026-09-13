use ash::vk::{
    self, DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType,
    Pipeline, PipelineLayout, PipelineShaderStageCreateInfo, ShaderStageFlags,
};
use std::sync::LazyLock;

use crate::backend::{
    renderer::vulkan::shaders::{
        Error as ShaderError,
        descriptor::{DescriptorAllocator, DescriptorSet},
    },
    vulkan::{Device, device::WeakDevice},
};

static CUSTOM_2D_BINDINGS: LazyLock<[DescriptorSetLayoutBinding<'static>; 2]> = LazyLock::new(|| {
    [
        DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(DescriptorType::STORAGE_IMAGE)
            .stage_flags(ShaderStageFlags::COMPUTE)
            .descriptor_count(1),
        DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(ShaderStageFlags::COMPUTE)
            .descriptor_count(1),
    ]
});

static CUSTOM_2D_SIZES: LazyLock<[DescriptorPoolSize; 2]> = LazyLock::new(|| {
    [
        DescriptorPoolSize::default()
            .ty(DescriptorType::STORAGE_IMAGE)
            .descriptor_count(64),
        DescriptorPoolSize::default()
            .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(64),
    ]
});

#[derive(Debug)]
pub struct VulkanCustomPipeline {
    device: WeakDevice,
    pipeline: Pipeline,
    layout: PipelineLayout,
    descriptor_layout: DescriptorSetLayout,
    desc_pool: DescriptorAllocator,
    push_constant_size: usize,
}

impl VulkanCustomPipeline {
    pub fn new(
        device: &Device,
        cache: vk::PipelineCache,
        spirv: &[u32],
        push_constant_size: usize,
    ) -> Result<Self, ShaderError> {
        let create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(CUSTOM_2D_BINDINGS.as_slice());
        let descriptor_layout = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(ShaderError::DescriptorSetLayout)?
        };

        let layouts = [descriptor_layout];
        let constants = if push_constant_size > 0 {
            vec![vk::PushConstantRange::default()
                .stage_flags(ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constant_size as u32)]
        } else {
            Vec::new()
        };

        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&constants);
        let layout = unsafe {
            device
                .vk()
                .create_pipeline_layout(&create_info, None)
                .map_err(ShaderError::PipelineLayout)?
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(ShaderError::Shader)?
        };

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(
                PipelineShaderStageCreateInfo::default()
                    .stage(ShaderStageFlags::COMPUTE)
                    .name(c"main")
                    .module(shader),
            )
            .layout(layout);

        let pipelines = unsafe {
            device
                .vk()
                .create_compute_pipelines(cache, &[create_info], None)
                .map_err(|(pipelines, res)| {
                    for p in pipelines {
                        device.vk().destroy_pipeline(p, None);
                    }
                    res
                })
                .map_err(ShaderError::Pipeline)?
        };

        unsafe {
            device.vk().destroy_shader_module(shader, None);
        }

        let desc_pool = DescriptorAllocator::new(device, descriptor_layout, &*CUSTOM_2D_SIZES);

        Ok(VulkanCustomPipeline {
            device: device.downgrade(),
            pipeline: pipelines[0],
            layout,
            descriptor_layout,
            desc_pool,
            push_constant_size,
        })
    }

    pub fn pipeline(&self) -> Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> PipelineLayout {
        self.layout
    }

    pub fn push_constant_size(&self) -> usize {
        self.push_constant_size
    }

    pub fn alloc_descriptor_set(&mut self) -> Result<DescriptorSet, ShaderError> {
        self.desc_pool.alloc_descriptor_set().map_err(Into::into)
    }
}

impl Drop for VulkanCustomPipeline {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe {
                device.vk().destroy_pipeline(self.pipeline, None);
                device.vk().destroy_pipeline_layout(self.layout, None);
                device
                    .vk()
                    .destroy_descriptor_set_layout(self.descriptor_layout, None);
            }
        }
    }
}
