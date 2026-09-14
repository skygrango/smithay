use std::collections::HashMap;

use ash::vk::{self, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo, ShaderStageFlags};
use include_bytes_aligned::include_bytes_aligned;

use crate::backend::{
    renderer::vulkan::shaders::descriptor::DescriptorAllocator,
    vulkan::{Device, device::WeakDevice},
};

mod clear;
mod descriptor;
mod hdr_texture;
mod texture;
use self::clear::*;
pub use self::descriptor::DescriptorSet;
use self::hdr_texture::*;
use self::texture::*;
pub use self::{clear::ClearPushConstants, hdr_texture::HdrTexPushConstants, texture::TexPushConstants};

pub const QUAD_VERT_SHADER: &[u8] =
    include_bytes_aligned!(32, concat!(env!("OUT_DIR"), "/vk/quad.vert.glsl"));

pub fn spirv_u32(shader: &[u8]) -> &[u32] {
    let len = shader.len();
    assert!(len.is_multiple_of(4));
    bytemuck::cast_slice(shader)
}

#[derive(Debug, Clone, Copy)]
pub struct FormatPipelines {
    pub clear_pipeline: Pipeline,
    pub clear_blend_pipeline: Pipeline,
    pub tex_pipeline: Pipeline,
    pub tex_blend_pipeline: Pipeline,
    pub hdr_tex_pipeline: Pipeline,
    pub hdr_tex_blend_pipeline: Pipeline,
}

#[derive(Debug)]
pub struct Pipelines {
    device: WeakDevice,
    cache: vk::PipelineCache,

    clear_layout: PipelineLayout,
    tex_layout: PipelineLayout,
    hdr_tex_layout: PipelineLayout,

    tex_desc_pool: DescriptorAllocator,
    hdr_tex_desc_pool: DescriptorAllocator,

    quad_vert: vk::ShaderModule,
    clear_frag: vk::ShaderModule,
    tex_frag: vk::ShaderModule,
    hdr_tex_frag: vk::ShaderModule,

    format_pipelines: HashMap<vk::Format, FormatPipelines>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinShader {
    Clear,
    Texture,
    HdrTexture,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to create pipeline cache")]
    CacheCreation(#[source] vk::Result),
    #[error("Failed to create shader module")]
    Shader(#[source] vk::Result),
    #[error("Failed to create descriptor set layout")]
    DescriptorSetLayout(#[source] vk::Result),
    #[error(transparent)]
    DescriptorSet(#[from] self::descriptor::Error),
    #[error("Failed to create pipeline layout")]
    PipelineLayout(#[source] vk::Result),
    #[error("Failed to create pipelines")]
    Pipeline(#[source] vk::Result),
}

impl Pipelines {
    pub fn new(device: &Device) -> Result<Self, Error> {
        let create_info = vk::PipelineCacheCreateInfo::default();
        let cache = unsafe {
            device
                .vk()
                .create_pipeline_cache(&create_info, None)
                .map_err(Error::CacheCreation)?
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32(QUAD_VERT_SHADER));
        let quad_vert = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(Error::Shader)?
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32(CLEAR_SHADER));
        let clear_frag = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(Error::Shader)?
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32(TEX_SHADER));
        let tex_frag = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(Error::Shader)?
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32(HDR_TEX_SHADER));
        let hdr_tex_frag = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(Error::Shader)?
        };

        let layout_flags = if device.vk_khr_push_descriptor().is_some() {
            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };

        // Clear pipeline layout (0 descriptor sets)
        let constants = [vk::PushConstantRange::default()
            .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<ClearPushConstants>() as u32)];
        let create_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&constants);
        let clear_layout = unsafe {
            device
                .vk()
                .create_pipeline_layout(&create_info, None)
                .map_err(Error::PipelineLayout)?
        };

        // Texture pipeline layout (1 descriptor set: sampler2D)
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(layout_flags)
            .bindings(TEX_BINDINGS.as_slice());
        let tex_descriptor_set = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(Error::DescriptorSetLayout)?
        };
        let tex_layouts = [tex_descriptor_set];

        let constants = [vk::PushConstantRange::default()
            .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<TexPushConstants>() as u32)];
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&tex_layouts)
            .push_constant_ranges(&constants);
        let tex_layout = unsafe {
            device
                .vk()
                .create_pipeline_layout(&create_info, None)
                .map_err(Error::PipelineLayout)?
        };
        let tex_desc_pool = DescriptorAllocator::new(device, tex_descriptor_set, &*TEX_SIZES);

        // HDR Texture pipeline layout (1 descriptor set: sampler2D)
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(layout_flags)
            .bindings(HDR_TEX_BINDINGS.as_slice());
        let hdr_tex_descriptor_set = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(Error::DescriptorSetLayout)?
        };
        let hdr_tex_layouts = [hdr_tex_descriptor_set];

        let constants = [vk::PushConstantRange::default()
            .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<HdrTexPushConstants>() as u32)];
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&hdr_tex_layouts)
            .push_constant_ranges(&constants);
        let hdr_tex_layout = unsafe {
            device
                .vk()
                .create_pipeline_layout(&create_info, None)
                .map_err(Error::PipelineLayout)?
        };
        let hdr_tex_desc_pool = DescriptorAllocator::new(device, hdr_tex_descriptor_set, &*HDR_TEX_SIZES);

        Ok(Pipelines {
            device: device.downgrade(),
            cache,

            clear_layout,
            tex_layout,
            hdr_tex_layout,

            tex_desc_pool,
            hdr_tex_desc_pool,

            quad_vert,
            clear_frag,
            tex_frag,
            hdr_tex_frag,

            format_pipelines: HashMap::new(),
        })
    }

    pub fn get_or_create_format_pipelines(&mut self, format: vk::Format) -> Result<FormatPipelines, Error> {
        if let Some(&p) = self.format_pipelines.get(&format) {
            return Ok(p);
        }

        let device = self
            .device
            .upgrade()
            .ok_or(Error::Pipeline(vk::Result::ERROR_DEVICE_LOST))?;

        let vertex_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::VERTEX)
            .name(c"main")
            .module(self.quad_vert);

        let clear_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.clear_frag);

        let tex_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.tex_frag);

        let hdr_tex_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.hdr_tex_frag);

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_attachment_formats = [format];
        let mut r0 =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_attachment_formats);
        let mut r1 =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_attachment_formats);
        let mut r2 =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_attachment_formats);
        let mut r3 =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_attachment_formats);
        let mut r4 =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_attachment_formats);
        let mut r5 =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_attachment_formats);

        let opaque_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let opaque_blend_state =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&opaque_attachment);

        let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let alpha_blend_state =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);

        let clear_stages = [vertex_stage, clear_stage];
        let tex_stages = [vertex_stage, tex_stage];
        let hdr_tex_stages = [vertex_stage, hdr_tex_stage];

        let base_ci = vk::GraphicsPipelineCreateInfo::default()
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .dynamic_state(&dynamic_state);

        let create_infos = [
            // 0: clear opaque
            base_ci
                .push_next(&mut r0)
                .stages(&clear_stages)
                .color_blend_state(&opaque_blend_state)
                .layout(self.clear_layout),
            // 1: clear blend
            base_ci
                .push_next(&mut r1)
                .stages(&clear_stages)
                .color_blend_state(&alpha_blend_state)
                .layout(self.clear_layout),
            // 2: tex opaque
            base_ci
                .push_next(&mut r2)
                .stages(&tex_stages)
                .color_blend_state(&opaque_blend_state)
                .layout(self.tex_layout),
            // 3: tex blend
            base_ci
                .push_next(&mut r3)
                .stages(&tex_stages)
                .color_blend_state(&alpha_blend_state)
                .layout(self.tex_layout),
            // 4: hdr_tex opaque
            base_ci
                .push_next(&mut r4)
                .stages(&hdr_tex_stages)
                .color_blend_state(&opaque_blend_state)
                .layout(self.hdr_tex_layout),
            // 5: hdr_tex blend
            base_ci
                .push_next(&mut r5)
                .stages(&hdr_tex_stages)
                .color_blend_state(&alpha_blend_state)
                .layout(self.hdr_tex_layout),
        ];

        let pipelines = unsafe {
            device
                .vk()
                .create_graphics_pipelines(self.cache, &create_infos, None)
                .map_err(|(pipelines, res)| {
                    for pipeline in pipelines {
                        device.vk().destroy_pipeline(pipeline, None);
                    }
                    res
                })
                .map_err(Error::Pipeline)?
        };

        let format_pipelines = FormatPipelines {
            clear_pipeline: pipelines[0],
            clear_blend_pipeline: pipelines[1],
            tex_pipeline: pipelines[2],
            tex_blend_pipeline: pipelines[3],
            hdr_tex_pipeline: pipelines[4],
            hdr_tex_blend_pipeline: pipelines[5],
        };

        self.format_pipelines.insert(format, format_pipelines);
        Ok(format_pipelines)
    }

    pub fn clear_pipeline_layout(&self) -> &PipelineLayout {
        &self.clear_layout
    }

    pub fn tex_pipeline_layout(&self) -> &PipelineLayout {
        &self.tex_layout
    }

    pub fn hdr_tex_pipeline_layout(&self) -> &PipelineLayout {
        &self.hdr_tex_layout
    }

    pub fn alloc_descriptor_set(&mut self, shader: BuiltinShader) -> Result<DescriptorSet, Error> {
        match shader {
            BuiltinShader::Clear => self.tex_desc_pool.alloc_descriptor_set().map_err(Into::into),
            BuiltinShader::Texture => self.tex_desc_pool.alloc_descriptor_set().map_err(Into::into),
            BuiltinShader::HdrTexture => self.hdr_tex_desc_pool.alloc_descriptor_set().map_err(Into::into),
        }
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe {
                for (_, p) in self.format_pipelines.drain() {
                    device.vk().destroy_pipeline(p.clear_pipeline, None);
                    device.vk().destroy_pipeline(p.clear_blend_pipeline, None);
                    device.vk().destroy_pipeline(p.tex_pipeline, None);
                    device.vk().destroy_pipeline(p.tex_blend_pipeline, None);
                    device.vk().destroy_pipeline(p.hdr_tex_pipeline, None);
                    device.vk().destroy_pipeline(p.hdr_tex_blend_pipeline, None);
                }
                device.vk().destroy_shader_module(self.quad_vert, None);
                device.vk().destroy_shader_module(self.clear_frag, None);
                device.vk().destroy_shader_module(self.tex_frag, None);
                device.vk().destroy_shader_module(self.hdr_tex_frag, None);

                device.vk().destroy_pipeline_layout(self.clear_layout, None);
                device.vk().destroy_pipeline_layout(self.tex_layout, None);
                device.vk().destroy_pipeline_layout(self.hdr_tex_layout, None);
                device.vk().destroy_pipeline_cache(self.cache, None);
            }
        }
    }
}
