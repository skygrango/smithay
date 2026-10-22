use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ash::vk::{
    self, BorderColor, CompareOp, Filter, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    SamplerAddressMode, SamplerCreateFlags, SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags,
};
use include_bytes_aligned::include_bytes_aligned;

use crate::backend::{
    renderer::vulkan::shaders::descriptor::DescriptorAllocator,
    vulkan::{Device, ycbcr::YcbcrKey},
};

mod clear;
mod descriptor;
mod hdr_texture;
mod outline;
mod texture;
use self::clear::*;
pub use self::descriptor::DescriptorSet;
use self::hdr_texture::*;
use self::outline::*;
use self::texture::*;
pub use self::{
    clear::ClearPushConstants,
    hdr_texture::{
        HDR_FLAG_HARDWARE_OFFLOAD, HDR_FLAG_INPUT_IS_HLG, HDR_FLAG_INPUT_IS_PQ,
        HDR_FLAG_INPUT_PRIMARIES_SHIFT, HDR_FLAG_SKIP_COLOR_TRANSFORM, HDR_FLAG_TARGET_IS_SDR,
        HdrTexPushConstants, SPEC_MODE_GENERIC, SPEC_MODE_LUT3D, SPEC_MODE_PASSTHROUGH, SPEC_MODE_PQ,
        SPEC_MODE_SDR,
    },
    outline::OutlinePushConstants,
    texture::TexPushConstants,
};

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
    pub hdr_passthrough_pipeline: Pipeline,
    pub hdr_passthrough_blend_pipeline: Pipeline,
    pub hdr_sdr_pipeline: Pipeline,
    pub hdr_sdr_blend_pipeline: Pipeline,
    pub hdr_pq_pipeline: Pipeline,
    pub hdr_pq_blend_pipeline: Pipeline,
    pub hdr_lut3d_pipeline: Pipeline,
    pub hdr_lut3d_blend_pipeline: Pipeline,
    pub outline_pipeline: Pipeline,
    pub outline_blend_pipeline: Pipeline,
}

#[derive(Debug, Clone, Copy)]
pub struct YcbcrFormatPipelines {
    pub tex_pipeline: Pipeline,
    pub tex_blend_pipeline: Pipeline,
    pub hdr_tex_pipeline: Pipeline,
    pub hdr_tex_blend_pipeline: Pipeline,
    pub hdr_passthrough_pipeline: Pipeline,
    pub hdr_passthrough_blend_pipeline: Pipeline,
    pub hdr_sdr_pipeline: Pipeline,
    pub hdr_sdr_blend_pipeline: Pipeline,
    pub hdr_pq_pipeline: Pipeline,
    pub hdr_pq_blend_pipeline: Pipeline,
}

#[derive(Debug)]
pub struct YcbcrEntry {
    pub tex_descriptor_set_layout: vk::DescriptorSetLayout,
    pub tex_layout: PipelineLayout,
    pub tex_desc_pool: DescriptorAllocator,

    pub hdr_tex_descriptor_set_layout: vk::DescriptorSetLayout,
    pub hdr_tex_layout: PipelineLayout,
    pub hdr_tex_desc_pool: DescriptorAllocator,

    pub format_pipelines: HashMap<(vk::Format, bool), YcbcrFormatPipelines>,
}

#[derive(Debug)]
pub struct SharedPipelines {
    cache: vk::PipelineCache,
    cache_path: Option<std::path::PathBuf>,

    clear_layout: PipelineLayout,
    tex_descriptor_set_layout: vk::DescriptorSetLayout,
    tex_layout: PipelineLayout,
    hdr_tex_descriptor_set_layout: vk::DescriptorSetLayout,
    hdr_tex_layout: PipelineLayout,
    outline_layout: PipelineLayout,

    quad_vert: vk::ShaderModule,
    clear_frag: vk::ShaderModule,
    tex_frag: vk::ShaderModule,
    hdr_tex_frag: vk::ShaderModule,
    outline_frag: vk::ShaderModule,

    texture_sampler: vk::Sampler,
    texture_sampler_nearest: vk::Sampler,

    format_pipelines: Mutex<HashMap<(vk::Format, bool), FormatPipelines>>,
    device: Device,
}

#[derive(Debug)]
pub struct Pipelines {
    tex_desc_pool: DescriptorAllocator,
    hdr_tex_desc_pool: DescriptorAllocator,
    local_format_pipelines: HashMap<(vk::Format, bool), FormatPipelines>,
    ycbcr_entries: HashMap<YcbcrKey, YcbcrEntry>,
    shared: Arc<SharedPipelines>,
}

fn get_pipeline_cache_path(device: &Device) -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("SMITHAY_VULKAN_PIPELINE_CACHE") {
        return Some(std::path::PathBuf::from(path));
    }

    let base_dir = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;

    let dir = base_dir.join("smithay");
    let _ = std::fs::create_dir_all(&dir);

    let uuid = device.pipeline_cache_uuid();
    let hex_uuid = uuid.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    Some(dir.join(format!("vulkan_pipelines_{}.bin", hex_uuid)))
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
    #[error("Failed to create sampler")]
    Sampler(#[source] vk::Result),
}

impl SharedPipelines {
    pub fn new(device: &Device) -> Result<Arc<Self>, Error> {
        let cache_path = get_pipeline_cache_path(device);
        let initial_data = cache_path.as_ref().and_then(|p| match std::fs::read(p) {
            Ok(data) if !data.is_empty() => Some(data),
            _ => None,
        });

        let mut create_info = vk::PipelineCacheCreateInfo::default();
        if let Some(ref data) = initial_data {
            create_info = create_info.initial_data(data);
        }

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

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32(OUTLINE_SHADER));
        let outline_frag = unsafe {
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

        // Outline pipeline layout (0 descriptor sets)
        let constants = [vk::PushConstantRange::default()
            .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<OutlinePushConstants>() as u32)];
        let create_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&constants);
        let outline_layout = unsafe {
            device
                .vk()
                .create_pipeline_layout(&create_info, None)
                .map_err(Error::PipelineLayout)?
        };

        // Texture pipeline layout (1 descriptor set: sampler2D)
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(layout_flags)
            .bindings(TEX_BINDINGS.as_slice());
        let tex_descriptor_set_layout = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(Error::DescriptorSetLayout)?
        };
        let tex_layouts = [tex_descriptor_set_layout];

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

        // HDR Texture pipeline layout (1 descriptor set: sampler2D)
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(layout_flags)
            .bindings(HDR_TEX_BINDINGS.as_slice());
        let hdr_tex_descriptor_set_layout = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(Error::DescriptorSetLayout)?
        };
        let hdr_tex_layouts = [hdr_tex_descriptor_set_layout];

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

        let texture_sampler = unsafe {
            device
                .vk()
                .create_sampler(
                    &SamplerCreateInfo::default()
                        .flags(SamplerCreateFlags::empty())
                        .mag_filter(Filter::LINEAR)
                        .min_filter(Filter::LINEAR)
                        .mipmap_mode(SamplerMipmapMode::NEAREST)
                        .address_mode_u(SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_w(SamplerAddressMode::CLAMP_TO_BORDER)
                        .mip_lod_bias(0.0)
                        .anisotropy_enable(false)
                        .max_anisotropy(0.0)
                        .compare_enable(false)
                        .compare_op(CompareOp::NEVER)
                        .min_lod(0.0)
                        .max_lod(0.0)
                        .border_color(BorderColor::FLOAT_TRANSPARENT_BLACK)
                        .unnormalized_coordinates(false),
                    None,
                )
                .map_err(Error::Sampler)?
        };

        let texture_sampler_nearest = unsafe {
            device
                .vk()
                .create_sampler(
                    &SamplerCreateInfo::default()
                        .flags(SamplerCreateFlags::empty())
                        .mag_filter(Filter::NEAREST)
                        .min_filter(Filter::NEAREST)
                        .mipmap_mode(SamplerMipmapMode::NEAREST)
                        .address_mode_u(SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_w(SamplerAddressMode::CLAMP_TO_BORDER)
                        .mip_lod_bias(0.0)
                        .anisotropy_enable(false)
                        .max_anisotropy(0.0)
                        .compare_enable(false)
                        .compare_op(CompareOp::NEVER)
                        .min_lod(0.0)
                        .max_lod(0.0)
                        .border_color(BorderColor::FLOAT_TRANSPARENT_BLACK)
                        .unnormalized_coordinates(false),
                    None,
                )
                .map_err(Error::Sampler)?
        };

        Ok(Arc::new(SharedPipelines {
            cache,
            cache_path,
            clear_layout,
            tex_descriptor_set_layout,
            tex_layout,
            hdr_tex_descriptor_set_layout,
            hdr_tex_layout,
            outline_layout,
            quad_vert,
            clear_frag,
            tex_frag,
            hdr_tex_frag,
            outline_frag,
            texture_sampler,
            texture_sampler_nearest,
            format_pipelines: Mutex::new(HashMap::new()),
            device: device.clone(),
        }))
    }

    pub fn texture_sampler(&self) -> vk::Sampler {
        self.texture_sampler
    }

    pub fn texture_sampler_nearest(&self) -> vk::Sampler {
        self.texture_sampler_nearest
    }

    pub fn save_cache(&self) {
        let Some(ref path) = self.cache_path else {
            return;
        };

        let data = unsafe {
            match self.device.vk().get_pipeline_cache_data(self.cache) {
                Ok(data) => data,
                Err(err) => {
                    tracing::debug!("Failed to get Vulkan pipeline cache data: {:?}", err);
                    return;
                }
            }
        };

        if data.is_empty() {
            return;
        }

        let temp_path = path.with_extension("tmp");
        if let Err(err) = std::fs::write(&temp_path, &data) {
            tracing::debug!(
                "Failed to write Vulkan pipeline cache to {:?}: {:?}",
                temp_path,
                err
            );
            return;
        }

        let _ = std::fs::rename(&temp_path, path);
    }

    fn get_or_create_format_pipelines(
        &self,
        format: vk::Format,
        has_depth: bool,
    ) -> Result<FormatPipelines, Error> {
        let mut guard = self.format_pipelines.lock().unwrap();
        if let Some(&p) = guard.get(&(format, has_depth)) {
            return Ok(p);
        }

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

        let outline_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.outline_frag);

        let spec_entry = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(std::mem::size_of::<u32>())];

        let spec_generic_data = SPEC_MODE_GENERIC.to_ne_bytes();
        let spec_generic = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_generic_data);
        let hdr_tex_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.hdr_tex_frag)
            .specialization_info(&spec_generic);

        let spec_passthrough_data = SPEC_MODE_PASSTHROUGH.to_ne_bytes();
        let spec_passthrough = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_passthrough_data);
        let hdr_passthrough_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.hdr_tex_frag)
            .specialization_info(&spec_passthrough);

        let spec_sdr_data = SPEC_MODE_SDR.to_ne_bytes();
        let spec_sdr = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_sdr_data);
        let hdr_sdr_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.hdr_tex_frag)
            .specialization_info(&spec_sdr);

        let spec_pq_data = SPEC_MODE_PQ.to_ne_bytes();
        let spec_pq = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_pq_data);
        let hdr_pq_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.hdr_tex_frag)
            .specialization_info(&spec_pq);

        let spec_lut3d_data = SPEC_MODE_LUT3D.to_ne_bytes();
        let spec_lut3d = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_lut3d_data);
        let hdr_lut3d_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.hdr_tex_frag)
            .specialization_info(&spec_lut3d);

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
        let depth_format = if has_depth {
            vk::Format::D16_UNORM
        } else {
            vk::Format::UNDEFINED
        };

        let opaque_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(has_depth)
            .depth_write_enable(has_depth)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let blend_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(has_depth)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

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
        let hdr_passthrough_stages = [vertex_stage, hdr_passthrough_stage];
        let hdr_sdr_stages = [vertex_stage, hdr_sdr_stage];
        let hdr_pq_stages = [vertex_stage, hdr_pq_stage];
        let hdr_lut3d_stages = [vertex_stage, hdr_lut3d_stage];
        let outline_stages = [vertex_stage, outline_stage];

        let base_ci = vk::GraphicsPipelineCreateInfo::default()
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .dynamic_state(&dynamic_state);

        let specs = [
            // 0, 1: clear
            (&clear_stages[..], false, self.clear_layout),
            (&clear_stages[..], true, self.clear_layout),
            // 2, 3: tex
            (&tex_stages[..], false, self.tex_layout),
            (&tex_stages[..], true, self.tex_layout),
            // 4, 5: hdr_tex
            (&hdr_tex_stages[..], false, self.hdr_tex_layout),
            (&hdr_tex_stages[..], true, self.hdr_tex_layout),
            // 6, 7: hdr_passthrough
            (&hdr_passthrough_stages[..], false, self.hdr_tex_layout),
            (&hdr_passthrough_stages[..], true, self.hdr_tex_layout),
            // 8, 9: hdr_sdr
            (&hdr_sdr_stages[..], false, self.hdr_tex_layout),
            (&hdr_sdr_stages[..], true, self.hdr_tex_layout),
            // 10, 11: hdr_pq
            (&hdr_pq_stages[..], false, self.hdr_tex_layout),
            (&hdr_pq_stages[..], true, self.hdr_tex_layout),
            // 12, 13: hdr_lut3d
            (&hdr_lut3d_stages[..], false, self.hdr_tex_layout),
            (&hdr_lut3d_stages[..], true, self.hdr_tex_layout),
            // 14, 15: outline
            (&outline_stages[..], false, self.outline_layout),
            (&outline_stages[..], true, self.outline_layout),
        ];

        let mut rendering_infos = (0..specs.len())
            .map(|_| {
                vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(&color_attachment_formats)
                    .depth_attachment_format(depth_format)
            })
            .collect::<Vec<_>>();

        let create_infos = specs
            .iter()
            .zip(rendering_infos.iter_mut())
            .map(|((stages, is_blend, layout), r_info)| {
                let (blend_state, depth_state) = if *is_blend {
                    (&alpha_blend_state, &blend_depth_stencil)
                } else {
                    (&opaque_blend_state, &opaque_depth_stencil)
                };
                base_ci
                    .push_next(r_info)
                    .stages(stages)
                    .color_blend_state(blend_state)
                    .depth_stencil_state(depth_state)
                    .layout(*layout)
            })
            .collect::<Vec<_>>();

        let pipelines = unsafe {
            self.device
                .vk()
                .create_graphics_pipelines(self.cache, &create_infos, None)
                .map_err(|(pipelines, res)| {
                    for pipeline in pipelines {
                        self.device.vk().destroy_pipeline(pipeline, None);
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
            hdr_passthrough_pipeline: pipelines[6],
            hdr_passthrough_blend_pipeline: pipelines[7],
            hdr_sdr_pipeline: pipelines[8],
            hdr_sdr_blend_pipeline: pipelines[9],
            hdr_pq_pipeline: pipelines[10],
            hdr_pq_blend_pipeline: pipelines[11],
            hdr_lut3d_pipeline: pipelines[12],
            hdr_lut3d_blend_pipeline: pipelines[13],
            outline_pipeline: pipelines[14],
            outline_blend_pipeline: pipelines[15],
        };

        guard.insert((format, has_depth), format_pipelines);
        drop(guard);
        self.save_cache();
        Ok(format_pipelines)
    }
}

impl Drop for SharedPipelines {
    fn drop(&mut self) {
        self.save_cache();
        let vk = self.device.vk();
        unsafe {
            for (_, p) in self.format_pipelines.get_mut().unwrap().drain() {
                vk.destroy_pipeline(p.clear_pipeline, None);
                vk.destroy_pipeline(p.clear_blend_pipeline, None);
                vk.destroy_pipeline(p.tex_pipeline, None);
                vk.destroy_pipeline(p.tex_blend_pipeline, None);
                vk.destroy_pipeline(p.hdr_tex_pipeline, None);
                vk.destroy_pipeline(p.hdr_tex_blend_pipeline, None);
                vk.destroy_pipeline(p.hdr_passthrough_pipeline, None);
                vk.destroy_pipeline(p.hdr_passthrough_blend_pipeline, None);
                vk.destroy_pipeline(p.hdr_sdr_pipeline, None);
                vk.destroy_pipeline(p.hdr_sdr_blend_pipeline, None);
                vk.destroy_pipeline(p.hdr_pq_pipeline, None);
                vk.destroy_pipeline(p.hdr_pq_blend_pipeline, None);
                vk.destroy_pipeline(p.hdr_lut3d_pipeline, None);
                vk.destroy_pipeline(p.hdr_lut3d_blend_pipeline, None);
                vk.destroy_pipeline(p.outline_pipeline, None);
                vk.destroy_pipeline(p.outline_blend_pipeline, None);
            }

            vk.destroy_sampler(self.texture_sampler, None);
            vk.destroy_sampler(self.texture_sampler_nearest, None);

            vk.destroy_shader_module(self.quad_vert, None);
            vk.destroy_shader_module(self.clear_frag, None);
            vk.destroy_shader_module(self.tex_frag, None);
            vk.destroy_shader_module(self.hdr_tex_frag, None);
            vk.destroy_shader_module(self.outline_frag, None);

            vk.destroy_pipeline_layout(self.clear_layout, None);
            vk.destroy_pipeline_layout(self.tex_layout, None);
            vk.destroy_pipeline_layout(self.hdr_tex_layout, None);
            vk.destroy_pipeline_layout(self.outline_layout, None);

            vk.destroy_descriptor_set_layout(self.tex_descriptor_set_layout, None);
            vk.destroy_descriptor_set_layout(self.hdr_tex_descriptor_set_layout, None);

            vk.destroy_pipeline_cache(self.cache, None);
        }
    }
}

impl Pipelines {
    pub fn from_shared(shared: Arc<SharedPipelines>) -> Self {
        let tex_desc_pool =
            DescriptorAllocator::new(&shared.device, shared.tex_descriptor_set_layout, &*TEX_SIZES);
        let hdr_tex_desc_pool = DescriptorAllocator::new(
            &shared.device,
            shared.hdr_tex_descriptor_set_layout,
            &*HDR_TEX_SIZES,
        );

        Pipelines {
            tex_desc_pool,
            hdr_tex_desc_pool,
            local_format_pipelines: HashMap::new(),
            ycbcr_entries: HashMap::new(),
            shared,
        }
    }

    pub fn shared(&self) -> &Arc<SharedPipelines> {
        &self.shared
    }

    pub fn get_or_create_format_pipelines(
        &mut self,
        format: vk::Format,
        has_depth: bool,
    ) -> Result<FormatPipelines, Error> {
        if let Some(&p) = self.local_format_pipelines.get(&(format, has_depth)) {
            return Ok(p);
        }

        let p = self.shared.get_or_create_format_pipelines(format, has_depth)?;
        self.local_format_pipelines.insert((format, has_depth), p);
        Ok(p)
    }

    pub fn clear_pipeline_layout(&self) -> &PipelineLayout {
        &self.shared.clear_layout
    }

    pub fn tex_pipeline_layout(&self) -> &PipelineLayout {
        &self.shared.tex_layout
    }

    pub fn hdr_tex_pipeline_layout(&self) -> &PipelineLayout {
        &self.shared.hdr_tex_layout
    }

    pub fn outline_pipeline_layout(&self) -> &PipelineLayout {
        &self.shared.outline_layout
    }

    pub fn alloc_descriptor_set(&mut self, shader: BuiltinShader) -> Result<DescriptorSet, Error> {
        match shader {
            BuiltinShader::Clear => self.tex_desc_pool.alloc_descriptor_set().map_err(Into::into),
            BuiltinShader::Texture => self.tex_desc_pool.alloc_descriptor_set().map_err(Into::into),
            BuiltinShader::HdrTexture => self.hdr_tex_desc_pool.alloc_descriptor_set().map_err(Into::into),
        }
    }

    pub fn get_or_create_ycbcr_entry(
        &mut self,
        key: YcbcrKey,
        sampler: vk::Sampler,
    ) -> Result<&mut YcbcrEntry, Error> {
        if !self.ycbcr_entries.contains_key(&key) {
            let device = &self.shared.device;

            let layout_flags = if device.vk_khr_push_descriptor().is_some() {
                vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
            } else {
                vk::DescriptorSetLayoutCreateFlags::empty()
            };

            // Texture pipeline layout (1 descriptor set: sampler2D with immutable sampler)
            let immutable_samplers = [sampler];
            let tex_binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(ShaderStageFlags::FRAGMENT)
                .descriptor_count(1)
                .immutable_samplers(&immutable_samplers)];
            let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(layout_flags)
                .bindings(&tex_binding);
            let tex_descriptor_set_layout = unsafe {
                device
                    .vk()
                    .create_descriptor_set_layout(&create_info, None)
                    .map_err(Error::DescriptorSetLayout)?
            };

            let constants = [vk::PushConstantRange::default()
                .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(std::mem::size_of::<TexPushConstants>() as u32)];
            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&tex_descriptor_set_layout))
                .push_constant_ranges(&constants);
            let tex_layout = unsafe {
                device
                    .vk()
                    .create_pipeline_layout(&create_info, None)
                    .map_err(Error::PipelineLayout)?
            };
            let tex_desc_pool = DescriptorAllocator::new(device, tex_descriptor_set_layout, &*TEX_SIZES);

            // HDR Texture pipeline layout (binding 0: immutable sampler, binding 1: standard sampler3D)
            let hdr_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(ShaderStageFlags::FRAGMENT)
                    .descriptor_count(1)
                    .immutable_samplers(&immutable_samplers),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(ShaderStageFlags::FRAGMENT)
                    .descriptor_count(1),
            ];
            let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(layout_flags)
                .bindings(&hdr_bindings);
            let hdr_tex_descriptor_set_layout = unsafe {
                device
                    .vk()
                    .create_descriptor_set_layout(&create_info, None)
                    .map_err(Error::DescriptorSetLayout)?
            };

            let constants = [vk::PushConstantRange::default()
                .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(std::mem::size_of::<HdrTexPushConstants>() as u32)];
            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&hdr_tex_descriptor_set_layout))
                .push_constant_ranges(&constants);
            let hdr_tex_layout = unsafe {
                device
                    .vk()
                    .create_pipeline_layout(&create_info, None)
                    .map_err(Error::PipelineLayout)?
            };
            let hdr_tex_desc_pool =
                DescriptorAllocator::new(device, hdr_tex_descriptor_set_layout, &*HDR_TEX_SIZES);

            self.ycbcr_entries.insert(
                key,
                YcbcrEntry {
                    tex_descriptor_set_layout,
                    tex_layout,
                    tex_desc_pool,
                    hdr_tex_descriptor_set_layout,
                    hdr_tex_layout,
                    hdr_tex_desc_pool,
                    format_pipelines: HashMap::new(),
                },
            );
        }

        Ok(self.ycbcr_entries.get_mut(&key).unwrap())
    }

    pub fn get_or_create_ycbcr_format_pipelines(
        &mut self,
        key: YcbcrKey,
        sampler: vk::Sampler,
        format: vk::Format,
        has_depth: bool,
    ) -> Result<YcbcrFormatPipelines, Error> {
        let (tex_layout, hdr_tex_layout) = {
            let entry = self.get_or_create_ycbcr_entry(key, sampler)?;
            if let Some(&p) = entry.format_pipelines.get(&(format, has_depth)) {
                return Ok(p);
            }
            (entry.tex_layout, entry.hdr_tex_layout)
        };

        let device = &self.shared.device;

        let vertex_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::VERTEX)
            .name(c"main")
            .module(self.shared.quad_vert);

        let tex_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.shared.tex_frag);

        let spec_entry = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(std::mem::size_of::<u32>())];

        let spec_generic_data = SPEC_MODE_GENERIC.to_ne_bytes();
        let spec_generic = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_generic_data);
        let hdr_tex_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.shared.hdr_tex_frag)
            .specialization_info(&spec_generic);

        let spec_passthrough_data = SPEC_MODE_PASSTHROUGH.to_ne_bytes();
        let spec_passthrough = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_passthrough_data);
        let hdr_passthrough_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.shared.hdr_tex_frag)
            .specialization_info(&spec_passthrough);

        let spec_sdr_data = SPEC_MODE_SDR.to_ne_bytes();
        let spec_sdr = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_sdr_data);
        let hdr_sdr_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.shared.hdr_tex_frag)
            .specialization_info(&spec_sdr);

        let spec_pq_data = SPEC_MODE_PQ.to_ne_bytes();
        let spec_pq = vk::SpecializationInfo::default()
            .map_entries(&spec_entry)
            .data(&spec_pq_data);
        let hdr_pq_stage = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .module(self.shared.hdr_tex_frag)
            .specialization_info(&spec_pq);

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
        let depth_format = if has_depth {
            vk::Format::D16_UNORM
        } else {
            vk::Format::UNDEFINED
        };

        let tex_stages = [vertex_stage, tex_stage];
        let hdr_tex_stages = [vertex_stage, hdr_tex_stage];
        let hdr_passthrough_stages = [vertex_stage, hdr_passthrough_stage];
        let hdr_sdr_stages = [vertex_stage, hdr_sdr_stage];
        let hdr_pq_stages = [vertex_stage, hdr_pq_stage];

        let opaque_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let opaque_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&opaque_blend_attachment);

        let alpha_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let alpha_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&alpha_blend_attachment);

        let opaque_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(has_depth)
            .depth_write_enable(has_depth)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let blend_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(has_depth)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let base_ci = vk::GraphicsPipelineCreateInfo::default()
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .dynamic_state(&dynamic_state);

        let specs = [
            // 0, 1: tex
            (&tex_stages[..], false, tex_layout),
            (&tex_stages[..], true, tex_layout),
            // 2, 3: hdr_tex
            (&hdr_tex_stages[..], false, hdr_tex_layout),
            (&hdr_tex_stages[..], true, hdr_tex_layout),
            // 4, 5: hdr_passthrough
            (&hdr_passthrough_stages[..], false, hdr_tex_layout),
            (&hdr_passthrough_stages[..], true, hdr_tex_layout),
            // 6, 7: hdr_sdr
            (&hdr_sdr_stages[..], false, hdr_tex_layout),
            (&hdr_sdr_stages[..], true, hdr_tex_layout),
            // 8, 9: hdr_pq
            (&hdr_pq_stages[..], false, hdr_tex_layout),
            (&hdr_pq_stages[..], true, hdr_tex_layout),
        ];

        let mut rendering_infos = (0..specs.len())
            .map(|_| {
                vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(&color_attachment_formats)
                    .depth_attachment_format(depth_format)
            })
            .collect::<Vec<_>>();

        let create_infos = specs
            .iter()
            .zip(rendering_infos.iter_mut())
            .map(|((stages, is_blend, layout), r_info)| {
                let (blend_state, depth_state) = if *is_blend {
                    (&alpha_blend_state, &blend_depth_stencil)
                } else {
                    (&opaque_blend_state, &opaque_depth_stencil)
                };
                base_ci
                    .push_next(r_info)
                    .stages(stages)
                    .color_blend_state(blend_state)
                    .depth_stencil_state(depth_state)
                    .layout(*layout)
            })
            .collect::<Vec<_>>();

        let pipelines = unsafe {
            device
                .vk()
                .create_graphics_pipelines(self.shared.cache, &create_infos, None)
                .map_err(|(pipelines, res)| {
                    for pipeline in pipelines {
                        device.vk().destroy_pipeline(pipeline, None);
                    }
                    res
                })
                .map_err(Error::Pipeline)?
        };

        let format_pipelines = YcbcrFormatPipelines {
            tex_pipeline: pipelines[0],
            tex_blend_pipeline: pipelines[1],
            hdr_tex_pipeline: pipelines[2],
            hdr_tex_blend_pipeline: pipelines[3],
            hdr_passthrough_pipeline: pipelines[4],
            hdr_passthrough_blend_pipeline: pipelines[5],
            hdr_sdr_pipeline: pipelines[6],
            hdr_sdr_blend_pipeline: pipelines[7],
            hdr_pq_pipeline: pipelines[8],
            hdr_pq_blend_pipeline: pipelines[9],
        };

        let entry = self.ycbcr_entries.get_mut(&key).unwrap();
        entry
            .format_pipelines
            .insert((format, has_depth), format_pipelines);
        self.shared.save_cache();
        Ok(format_pipelines)
    }

    pub fn ycbcr_pipeline_layout(&self, key: YcbcrKey, is_hdr: bool) -> Option<PipelineLayout> {
        let entry = self.ycbcr_entries.get(&key)?;
        Some(if is_hdr {
            entry.hdr_tex_layout
        } else {
            entry.tex_layout
        })
    }

    pub fn alloc_ycbcr_descriptor_set(
        &mut self,
        key: YcbcrKey,
        sampler: vk::Sampler,
        is_hdr: bool,
    ) -> Result<DescriptorSet, Error> {
        let entry = self.get_or_create_ycbcr_entry(key, sampler)?;
        if is_hdr {
            entry.hdr_tex_desc_pool.alloc_descriptor_set().map_err(Into::into)
        } else {
            entry.tex_desc_pool.alloc_descriptor_set().map_err(Into::into)
        }
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        let vk = self.shared.device.vk();
        unsafe {
            for (_, mut entry) in self.ycbcr_entries.drain() {
                for (_, p) in entry.format_pipelines.drain() {
                    vk.destroy_pipeline(p.tex_pipeline, None);
                    vk.destroy_pipeline(p.tex_blend_pipeline, None);
                    vk.destroy_pipeline(p.hdr_tex_pipeline, None);
                    vk.destroy_pipeline(p.hdr_tex_blend_pipeline, None);
                    vk.destroy_pipeline(p.hdr_passthrough_pipeline, None);
                    vk.destroy_pipeline(p.hdr_passthrough_blend_pipeline, None);
                    vk.destroy_pipeline(p.hdr_sdr_pipeline, None);
                    vk.destroy_pipeline(p.hdr_sdr_blend_pipeline, None);
                    vk.destroy_pipeline(p.hdr_pq_pipeline, None);
                    vk.destroy_pipeline(p.hdr_pq_blend_pipeline, None);
                }
                drop(entry.tex_desc_pool);
                drop(entry.hdr_tex_desc_pool);
                vk.destroy_pipeline_layout(entry.tex_layout, None);
                vk.destroy_pipeline_layout(entry.hdr_tex_layout, None);
                vk.destroy_descriptor_set_layout(entry.tex_descriptor_set_layout, None);
                vk.destroy_descriptor_set_layout(entry.hdr_tex_descriptor_set_layout, None);
            }
        }
    }
}
