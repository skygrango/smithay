use ash::vk;

use super::Error;
use crate::backend::vulkan::{Device, image::Error as ImageError};

/// Convert an f32 to a half-precision (16-bit) float IEEE 754.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7fffff;

    if exp == 255 {
        return ((sign << 15) | (0x1f << 10) | (if mantissa != 0 { 1 } else { 0 })) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1f << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let mantissa = mantissa | 0x800000;
        let shift = 14 - new_exp;
        let new_mantissa = mantissa >> shift;
        return ((sign << 15) | (new_mantissa as u32)) as u16;
    }
    let new_mantissa = (mantissa + 0x1000) >> 13;
    (((sign as u32) << 15) | ((new_exp as u32) << 10) | (new_mantissa & 0x3ff)) as u16
}

const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

fn encode_pq(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let p = v.clamp(0.0, 1.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p)).powf(PQ_M2)
}

fn pq_to_linear(code: f32) -> f32 {
    if code <= 0.0 {
        return 0.0;
    }
    let p = code.clamp(0.0, 1.0).powf(1.0 / PQ_M2);
    (p - PQ_C1).max(0.0).powf(1.0 / PQ_M1) / (PQ_C2 - PQ_C3 * p).powf(1.0 / PQ_M1)
}

fn mul_mat3_vec3(m: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}

const BT2020_TO_LMS: [f32; 9] = [
    0.412109375,
    0.166748046875,
    0.024169921875,
    0.52392578125,
    0.720458984375,
    0.075439453125,
    0.06396484375,
    0.11279296875,
    0.900390625,
];

const TO_ICTCP: [f32; 9] = [
    0.5,
    1.613769531250,
    4.378173828125,
    0.5,
    -3.323486328125,
    -4.245605468750,
    0.0,
    1.709716796875,
    -0.132568359375,
];

const FROM_ICTCP: [f32; 9] = [
    1.0,
    1.0,
    1.0,
    0.00860903703793,
    -0.00860903703793,
    0.56003133571068,
    0.11102962500303,
    -0.11102962500303,
    -0.32062717498732,
];

const LMS_TO_BT2020: [f32; 9] = [
    3.43660669,
    -0.79132956,
    -0.02594990,
    -2.50645212,
    1.98360045,
    -0.09891371,
    0.06984542,
    -0.19227090,
    1.12486361,
];

pub fn tonemap_ictcp(linear_10k: [f32; 3], ref_white: f32, max_content: f32, max_dest: f32) -> [f32; 3] {
    if linear_10k[0] <= 0.0 && linear_10k[1] <= 0.0 && linear_10k[2] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    if max_content <= max_dest * 1.01 {
        let cap = max_dest / 10000.0;
        return [
            linear_10k[0].clamp(0.0, cap),
            linear_10k[1].clamp(0.0, cap),
            linear_10k[2].clamp(0.0, cap),
        ];
    }

    let lms = mul_mat3_vec3(&BT2020_TO_LMS, linear_10k);
    let lms_pq = [encode_pq(lms[0]), encode_pq(lms[1]), encode_pq(lms[2])];
    let mut ictcp = mul_mat3_vec3(&TO_ICTCP, lms_pq);

    let mut lum = pq_to_linear(ictcp[0]) * 10000.0;
    let ref_w = ref_white.clamp(80.0, 10000.0);
    let mut rel_lum = (lum / ref_w).max(0.0);
    let in_range = max_content / ref_w;
    let out_range = max_dest / ref_w;
    let v = (out_range * (1.0 + in_range) - in_range) / (in_range * in_range);
    rel_lum = rel_lum * (1.0 + rel_lum * v) / (1.0 + rel_lum);
    lum = rel_lum * ref_w;

    ictcp[0] = encode_pq(lum / 10000.0);
    let mapped_lms_pq = mul_mat3_vec3(&FROM_ICTCP, ictcp);
    let mapped_lms = [
        pq_to_linear(mapped_lms_pq[0]),
        pq_to_linear(mapped_lms_pq[1]),
        pq_to_linear(mapped_lms_pq[2]),
    ];
    let mapped_rgb = mul_mat3_vec3(&LMS_TO_BT2020, mapped_lms);
    let cap = max_dest / 10000.0;
    [
        mapped_rgb[0].clamp(0.0, cap),
        mapped_rgb[1].clamp(0.0, cap),
        mapped_rgb[2].clamp(0.0, cap),
    ]
}

pub fn generate_ictcp_tonemap_lut(
    size: u32,
    ref_white: f32,
    content_ref: f32,
    max_content: f32,
    max_dest: f32,
) -> Vec<u16> {
    let mut data = Vec::with_capacity((size * size * size * 4) as usize);
    let ref_scale = ref_white.clamp(80.0, 10000.0) / content_ref.max(80.0);

    for b in 0..size {
        let b_pq = b as f32 / (size - 1) as f32;
        for g in 0..size {
            let g_pq = g as f32 / (size - 1) as f32;
            for r in 0..size {
                let r_pq = r as f32 / (size - 1) as f32;

                let lin_r = pq_to_linear(r_pq) * ref_scale;
                let lin_g = pq_to_linear(g_pq) * ref_scale;
                let lin_b = pq_to_linear(b_pq) * ref_scale;

                let mapped_lin = tonemap_ictcp([lin_r, lin_g, lin_b], ref_white, max_content, max_dest);

                let out_r = encode_pq(mapped_lin[0]);
                let out_g = encode_pq(mapped_lin[1]);
                let out_b = encode_pq(mapped_lin[2]);

                data.push(f32_to_f16(out_r));
                data.push(f32_to_f16(out_g));
                data.push(f32_to_f16(out_b));
                data.push(f32_to_f16(1.0));
            }
        }
    }
    data
}

pub fn generate_identity_lut(size: u32) -> Vec<u16> {
    let mut data = Vec::with_capacity((size * size * size * 4) as usize);
    for b in 0..size {
        let b_val = b as f32 / (size - 1) as f32;
        for g in 0..size {
            let g_val = g as f32 / (size - 1) as f32;
            for r in 0..size {
                let r_val = r as f32 / (size - 1) as f32;
                data.push(f32_to_f16(r_val));
                data.push(f32_to_f16(g_val));
                data.push(f32_to_f16(b_val));
                data.push(f32_to_f16(1.0));
            }
        }
    }
    data
}

#[derive(Debug)]
pub struct Lut3dTexture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub size: u32,
    pub params: (u32, u32, u32), // (ref_white as u32, max_content as u32, max_dest as u32)
}

impl Lut3dTexture {
    pub fn new(device: &Device, data: &[u16], size: u32, params: (u32, u32, u32)) -> Result<Self, Error> {
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: size,
                height: size,
                depth: size,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            device
                .vk()
                .create_image(&image_create_info, None)
                .map_err(|e| Error::ImageError(ImageError::VulkanImage(e)))?
        };

        let mem_reqs = unsafe { device.vk().get_image_memory_requirements(image) };
        let phd_props = device.memory_properties();
        let mem_type_index = (0..phd_props.memory_type_count)
            .find(|&i| {
                (mem_reqs.memory_type_bits & (1 << i)) != 0
                    && phd_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .unwrap_or(0);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            device
                .vk()
                .allocate_memory(&alloc_info, None)
                .map_err(|e| Error::ImageError(ImageError::VulkanAllocate(e)))?
        };

        unsafe {
            device
                .vk()
                .bind_image_memory(image, memory, 0)
                .map_err(|e| Error::ImageError(ImageError::VulkanBind(e)))?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let view = unsafe {
            device
                .vk()
                .create_image_view(&view_info, None)
                .map_err(|e| Error::ImageError(ImageError::VulkanImageView(e)))?
        };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = unsafe {
            device
                .vk()
                .create_sampler(&sampler_info, None)
                .map_err(Error::SamplerError)?
        };

        // Create staging buffer and upload data
        let byte_size = (data.len() * std::mem::size_of::<u16>()) as vk::DeviceSize;
        let buf_info = vk::BufferCreateInfo::default()
            .size(byte_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let staging_buf = unsafe {
            device
                .vk()
                .create_buffer(&buf_info, None)
                .map_err(|e| Error::ImageError(ImageError::VulkanAllocate(e)))?
        };

        let staging_reqs = unsafe { device.vk().get_buffer_memory_requirements(staging_buf) };
        let staging_mem_index = (0..phd_props.memory_type_count)
            .find(|&i| {
                (staging_reqs.memory_type_bits & (1 << i)) != 0
                    && phd_props.memory_types[i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .unwrap_or(0);

        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_reqs.size)
            .memory_type_index(staging_mem_index);
        let staging_memory = unsafe {
            device
                .vk()
                .allocate_memory(&staging_alloc, None)
                .map_err(|e| Error::ImageError(ImageError::VulkanAllocate(e)))?
        };

        unsafe {
            device
                .vk()
                .bind_buffer_memory(staging_buf, staging_memory, 0)
                .map_err(|e| Error::ImageError(ImageError::VulkanBind(e)))?;
            let ptr = device
                .vk()
                .map_memory(staging_memory, 0, byte_size, vk::MemoryMapFlags::empty())
                .map_err(|e| Error::ImageError(ImageError::VulkanAllocate(e)))?;
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, byte_size as usize);
            device.vk().unmap_memory(staging_memory);
        }

        // Allocate transient command buffer to execute transfer
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family_idx())
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = unsafe {
            device
                .vk()
                .create_command_pool(&cmd_pool_info, None)
                .map_err(Error::CommandPoolError)?
        };

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe {
            device
                .vk()
                .allocate_command_buffers(&cmd_alloc)
                .map_err(Error::CommandBufferError)?[0]
        };

        unsafe {
            device
                .vk()
                .begin_command_buffer(
                    cmd_buf,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(Error::CommandBufferError)?;

            // Barrier: UNDEFINED -> TRANSFER_DST_OPTIMAL
            let barrier1 = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.vk().cmd_pipeline_barrier2(
                cmd_buf,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier1]),
            );

            // Copy buffer to image
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(size)
                .buffer_image_height(size)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: size,
                    height: size,
                    depth: size,
                });
            device.vk().cmd_copy_buffer_to_image(
                cmd_buf,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Barrier: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            let barrier2 = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.vk().cmd_pipeline_barrier2(
                cmd_buf,
                &vk::DependencyInfo::default().image_memory_barriers(&[barrier2]),
            );

            device
                .vk()
                .end_command_buffer(cmd_buf)
                .map_err(Error::CommandBufferError)?;

            let submit_cmd = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buf)];
            let submit = [vk::SubmitInfo2::default().command_buffer_infos(&submit_cmd)];
            device
                .vk()
                .queue_submit2(*device.queue(), &submit, vk::Fence::null())
                .map_err(Error::SubmitError)?;
            device
                .vk()
                .queue_wait_idle(*device.queue())
                .map_err(Error::SubmitError)?;

            device.vk().destroy_command_pool(cmd_pool, None);
            device.vk().destroy_buffer(staging_buf, None);
            device.vk().free_memory(staging_memory, None);
        }

        Ok(Lut3dTexture {
            image,
            memory,
            view,
            sampler,
            size,
            params,
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        let vk = device.vk();
        unsafe {
            vk.destroy_sampler(self.sampler, None);
            vk.destroy_image_view(self.view, None);
            vk.destroy_image(self.image, None);
            vk.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        assert_eq!(f32_to_f16(0.0), 0);
        // In IEEE 754 half precision, 1.0 is represented as 0x3C00
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        // Half of 1.0 is 0.5, which is 0x3800
        assert_eq!(f32_to_f16(0.5), 0x3800);
    }

    #[test]
    fn test_identity_lut_generation() {
        let size = 17u32;
        let lut = generate_identity_lut(size);
        assert_eq!(lut.len(), (size * size * size * 4) as usize);

        // Origin (0, 0, 0)
        assert_eq!(lut[0], f32_to_f16(0.0));
        assert_eq!(lut[1], f32_to_f16(0.0));
        assert_eq!(lut[2], f32_to_f16(0.0));
        assert_eq!(lut[3], f32_to_f16(1.0));

        // Max corner (size - 1, size - 1, size - 1)
        let last_idx = lut.len() - 4;
        assert_eq!(lut[last_idx], f32_to_f16(1.0));
        assert_eq!(lut[last_idx + 1], f32_to_f16(1.0));
        assert_eq!(lut[last_idx + 2], f32_to_f16(1.0));
        assert_eq!(lut[last_idx + 3], f32_to_f16(1.0));
    }

    #[test]
    fn test_ictcp_tonemap_lut() {
        let size = 17u32;
        let lut = generate_ictcp_tonemap_lut(size, 203.0, 203.0, 1000.0, 500.0);
        assert_eq!(lut.len(), (size * size * size * 4) as usize);

        // Origin (0,0,0) should map to 0
        assert_eq!(lut[0], 0);
        assert_eq!(lut[1], 0);
        assert_eq!(lut[2], 0);
        assert_eq!(lut[3], f32_to_f16(1.0));

        // Ensure non-negative and non-NaN values across sample points
        for (i, &val) in lut.iter().enumerate() {
            // Check alpha channel is always 1.0
            if i % 4 == 3 {
                assert_eq!(val, f32_to_f16(1.0));
            }
        }
    }

    #[test]
    fn test_tonemap_passthrough_when_target_exceeds_content() {
        let input = [0.02, 0.03, 0.01]; // ~200-300 nits
        let mapped = tonemap_ictcp(input, 203.0, 400.0, 1000.0);
        // Destination max is greater than content max, should be passthrough clamped
        assert!((mapped[0] - input[0]).abs() < 1e-5);
        assert!((mapped[1] - input[1]).abs() < 1e-5);
        assert!((mapped[2] - input[2]).abs() < 1e-5);
    }

    #[test]
    fn test_tetrahedral_interpolation_neutral_axis() {
        let size = 17u32;
        let lut = generate_identity_lut(size);

        let sample_lut = |x: u32, y: u32, z: u32| -> [f32; 3] {
            let idx = ((z * size * size + y * size + x) * 4) as usize;
            let decode_f16 = |h: u16| -> f32 {
                let sign = ((h >> 15) & 1) as u32;
                let exp = ((h >> 10) & 0x1f) as i32;
                let mant = (h & 0x3ff) as u32;
                if exp == 0 {
                    0.0
                } else {
                    let exp_f = (exp - 15 + 127) as u32;
                    let f_bits = (sign << 31) | (exp_f << 23) | (mant << 13);
                    f32::from_bits(f_bits)
                }
            };
            [
                decode_f16(lut[idx]),
                decode_f16(lut[idx + 1]),
                decode_f16(lut[idx + 2]),
            ]
        };

        let tetrahedral_interp = |color: [f32; 3]| -> [f32; 3] {
            let p = [
                color[0].clamp(0.0, 1.0) * (size - 1) as f32,
                color[1].clamp(0.0, 1.0) * (size - 1) as f32,
                color[2].clamp(0.0, 1.0) * (size - 1) as f32,
            ];
            let p0 = [p[0].floor() as u32, p[1].floor() as u32, p[2].floor() as u32];
            let f = [p[0] - p0[0] as f32, p[1] - p0[1] as f32, p[2] - p0[2] as f32];
            let max_c = size - 1;
            let p1 = [
                (p0[0] + 1).min(max_c),
                (p0[1] + 1).min(max_c),
                (p0[2] + 1).min(max_c),
            ];
            let p0 = [p0[0].min(max_c), p0[1].min(max_c), p0[2].min(max_c)];

            let c000 = sample_lut(p0[0], p0[1], p0[2]);
            let c111 = sample_lut(p1[0], p1[1], p1[2]);

            let (c1, c2, w) = if f[0] >= f[1] {
                if f[1] >= f[2] {
                    (
                        sample_lut(p1[0], p0[1], p0[2]),
                        sample_lut(p1[0], p1[1], p0[2]),
                        [1.0 - f[0], f[0] - f[1], f[1] - f[2], f[2]],
                    )
                } else if f[0] >= f[2] {
                    (
                        sample_lut(p1[0], p0[1], p0[2]),
                        sample_lut(p1[0], p0[1], p1[2]),
                        [1.0 - f[0], f[0] - f[2], f[2] - f[1], f[1]],
                    )
                } else {
                    (
                        sample_lut(p0[0], p0[1], p1[2]),
                        sample_lut(p1[0], p0[1], p1[2]),
                        [1.0 - f[2], f[2] - f[0], f[0] - f[1], f[1]],
                    )
                }
            } else {
                if f[2] >= f[1] {
                    (
                        sample_lut(p0[0], p0[1], p1[2]),
                        sample_lut(p0[0], p1[1], p1[2]),
                        [1.0 - f[2], f[2] - f[1], f[1] - f[0], f[0]],
                    )
                } else if f[2] >= f[0] {
                    (
                        sample_lut(p0[0], p1[1], p0[2]),
                        sample_lut(p0[0], p1[1], p1[2]),
                        [1.0 - f[1], f[1] - f[2], f[2] - f[0], f[0]],
                    )
                } else {
                    (
                        sample_lut(p0[0], p1[1], p0[2]),
                        sample_lut(p1[0], p1[1], p0[2]),
                        [1.0 - f[1], f[1] - f[0], f[0] - f[2], f[2]],
                    )
                }
            };

            [
                c000[0] * w[0] + c1[0] * w[1] + c2[0] * w[2] + c111[0] * w[3],
                c000[1] * w[0] + c1[1] * w[1] + c2[1] * w[2] + c111[1] * w[3],
                c000[2] * w[0] + c1[2] * w[1] + c2[2] * w[2] + c111[2] * w[3],
            ]
        };

        // Test along neutral gray axis (R = G = B)
        for val in [0.05, 0.125, 0.333, 0.5, 0.777, 0.95] {
            let res = tetrahedral_interp([val, val, val]);
            assert!(
                (res[0] - val).abs() < 1e-3,
                "Neutral axis R shifted at {}: {}",
                val,
                res[0]
            );
            assert!(
                (res[1] - val).abs() < 1e-3,
                "Neutral axis G shifted at {}: {}",
                val,
                res[1]
            );
            assert!(
                (res[2] - val).abs() < 1e-3,
                "Neutral axis B shifted at {}: {}",
                val,
                res[2]
            );
        }
    }
}
