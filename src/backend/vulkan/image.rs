use std::{fmt, io, os::fd::IntoRawFd, sync::Arc};

pub use ash::vk::ImageUsageFlags;
use ash::vk::{self, ImageTiling, MemoryPropertyFlags};
#[cfg(feature = "backend_drm")]
use drm::node::DrmNode;

use super::device::WeakDevice;
use crate::backend::{
    allocator::{Buffer, Format, Fourcc, Modifier, dmabuf::Dmabuf, format::has_alpha},
    vulkan::{Device, format::component_mapping_for_format},
};

/// Vulkan image object.
///
/// The underlying image may be exportable as a dmabuf.
#[derive(Clone)]
pub struct VulkanImage {
    pub(crate) inner: Arc<ImageInner>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: vk::Format,
    pub(crate) has_alpha: bool,
    pub(crate) mem_types: MemoryPropertyFlags,
    pub(crate) usage: ImageUsageFlags,
    pub(crate) tiling: ImageTiling,
    pub(crate) drm: Option<Format>,
    #[cfg(feature = "backend_drm")]
    pub(crate) node: Option<DrmNode>,
}

impl fmt::Debug for VulkanImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("has_alpha", &self.has_alpha)
            .field("usage", &self.usage)
            .field("drm", &self.drm)
            .field("inner", &self.inner)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
/// Errors created when creating or manipulating `VulkanImage`s.
pub enum Error {
    #[error("Size provided isn't valid")]
    InvalidSize,
    #[error("Format not supported")]
    UnsupportedFormat,
    #[error("Invalid or unsupported (distinct planes) dmabuf")]
    UnsupportedDmabuf,
    #[error("Missing or invalid image usage flags for the requested operation")]
    MissingOrInvalidUsage,
    #[error("Unable to find a supported memory type for the allocation")]
    NoMemoryAvailable,
    #[error("Creating vulkan image failed")]
    VulkanImage(#[source] vk::Result),
    #[error("Querying the vulkan image's modifier failed")]
    VulkanModifierQuery(#[source] vk::Result),
    #[error("Allocating vulkan device memory for the image failed")]
    VulkanAllocate(#[source] vk::Result),
    #[error("Binding the vulkan device memory to the image failed")]
    VulkanBind(#[source] vk::Result),
    #[error("Creating a vulkan image view")]
    VulkanImageView(#[source] vk::Result),
    #[error("Failed to clone the dmabuf fd")]
    DmabufFdError(#[source] io::Error),
}

impl VulkanImage {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        linear: bool,
    ) -> Result<Self, Error> {
        Self::new_internal(
            device,
            width,
            height,
            format,
            None,
            usage,
            linear,
            Option::<Option<Modifier>>::None,
            None,
        )
    }

    pub fn new_with_fourcc(
        device: &Device,
        width: u32,
        height: u32,
        format: Fourcc,
        usage: vk::ImageUsageFlags,
        linear: bool,
    ) -> Result<Self, Error> {
        let vk_format = super::format::get_vk_format(format).ok_or(Error::UnsupportedFormat)?;
        Self::new_internal(
            device,
            width,
            height,
            vk_format,
            Some(format),
            usage,
            linear,
            Option::<Option<Modifier>>::None,
            None,
        )
    }

    pub fn new_exportable(
        device: &Device,
        width: u32,
        height: u32,
        format: Fourcc,
        modifiers: impl Iterator<Item = Modifier>,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, Error> {
        let vk_format = super::format::get_vk_format(format).ok_or(Error::UnsupportedFormat)?;
        Self::new_internal(
            device,
            width,
            height,
            vk_format,
            Some(format),
            usage,
            false,
            Some(modifiers),
            None,
        )
    }

    pub fn new_from_dmabuf(
        device: &Device,
        dmabuf: &Dmabuf,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, Error> {
        let width = dmabuf.width();
        let height = dmabuf.height();
        let vk_format = super::format::get_vk_format(dmabuf.format().code).ok_or(Error::UnsupportedFormat)?;

        // TODO: Handle distinct dmabuf formats
        // (see vkBindImageMemory2 and VkBindImagePlaneMemoryInfo)
        let handles = dmabuf.handles().collect::<Vec<_>>();
        let ino = rustix::fs::fstat(handles[0])
            .map_err(|_| Error::UnsupportedDmabuf)?
            .st_ino;
        if handles
            .iter()
            .skip(1)
            .any(|h| rustix::fs::fstat(h).is_ok_and(|s| s.st_ino != ino))
        {
            return Err(Error::UnsupportedFormat);
        }

        Self::new_internal(
            device,
            width,
            height,
            vk_format,
            Some(dmabuf.format().code),
            usage,
            false,
            Option::<Option<Modifier>>::None,
            Some(dmabuf),
        )
    }

    fn new_internal(
        device: &Device,
        width: u32,
        height: u32,
        vk_format: vk::Format,
        fourcc: Option<Fourcc>,
        vk_usage: vk::ImageUsageFlags,
        linear: bool,
        modifiers: Option<impl IntoIterator<Item = Modifier>>,
        dmabuf: Option<&Dmabuf>,
    ) -> Result<Self, Error> {
        let modifiers = modifiers.map(|modifiers| modifiers.into_iter().map(u64::from).collect::<Vec<_>>());
        let fourcc = dmabuf.map(|d| d.format().code).or(fourcc);
        let has_alpha = fourcc.is_none_or(|fourcc| has_alpha(fourcc));
        let mut modifier_list = modifiers.as_deref().map(|modifiers| {
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(modifiers)
        });
        let has_explicit_modifier = dmabuf
            .map(|dmabuf| dmabuf.format().modifier != Modifier::Invalid)
            .unwrap_or(false);
        let tiling = if modifiers.is_some() || has_explicit_modifier {
            vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT
        } else if linear {
            vk::ImageTiling::LINEAR
        } else {
            vk::ImageTiling::OPTIMAL
        };

        let mut image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .usage(vk_usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .tiling(tiling);

        let plane_layouts: Vec<vk::SubresourceLayout>;
        let mut modifier_image_create_info: vk::ImageDrmFormatModifierExplicitCreateInfoEXT<'_>;
        let mut external_image_create_info: vk::ExternalMemoryImageCreateInfo<'_>;

        if let Some(modifier_list) = modifier_list
            .as_mut()
            .filter(|_| tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        {
            image_create_info = image_create_info.push_next(modifier_list);
        }

        if let Some(dmabuf) = dmabuf {
            if has_explicit_modifier {
                plane_layouts = dmabuf
                    .offsets()
                    .zip(dmabuf.strides())
                    .map(|(offset, stride)| {
                        vk::SubresourceLayout::default()
                            .offset(offset as u64)
                            .row_pitch(stride as u64)
                    })
                    .collect::<Vec<_>>();

                modifier_image_create_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                    .drm_format_modifier(dmabuf.format().modifier.into())
                    .plane_layouts(&plane_layouts);

                image_create_info = image_create_info.push_next(&mut modifier_image_create_info);
            }
        };

        if modifiers.is_some() || dmabuf.is_some() {
            external_image_create_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            image_create_info = image_create_info.push_next(&mut external_image_create_info);
        }

        let mut inner = ImageInner {
            image: unsafe {
                device
                    .vk()
                    .create_image(&image_create_info, None)
                    .map_err(|err| {
                        tracing::error!(
                            "VulkanImage::new_internal: create_image failed: {:?}, width={}, height={}, vk_format={:?}, vk_usage={:?}, tiling={:?}",
                            err, width, height, vk_format, vk_usage, tiling
                        );
                        Error::VulkanImage(err)
                    })?
            },
            memory: vk::DeviceMemory::null(),
            memory_offset: 0,
            suballocated: false,
            allocation_size: 0,
            persistent_mapping: std::sync::Mutex::new(None),
            device: device.downgrade(),
            view: None,
            current_layout: std::sync::atomic::AtomicI32::new(vk::ImageLayout::UNDEFINED.as_raw()),
            needs_acquire: std::sync::atomic::AtomicBool::new(true),
            dmabuf_exportable: modifiers.is_some() || dmabuf.is_some(),
            dmabuf_plane_count: dmabuf.map(|dmabuf| dmabuf.num_planes() as u32).unwrap_or(0),
        };

        let drm_format = fourcc
            .map(|fourcc| {
                if tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT {
                    let mut image_modifier_properties = vk::ImageDrmFormatModifierPropertiesEXT::default();

                    unsafe {
                        device
                            .vk_ext_image_drm_format_modifier()
                            .expect("Required extensions contains ext_image_format_modifier")
                            .get_image_drm_format_modifier_properties(
                                inner.image,
                                &mut image_modifier_properties,
                            )
                            .map_err(Error::VulkanModifierQuery)?
                    };

                    let format = Format {
                        code: fourcc,
                        modifier: Modifier::from(image_modifier_properties.drm_format_modifier),
                    };

                    // Now that we know the format, get the number of planes
                    let format_plane_count = device
                        .formats()
                        .find(|entry| entry.format == format)
                        .map(|entry| entry.modifier_properties.drm_format_modifier_plane_count)
                        .unwrap_or(1);
                    inner.dmabuf_plane_count = format_plane_count;

                    Ok(format)
                } else {
                    let modifier = if linear {
                        Modifier::Linear
                    } else {
                        Modifier::Invalid
                    };
                    Ok(Format {
                        code: fourcc,
                        modifier,
                    })
                }
            })
            .transpose()?;

        // Allocate image memory with dedicated requirements query
        let mut dedicated_reqs = vk::MemoryDedicatedRequirements::default();
        let mut mem_reqs2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_reqs);
        let image_mem_req_info = vk::ImageMemoryRequirementsInfo2::default().image(inner.image);
        unsafe {
            device
                .vk()
                .get_image_memory_requirements2(&image_mem_req_info, &mut mem_reqs2);
        }
        let memory_reqs = mem_reqs2.memory_requirements;
        let mut alloc_create_info = vk::MemoryAllocateInfo::default().allocation_size(memory_reqs.size);

        let mut mem_bits = None;
        if linear {
            let mut best_index = None;
            let mut best_flags = MemoryPropertyFlags::empty();
            for (i, types) in device
                .memory_properties()
                .memory_types_as_slice()
                .iter()
                .enumerate()
            {
                if (memory_reqs.memory_type_bits & (1 << i)) != 0
                    && types.property_flags.contains(MemoryPropertyFlags::HOST_VISIBLE)
                {
                    let flags = types.property_flags;
                    if flags.contains(
                        MemoryPropertyFlags::HOST_VISIBLE
                            | MemoryPropertyFlags::HOST_COHERENT
                            | MemoryPropertyFlags::DEVICE_LOCAL,
                    ) {
                        best_index = Some(i as u32);
                        best_flags = flags;
                        break;
                    } else if flags
                        .contains(MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT)
                    {
                        if best_index.is_none() || !best_flags.contains(MemoryPropertyFlags::HOST_COHERENT) {
                            best_index = Some(i as u32);
                            best_flags = flags;
                        }
                    } else if best_index.is_none() {
                        best_index = Some(i as u32);
                        best_flags = flags;
                    }
                }
            }
            if let Some(index) = best_index {
                alloc_create_info = alloc_create_info.memory_type_index(index);
                mem_bits = Some(best_flags);
            }
        }

        if mem_bits.is_none() {
            for (i, types) in device
                .memory_properties()
                .memory_types_as_slice()
                .iter()
                .enumerate()
            {
                if (memory_reqs.memory_type_bits & (1 << i)) != 0
                    && types.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    alloc_create_info = alloc_create_info.memory_type_index(i as u32);
                    mem_bits = Some(types.property_flags.clone());
                    break;
                }
            }
        }

        if mem_bits.is_none() {
            for (i, types) in device
                .memory_properties()
                .memory_types_as_slice()
                .iter()
                .enumerate()
            {
                if (memory_reqs.memory_type_bits & (1 << i)) != 0 {
                    alloc_create_info = alloc_create_info.memory_type_index(i as u32);
                    mem_bits = Some(types.property_flags.clone());
                    break;
                }
            }
        }

        let Some(mem_bits) = mem_bits else {
            if width > 1 || height > 1 {
                tracing::error!(
                    "VulkanImage::new_internal: NoMemoryAvailable! width={}, height={}, fourcc={:?}, vk_format={:?}, vk_usage={:?}, linear={}, tiling={:?}, memory_type_bits={:#b}",
                    width,
                    height,
                    fourcc,
                    vk_format,
                    vk_usage,
                    linear,
                    tiling,
                    memory_reqs.memory_type_bits
                );
            } else {
                tracing::debug!(
                    "VulkanImage::new_internal probe NoMemoryAvailable: vk_usage={:?}, linear={}, tiling={:?}, memory_type_bits={:#b}",
                    vk_usage,
                    linear,
                    tiling,
                    memory_reqs.memory_type_bits
                );
            }
            return Err(Error::NoMemoryAvailable);
        };

        let is_dedicated = modifiers.is_some()
            || dmabuf.is_some()
            || inner.dmabuf_exportable
            || vk_usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
            || dedicated_reqs.requires_dedicated_allocation != 0
            || dedicated_reqs.prefers_dedicated_allocation != 0;

        let mut import_memory_info: vk::ImportMemoryFdInfoKHR<'_>;
        let mut memory_export_info: vk::ExportMemoryAllocateInfo<'_>;
        let mut memory_dedicated_info: vk::MemoryDedicatedAllocateInfo<'_>;

        if is_dedicated {
            memory_dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(inner.image);
            alloc_create_info = alloc_create_info.push_next(&mut memory_dedicated_info);

            if inner.dmabuf_exportable && dmabuf.is_none() {
                memory_export_info = vk::ExportMemoryAllocateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                alloc_create_info = alloc_create_info.push_next(&mut memory_export_info);
            }

            if let Some(dmabuf) = dmabuf {
                let handle = dmabuf.handles().next().unwrap();
                import_memory_info = vk::ImportMemoryFdInfoKHR::default()
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                    .fd(handle
                        .try_clone_to_owned()
                        .map_err(Error::DmabufFdError)?
                        .into_raw_fd());
                alloc_create_info = alloc_create_info.push_next(&mut import_memory_info);
            }

            unsafe {
                inner.memory = device
                    .vk()
                    .allocate_memory(&alloc_create_info, None)
                    .map_err(|err| {
                        tracing::error!(
                            "VulkanImage::new_internal dedicated allocate_memory failed: {:?}, size={}, type_index={}",
                            err,
                            alloc_create_info.allocation_size,
                            alloc_create_info.memory_type_index
                        );
                        Error::VulkanAllocate(err)
                    })?;
                inner.memory_offset = 0;
                inner.suballocated = false;
                inner.allocation_size = memory_reqs.size;

                device
                    .vk()
                    .bind_image_memory(inner.image, inner.memory, 0)
                    .map_err(|err| {
                        tracing::error!("VulkanImage::new_internal: bind_image_memory failed: {:?}", err);
                        Error::VulkanBind(err)
                    })?;
            }
        } else {
            let suballoc_res = device.suballocate_memory(
                memory_reqs.size,
                memory_reqs.alignment,
                alloc_create_info.memory_type_index,
            );

            match suballoc_res {
                Ok((memory, memory_offset, mapped_ptr)) => {
                    inner.memory = memory;
                    inner.memory_offset = memory_offset;
                    inner.suballocated = true;
                    inner.allocation_size = memory_reqs.size;
                    if let Some(ptr) = mapped_ptr {
                        *inner.persistent_mapping.lock().unwrap() = Some(MappedPointer(ptr));
                    }

                    unsafe {
                        device
                            .vk()
                            .bind_image_memory(inner.image, inner.memory, memory_offset)
                            .map_err(|err| {
                                tracing::error!(
                                    "VulkanImage::new_internal: bind_image_memory (suballocated) failed: {:?}",
                                    err
                                );
                                Error::VulkanBind(err)
                            })?;
                    }
                }
                Err(_) => unsafe {
                    inner.memory = device
                            .vk()
                            .allocate_memory(&alloc_create_info, None)
                            .map_err(|err| {
                                tracing::error!(
                                    "VulkanImage::new_internal: fallback direct allocate_memory failed: {:?}, size={}, type_index={}",
                                    err,
                                    alloc_create_info.allocation_size,
                                    alloc_create_info.memory_type_index
                                );
                                Error::VulkanAllocate(err)
                            })?;
                    inner.memory_offset = 0;
                    inner.suballocated = false;
                    inner.allocation_size = memory_reqs.size;

                    device
                        .vk()
                        .bind_image_memory(inner.image, inner.memory, 0)
                        .map_err(|err| {
                            tracing::error!(
                                "VulkanImage::new_internal: fallback bind_image_memory failed: {:?}",
                                err
                            );
                            Error::VulkanBind(err)
                        })?;
                },
            }
        }

        let is_depth = vk_usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            || vk_format == vk::Format::D16_UNORM
            || vk_format == vk::Format::D32_SFLOAT
            || vk_format == vk::Format::D24_UNORM_S8_UINT;

        if vk_usage.contains(vk::ImageUsageFlags::SAMPLED)
            || vk_usage.contains(vk::ImageUsageFlags::STORAGE)
            || vk_usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            || is_depth
        {
            let aspect_mask = if is_depth {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let info = vk::ImageViewCreateInfo::default()
                .image(inner.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk_format)
                .components(
                    if is_depth
                        || vk_usage.contains(vk::ImageUsageFlags::STORAGE)
                        || vk_usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    {
                        vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        }
                    } else {
                        component_mapping_for_format(vk_format, has_alpha)
                    },
                )
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(aspect_mask)
                        .level_count(1)
                        .layer_count(1),
                );
            inner.view = Some(unsafe {
                device
                    .vk()
                    .create_image_view(&info, None)
                    .map_err(Error::VulkanImageView)?
            });
        }

        Ok(VulkanImage {
            inner: Arc::new(inner),
            width,
            height,
            format: vk_format,
            drm: drm_format.or_else(|| {
                super::format::get_drm_format(vk_format).map(|fourcc| Format {
                    code: fourcc,
                    modifier: Modifier::Invalid,
                })
            }),
            mem_types: mem_bits,
            has_alpha,
            usage: vk_usage,
            tiling,
            #[cfg(feature = "backend_drm")]
            node: if let Some(dmabuf) = dmabuf {
                dmabuf.node()
            } else {
                device.node()
            },
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    pub fn vk(&self) -> &vk::Image {
        &self.inner.image
    }

    pub fn vk_view(&self) -> Option<&vk::ImageView> {
        self.inner.view.as_ref()
    }

    pub fn vk_usage(&self) -> ImageUsageFlags {
        self.usage
    }

    pub fn mem_bits(&self) -> MemoryPropertyFlags {
        self.mem_types
    }

    pub fn is_linear(&self) -> bool {
        self.tiling == ImageTiling::LINEAR
            || (self.tiling == ImageTiling::DRM_FORMAT_MODIFIER_EXT
                && self
                    .drm
                    .as_ref()
                    .is_some_and(|format| format.modifier == Modifier::Linear))
    }

    pub fn current_layout(&self) -> vk::ImageLayout {
        vk::ImageLayout::from_raw(
            self.inner
                .current_layout
                .load(std::sync::atomic::Ordering::SeqCst),
        )
    }

    pub fn set_current_layout(&self, layout: vk::ImageLayout) {
        self.inner
            .current_layout
            .store(layout.as_raw(), std::sync::atomic::Ordering::SeqCst);
    }

    pub fn needs_acquire(&self) -> bool {
        self.inner
            .needs_acquire
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn set_needs_acquire(&self, val: bool) {
        self.inner
            .needs_acquire
            .store(val, std::sync::atomic::Ordering::Release);
    }

    pub fn dmabuf_exportable(&self) -> bool {
        self.inner.dmabuf_exportable
    }

    pub fn memory_offset(&self) -> vk::DeviceSize {
        self.inner.memory_offset
    }

    pub fn is_suballocated(&self) -> bool {
        self.inner.suballocated
    }

    /// Export the image as a DMA-BUF if exportable.
    pub fn export(&self) -> Result<Dmabuf, crate::backend::allocator::vulkan::ExportError> {
        crate::backend::allocator::dmabuf::AsDmabuf::export(self)
    }

    /// Returns a persistent pointer to the mapped memory of this image.
    /// The pointer is offset to the start of this image within its memory allocation/block.
    pub fn map(&self) -> Result<*mut u8, vk::Result> {
        self.inner.map()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MappedPointer(pub *mut u8);
unsafe impl Send for MappedPointer {}
unsafe impl Sync for MappedPointer {}

#[derive(Debug)]
pub(crate) struct ImageInner {
    pub(crate) image: vk::Image,
    // might be a null handle
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) memory_offset: vk::DeviceSize,
    pub(crate) suballocated: bool,
    pub(crate) allocation_size: vk::DeviceSize,
    pub(crate) persistent_mapping: std::sync::Mutex<Option<MappedPointer>>,
    pub(crate) device: WeakDevice,
    pub(crate) view: Option<vk::ImageView>,
    pub(crate) current_layout: std::sync::atomic::AtomicI32,
    pub(crate) needs_acquire: std::sync::atomic::AtomicBool,

    pub(crate) dmabuf_exportable: bool,
    pub(crate) dmabuf_plane_count: u32,
}

impl ImageInner {
    pub fn map(&self) -> Result<*mut u8, vk::Result> {
        let mut guard = self.persistent_mapping.lock().unwrap();
        if let Some(mapped) = *guard {
            return Ok(mapped.0);
        }

        if self.memory == vk::DeviceMemory::null() {
            return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
        }

        let device = self.device.upgrade().ok_or(vk::Result::ERROR_DEVICE_LOST)?;
        let vk = device.vk();

        let ptr =
            unsafe { vk.map_memory(self.memory, 0, self.allocation_size, vk::MemoryMapFlags::empty())? };

        let ptr = unsafe { (ptr as *mut u8).add(self.memory_offset as usize) };
        *guard = Some(MappedPointer(ptr));
        Ok(ptr)
    }
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe {
                let vk = device.vk();
                if let Some(view) = self.view.as_ref() {
                    vk.destroy_image_view(*view, None);
                }
                vk.destroy_image(self.image, None);
                if self.memory != vk::DeviceMemory::null() {
                    if self.suballocated {
                        device.free_suballocation(self.memory, self.memory_offset, self.allocation_size);
                    } else {
                        if let Ok(mut lock) = self.persistent_mapping.lock() {
                            if lock.take().is_some() {
                                vk.unmap_memory(self.memory);
                            }
                        }
                        vk.free_memory(self.memory, None);
                    }
                }
            }
        }
    }
}
