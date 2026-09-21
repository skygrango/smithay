use crate::backend::allocator::Fourcc;
use crate::backend::vulkan::format::get_vk_format;
use ash::vk;

#[cfg(feature = "wayland_frontend")]
use crate::wayland::color::representation::{ChromaLocation, Coefficients, ColorRepresentation, Range};

/// Key identifying a YCbCr sampler conversion configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct YcbcrKey {
    /// The Vulkan format of the image.
    pub format: vk::Format,
    /// YCbCr model conversion (e.g. BT.709, BT.2020).
    pub model: vk::SamplerYcbcrModelConversion,
    /// Numerical range of the YCbCr components (Narrow vs Full).
    pub range: vk::SamplerYcbcrRange,
    /// Horizontal chroma location.
    pub x_chroma_offset: vk::ChromaLocation,
    /// Vertical chroma location.
    pub y_chroma_offset: vk::ChromaLocation,
}

impl YcbcrKey {
    /// Returns default YCbCr conversion parameters for a given DRM fourcc.
    ///
    /// Defaults:
    /// - NV12 / NV16 / YUV420: BT.709 Narrow Range, Cosited Even (horizontal), Midpoint (vertical)
    /// - P010: BT.2020 Narrow Range, Cosited Even (horizontal), Midpoint (vertical)
    pub fn default_for_fourcc(fourcc: Fourcc) -> Option<Self> {
        let format = get_vk_format(fourcc)?;
        match fourcc {
            Fourcc::Nv12 | Fourcc::Nv16 | Fourcc::Yuv420 | Fourcc::Yuv422 | Fourcc::Yuv444 => Some(Self {
                format,
                model: vk::SamplerYcbcrModelConversion::YCBCR_709,
                range: vk::SamplerYcbcrRange::ITU_NARROW,
                x_chroma_offset: vk::ChromaLocation::COSITED_EVEN,
                y_chroma_offset: vk::ChromaLocation::MIDPOINT,
            }),
            Fourcc::P010 | Fourcc::P210 | Fourcc::P012 | Fourcc::P016 | Fourcc::Q410 => Some(Self {
                format,
                model: vk::SamplerYcbcrModelConversion::YCBCR_2020,
                range: vk::SamplerYcbcrRange::ITU_NARROW,
                x_chroma_offset: vk::ChromaLocation::COSITED_EVEN,
                y_chroma_offset: vk::ChromaLocation::MIDPOINT,
            }),
            _ => None,
        }
    }

    #[cfg(feature = "wayland_frontend")]
    /// Derives YCbCr conversion parameters from client-provided Wayland ColorRepresentation metadata.
    pub fn from_color_representation(fourcc: Fourcc, color_rep: &ColorRepresentation) -> Option<Self> {
        let format = get_vk_format(fourcc)?;

        let model = match color_rep.coefficients {
            Some(Coefficients::Bt709) => vk::SamplerYcbcrModelConversion::YCBCR_709,
            Some(Coefficients::Bt601) => vk::SamplerYcbcrModelConversion::YCBCR_601,
            Some(Coefficients::Bt2020) => vk::SamplerYcbcrModelConversion::YCBCR_2020,
            Some(Coefficients::Identity) => vk::SamplerYcbcrModelConversion::RGB_IDENTITY,
            None => match fourcc {
                Fourcc::P010 | Fourcc::P210 | Fourcc::P012 | Fourcc::P016 | Fourcc::Q410 => {
                    vk::SamplerYcbcrModelConversion::YCBCR_2020
                }
                _ => vk::SamplerYcbcrModelConversion::YCBCR_709,
            },
            _ => vk::SamplerYcbcrModelConversion::YCBCR_709,
        };

        let range = match color_rep.range {
            Some(Range::Full) => vk::SamplerYcbcrRange::ITU_FULL,
            Some(Range::Limited) => vk::SamplerYcbcrRange::ITU_NARROW,
            None => vk::SamplerYcbcrRange::ITU_NARROW,
        };

        let (x_chroma_offset, y_chroma_offset) = match color_rep.chroma_location {
            Some(ChromaLocation::Type0) => (vk::ChromaLocation::COSITED_EVEN, vk::ChromaLocation::MIDPOINT),
            Some(ChromaLocation::Type1) => (vk::ChromaLocation::MIDPOINT, vk::ChromaLocation::MIDPOINT),
            Some(ChromaLocation::Type2) => {
                (vk::ChromaLocation::COSITED_EVEN, vk::ChromaLocation::COSITED_EVEN)
            }
            Some(ChromaLocation::Type3) => (vk::ChromaLocation::MIDPOINT, vk::ChromaLocation::COSITED_EVEN),
            Some(ChromaLocation::Type4) => {
                (vk::ChromaLocation::COSITED_EVEN, vk::ChromaLocation::COSITED_EVEN)
            }
            Some(ChromaLocation::Type5) => (vk::ChromaLocation::MIDPOINT, vk::ChromaLocation::COSITED_EVEN),
            None => (vk::ChromaLocation::COSITED_EVEN, vk::ChromaLocation::MIDPOINT),
        };

        Some(Self {
            format,
            model,
            range,
            x_chroma_offset,
            y_chroma_offset,
        })
    }
}

/// An allocated Vulkan YCbCr conversion and its corresponding immutable sampler.
#[derive(Debug)]
pub struct YcbcrConversion {
    /// The conversion configuration key.
    pub key: YcbcrKey,
    /// The Vulkan YCbCr conversion handle.
    pub conversion: vk::SamplerYcbcrConversion,
    /// The immutable sampler created with the YCbCr conversion.
    pub sampler: vk::Sampler,
}
