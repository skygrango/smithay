//! Types for configuring the color pipeline of a connector.
//!
//! These types describe the wide-gamut / HDR signalling state of a DRM connector, i.e. the
//! values of the optional `Colorspace`, `HDR_OUTPUT_METADATA` and `max bpc` connector
//! properties. They are used with [`DrmSurface::use_color_state`](super::DrmSurface::use_color_state)
//! (and the corresponding
//! [`DrmCompositor::use_color_state`](super::compositor::DrmCompositor::use_color_state)),
//! which stages the state so that it is applied as part of the *same* atomic commit as the
//! mode and plane state. Some drivers (notably nvidia) treat a `Colorspace` change as a full
//! modeset and misbehave when it is committed on its own, so smithay never issues standalone
//! connector-property commits for these.
//!
//! Whether a connector supports these properties can be queried with
//! [`DrmSurface::supported_colorspaces`](super::DrmSurface::supported_colorspaces),
//! [`DrmSurface::hdr_metadata_supported`](super::DrmSurface::hdr_metadata_supported) and
//! [`DrmSurface::max_bpc_range`](super::DrmSurface::max_bpc_range).
//!
//! Capabilities of the connected sink (whether it accepts a PQ EOTF, its desired luminance
//! range, BT.2020 signal support) should be read from its EDID, e.g. via
//! `smithay-drm-extras`' `display_info` module and libdisplay-info's
//! `Info::hdr_static_metadata()` / `Info::supported_signal_colorimetry()`.

/// Value of the `Colorspace` connector property.
///
/// This selects the colorimetry signalled to the sink in the AVI infoframe (HDMI) or MSA/SDP
/// (DisplayPort). It does not perform any color conversion; the submitted framebuffer contents
/// are expected to already be encoded in the signalled colorspace.
///
/// Only a portable subset of the kernel's colorspace values is exposed. The property's enum
/// values are driver-defined and are resolved by name at set time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Colorspace {
    /// The default colorspace of the connector (`Default`), typically sRGB/BT.709.
    #[default]
    Default,
    /// ITU-R BT.2020 RGB colorimetry (`BT2020_RGB`).
    Bt2020Rgb,
    /// ITU-R BT.2020 YCbCr colorimetry (`BT2020_YCC`).
    Bt2020Ycc,
    /// DCI-P3 RGB colorimetry with D65 white point (`DCI-P3_RGB_D65`).
    DciP3RgbD65,
    /// A colorspace not modelled by smithay.
    ///
    /// Only ever returned when reading back the current state of a connector; requesting it
    /// in [`use_color_state`](super::DrmSurface::use_color_state) fails.
    Unknown,
}

impl Colorspace {
    /// The kernel's name for this colorspace in the `Colorspace` property enum.
    ///
    /// Returns `None` for [`Colorspace::Unknown`].
    pub fn kernel_name(&self) -> Option<&'static str> {
        Some(match self {
            Colorspace::Default => "Default",
            Colorspace::Bt2020Rgb => "BT2020_RGB",
            Colorspace::Bt2020Ycc => "BT2020_YCC",
            Colorspace::DciP3RgbD65 => "DCI-P3_RGB_D65",
            Colorspace::Unknown => return None,
        })
    }

    pub(super) fn from_kernel_name(name: &str) -> Option<Self> {
        Some(match name {
            "Default" => Colorspace::Default,
            "BT2020_RGB" => Colorspace::Bt2020Rgb,
            "BT2020_YCC" => Colorspace::Bt2020Ycc,
            "DCI-P3_RGB_D65" => Colorspace::DciP3RgbD65,
            _ => return None,
        })
    }
}

/// Electro-optical transfer function of the HDR metadata infoframe, per CTA-861.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Eotf {
    /// Traditional gamma, SDR luminance range.
    TraditionalSdr,
    /// Traditional gamma, HDR luminance range.
    TraditionalHdr,
    /// SMPTE ST 2084, a.k.a. Perceptual Quantizer (PQ).
    SmpteSt2084,
    /// Hybrid Log-Gamma (HLG), per BT.2100.
    Hlg,
}

impl Eotf {
    fn to_raw(self) -> u8 {
        match self {
            Eotf::TraditionalSdr => 0,
            Eotf::TraditionalHdr => 1,
            Eotf::SmpteSt2084 => 2,
            Eotf::Hlg => 3,
        }
    }

    fn from_raw(raw: u8) -> Option<Self> {
        Some(match raw {
            0 => Eotf::TraditionalSdr,
            1 => Eotf::TraditionalHdr,
            2 => Eotf::SmpteSt2084,
            3 => Eotf::Hlg,
            _ => return None,
        })
    }
}

/// A CIE 1931 xy chromaticity coordinate in CTA-861.3 units, i.e. the floating-point
/// coordinate scaled by 50000 (increments of 0.00002).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CtaCoordinate {
    /// The x coordinate, scaled by 50000.
    pub x: u16,
    /// The y coordinate, scaled by 50000.
    pub y: u16,
}

impl CtaCoordinate {
    /// Converts a floating point CIE 1931 xy coordinate into CTA-861.3 units.
    pub fn from_xy(x: f64, y: f64) -> Self {
        Self {
            x: (x * 50000.0).round() as u16,
            y: (y * 50000.0).round() as u16,
        }
    }

    /// The BT.2020 red primary (0.708, 0.292).
    pub const BT2020_RED: Self = Self { x: 35400, y: 14600 };
    /// The BT.2020 green primary (0.170, 0.797).
    pub const BT2020_GREEN: Self = Self { x: 8500, y: 39850 };
    /// The BT.2020 blue primary (0.131, 0.046).
    pub const BT2020_BLUE: Self = Self { x: 6550, y: 2300 };
    /// The D65 white point (0.3127, 0.3290).
    pub const D65_WHITE: Self = Self { x: 15635, y: 16450 };
}

/// Static HDR metadata (CTA-861.3 Static Metadata Type 1) for the `HDR_OUTPUT_METADATA`
/// connector property.
///
/// This describes the mastering display and content light levels of the submitted frames to
/// the sink. All fields use the raw infoframe units, so no rounding is hidden from callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HdrOutputMetadata {
    /// The electro-optical transfer function the frame contents are encoded with.
    pub eotf: Eotf,
    /// Chromaticity of the mastering display's red, green and blue primaries.
    pub display_primaries: [CtaCoordinate; 3],
    /// Chromaticity of the mastering display's white point.
    pub white_point: CtaCoordinate,
    /// Maximum luminance of the mastering display, in cd/m².
    pub max_display_mastering_luminance: u16,
    /// Minimum luminance of the mastering display, in 0.0001 cd/m² units.
    pub min_display_mastering_luminance: u16,
    /// Maximum content light level, in cd/m².
    pub max_cll: u16,
    /// Maximum frame-average light level, in cd/m².
    pub max_fall: u16,
}

impl HdrOutputMetadata {
    /// Convenience constructor for the most common HDR10-style signal: PQ transfer function
    /// with BT.2020 mastering primaries and a D65 white point.
    ///
    /// Luminance values should be clamped to what the sink advertises in its EDID HDR static
    /// metadata block.
    pub fn pq_bt2020(max_luminance: u16, min_luminance: u16, max_cll: u16, max_fall: u16) -> Self {
        Self {
            eotf: Eotf::SmpteSt2084,
            display_primaries: [
                CtaCoordinate::BT2020_RED,
                CtaCoordinate::BT2020_GREEN,
                CtaCoordinate::BT2020_BLUE,
            ],
            white_point: CtaCoordinate::D65_WHITE,
            max_display_mastering_luminance: max_luminance,
            min_display_mastering_luminance: min_luminance,
            max_cll,
            max_fall,
        }
    }
}

/// Desired color pipeline configuration of a connector.
///
/// The default value describes plain SDR signalling: default colorspace, no HDR metadata and
/// the `max bpc` property left untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ConnectorColorState {
    /// The colorimetry to signal via the `Colorspace` property.
    pub colorspace: Colorspace,
    /// The HDR static metadata to signal via the `HDR_OUTPUT_METADATA` property.
    ///
    /// `None` disables the HDR infoframe (the property is set to no blob).
    pub hdr_metadata: Option<HdrOutputMetadata>,
    /// The maximum bits per component to allow on the link via the `max bpc` property.
    ///
    /// `None` leaves the property at its current value.
    pub max_bpc: Option<u32>,
}

/// Entry of a DRM CRTC hardware lookup table (`struct drm_color_lut`).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct DrmColorLut {
    /// Red component value (0..65535).
    pub red: u16,
    /// Green component value (0..65535).
    pub green: u16,
    /// Blue component value (0..65535).
    pub blue: u16,
    /// Reserved field required by the kernel UAPI struct alignment.
    pub reserved: u16,
}

impl DrmColorLut {
    /// Creates a LUT entry from normalized [0.0, 1.0] RGB values.
    pub fn from_rgb(r: f32, g: f32, b: f32) -> Self {
        let to_u16 = |v: f32| (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
        Self {
            red: to_u16(r),
            green: to_u16(g),
            blue: to_u16(b),
            reserved: 0,
        }
    }

    /// Generates a ST 2084 (PQ) hardware gamma LUT of the given size.
    /// Maps input linear radiance [0.0, 1.0] (where 1.0 = reference_white)
    /// to output PQ code values [0.0, 1.0] (where 1.0 = 10,000 cd/m²).
    pub fn create_pq_lut(size: usize, reference_white: f32) -> Vec<Self> {
        let mut lut = Vec::with_capacity(size);
        let scale = reference_white.clamp(80.0, 10_000.0) / 10_000.0;
        let denom = (size - 1).max(1) as f32;
        for i in 0..size {
            let input = (i as f32) / denom;
            let val = crate::backend::renderer::gles::hdr::encode_pq(input * scale);
            lut.push(Self::from_rgb(val, val, val));
        }
        lut
    }

    /// Generates a linear identity hardware LUT.
    pub fn create_identity_lut(size: usize) -> Vec<Self> {
        let mut lut = Vec::with_capacity(size);
        let denom = (size - 1).max(1) as f32;
        for i in 0..size {
            let val = (i as f32) / denom;
            lut.push(Self::from_rgb(val, val, val));
        }
        lut
    }
}

/// A 3x3 color transformation matrix for the DRM CRTC `CTM` property (`struct drm_color_ctm`).
///
/// Matrix coefficients are in S31.32 sign-magnitude format (bit 63 is sign,
/// bits 62..32 are integer, bits 31..0 are fractional part).
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DrmColorCtm {
    /// 3x3 row-major matrix entries.
    pub matrix: [u64; 9],
}

impl Default for DrmColorCtm {
    fn default() -> Self {
        Self::identity()
    }
}

impl DrmColorCtm {
    /// Converts a floating-point value to DRM S31.32 sign-magnitude fixed-point format.
    pub fn to_s31_32(val: f64) -> u64 {
        let sign = if val < 0.0 { 1u64 << 63 } else { 0 };
        let abs_val = val.abs();
        let integer = (abs_val.floor() as u64) & 0x7fff_ffff;
        let fraction = ((abs_val.fract() * ((1u64 << 32) as f64)).round() as u64) & 0xffff_ffff;
        sign | (integer << 32) | fraction
    }

    /// Converts a DRM S31.32 sign-magnitude fixed-point value back to floating-point.
    pub fn from_s31_32(val: u64) -> f64 {
        let is_negative = (val & (1u64 << 63)) != 0;
        let integer = ((val >> 32) & 0x7fff_ffff) as f64;
        let fraction = (val & 0xffff_ffff) as f64 / ((1u64 << 32) as f64);
        let mag = integer + fraction;
        if is_negative { -mag } else { mag }
    }

    /// Creates an identity color transformation matrix.
    pub fn identity() -> Self {
        let one = Self::to_s31_32(1.0);
        Self {
            matrix: [one, 0, 0, 0, one, 0, 0, 0, one],
        }
    }

    /// Creates a CTM from a 3x3 row-major floating point array.
    pub fn from_3x3(m: [[f64; 3]; 3]) -> Self {
        Self {
            matrix: [
                Self::to_s31_32(m[0][0]),
                Self::to_s31_32(m[0][1]),
                Self::to_s31_32(m[0][2]),
                Self::to_s31_32(m[1][0]),
                Self::to_s31_32(m[1][1]),
                Self::to_s31_32(m[1][2]),
                Self::to_s31_32(m[2][0]),
                Self::to_s31_32(m[2][1]),
                Self::to_s31_32(m[2][2]),
            ],
        }
    }

    /// Linear Rec.709 to BT.2020 color gamut matrix (row-major).
    pub fn rec709_to_bt2020() -> Self {
        Self::from_3x3([
            [0.6274040, 0.3292820, 0.0433136],
            [0.0690970, 0.9195400, 0.0113612],
            [0.0163916, 0.0880132, 0.8955950],
        ])
    }
}

/// CRTC hardware color management pipeline configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CrtcColorState {
    /// Post-blending gamma lookup table (GAMMA_LUT).
    pub gamma_lut: Option<Vec<DrmColorLut>>,
    /// Post-blending color transformation matrix (CTM).
    pub ctm: Option<DrmColorCtm>,
    /// Pre-blending degamma lookup table (DEGAMMA_LUT).
    pub degamma_lut: Option<Vec<DrmColorLut>>,
}

pub(super) mod ffi {
    //! Binary layout of the kernel's `HDR_OUTPUT_METADATA` blob
    //! (`struct hdr_output_metadata` in `include/uapi/drm/drm_mode.h`).

    /// Static Metadata Type 1 descriptor id (CTA-861.3).
    const HDMI_STATIC_METADATA_TYPE1: u32 = 0;

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct HdrColorPoint {
        pub x: u16,
        pub y: u16,
    }

    /// `struct hdr_metadata_infoframe`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct HdrMetadataInfoframe {
        pub eotf: u8,
        pub metadata_type: u8,
        pub display_primaries: [HdrColorPoint; 3],
        pub white_point: HdrColorPoint,
        pub max_display_mastering_luminance: u16,
        pub min_display_mastering_luminance: u16,
        pub max_cll: u16,
        pub max_fall: u16,
    }

    /// `struct hdr_output_metadata`. Passed to the kernel verbatim as the blob contents,
    /// so the layout must match exactly.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct HdrOutputMetadata {
        pub metadata_type: u32,
        pub hdmi_metadata_type1: HdrMetadataInfoframe,
    }

    impl From<super::HdrOutputMetadata> for HdrOutputMetadata {
        fn from(meta: super::HdrOutputMetadata) -> Self {
            let coord = |c: super::CtaCoordinate| HdrColorPoint { x: c.x, y: c.y };
            HdrOutputMetadata {
                metadata_type: HDMI_STATIC_METADATA_TYPE1,
                hdmi_metadata_type1: HdrMetadataInfoframe {
                    eotf: meta.eotf.to_raw(),
                    metadata_type: HDMI_STATIC_METADATA_TYPE1 as u8,
                    display_primaries: [
                        coord(meta.display_primaries[0]),
                        coord(meta.display_primaries[1]),
                        coord(meta.display_primaries[2]),
                    ],
                    white_point: coord(meta.white_point),
                    max_display_mastering_luminance: meta.max_display_mastering_luminance,
                    min_display_mastering_luminance: meta.min_display_mastering_luminance,
                    max_cll: meta.max_cll,
                    max_fall: meta.max_fall,
                },
            }
        }
    }

    impl HdrOutputMetadata {
        /// Parses the contents of an `HDR_OUTPUT_METADATA` blob.
        ///
        /// Returns `None` if the blob is too short, describes a different metadata type or
        /// uses an EOTF unknown to us.
        pub fn parse(bytes: &[u8]) -> Option<super::HdrOutputMetadata> {
            // The infoframe is valid without the trailing struct padding, so only require the
            // payload itself.
            const PAYLOAD_LEN: usize = 30;
            if bytes.len() < PAYLOAD_LEN {
                return None;
            }
            let u16_at = |off: usize| u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            let coord_at = |off: usize| super::CtaCoordinate {
                x: u16_at(off),
                y: u16_at(off + 2),
            };

            let metadata_type = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if metadata_type != HDMI_STATIC_METADATA_TYPE1 {
                return None;
            }

            Some(super::HdrOutputMetadata {
                eotf: super::Eotf::from_raw(bytes[4])?,
                display_primaries: [coord_at(6), coord_at(10), coord_at(14)],
                white_point: coord_at(18),
                max_display_mastering_luminance: u16_at(22),
                min_display_mastering_luminance: u16_at(24),
                max_cll: u16_at(26),
                max_fall: u16_at(28),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdr_output_metadata_layout() {
        // The blob is passed to the kernel verbatim, so the layout must match
        // `struct hdr_output_metadata` exactly: 4-byte type + 26-byte infoframe, padded to 32.
        assert_eq!(std::mem::size_of::<ffi::HdrOutputMetadata>(), 32);

        let meta: ffi::HdrOutputMetadata = HdrOutputMetadata::pq_bt2020(500, 50, 400, 300).into();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &meta as *const ffi::HdrOutputMetadata as *const u8,
                std::mem::size_of::<ffi::HdrOutputMetadata>(),
            )
        };

        // metadata_type: u32 = HDMI_STATIC_METADATA_TYPE1 (0)
        assert_eq!(&bytes[0..4], &0u32.to_le_bytes());
        // infoframe.eotf = SMPTE ST 2084 (PQ) = 2
        assert_eq!(bytes[4], 2);
        // infoframe.metadata_type = 0 (Static Metadata Type 1)
        assert_eq!(bytes[5], 0);
        // display_primaries[0] (red) = BT.2020 (0.708, 0.292) * 50000
        assert_eq!(&bytes[6..8], &35400u16.to_le_bytes());
        assert_eq!(&bytes[8..10], &14600u16.to_le_bytes());
        // display_primaries[1] (green) = (0.170, 0.797) * 50000
        assert_eq!(&bytes[10..12], &8500u16.to_le_bytes());
        assert_eq!(&bytes[12..14], &39850u16.to_le_bytes());
        // display_primaries[2] (blue) = (0.131, 0.046) * 50000
        assert_eq!(&bytes[14..16], &6550u16.to_le_bytes());
        assert_eq!(&bytes[16..18], &2300u16.to_le_bytes());
        // white_point = D65 (0.3127, 0.3290) * 50000
        assert_eq!(&bytes[18..20], &15635u16.to_le_bytes());
        assert_eq!(&bytes[20..22], &16450u16.to_le_bytes());
        // luminances and light levels
        assert_eq!(&bytes[22..24], &500u16.to_le_bytes());
        assert_eq!(&bytes[24..26], &50u16.to_le_bytes());
        assert_eq!(&bytes[26..28], &400u16.to_le_bytes());
        assert_eq!(&bytes[28..30], &300u16.to_le_bytes());
    }

    #[test]
    fn hdr_output_metadata_roundtrip() {
        let meta = HdrOutputMetadata::pq_bt2020(1000, 1, 800, 400);
        let raw: ffi::HdrOutputMetadata = meta.into();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &raw as *const ffi::HdrOutputMetadata as *const u8,
                std::mem::size_of::<ffi::HdrOutputMetadata>(),
            )
        };
        assert_eq!(ffi::HdrOutputMetadata::parse(bytes), Some(meta));
    }

    #[test]
    fn cta_coordinate_from_xy() {
        assert_eq!(CtaCoordinate::from_xy(0.708, 0.292), CtaCoordinate::BT2020_RED);
        assert_eq!(CtaCoordinate::from_xy(0.3127, 0.3290), CtaCoordinate::D65_WHITE);
    }

    #[test]
    fn colorspace_names_roundtrip() {
        for cs in [
            Colorspace::Default,
            Colorspace::Bt2020Rgb,
            Colorspace::Bt2020Ycc,
            Colorspace::DciP3RgbD65,
        ] {
            assert_eq!(Colorspace::from_kernel_name(cs.kernel_name().unwrap()), Some(cs));
        }
        assert_eq!(Colorspace::Unknown.kernel_name(), None);
    }

    #[test]
    fn drm_color_lut_layout_and_generation() {
        assert_eq!(std::mem::size_of::<DrmColorLut>(), 8);
        assert_eq!(std::mem::size_of::<DrmColorCtm>(), 72);

        let lut = DrmColorLut::create_pq_lut(1024, 203.0);
        assert_eq!(lut.len(), 1024);
        assert_eq!(
            lut[0],
            DrmColorLut {
                red: 0,
                green: 0,
                blue: 0,
                reserved: 0
            }
        );

        // Diffuse white (203 nits) at top of input range corresponds to ~0.5807 in PQ code
        // 0.58068 * 65535 = 38055
        let white_val = lut[1023].red;
        assert!((white_val as i32 - 38055).abs() < 50, "white_val={white_val}");
    }

    #[test]
    fn drm_color_ctm_s31_32_conversion() {
        assert_eq!(DrmColorCtm::to_s31_32(1.0), 1u64 << 32);
        assert_eq!(DrmColorCtm::to_s31_32(-1.0), (1u64 << 63) | (1u64 << 32));
        assert_eq!(DrmColorCtm::to_s31_32(0.5), 1u64 << 31);
        assert_eq!(DrmColorCtm::to_s31_32(-0.5), (1u64 << 63) | (1u64 << 31));

        for val in [0.0, 1.0, -1.0, 0.5, -0.5, 0.627404, -0.0163916] {
            let encoded = DrmColorCtm::to_s31_32(val);
            let decoded = DrmColorCtm::from_s31_32(encoded);
            assert!((decoded - val).abs() < 1e-9, "val={val} decoded={decoded}");
        }

        let ctm = DrmColorCtm::rec709_to_bt2020();
        let r0 = DrmColorCtm::from_s31_32(ctm.matrix[0])
            + DrmColorCtm::from_s31_32(ctm.matrix[1])
            + DrmColorCtm::from_s31_32(ctm.matrix[2]);
        let r1 = DrmColorCtm::from_s31_32(ctm.matrix[3])
            + DrmColorCtm::from_s31_32(ctm.matrix[4])
            + DrmColorCtm::from_s31_32(ctm.matrix[5]);
        let r2 = DrmColorCtm::from_s31_32(ctm.matrix[6])
            + DrmColorCtm::from_s31_32(ctm.matrix[7])
            + DrmColorCtm::from_s31_32(ctm.matrix[8]);
        assert!((r0 - 1.0).abs() < 2e-6);
        assert!((r1 - 1.0).abs() < 2e-6);
        assert!((r2 - 1.0).abs() < 2e-6);
    }
}
