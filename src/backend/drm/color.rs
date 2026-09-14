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

use crate::backend::allocator::format::FormatSet;

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
#[derive(Debug, Clone, Copy, Default)]
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
    /// Optional reference white level in nits (cd/m²) for HDR10 PQ LUT generation.
    /// If `None` and HDR is active, defaults to 203.0 nits (standard SDR reference white).
    pub reference_white: Option<f32>,
}

impl PartialEq for ConnectorColorState {
    fn eq(&self, other: &Self) -> bool {
        self.colorspace == other.colorspace
            && self.hdr_metadata == other.hdr_metadata
            && self.max_bpc == other.max_bpc
            && match (self.reference_white, other.reference_white) {
                (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for ConnectorColorState {}

impl std::hash::Hash for ConnectorColorState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.colorspace.hash(state);
        self.hdr_metadata.hash(state);
        self.max_bpc.hash(state);
        self.reference_white.map(|w| w.to_bits()).hash(state);
    }
}

impl ConnectorColorState {
    /// Returns true if this state configures HDR output (Colorspace BT.2020 and HDR metadata present).
    pub fn is_hdr(&self) -> bool {
        self.colorspace == Colorspace::Bt2020Rgb && self.hdr_metadata.is_some()
    }
}

/// Encodes normalized absolute luminance (0.0 to 1.0, where 1.0 = 10,000 cd/m²) to a ST 2084 (PQ) code value.
#[inline]
pub fn encode_pq(value: f32) -> f32 {
    if value <= 0.0 {
        return 0.0;
    }
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.6875;
    let p = value.powf(M1);
    ((C1 + C2 * p) / (1.0 + C3 * p)).powf(M2)
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
            let val = encode_pq(input * scale);
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

    /// Generates an sRGB degamma hardware lookup table (non-linear sRGB to linear radiance [0.0, 1.0]).
    pub fn create_srgb_degamma_lut(size: usize) -> Vec<Self> {
        let mut lut = Vec::with_capacity(size);
        let denom = (size - 1).max(1) as f32;
        for i in 0..size {
            let input = (i as f32) / denom;
            let linear = if input <= 0.04045 {
                input / 12.92
            } else {
                ((input + 0.055) / 1.055).powf(2.4)
            };
            lut.push(Self::from_rgb(linear, linear, linear));
        }
        lut
    }

    /// Generates an sRGB gamma hardware lookup table (linear radiance [0.0, 1.0] to non-linear sRGB).
    pub fn create_srgb_gamma_lut(size: usize) -> Vec<Self> {
        let mut lut = Vec::with_capacity(size);
        let denom = (size - 1).max(1) as f32;
        for i in 0..size {
            let linear = (i as f32) / denom;
            let non_linear = if linear <= 0.0031308 {
                12.92 * linear
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            lut.push(Self::from_rgb(non_linear, non_linear, non_linear));
        }
        lut
    }

    /// Generates a SMPTE ST.2084 PQ degamma hardware lookup table (PQ code values [0.0, 1.0] to linear luminance [0.0, 1.0]).
    pub fn create_pq_degamma_lut(size: usize) -> Vec<Self> {
        let m1 = 2610.0 / 16384.0;
        let m2 = (2523.0 / 4096.0) * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = (2413.0 / 4096.0) * 32.0;
        let c3 = (2392.0 / 4096.0) * 32.0;
        let mut lut = Vec::with_capacity(size);
        let denom = (size - 1).max(1) as f64;
        for i in 0..size {
            let v = (i as f64) / denom;
            let v_m2 = v.powf(1.0 / m2);
            let num = (v_m2 - c1).max(0.0);
            let den = (c2 - c3 * v_m2).max(1e-6);
            let linear = (num / den).powf(1.0 / m1) as f32;
            lut.push(Self::from_rgb(linear, linear, linear));
        }
        lut
    }

    /// Generates an ARIB STD-B67 HLG degamma hardware lookup table (HLG code values [0.0, 1.0] to linear radiance [0.0, 1.0]).
    pub fn create_hlg_degamma_lut(size: usize) -> Vec<Self> {
        let a = 0.17883277f32;
        let b = 1.0 - 4.0 * a;
        let c = 0.5 - a * ((4.0 * a).ln());
        let mut lut = Vec::with_capacity(size);
        let denom = (size - 1).max(1) as f32;
        for i in 0..size {
            let e = (i as f32) / denom;
            let linear = if e <= 0.5 {
                (e * e) / 3.0
            } else {
                (((e - c) / a).exp() + b) / 12.0
            };
            lut.push(Self::from_rgb(linear, linear, linear));
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

    /// Linear Rec.709 to BT.2020 color gamut matrix scaled by a luminance factor.
    pub fn rec709_to_bt2020_scaled(scale: f64) -> Self {
        Self::from_3x3([
            [0.6274040 * scale, 0.3292820 * scale, 0.0433136 * scale],
            [0.0690970 * scale, 0.9195400 * scale, 0.0113612 * scale],
            [0.0163916 * scale, 0.0880132 * scale, 0.8955950 * scale],
        ])
    }

    /// Linear BT.2020 to Rec.709 color gamut matrix (row-major).
    pub fn bt2020_to_rec709() -> Self {
        Self::from_3x3([
            [1.6604910, -0.5876411, -0.0728499],
            [-0.1245505, 1.1328999, -0.0083494],
            [-0.0181508, -0.1005789, 1.1187297],
        ])
    }

    /// Linear BT.2020 to Rec.709 color gamut matrix scaled by a luminance factor.
    pub fn bt2020_to_rec709_scaled(scale: f64) -> Self {
        Self::from_3x3([
            [1.6604910 * scale, -0.5876411 * scale, -0.0728499 * scale],
            [-0.1245505 * scale, 1.1328999 * scale, -0.0083494 * scale],
            [-0.0181508 * scale, -0.1005789 * scale, 1.1187297 * scale],
        ])
    }

    /// Identity matrix scaled by a luminance factor.
    pub fn identity_scaled(scale: f64) -> Self {
        Self::from_3x3([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]])
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

/// CRTC hardware color capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CrtcColorCapabilities {
    /// Whether the CRTC supports a post-blending gamma lookup table (GAMMA_LUT).
    pub has_gamma_lut: bool,
    /// Maximum size of the hardware GAMMA_LUT in entries.
    pub gamma_lut_size: u64,
    /// Whether the CRTC supports a pre-blending degamma lookup table (DEGAMMA_LUT).
    pub has_degamma_lut: bool,
    /// Maximum size of the hardware DEGAMMA_LUT in entries.
    pub degamma_lut_size: u64,
    /// Whether the CRTC supports a hardware color transformation matrix (CTM).
    pub has_ctm: bool,
}

/// Color transformation to be executed on the KMS plane or CRTC for direct scanout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneColorConversion {
    /// scRGB (Extended Linear Rec.709 FP16) to BT.2020 PQ.
    ScRgbToPq { reference_white: u16 },
    /// SDR (sRGB Rec.709 8/10-bit) to BT.2020 PQ.
    SrgbToPq { reference_white: u16 },
    /// HLG to BT.2020 PQ.
    HlgToPq { reference_white: u16 },
    /// scRGB (Extended Linear Rec.709 FP16) to SDR Rec.709 sRGB.
    ScRgbToSrgb,
    /// BT.2020 PQ (HDR10) to SDR Rec.709 sRGB.
    PqToSrgb,
    /// HLG to SDR Rec.709 sRGB.
    HlgToSrgb,
}

impl PlaneColorConversion {
    /// Builds the CRTC color state needed to perform this conversion in display hardware.
    pub fn to_crtc_color_state(&self, gamma_lut_size: usize, degamma_lut_size: usize) -> CrtcColorState {
        let gamma_size = if gamma_lut_size > 0 { gamma_lut_size } else { 4096 };
        match self {
            PlaneColorConversion::ScRgbToPq { reference_white } => {
                let ref_white = if *reference_white > 0 {
                    *reference_white
                } else {
                    80
                };
                // scRGB values span from negative (out-of-Rec.709 wide-gamut colors) to
                // > 1.0 (HDR highlights up to 125.0 for 10,000 cd/m² with 80 cd/m² nominal white).
                // Hardware 1D LUT (GAMMA_LUT) input domain is fixed to [0.0, 1.0].
                // We pre-scale in CTM by (ref_white / 10,000.0) so that all positive luminance
                // up to 10,000 cd/m² maps continuously into [0.0, 1.0] without clamping highlights,
                // and Rec.709 primaries rotate into BT.2020 primaries turning wide-gamut coordinates
                // non-negative.
                let scale = (ref_white as f64) / 10000.0;
                let ctm = DrmColorCtm::rec709_to_bt2020_scaled(scale);
                let gamma_lut = DrmColorLut::create_pq_lut(gamma_size, 10000.0);
                CrtcColorState {
                    degamma_lut: None,
                    ctm: Some(ctm),
                    gamma_lut: Some(gamma_lut),
                }
            }
            PlaneColorConversion::SrgbToPq { reference_white } => {
                let ref_white = if *reference_white > 0 {
                    *reference_white
                } else {
                    203
                };
                // SDR input is strictly in [0.0, 1.0] (no out-of-gamut negative values, no >1.0 highlights).
                // DEGAMMA linearizes non-linear sRGB into linear [0.0, 1.0].
                // CTM rotates Rec.709 primaries into BT.2020 primaries and pre-scales by
                // (ref_white / 10,000.0) so that linear radiance maps into [0.0, 1.0], matching
                // the ScRgbToPq pipeline.
                // GAMMA_LUT encodes the canonical SMPTE ST 2084 PQ curve from 0.0 to 10,000 cd/m²,
                // providing correct contrast, deep blacks, and preventing washed-out visuals.
                let scale = (ref_white as f64) / 10000.0;
                let ctm = DrmColorCtm::rec709_to_bt2020_scaled(scale);
                let degamma_lut = if degamma_lut_size > 0 {
                    Some(DrmColorLut::create_srgb_degamma_lut(degamma_lut_size))
                } else {
                    None
                };
                let gamma_lut = DrmColorLut::create_pq_lut(gamma_size, 10000.0);
                CrtcColorState {
                    degamma_lut,
                    ctm: Some(ctm),
                    gamma_lut: Some(gamma_lut),
                }
            }
            PlaneColorConversion::HlgToPq { reference_white } => {
                let ref_white = if *reference_white > 0 {
                    *reference_white
                } else {
                    1000
                };
                // HLG is already BT.2020 color primaries, input strictly in [0.0, 1.0].
                // DEGAMMA converts HLG to linear [0.0, 1.0].
                // CTM scales linear radiance into [0.0, 1.0] by (ref_white / 10,000.0).
                // GAMMA_LUT encodes canonical ST 2084 PQ.
                let scale = (ref_white as f64) / 10000.0;
                let ctm = DrmColorCtm::identity_scaled(scale);
                let degamma_lut = if degamma_lut_size > 0 {
                    Some(DrmColorLut::create_hlg_degamma_lut(degamma_lut_size))
                } else {
                    None
                };
                let gamma_lut = DrmColorLut::create_pq_lut(gamma_size, 10000.0);
                CrtcColorState {
                    degamma_lut,
                    ctm: Some(ctm),
                    gamma_lut: Some(gamma_lut),
                }
            }
            PlaneColorConversion::ScRgbToSrgb => {
                let gamma_lut = DrmColorLut::create_srgb_gamma_lut(gamma_size);
                CrtcColorState {
                    degamma_lut: None,
                    ctm: None,
                    gamma_lut: Some(gamma_lut),
                }
            }
            PlaneColorConversion::PqToSrgb => {
                let degamma_lut = if degamma_lut_size > 0 {
                    Some(DrmColorLut::create_pq_degamma_lut(degamma_lut_size))
                } else {
                    None
                };
                let ctm = DrmColorCtm::bt2020_to_rec709();
                let gamma_lut = DrmColorLut::create_srgb_gamma_lut(gamma_size);
                CrtcColorState {
                    degamma_lut,
                    ctm: Some(ctm),
                    gamma_lut: Some(gamma_lut),
                }
            }
            PlaneColorConversion::HlgToSrgb => {
                let degamma_lut = if degamma_lut_size > 0 {
                    Some(DrmColorLut::create_hlg_degamma_lut(degamma_lut_size))
                } else {
                    None
                };
                let ctm = DrmColorCtm::bt2020_to_rec709();
                let gamma_lut = DrmColorLut::create_srgb_gamma_lut(gamma_size);
                CrtcColorState {
                    degamma_lut,
                    ctm: Some(ctm),
                    gamma_lut: Some(gamma_lut),
                }
            }
        }
    }
}

/// The scanout execution plan evaluated for a fullscreen client surface,
/// following the strict efficiency hierarchy:
/// DirectPassthrough -> PlaneColorop -> CrtcHardware -> VulkanFastDirectFlip
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScanoutPlan {
    /// 1. Passthrough direct scanout: Client color characteristics match the output pipeline.
    /// Zero transformation on both GPU and Display Engine (0% GPU, 0% display color pipe).
    /// (Native BT.2020 PQ on HDR output, or sRGB/Rec.709 on SDR output).
    #[default]
    DirectPassthrough,
    /// 2. DRM Plane Colorop direct scanout:
    /// Transformation occurs on the KMS Plane COLOR_PIPELINE before blending (0% GPU).
    PlaneColorop(PlaneColorConversion),
    /// 3. DRM CRTC Color Management direct scanout:
    /// Transformation occurs on the KMS CRTC (DEGAMMA_LUT, CTM, GAMMA_LUT) after blending (0% GPU).
    CrtcHardware(PlaneColorConversion),
    /// 4. Fast GPU Shader Scanout:
    /// Fallback path when hardware Colorop/CRTC cannot handle or atomic test fails.
    /// Vulkan renderer compute/fragment shader directly transforms 1:1 into swapchain scanout buffer.
    VulkanFastDirectFlip,
}

impl ScanoutPlan {
    /// Returns whether this plan attempts zero-copy direct scanout on the primary plane.
    #[inline]
    pub fn allows_primary_scanout(&self) -> bool {
        matches!(
            self,
            ScanoutPlan::DirectPassthrough | ScanoutPlan::PlaneColorop(_) | ScanoutPlan::CrtcHardware(_)
        )
    }

    /// Returns the hardware color conversion if this plan requires CRTC color state modification.
    #[inline]
    pub fn requires_crtc_color_state(&self) -> Option<PlaneColorConversion> {
        match self {
            ScanoutPlan::CrtcHardware(conv) | ScanoutPlan::PlaneColorop(conv) => Some(*conv),
            _ => None,
        }
    }

    /// Returns the hardware color conversion if this plan requires Plane COLOR_PIPELINE modification.
    #[inline]
    pub fn requires_plane_colorop(&self) -> Option<PlaneColorConversion> {
        match self {
            ScanoutPlan::PlaneColorop(conv) => Some(*conv),
            _ => None,
        }
    }
}

/// Overall DRM hardware scanout capabilities for a display surface.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DrmScanoutCapabilities {
    /// Color capabilities of the driving CRTC.
    pub crtc_color: CrtcColorCapabilities,
    /// Whether the primary plane supports DRM COLOR_PIPELINE (colorop).
    pub supports_plane_colorop: bool,
    /// Supported pixel formats on the primary plane.
    pub primary_plane_formats: FormatSet,
    /// Whether the primary plane supports FP16 formats (e.g. ABGR16161616F / XBGR16161616F).
    pub supports_fp16: bool,
    /// Whether the primary plane supports 10-bit formats (e.g. XRGB2101010 / XBGR2101010).
    pub supports_10bit: bool,
}

impl DrmScanoutCapabilities {
    /// Whether the hardware can directly scan out scRGB FP16 content via CRTC color management.
    pub fn supports_scrgb_hardware_scanout(&self) -> bool {
        self.supports_fp16 && self.crtc_color.has_gamma_lut && self.crtc_color.has_ctm
    }

    /// Whether the hardware can directly scan out SDR content onto an HDR output via CRTC color management.
    pub fn supports_sdr_to_hdr_hardware_scanout(&self) -> bool {
        self.crtc_color.has_degamma_lut && self.crtc_color.has_gamma_lut && self.crtc_color.has_ctm
    }

    /// Whether the hardware can directly scan out HLG content onto an HDR output via CRTC color management.
    pub fn supports_hlg_to_hdr_hardware_scanout(&self) -> bool {
        self.crtc_color.has_degamma_lut && self.crtc_color.has_gamma_lut
    }

    /// Evaluates the appropriate scanout plan for a given content image description,
    /// following the strict efficiency hierarchy:
    /// DirectPassthrough -> PlaneColorop -> CrtcHardware -> VulkanFastDirectFlip
    pub fn evaluate_scanout_plan(
        &self,
        output_hdr_enabled: bool,
        desc: Option<&crate::wayland::color::management::ImageDescription>,
        output_reference_white: u16,
    ) -> ScanoutPlan {
        if !output_hdr_enabled {
            match desc {
                Some(desc)
                    if desc.windows_scrgb
                        || desc.transfer
                            == crate::wayland::color::management::TransferFunction::ExtLinear =>
                {
                    // scRGB FP16 on SDR output
                    let conv = PlaneColorConversion::ScRgbToSrgb;
                    if self.supports_plane_colorop && self.supports_fp16 {
                        ScanoutPlan::PlaneColorop(conv)
                    } else if self.supports_fp16 && self.crtc_color.has_gamma_lut {
                        ScanoutPlan::CrtcHardware(conv)
                    } else {
                        ScanoutPlan::VulkanFastDirectFlip
                    }
                }
                Some(desc) if desc.is_pq_bt2020() => {
                    // PQ BT.2020 on SDR output
                    let conv = PlaneColorConversion::PqToSrgb;
                    if self.supports_plane_colorop {
                        ScanoutPlan::PlaneColorop(conv)
                    } else if self.crtc_color.has_degamma_lut
                        && self.crtc_color.has_ctm
                        && self.crtc_color.has_gamma_lut
                    {
                        ScanoutPlan::CrtcHardware(conv)
                    } else {
                        ScanoutPlan::VulkanFastDirectFlip
                    }
                }
                Some(desc) if desc.transfer == crate::wayland::color::management::TransferFunction::Hlg => {
                    // HLG on SDR output
                    let conv = PlaneColorConversion::HlgToSrgb;
                    if self.supports_plane_colorop {
                        ScanoutPlan::PlaneColorop(conv)
                    } else if self.crtc_color.has_degamma_lut
                        && self.crtc_color.has_ctm
                        && self.crtc_color.has_gamma_lut
                    {
                        ScanoutPlan::CrtcHardware(conv)
                    } else {
                        ScanoutPlan::VulkanFastDirectFlip
                    }
                }
                Some(desc) if desc.is_hdr() => {
                    // Generic HDR on SDR requires shader tonemapping
                    ScanoutPlan::VulkanFastDirectFlip
                }
                _ => {
                    // Standard SDR on SDR output -> Zero transform
                    ScanoutPlan::DirectPassthrough
                }
            }
        } else {
            // HDR output active (BT.2020 PQ signal)
            match desc {
                Some(desc) if desc.is_pq_bt2020() => {
                    // Tier 1: Native PQ BT.2020 matches output HDR pipeline directly.
                    // Zero GPU, zero plane/CRTC color transformation.
                    ScanoutPlan::DirectPassthrough
                }
                Some(desc)
                    if desc.windows_scrgb
                        || desc.transfer
                            == crate::wayland::color::management::TransferFunction::ExtLinear =>
                {
                    // scRGB FP16 on HDR
                    let ref_white = desc.luminances_or_default().2 as u16;
                    let conv = PlaneColorConversion::ScRgbToPq {
                        reference_white: if ref_white > 0 {
                            ref_white
                        } else if output_reference_white > 0 {
                            output_reference_white
                        } else {
                            203
                        },
                    };
                    if self.supports_plane_colorop && self.supports_fp16 {
                        // Tier 2A: Plane COLOR_PIPELINE (colorop)
                        ScanoutPlan::PlaneColorop(conv)
                    } else if self.supports_scrgb_hardware_scanout() {
                        // Tier 2B: CRTC Color Management
                        ScanoutPlan::CrtcHardware(conv)
                    } else {
                        // Tier 3: Vulkan Shader Fast Flip
                        ScanoutPlan::VulkanFastDirectFlip
                    }
                }
                Some(desc) if desc.transfer == crate::wayland::color::management::TransferFunction::Hlg => {
                    // HLG on HDR
                    let ref_white = desc.luminances_or_default().2 as u16;
                    let conv = PlaneColorConversion::HlgToPq {
                        reference_white: if ref_white > 0 {
                            ref_white
                        } else if output_reference_white > 0 {
                            output_reference_white
                        } else {
                            203
                        },
                    };
                    if self.supports_plane_colorop {
                        // Tier 2A: Plane COLOR_PIPELINE (colorop)
                        ScanoutPlan::PlaneColorop(conv)
                    } else if self.supports_hlg_to_hdr_hardware_scanout() {
                        // Tier 2B: CRTC Color Management
                        ScanoutPlan::CrtcHardware(conv)
                    } else {
                        // Tier 3: Vulkan Shader Fast Flip
                        ScanoutPlan::VulkanFastDirectFlip
                    }
                }
                Some(desc) if !desc.is_hdr() => {
                    // Tagged SDR on HDR
                    let effective_ref_white = if output_reference_white > 0 {
                        output_reference_white
                    } else {
                        203
                    };
                    let conv = PlaneColorConversion::SrgbToPq {
                        reference_white: effective_ref_white,
                    };
                    if self.supports_plane_colorop {
                        // Tier 2A: Plane COLOR_PIPELINE (colorop)
                        ScanoutPlan::PlaneColorop(conv)
                    } else if self.supports_sdr_to_hdr_hardware_scanout() {
                        // Tier 2B: CRTC Color Management
                        ScanoutPlan::CrtcHardware(conv)
                    } else {
                        // Tier 3: Vulkan Shader Fast Flip
                        ScanoutPlan::VulkanFastDirectFlip
                    }
                }
                None => {
                    // Untagged SDR on HDR
                    let effective_ref_white = if output_reference_white > 0 {
                        output_reference_white
                    } else {
                        203
                    };
                    let conv = PlaneColorConversion::SrgbToPq {
                        reference_white: effective_ref_white,
                    };
                    if self.supports_plane_colorop {
                        // Tier 2A: Plane COLOR_PIPELINE (colorop)
                        ScanoutPlan::PlaneColorop(conv)
                    } else if self.supports_sdr_to_hdr_hardware_scanout() {
                        // Tier 2B: CRTC Color Management
                        ScanoutPlan::CrtcHardware(conv)
                    } else {
                        // Tier 3: Vulkan Shader Fast Flip
                        ScanoutPlan::VulkanFastDirectFlip
                    }
                }
                _ => ScanoutPlan::VulkanFastDirectFlip,
            }
        }
    }
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

    #[test]
    fn connector_color_state_is_hdr_and_equality() {
        let sdr = ConnectorColorState::default();
        assert!(!sdr.is_hdr());
        assert_eq!(sdr.reference_white, None);

        let hdr = ConnectorColorState {
            colorspace: Colorspace::Bt2020Rgb,
            hdr_metadata: Some(HdrOutputMetadata::pq_bt2020(1000, 1, 1000, 400)),
            max_bpc: Some(10),
            reference_white: Some(250.0),
        };
        assert!(hdr.is_hdr());

        let mut hdr2 = hdr;
        assert_eq!(hdr, hdr2);

        hdr2.reference_white = Some(300.0);
        assert_ne!(hdr, hdr2);
    }

    #[test]
    fn srgb_degamma_lut_generation() {
        let lut = DrmColorLut::create_srgb_degamma_lut(1024);
        assert_eq!(lut.len(), 1024);
        assert_eq!(lut[0].red, 0);
        assert_eq!(lut[1023].red, 65535);

        // Mid-gray test: sRGB 0.5 (~128/255) decodes to ~0.214 in linear
        // 0.214 * 65535 = ~14024
        let mid = lut[512].red;
        assert!((mid as i32 - 14024).abs() < 100, "mid={mid}");
    }

    #[test]
    fn drm_color_ctm_scaled() {
        let scale = 0.5;
        let ctm = DrmColorCtm::rec709_to_bt2020_scaled(scale);
        let r0 = DrmColorCtm::from_s31_32(ctm.matrix[0])
            + DrmColorCtm::from_s31_32(ctm.matrix[1])
            + DrmColorCtm::from_s31_32(ctm.matrix[2]);
        assert!((r0 - scale).abs() < 2e-6, "r0={r0} scale={scale}");
    }

    #[test]
    fn scanout_plan_evaluation() {
        use crate::wayland::color::management::ImageDescription;
        let mut caps = DrmScanoutCapabilities {
            crtc_color: CrtcColorCapabilities {
                has_gamma_lut: true,
                gamma_lut_size: 4096,
                has_degamma_lut: true,
                degamma_lut_size: 4096,
                has_ctm: true,
            },
            supports_plane_colorop: false,
            primary_plane_formats: FormatSet::default(),
            supports_fp16: true,
            supports_10bit: true,
        };

        // 1. Tier 1: SDR on SDR (zero transform)
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::SRGB), 203);
        assert_eq!(plan, ScanoutPlan::DirectPassthrough);

        // 2. Tier 1: PQ BT.2020 on HDR (zero transform)
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_BT2100), 203);
        assert_eq!(plan, ScanoutPlan::DirectPassthrough);

        // --- Target: SDR Output ---
        // 1. SDR sRGB on SDR output -> DirectPassthrough
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::SRGB), 203);
        assert_eq!(plan, ScanoutPlan::DirectPassthrough);

        // 2. scRGB on SDR output:
        // 2a. PlaneColorop
        caps.supports_plane_colorop = true;
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert_eq!(plan, ScanoutPlan::PlaneColorop(PlaneColorConversion::ScRgbToSrgb));
        // 2b. CrtcHardware
        caps.supports_plane_colorop = false;
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert_eq!(plan, ScanoutPlan::CrtcHardware(PlaneColorConversion::ScRgbToSrgb));
        // 2c. Fallback to Shader (no FP16)
        caps.supports_fp16 = false;
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert_eq!(plan, ScanoutPlan::VulkanFastDirectFlip);
        caps.supports_fp16 = true;

        // 3. PQ BT.2020 on SDR output:
        // 3a. PlaneColorop
        caps.supports_plane_colorop = true;
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::WINDOWS_BT2100), 203);
        assert_eq!(plan, ScanoutPlan::PlaneColorop(PlaneColorConversion::PqToSrgb));
        // 3b. CrtcHardware
        caps.supports_plane_colorop = false;
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::WINDOWS_BT2100), 203);
        assert_eq!(plan, ScanoutPlan::CrtcHardware(PlaneColorConversion::PqToSrgb));
        // 3c. Fallback to Shader (missing DEGAMMA)
        caps.crtc_color.has_degamma_lut = false;
        let plan = caps.evaluate_scanout_plan(false, Some(&ImageDescription::WINDOWS_BT2100), 203);
        assert_eq!(plan, ScanoutPlan::VulkanFastDirectFlip);
        caps.crtc_color.has_degamma_lut = true;

        // 4. HLG on SDR output:
        use crate::wayland::color::management::TransferFunction;
        let mut hlg_desc = ImageDescription::WINDOWS_BT2100;
        hlg_desc.transfer = TransferFunction::Hlg;
        // 4a. PlaneColorop
        caps.supports_plane_colorop = true;
        let plan = caps.evaluate_scanout_plan(false, Some(&hlg_desc), 203);
        assert_eq!(plan, ScanoutPlan::PlaneColorop(PlaneColorConversion::HlgToSrgb));
        // 4b. CrtcHardware
        caps.supports_plane_colorop = false;
        let plan = caps.evaluate_scanout_plan(false, Some(&hlg_desc), 203);
        assert_eq!(plan, ScanoutPlan::CrtcHardware(PlaneColorConversion::HlgToSrgb));
        // 4c. Fallback to Shader (missing CTM)
        caps.crtc_color.has_ctm = false;
        let plan = caps.evaluate_scanout_plan(false, Some(&hlg_desc), 203);
        assert_eq!(plan, ScanoutPlan::VulkanFastDirectFlip);
        caps.crtc_color.has_ctm = true;

        // --- Target: HDR Output ---
        // 5. PQ BT.2020 on HDR output -> DirectPassthrough
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_BT2100), 203);
        assert_eq!(plan, ScanoutPlan::DirectPassthrough);

        // 6. scRGB on HDR output:
        // 6a. PlaneColorop
        caps.supports_plane_colorop = true;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::PlaneColorop(PlaneColorConversion::ScRgbToPq { .. })
        ));
        // 6b. CrtcHardware
        caps.supports_plane_colorop = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::CrtcHardware(PlaneColorConversion::ScRgbToPq { .. })
        ));
        // 6c. Fallback to Shader (no CTM)
        caps.crtc_color.has_ctm = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert_eq!(plan, ScanoutPlan::VulkanFastDirectFlip);
        caps.crtc_color.has_ctm = true;

        // 7. sRGB on HDR output:
        // 7a. PlaneColorop
        caps.supports_plane_colorop = true;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::SRGB), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::PlaneColorop(PlaneColorConversion::SrgbToPq { .. })
        ));
        // 7b. CrtcHardware
        caps.supports_plane_colorop = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::SRGB), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::CrtcHardware(PlaneColorConversion::SrgbToPq { .. })
        ));
        // 7c. Fallback to Shader (no DEGAMMA)
        caps.crtc_color.has_degamma_lut = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::SRGB), 203);
        assert_eq!(plan, ScanoutPlan::VulkanFastDirectFlip);
        caps.crtc_color.has_degamma_lut = true;

        // 8. HLG on HDR output:
        // 8a. PlaneColorop
        caps.supports_plane_colorop = true;
        let plan = caps.evaluate_scanout_plan(true, Some(&hlg_desc), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::PlaneColorop(PlaneColorConversion::HlgToPq { .. })
        ));
        // 8b. CrtcHardware
        caps.supports_plane_colorop = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&hlg_desc), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::CrtcHardware(PlaneColorConversion::HlgToPq { .. })
        ));
        // 8c. Fallback to Shader (no DEGAMMA)
        caps.crtc_color.has_degamma_lut = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&hlg_desc), 203);
        assert_eq!(plan, ScanoutPlan::VulkanFastDirectFlip);
        caps.crtc_color.has_degamma_lut = true;
    }

    #[test]
    fn test_scrgb_to_pq_hardware_conversion_extended_range_and_negative() {
        let conv = PlaneColorConversion::ScRgbToPq { reference_white: 203 };
        let state = conv.to_crtc_color_state(4096, 4096);
        assert!(state.degamma_lut.is_none());
        assert!(state.ctm.is_some());
        assert!(state.gamma_lut.is_some());

        let ctm = state.ctm.unwrap();
        let gamma_lut = state.gamma_lut.unwrap();
        assert_eq!(gamma_lut.len(), 4096);

        // Helper to apply 3x3 CTM to RGB vector
        let apply_ctm = |rgb: [f64; 3]| -> [f64; 3] {
            let m00 = DrmColorCtm::from_s31_32(ctm.matrix[0]);
            let m01 = DrmColorCtm::from_s31_32(ctm.matrix[1]);
            let m02 = DrmColorCtm::from_s31_32(ctm.matrix[2]);
            let m10 = DrmColorCtm::from_s31_32(ctm.matrix[3]);
            let m11 = DrmColorCtm::from_s31_32(ctm.matrix[4]);
            let m12 = DrmColorCtm::from_s31_32(ctm.matrix[5]);
            let m20 = DrmColorCtm::from_s31_32(ctm.matrix[6]);
            let m21 = DrmColorCtm::from_s31_32(ctm.matrix[7]);
            let m22 = DrmColorCtm::from_s31_32(ctm.matrix[8]);
            [
                m00 * rgb[0] + m01 * rgb[1] + m02 * rgb[2],
                m10 * rgb[0] + m11 * rgb[1] + m12 * rgb[2],
                m20 * rgb[0] + m21 * rgb[1] + m22 * rgb[2],
            ]
        };

        // 1. Negative values test (wide color gamut in Rec.709 coordinates)
        // BT.2020 pure green in Rec.709 primaries has negative Red and Blue:
        // [-0.4677, 1.0772, -0.0298]
        let wide_gamut_green_scrgb = [-0.4677, 1.0772, -0.0298];
        let transformed = apply_ctm(wide_gamut_green_scrgb);
        // After rec709_to_bt2020 rotation and luminance scaling, coordinates must be non-negative!
        assert!(transformed[0] >= -1e-4, "Red was negative: {}", transformed[0]);
        assert!(
            transformed[1] > 0.0,
            "Green should be positive: {}",
            transformed[1]
        );
        assert!(transformed[2] >= -1e-4, "Blue was negative: {}", transformed[2]);

        // 2. Nominal white test: scRGB 1.0 maps to 203 nits in PQ
        let white_scrgb = [1.0, 1.0, 1.0];
        let white_out = apply_ctm(white_scrgb);
        let expected_linear = 203.0 / 10000.0;
        assert!((white_out[0] - expected_linear).abs() < 1e-5);
        assert!((white_out[1] - expected_linear).abs() < 1e-5);
        assert!((white_out[2] - expected_linear).abs() < 1e-5);

        // 3. Extended range HDR highlight test: values exceeding 1.0
        // E.g. 1000 nits highlight: 1000 / 203 = ~4.926 in scRGB
        let highlight_scrgb = [4.926108, 4.926108, 4.926108];
        let highlight_out = apply_ctm(highlight_scrgb);
        // 1000 / 10000 = 0.1, which MUST be <= 1.0 so hardware 1D LUT does NOT clamp it!
        assert!((highlight_out[0] - 0.100).abs() < 1e-3);
        assert!(
            highlight_out[0] <= 1.0,
            "Highlight must not exceed LUT domain [0, 1]!"
        );

        // 4. Maximum HDR luminance (10,000 nits): 10000 / 203 = 49.261 in scRGB
        let max_hdr_scrgb = [49.26108, 49.26108, 49.26108];
        let max_out = apply_ctm(max_hdr_scrgb);
        assert!((max_out[0] - 1.000).abs() < 1e-3, "10,000 nits must map to 1.0!");

        // 5. Verify GAMMA_LUT encodes ST 2084 PQ continuously without clipping
        // At index 0 (0.0): 0
        assert_eq!(gamma_lut[0].red, 0);
        // At index corresponding to 0.1 (1000 nits): PQ(0.1) = ~0.7518 -> ~49270 in u16
        let idx_1000nits = (0.1 * 4095.0) as usize;
        let pq_1000 = gamma_lut[idx_1000nits].red;
        assert!((pq_1000 as i32 - 49270).abs() < 200, "pq_1000={pq_1000}");
        // At index 4095 (1.0 = 10,000 nits): PQ(1.0) = 1.0 -> 65535 in u16
        assert_eq!(gamma_lut[4095].red, 65535);
    }

    #[test]
    fn test_srgb_to_pq_hardware_conversion_full_precision_and_black_level() {
        let conv = PlaneColorConversion::SrgbToPq { reference_white: 335 };
        let state = conv.to_crtc_color_state(4096, 4096);
        assert!(state.degamma_lut.is_some());
        assert!(state.ctm.is_some());
        assert!(state.gamma_lut.is_some());

        let degamma_lut = state.degamma_lut.unwrap();
        let ctm = state.ctm.unwrap();
        let gamma_lut = state.gamma_lut.unwrap();

        assert_eq!(degamma_lut.len(), 4096);
        assert_eq!(gamma_lut.len(), 4096);

        // 1. DEGAMMA linearizes sRGB
        assert_eq!(degamma_lut[0].red, 0);
        assert_eq!(degamma_lut[4095].red, 65535);

        // 2. CTM is scaled Rec.709 to BT.2020 matrix (row sums = scale)
        let scale = 335.0 / 10000.0;
        let m00 = DrmColorCtm::from_s31_32(ctm.matrix[0]);
        let m01 = DrmColorCtm::from_s31_32(ctm.matrix[1]);
        let m02 = DrmColorCtm::from_s31_32(ctm.matrix[2]);
        let r0 = m00 + m01 + m02;
        assert!(
            (r0 - scale).abs() < 2e-6,
            "r0={r0} must be scale={scale} (scaled CTM)"
        );

        // 3. GAMMA_LUT maps linear [0.0, 1.0] to canonical PQ [0.0, 10,000 nits]
        // Entry 0 must be STRICTLY 0 (0.0 nits true black, no washed out / lifted black)
        assert_eq!(gamma_lut[0].red, 0);
        assert_eq!(gamma_lut[0].green, 0);
        assert_eq!(gamma_lut[0].blue, 0);

        // Entry 4095 is 10,000 nits -> 65535
        assert_eq!(gamma_lut[4095].red, 65535);

        // At index corresponding to 335 nits (335 / 10000 = 0.0335):
        let idx_335 = (scale * 4095.0) as usize;
        let expected_white = (encode_pq(335.0 / 10000.0) * 65535.0).round() as u16;
        assert!((gamma_lut[idx_335].red as i32 - expected_white as i32).abs() < 500);
    }

    #[test]
    fn test_print_hardware_color_support() {
        use crate::backend::allocator::format::FormatSet;
        use crate::backend::drm::color::{CrtcColorCapabilities, DrmScanoutCapabilities, ScanoutPlan};
        use crate::backend::drm::device::DrmDeviceFd;
        use crate::utils::DeviceFd;
        use crate::wayland::color::management::{ImageDescription, TransferFunction};
        use drm::Device as BasicDevice;
        use drm::control::Device as ControlDevice;
        use std::fs::OpenOptions;
        use std::os::unix::io::OwnedFd;

        println!(
            "\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                      DRM HARDWARE COLOR MANAGEMENT SCANOUT REPORT                         ║"
        );
        println!(
            "╚═══════════════════════════════════════════════════════════════════════════════════════════╝"
        );

        let candidates = ["/dev/dri/card1", "/dev/dri/card0", "/dev/dri/card2"];
        let mut found_any = false;

        for path in candidates {
            let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else {
                continue;
            };
            found_any = true;
            let device_fd = DeviceFd::from(OwnedFd::from(file));
            let drm_fd = DrmDeviceFd::new(device_fd);
            let _ = drm_fd.set_client_capability(drm::ClientCapability::UniversalPlanes, true);
            let _ = drm_fd.set_client_capability(drm::ClientCapability::Atomic, true);
            let _ = drm_fd.set_client_capability(drm::ClientCapability::PlaneColorPipeline, true);

            let driver = drm_fd
                .get_driver()
                .map(|d| {
                    format!(
                        "{} ({}) - {}",
                        d.name().to_string_lossy(),
                        d.date().to_string_lossy(),
                        d.description().to_string_lossy()
                    )
                })
                .unwrap_or_else(|_| "Unknown Driver".to_string());

            println!("\n[DRM Node: {}]", path);
            println!("  Driver: {}", driver);

            let res = match drm_fd.resource_handles() {
                Ok(r) => r,
                Err(err) => {
                    println!("  Failed to get resources: {}", err);
                    continue;
                }
            };
            let planes = drm_fd.plane_handles().unwrap_or_default();

            println!(
                "  Resources: {} Connectors, {} CRTCs, {} Planes",
                res.connectors().len(),
                res.crtcs().len(),
                planes.len()
            );

            // Connectors inspection
            println!("\n  --- Connectors ---");
            for conn_handle in res.connectors() {
                if let Ok(conn_info) = drm_fd.get_connector(*conn_handle, false) {
                    let name = format!("{:?}", conn_info.interface());
                    let state = format!("{:?}", conn_info.state());
                    println!("    • Connector {:?} ({}): State = {}", conn_handle, name, state);

                    if let Ok(props) = drm_fd.get_properties(*conn_handle) {
                        let (prop_ids, prop_vals) = props.as_props_and_values();
                        for (prop_id, val) in prop_ids.iter().zip(prop_vals.iter()) {
                            if let Ok(info) = drm_fd.get_property(*prop_id) {
                                let pname = info.name().to_string_lossy();
                                if pname == "Colorspace" {
                                    let mut enums = Vec::new();
                                    if let drm::control::property::ValueType::Enum(items) = info.value_type()
                                    {
                                        let (_, enum_values) = items.values();
                                        for item in enum_values {
                                            enums.push(item.name().to_string_lossy().into_owned());
                                        }
                                    }
                                    println!(
                                        "      - Colorspace property present (current raw val: {}). Supported: {:?}",
                                        val, enums
                                    );
                                } else if pname == "HDR_OUTPUT_METADATA" {
                                    println!(
                                        "      - HDR_OUTPUT_METADATA property present (blob id: {})",
                                        val
                                    );
                                } else if pname == "max bpc" {
                                    println!("      - max bpc property present (current val: {})", val);
                                }
                            }
                        }
                    }
                }
            }

            // CRTCs inspection
            println!("\n  --- CRTCs Color Pipeline Capabilities ---");
            let mut best_crtc_color = CrtcColorCapabilities::default();
            for crtc_handle in res.crtcs() {
                let mut has_gamma = false;
                let mut has_degamma = false;
                let mut has_ctm = false;
                let mut has_vrr = false;
                let mut gamma_size = 0u64;
                let mut degamma_size = 0u64;

                if let Ok(props) = drm_fd.get_properties(*crtc_handle) {
                    let (prop_ids, prop_vals) = props.as_props_and_values();
                    for (prop_id, val) in prop_ids.iter().zip(prop_vals.iter()) {
                        if let Ok(info) = drm_fd.get_property(*prop_id) {
                            let pname = info.name().to_string_lossy();
                            match pname.as_ref() {
                                "GAMMA_LUT" => has_gamma = true,
                                "GAMMA_LUT_SIZE" => gamma_size = *val,
                                "DEGAMMA_LUT" => has_degamma = true,
                                "DEGAMMA_LUT_SIZE" => degamma_size = *val,
                                "CTM" => has_ctm = true,
                                "VRR_ENABLED" => has_vrr = true,
                                _ => {}
                            }
                        }
                    }
                }

                println!("    • CRTC {:?}:", crtc_handle);
                println!(
                    "        GAMMA_LUT:     {} (size: {} entries)",
                    if has_gamma { "SUPPORTED" } else { "NO" },
                    gamma_size
                );
                println!(
                    "        DEGAMMA_LUT:   {} (size: {} entries)",
                    if has_degamma { "SUPPORTED" } else { "NO" },
                    degamma_size
                );
                println!(
                    "        CTM (3x3 S31): {}",
                    if has_ctm { "SUPPORTED" } else { "NO" }
                );
                println!(
                    "        VRR_ENABLED:   {}",
                    if has_vrr { "SUPPORTED" } else { "NO" }
                );

                if has_gamma && gamma_size >= best_crtc_color.gamma_lut_size {
                    best_crtc_color = CrtcColorCapabilities {
                        has_gamma_lut: has_gamma,
                        gamma_lut_size: gamma_size,
                        has_degamma_lut: has_degamma,
                        degamma_lut_size: degamma_size,
                        has_ctm,
                    };
                }
            }

            // Planes inspection
            println!("\n  --- Planes Color & Format Capabilities ---");
            let mut primary_supports_fp16 = false;
            let mut primary_supports_10bit = false;
            let mut primary_supports_colorop = false;
            let primary_format_set = FormatSet::default();

            for plane_handle in &planes {
                if let Ok(plane_info) = drm_fd.get_plane(*plane_handle) {
                    let mut plane_type = "Overlay";
                    let mut has_colorop = false;

                    if let Ok(props) = drm_fd.get_properties(*plane_handle) {
                        let (prop_ids, prop_vals) = props.as_props_and_values();
                        for (prop_id, val) in prop_ids.iter().zip(prop_vals.iter()) {
                            if let Ok(info) = drm_fd.get_property(*prop_id) {
                                let pname = info.name().to_string_lossy();
                                if pname == "type" {
                                    plane_type = match *val {
                                        1 => "Primary",
                                        2 => "Cursor",
                                        _ => "Overlay",
                                    };
                                } else if pname == "COLOR_PIPELINE" {
                                    has_colorop = true;
                                }
                            }
                        }
                    }

                    let mut has_fp16 = false;
                    let mut has_10bit = false;
                    for raw_fmt in plane_info.formats() {
                        if let Ok(fourcc) = drm_fourcc::DrmFourcc::try_from(*raw_fmt) {
                            match fourcc {
                                drm_fourcc::DrmFourcc::Abgr16161616f
                                | drm_fourcc::DrmFourcc::Xbgr16161616f
                                | drm_fourcc::DrmFourcc::Argb16161616f
                                | drm_fourcc::DrmFourcc::Xrgb16161616f => has_fp16 = true,
                                drm_fourcc::DrmFourcc::Xbgr2101010
                                | drm_fourcc::DrmFourcc::Abgr2101010
                                | drm_fourcc::DrmFourcc::Xrgb2101010
                                | drm_fourcc::DrmFourcc::Argb2101010 => has_10bit = true,
                                _ => {}
                            }
                        }
                    }

                    println!(
                        "    • Plane {:?} [{}]: Formats: {}, FP16: {}, 10-bit: {}, COLOR_PIPELINE: {}",
                        plane_handle,
                        plane_type,
                        plane_info.formats().len(),
                        if has_fp16 { "YES" } else { "NO" },
                        if has_10bit { "YES" } else { "NO" },
                        if has_colorop { "YES" } else { "NO" }
                    );

                    if plane_type == "Primary" {
                        primary_supports_fp16 = has_fp16;
                        primary_supports_10bit = has_10bit;
                        primary_supports_colorop = has_colorop;
                    }
                }
            }

            let scanout_caps = DrmScanoutCapabilities {
                crtc_color: best_crtc_color,
                supports_plane_colorop: primary_supports_colorop,
                primary_plane_formats: primary_format_set,
                supports_fp16: primary_supports_fp16,
                supports_10bit: primary_supports_10bit,
            };

            println!("\n  ══════════════════════════════════════════════════════════════════════════");
            println!("   HARDWARE COLOR CONVERSION SCANOUT MATRIX EVALUATION");
            println!("  ══════════════════════════════════════════════════════════════════════════");

            let test_cases = [
                (
                    "SDR Display (sRGB)",
                    false,
                    "sRGB (8/10-bit SDR)",
                    Some(ImageDescription::SRGB),
                ),
                (
                    "SDR Display (sRGB)",
                    false,
                    "scRGB (FP16 Linear)",
                    Some(ImageDescription::WINDOWS_SCRGB),
                ),
                (
                    "SDR Display (sRGB)",
                    false,
                    "PQ BT.2020 (HDR10)",
                    Some(ImageDescription::WINDOWS_BT2100),
                ),
                ("SDR Display (sRGB)", false, "HLG BT.2020", {
                    let mut h = ImageDescription::WINDOWS_BT2100;
                    h.transfer = TransferFunction::Hlg;
                    Some(h)
                }),
                (
                    "HDR Display (BT.2020 PQ)",
                    true,
                    "PQ BT.2020 (HDR10)",
                    Some(ImageDescription::WINDOWS_BT2100),
                ),
                (
                    "HDR Display (BT.2020 PQ)",
                    true,
                    "scRGB (FP16 Linear)",
                    Some(ImageDescription::WINDOWS_SCRGB),
                ),
                (
                    "HDR Display (BT.2020 PQ)",
                    true,
                    "sRGB (Tagged SDR)",
                    Some(ImageDescription::SRGB),
                ),
                ("HDR Display (BT.2020 PQ)", true, "HLG BT.2020", {
                    let mut h = ImageDescription::WINDOWS_BT2100;
                    h.transfer = TransferFunction::Hlg;
                    Some(h)
                }),
            ];

            let mut current_target = "";
            for (target_name, is_hdr, content_name, desc) in &test_cases {
                if current_target != *target_name {
                    current_target = *target_name;
                    println!("\n  ▶ Target: {}", target_name);
                }
                let plan = scanout_caps.evaluate_scanout_plan(*is_hdr, desc.as_ref(), 203);
                let plan_str = match plan {
                    ScanoutPlan::DirectPassthrough => "DirectPassthrough (0% GPU, 0% CRTC/Plane)".to_string(),
                    ScanoutPlan::PlaneColorop(conv) => {
                        format!("PlaneColorop (DRM Plane COLOR_PIPELINE: {:?})", conv)
                    }
                    ScanoutPlan::CrtcHardware(conv) => {
                        let state = conv.to_crtc_color_state(
                            scanout_caps.crtc_color.gamma_lut_size as usize,
                            scanout_caps.crtc_color.degamma_lut_size as usize,
                        );
                        format!(
                            "CrtcHardware (DEGAMMA: {}, CTM: {}, GAMMA: {})",
                            if state.degamma_lut.is_some() {
                                format!("{} entries", scanout_caps.crtc_color.degamma_lut_size)
                            } else {
                                "None".to_string()
                            },
                            if state.ctm.is_some() {
                                "S31.32 matrix"
                            } else {
                                "None"
                            },
                            if state.gamma_lut.is_some() {
                                format!("{} entries", scanout_caps.crtc_color.gamma_lut_size)
                            } else {
                                "None".to_string()
                            }
                        )
                    }
                    ScanoutPlan::VulkanFastDirectFlip => {
                        "VulkanFastDirectFlip (GPU Compute Shader Fallback)".to_string()
                    }
                };
                println!("    • {:<22} ➔ {}", content_name, plan_str);
            }
            println!("\n  ══════════════════════════════════════════════════════════════════════════\n");
        }

        if !found_any {
            println!("  [NOTE] No accessible DRM card node found in /dev/dri/ (running in container/CI).");
        }
    }
}
