use std::ops::Mul;

/// A four-component color representing pre-multiplied RGBA color values
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Color32F([f32; 4]);

impl Color32F {
    /// Initialize a new [`Color32F`]
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }
}

impl Color32F {
    /// Transparent color
    pub const TRANSPARENT: Color32F = Color32F::new(0.0, 0.0, 0.0, 0.0);

    /// Solid black color
    pub const BLACK: Color32F = Color32F::new(0f32, 0f32, 0f32, 1f32);
}

impl Color32F {
    /// Red color component
    #[inline]
    pub fn r(&self) -> f32 {
        self.0[0]
    }

    /// Green color component
    #[inline]
    pub fn g(&self) -> f32 {
        self.0[1]
    }

    /// Blue color component
    #[inline]
    pub fn b(&self) -> f32 {
        self.0[2]
    }

    /// Alpha color component
    #[inline]
    pub fn a(&self) -> f32 {
        self.0[3]
    }

    /// Color components
    #[inline]
    pub fn components(self) -> [f32; 4] {
        self.0
    }
}

impl Color32F {
    /// Test if the color represents a opaque color
    #[inline]
    pub fn is_opaque(&self) -> bool {
        self.a() == 1f32
    }
}

impl From<[f32; 4]> for Color32F {
    #[inline]
    fn from(value: [f32; 4]) -> Self {
        Self(value)
    }
}

impl Mul<f32> for Color32F {
    type Output = Color32F;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.r() * rhs, self.g() * rhs, self.b() * rhs, self.a() * rhs)
    }
}

/// Output HDR target configuration for the renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrOutputConfig {
    /// Target output reference white in cd/m² (diffuse white luminance).
    pub reference_white: f32,
    /// SDR gamma exponent (0.0 = sRGB transfer function, 2.2 = pure power 2.2, 2.4 = BT.1886).
    pub sdr_gamma: f32,
    /// Gamut stretch factor (0.0 = accurate colorimetric BT.709->BT.2020, 1.0 = native vivid gamut).
    pub gamut_stretch: f32,
    /// Maximum output destination peak luminance in cd/m².
    pub max_luminance: f32,
    /// Whether the DRM CRTC hardware pipeline offloads EOTF encoding via GAMMA_LUT.
    /// When true, shaders output linear light without calling encode_pq.
    pub hardware_offload: bool,
    /// Whether the output target is SDR (sRGB / Rec.709).
    /// When true, SDR surfaces are passed through untransformed, and HDR surfaces
    /// are tone-mapped to SDR sRGB.
    pub is_sdr: bool,
}

impl Default for HdrOutputConfig {
    fn default() -> Self {
        Self {
            reference_white: 203.0,
            sdr_gamma: 2.2,
            gamut_stretch: 0.0,
            max_luminance: 1000.0,
            hardware_offload: false,
            is_sdr: false,
        }
    }
}

impl HdrOutputConfig {
    /// Create a new HDR output configuration.
    pub fn new(
        reference_white: f32,
        sdr_gamma: f32,
        gamut_stretch: f32,
        max_luminance: f32,
        hardware_offload: bool,
    ) -> Self {
        Self {
            reference_white,
            sdr_gamma,
            gamut_stretch,
            max_luminance,
            hardware_offload,
            is_sdr: false,
        }
    }

    /// Create an SDR output configuration with HDR surface tone-mapping enabled.
    pub fn sdr_tonemapping() -> Self {
        Self::sdr_tonemapping_with_reference(203.0)
    }

    /// Create an SDR output configuration with HDR surface tone-mapping enabled and custom reference white.
    pub fn sdr_tonemapping_with_reference(reference_white: f32) -> Self {
        Self {
            reference_white,
            sdr_gamma: 0.0, // standard piecewise sRGB
            gamut_stretch: 0.0,
            max_luminance: reference_white,
            hardware_offload: false,
            is_sdr: true,
        }
    }
}

/// Modified Reinhard tone-mapping curve on luminance in cd/m² (mirroring KWin's ICtCp tonemapping).
pub fn tonemap_reinhard(luminance: f32, reference_white: f32, max_content: f32, max_destination: f32) -> f32 {
    if max_content <= max_destination * 1.01 {
        return luminance.clamp(0.0, max_destination);
    }
    let rel_lum = (luminance / reference_white).max(0.0);
    let input_range = max_content / reference_white;
    let output_range = max_destination / reference_white;
    let v = (output_range * (1.0 + input_range) - input_range) / (input_range * input_range);
    let mapped_rel = rel_lum * (1.0 + rel_lum * v) / (1.0 + rel_lum);
    (mapped_rel * reference_white).clamp(0.0, max_destination)
}

/// CPU mirror of `decode_sdr` in the HDR shaders: `gamma == 0.0` is the
/// piecewise sRGB curve, anything else a pure power law.
pub fn decode_sdr(value: f32, gamma: f32) -> f32 {
    if gamma > 0.0 {
        value.max(0.0).powf(gamma)
    } else if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Transforms an sRGB / SDR solid color to PQ / BT.2020 matching the HDR output target.
/// Encodes normalized absolute luminance (0.0 to 1.0, where 1.0 = 10,000 cd/m²) to a ST 2084 (PQ) code value.
pub fn encode_pq(value: f32) -> f32 {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.6875;
    let p = value.max(0.0).powf(M1);
    ((C1 + C2 * p) / (1.0 + C3 * p)).powf(M2)
}

/// Decodes a ST 2084 (PQ) code value to normalized absolute luminance (0.0 to 1.0, where 1.0 = 10,000 cd/m²).
pub fn decode_pq(code: f32) -> f32 {
    const M1_INV: f32 = 1.0 / 0.159_301_76;
    const M2_INV: f32 = 1.0 / 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.6875;
    let p = code.clamp(0.0, 1.0).powf(M2_INV);
    ((p - C1).max(0.0) / (C2 - C3 * p)).powf(M1_INV)
}

/// Computes the optical alpha compensation in PQ space.
pub fn optical_alpha_pq(alpha: f32, reference_white: f32) -> f32 {
    let a = alpha.clamp(0.0, 1.0);
    if a <= 0.0 {
        return 0.0;
    }
    if a >= 1.0 {
        return 1.0;
    }
    let white_norm = reference_white.clamp(80.0, 10_000.0) / 10_000.0;
    let pq_white = encode_pq(white_norm);
    let pq_lum = encode_pq(white_norm * a);
    (pq_lum / pq_white.max(0.001)).clamp(0.0, 1.0)
}

/// Transforms an SDR color to PQ space with optical alpha compensation.
pub fn sdr_color_to_pq(
    color: Color32F,
    reference_white: f32,
    sdr_gamma: f32,
    gamut_stretch: f32,
) -> Color32F {
    let alpha = color.a();
    let unpremultiply = |value: f32| if alpha > 0.00001 { value / alpha } else { 0.0 };
    let decode = |value: f32| decode_sdr(value, sdr_gamma);

    let r = decode(unpremultiply(color.r()));
    let g = decode(unpremultiply(color.g()));
    let b = decode(unpremultiply(color.b()));
    let stretch = gamut_stretch.clamp(0.0, 1.0);
    let mix = |converted: f32, native: f32| converted + (native - converted) * stretch;
    let scale = reference_white.clamp(80.0, 10_000.0) / 10_000.0;
    let eff_alpha = alpha;
    Color32F::new(
        encode_pq(mix(0.627404 * r + 0.329282 * g + 0.043314 * b, r) * scale) * eff_alpha,
        encode_pq(mix(0.069097 * r + 0.919540 * g + 0.011362 * b, g) * scale) * eff_alpha,
        encode_pq(mix(0.016392 * r + 0.088013 * g + 0.895595 * b, b) * scale) * eff_alpha,
        eff_alpha,
    )
}

/// Transforms an SDR solid color to the HDR target space, accounting for hardware offload.
pub fn sdr_color_to_hdr(
    color: Color32F,
    reference_white: f32,
    sdr_gamma: f32,
    gamut_stretch: f32,
    hardware_offload: bool,
    is_sdr: bool,
) -> Color32F {
    if is_sdr {
        return color;
    }
    if hardware_offload {
        let alpha = color.a();
        let unpremultiply = |value: f32| if alpha > 0.00001 { value / alpha } else { 0.0 };
        let decode = |value: f32| decode_sdr(value, sdr_gamma);

        let r = decode(unpremultiply(color.r()));
        let g = decode(unpremultiply(color.g()));
        let b = decode(unpremultiply(color.b()));
        let stretch = gamut_stretch.clamp(0.0, 1.0);
        let mix = |converted: f32, native: f32| converted + (native - converted) * stretch;
        Color32F::new(
            mix(0.627404 * r + 0.329282 * g + 0.043314 * b, r) * alpha,
            mix(0.069097 * r + 0.919540 * g + 0.011362 * b, g) * alpha,
            mix(0.016392 * r + 0.088013 * g + 0.895595 * b, b) * alpha,
            alpha,
        )
    } else {
        sdr_color_to_pq(color, reference_white, sdr_gamma, gamut_stretch)
    }
}

/// Helper function to transform an sRGB color using standard defaults (Gamma 2.2).
pub fn srgb_color_to_pq(color: Color32F, reference_white: f32) -> Color32F {
    sdr_color_to_pq(color, reference_white, 2.2, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdr_tonemapping_with_reference() {
        let config = HdrOutputConfig::sdr_tonemapping_with_reference(350.0);
        assert_eq!(config.reference_white, 350.0);
        assert_eq!(config.max_luminance, 350.0);
        assert!(config.is_sdr);
        assert_eq!(config.sdr_gamma, 0.0);
    }
}
