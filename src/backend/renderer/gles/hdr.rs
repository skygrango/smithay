//! Built-in HDR and Color Management helpers for GLES rendering.

use crate::backend::renderer::Color32F;
use crate::backend::renderer::gles::Uniform;
use crate::backend::renderer::gles::uniform::UniformValue;
use crate::wayland::color::management::{ImageDescription, Primaries, TransferFunction};

/// Output HDR target configuration for the renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrOutputConfig {
    /// Target output reference white in cd/m² (diffuse white luminance).
    pub reference_white: f32,
    /// SDR gamma exponent (0.0 = sRGB transfer function, 2.2 = pure power 2.2, 2.4 = BT.1886).
    pub sdr_gamma: f32,
    /// Gamut stretch factor (0.0 = accurate colorimetric BT.709->BT.2020, 1.0 = native vivid gamut).
    pub gamut_stretch: f32,
}

impl Default for HdrOutputConfig {
    fn default() -> Self {
        Self {
            reference_white: 203.0,
            sdr_gamma: 2.2,
            gamut_stretch: 0.0,
        }
    }
}

impl HdrOutputConfig {
    /// Create a new HDR output configuration.
    pub fn new(reference_white: f32, sdr_gamma: f32, gamut_stretch: f32) -> Self {
        Self {
            reference_white,
            sdr_gamma,
            gamut_stretch,
        }
    }
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
pub fn sdr_color_to_pq(
    color: Color32F,
    reference_white: f32,
    sdr_gamma: f32,
    gamut_stretch: f32,
) -> Color32F {
    let alpha = color.a();
    let unpremultiply = |value: f32| if alpha > 0.00001 { value / alpha } else { 0.0 };
    let decode = |value: f32| decode_sdr(value, sdr_gamma);
    let pq = |value: f32| {
        const M1: f32 = 0.159_301_76;
        const M2: f32 = 78.84375;
        const C1: f32 = 0.8359375;
        const C2: f32 = 18.851_563;
        const C3: f32 = 18.6875;
        let p = value.max(0.0).powf(M1);
        ((C1 + C2 * p) / (1.0 + C3 * p)).powf(M2)
    };

    let r = decode(unpremultiply(color.r()));
    let g = decode(unpremultiply(color.g()));
    let b = decode(unpremultiply(color.b()));
    let stretch = gamut_stretch.clamp(0.0, 1.0);
    let mix = |converted: f32, native: f32| converted + (native - converted) * stretch;
    let scale = reference_white.clamp(80.0, 10_000.0) / 10_000.0;
    Color32F::new(
        pq(mix(0.627404 * r + 0.329282 * g + 0.043314 * b, r) * scale) * alpha,
        pq(mix(0.069097 * r + 0.919540 * g + 0.011362 * b, g) * scale) * alpha,
        pq(mix(0.016392 * r + 0.088013 * g + 0.895595 * b, b) * scale) * alpha,
        alpha,
    )
}

/// Helper function to transform an sRGB color using standard defaults (Gamma 2.2).
pub fn srgb_color_to_pq(color: Color32F, reference_white: f32) -> Color32F {
    sdr_color_to_pq(color, reference_white, 2.2, 0.0)
}

/// Adjusts HDR texture shader uniforms for a surface based on its committed ImageDescription.
pub fn update_hdr_surface_uniforms(
    desc: Option<&ImageDescription>,
    uniforms: &mut [Uniform<'static>],
    config: &HdrOutputConfig,
) {
    if let Some(desc) = desc {
        if desc.transfer == TransferFunction::Hlg {
            let content_reference = desc
                .luminances
                .map(|(_min, _max, reference)| reference.max(80) as f32)
                .unwrap_or(203.0);
            for uniform in uniforms.iter_mut() {
                match uniform.name.as_ref() {
                    "hdr_reference_white" => uniform.value = UniformValue::_1f(config.reference_white),
                    "hdr_sdr_gamma" => uniform.value = UniformValue::_1f(config.sdr_gamma),
                    "hdr_gamut_stretch" => uniform.value = UniformValue::_1f(config.gamut_stretch),
                    "hdr_input_pq" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_hlg" => uniform.value = UniformValue::_1f(1.0),
                    "hdr_input_primaries" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_content_reference" => uniform.value = UniformValue::_1f(content_reference),
                    _ => {}
                }
            }
        } else if desc.is_pq_bt2020() {
            let content_reference = desc
                .luminances
                .map(|(_min, _max, reference)| reference.max(80) as f32)
                .unwrap_or(203.0);
            for uniform in uniforms.iter_mut() {
                match uniform.name.as_ref() {
                    "hdr_reference_white" => uniform.value = UniformValue::_1f(config.reference_white),
                    "hdr_sdr_gamma" => uniform.value = UniformValue::_1f(config.sdr_gamma),
                    "hdr_gamut_stretch" => uniform.value = UniformValue::_1f(config.gamut_stretch),
                    "hdr_input_pq" => uniform.value = UniformValue::_1f(1.0),
                    "hdr_input_hlg" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_primaries" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_content_reference" => uniform.value = UniformValue::_1f(content_reference),
                    _ => {}
                }
            }
        } else if desc.windows_scrgb || desc.transfer == TransferFunction::ExtLinear {
            for uniform in uniforms.iter_mut() {
                match uniform.name.as_ref() {
                    "hdr_reference_white" => uniform.value = UniformValue::_1f(80.0),
                    "hdr_sdr_gamma" => uniform.value = UniformValue::_1f(1.0),
                    "hdr_gamut_stretch" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_pq" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_hlg" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_primaries" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_content_reference" => uniform.value = UniformValue::_1f(203.0),
                    _ => {}
                }
            }
        } else {
            let sdr_gamma = match desc.transfer {
                TransferFunction::Bt1886 => 2.4,
                TransferFunction::Gamma22 => 2.2,
                TransferFunction::CompoundPower24 | TransferFunction::Srgb => 0.0,
                _ => config.sdr_gamma,
            };
            let primaries_mode = match desc.primaries.named {
                Some(Primaries::DisplayP3) => 1.0,
                Some(Primaries::Bt2020) => 2.0,
                _ => 0.0,
            };
            let gamut_stretch = if primaries_mode > 0.5 {
                0.0
            } else {
                config.gamut_stretch
            };
            for uniform in uniforms.iter_mut() {
                match uniform.name.as_ref() {
                    "hdr_reference_white" => uniform.value = UniformValue::_1f(config.reference_white),
                    "hdr_sdr_gamma" => uniform.value = UniformValue::_1f(sdr_gamma),
                    "hdr_gamut_stretch" => uniform.value = UniformValue::_1f(gamut_stretch),
                    "hdr_input_pq" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_hlg" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_primaries" => uniform.value = UniformValue::_1f(primaries_mode),
                    "hdr_content_reference" => uniform.value = UniformValue::_1f(203.0),
                    _ => {}
                }
            }
        }
    } else {
        // Untagged default SDR surface
        for uniform in uniforms.iter_mut() {
            match uniform.name.as_ref() {
                "hdr_reference_white" => uniform.value = UniformValue::_1f(config.reference_white),
                "hdr_sdr_gamma" => uniform.value = UniformValue::_1f(config.sdr_gamma),
                "hdr_gamut_stretch" => uniform.value = UniformValue::_1f(config.gamut_stretch),
                "hdr_input_pq" => uniform.value = UniformValue::_1f(0.0),
                "hdr_input_hlg" => uniform.value = UniformValue::_1f(0.0),
                "hdr_input_primaries" => uniform.value = UniformValue::_1f(0.0),
                "hdr_content_reference" => uniform.value = UniformValue::_1f(203.0),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_endpoints_decode_to_linear_endpoints() {
        assert_eq!(decode_sdr(0.0, 0.0), 0.0);
        assert!((decode_sdr(1.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gamma22_decode_keeps_shadows_darker_than_srgb_toe() {
        let srgb = decode_sdr(0.03, 0.0);
        let gamma = decode_sdr(0.03, 2.2);
        assert!((srgb - 0.03 / 12.92).abs() < 1e-6);
        assert!(gamma < srgb / 2.0, "gamma22={gamma} srgb={srgb}");

        assert_eq!(decode_sdr(0.0, 2.2), 0.0);
        assert!((decode_sdr(1.0, 2.2) - 1.0).abs() < 1e-6);
        let white = sdr_color_to_pq(Color32F::new(1.0, 1.0, 1.0, 1.0), 203.0, 2.2, 0.0);
        assert!((white.r() - 0.5807).abs() < 0.0003);
    }

    #[test]
    fn gamut_stretch_moves_primaries_toward_native() {
        let red = Color32F::new(1.0, 0.0, 0.0, 1.0);
        let colorimetric = sdr_color_to_pq(red, 203.0, 2.2, 0.0);
        let native = sdr_color_to_pq(red, 203.0, 2.2, 1.0);
        assert!(native.r() > colorimetric.r());
    }

    #[test]
    fn solid_color_hdr_path_matches_shader_white_point() {
        let white = srgb_color_to_pq(Color32F::new(1.0, 1.0, 1.0, 1.0), 203.0);
        assert!((white.r() - 0.5807).abs() < 0.0003);
        assert!((white.r() - white.g()).abs() < 0.0001);
        assert!((white.g() - white.b()).abs() < 0.0001);

        let transparent = srgb_color_to_pq(Color32F::new(0.0, 0.0, 0.0, 0.0), 203.0);
        assert_eq!(transparent.a(), 0.0);
        assert!(transparent.r().is_finite());
    }

    fn rec709_to_bt2020(rgb: [f64; 3]) -> [f64; 3] {
        [
            0.6274040 * rgb[0] + 0.3292820 * rgb[1] + 0.0433136 * rgb[2],
            0.0690970 * rgb[0] + 0.9195400 * rgb[1] + 0.0113612 * rgb[2],
            0.0163916 * rgb[0] + 0.0880132 * rgb[1] + 0.8955950 * rgb[2],
        ]
    }

    #[test]
    fn rec709_to_bt2020_preserves_neutral_white() {
        let white = rec709_to_bt2020([1.0, 1.0, 1.0]);
        assert!(white.into_iter().all(|c| (c - 1.0).abs() < 2e-6));
    }

    fn st2084_encode(luminance_nits: f64) -> f64 {
        let y = (luminance_nits / 10_000.0).max(0.0);
        let m1 = 0.1593017578125;
        let m2 = 78.84375;
        let c1 = 0.8359375;
        let c2 = 18.8515625;
        let c3 = 18.6875;
        let p = y.powf(m1);
        ((c1 + c2 * p) / (1.0 + c3 * p)).powf(m2)
    }

    #[test]
    fn st2084_matches_reference_code_values() {
        assert!((st2084_encode(100.0) - 0.5081).abs() < 0.0002);
        assert!((st2084_encode(203.0) - 0.5807).abs() < 0.0002);
        assert!((st2084_encode(1_000.0) - 0.7518).abs() < 0.0002);
        assert!((st2084_encode(10_000.0) - 1.0).abs() < 1e-12);
    }

    fn st2084_decode(code: f64) -> f64 {
        let m1 = 0.1593017578125;
        let m2 = 78.84375;
        let c1 = 0.8359375;
        let c2 = 18.8515625;
        let c3 = 18.6875;
        let p = code.clamp(0.0, 1.0).powf(1.0 / m2);
        let y = ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1);
        y * 10_000.0
    }

    fn pq_rescale(code: f64, content_ref: f64, ref_white: f64) -> f64 {
        let ref_scale = ref_white.clamp(80.0, 10_000.0) / content_ref.max(80.0);
        if (ref_scale - 1.0).abs() < 0.001 {
            return code;
        }
        let linear = st2084_decode(code) * ref_scale;
        st2084_encode(linear)
    }

    #[test]
    fn pq_rescale_preserves_code_when_reference_whites_match() {
        let code_203 = st2084_encode(203.0);
        assert_eq!(pq_rescale(code_203, 203.0, 203.0), code_203);
    }

    #[test]
    fn pq_rescale_scales_diffuse_white_to_target_reference_white() {
        let code_203 = st2084_encode(203.0);
        let rescaled_to_300 = pq_rescale(code_203, 203.0, 300.0);
        let decoded_nits = st2084_decode(rescaled_to_300);
        assert!((decoded_nits - 300.0).abs() < 0.1);
    }

    fn hlg_to_scene(e: f64) -> f64 {
        let a = 0.17883277;
        let b = 0.28466892;
        let c = 0.55991073;
        let e = e.clamp(0.0, 1.0);
        if e <= 0.5 {
            (e * e) / 3.0
        } else {
            (((e - c) / a).exp() + b) / 12.0
        }
    }

    fn hlg_to_nits(e: f64) -> f64 {
        let scene = hlg_to_scene(e);
        let ys = scene;
        let gain = ys.max(1e-6).powf(0.2) * 1000.0;
        scene * gain
    }

    #[test]
    fn hlg_matches_itu_bt2100_and_bt2408_levels() {
        assert_eq!(hlg_to_scene(0.0), 0.0);
        assert_eq!(hlg_to_nits(0.0), 0.0);
        assert!((hlg_to_scene(0.5) - 0.25 / 3.0).abs() < 1e-6);

        let white_nits = hlg_to_nits(0.75);
        assert!((white_nits - 203.0).abs() < 1.0);

        let peak_nits = hlg_to_nits(1.0);
        assert!((peak_nits - 1000.0).abs() < 1.0);
    }

    fn p3_to_bt2020(rgb: [f64; 3]) -> [f64; 3] {
        [
            0.7538330 * rgb[0] + 0.1985974 * rgb[1] + 0.0475696 * rgb[2],
            0.0457438 * rgb[0] + 0.9417772 * rgb[1] + 0.0124789 * rgb[2],
            -0.0012103 * rgb[0] + 0.0176017 * rgb[1] + 0.9836086 * rgb[2],
        ]
    }

    #[test]
    fn display_p3_to_bt2020_preserves_neutral_white() {
        let white = p3_to_bt2020([1.0, 1.0, 1.0]);
        assert!(white.into_iter().all(|c| (c - 1.0).abs() < 2e-6));
    }
}
