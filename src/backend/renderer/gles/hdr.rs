//! Built-in HDR and Color Management helpers for GLES rendering.

use crate::backend::renderer::gles::Uniform;
use crate::backend::renderer::gles::uniform::UniformValue;
use crate::wayland::color::management::{ImageDescription, Primaries, TransferFunction};

pub use crate::backend::renderer::color::{
    HdrOutputConfig, decode_pq, decode_sdr, encode_pq, optical_alpha_pq, sdr_color_to_hdr, sdr_color_to_pq,
    srgb_color_to_pq, tonemap_reinhard,
};

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
            let max_content = desc
                .luminances
                .map(|(_min, max, _)| max as f32)
                .or_else(|| desc.max_cll.map(|v| v as f32))
                .or_else(|| desc.mastering_luminance.map(|(_, max)| max as f32))
                .unwrap_or(1000.0);
            for uniform in uniforms.iter_mut() {
                match uniform.name.as_ref() {
                    "hdr_reference_white" => uniform.value = UniformValue::_1f(config.reference_white),
                    "hdr_sdr_gamma" => uniform.value = UniformValue::_1f(config.sdr_gamma),
                    "hdr_gamut_stretch" => uniform.value = UniformValue::_1f(config.gamut_stretch),
                    "hdr_input_pq" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_hlg" => uniform.value = UniformValue::_1f(1.0),
                    "hdr_input_primaries" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_content_reference" => uniform.value = UniformValue::_1f(content_reference),
                    "hdr_max_content_luminance" => uniform.value = UniformValue::_1f(max_content),
                    "hdr_max_destination_luminance" => {
                        uniform.value = UniformValue::_1f(config.max_luminance)
                    }
                    _ => {}
                }
            }
        } else if desc.is_pq_bt2020() {
            let content_reference = desc
                .luminances
                .map(|(_min, _max, reference)| reference.max(80) as f32)
                .unwrap_or(203.0);
            let max_content = desc
                .luminances
                .map(|(_min, max, _)| max as f32)
                .or_else(|| desc.max_cll.map(|v| v as f32))
                .or_else(|| desc.mastering_luminance.map(|(_, max)| max as f32))
                .unwrap_or(1000.0);
            for uniform in uniforms.iter_mut() {
                match uniform.name.as_ref() {
                    "hdr_reference_white" => uniform.value = UniformValue::_1f(config.reference_white),
                    "hdr_sdr_gamma" => uniform.value = UniformValue::_1f(config.sdr_gamma),
                    "hdr_gamut_stretch" => uniform.value = UniformValue::_1f(config.gamut_stretch),
                    "hdr_input_pq" => uniform.value = UniformValue::_1f(1.0),
                    "hdr_input_hlg" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_input_primaries" => uniform.value = UniformValue::_1f(0.0),
                    "hdr_content_reference" => uniform.value = UniformValue::_1f(content_reference),
                    "hdr_max_content_luminance" => uniform.value = UniformValue::_1f(max_content),
                    "hdr_max_destination_luminance" => {
                        uniform.value = UniformValue::_1f(config.max_luminance)
                    }
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
                    "hdr_max_content_luminance" => uniform.value = UniformValue::_1f(config.max_luminance),
                    "hdr_max_destination_luminance" => {
                        uniform.value = UniformValue::_1f(config.max_luminance)
                    }
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
                    "hdr_max_content_luminance" => uniform.value = UniformValue::_1f(config.reference_white),
                    "hdr_max_destination_luminance" => {
                        uniform.value = UniformValue::_1f(config.max_luminance)
                    }
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
                "hdr_max_content_luminance" => uniform.value = UniformValue::_1f(config.reference_white),
                "hdr_max_destination_luminance" => uniform.value = UniformValue::_1f(config.max_luminance),
                _ => {}
            }
        }
    }

    for uniform in uniforms.iter_mut() {
        if uniform.name.as_ref() == "hdr_hardware_offload" {
            uniform.value = UniformValue::_1f(if config.hardware_offload { 1.0 } else { 0.0 });
        } else if uniform.name.as_ref() == "hdr_target_is_sdr" {
            uniform.value = UniformValue::_1f(if config.is_sdr { 1.0 } else { 0.0 });
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

    #[test]
    fn optical_alpha_preserves_perceptual_luminance() {
        let ref_white = 203.0;
        let white_norm = ref_white / 10_000.0;
        let pq_white = encode_pq(white_norm);

        // 50% opacity
        let alpha = 0.5;
        let eff_alpha = optical_alpha_pq(alpha, ref_white);
        let blended_code = pq_white * eff_alpha;
        let luminance = decode_pq(blended_code) * 10_000.0;

        // With optical alpha, 50% opacity of 203 cd/m² produces ~101.5 cd/m² (half luminance).
        assert!(
            (luminance - 101.5).abs() < 1.0,
            "Expected ~101.5 nits, got {luminance}"
        );

        // Without optical alpha (linear code blending), 50% opacity produces only ~5.4 cd/m²!
        let uncompensated_code = pq_white * alpha;
        let uncompensated_lum = decode_pq(uncompensated_code) * 10_000.0;
        assert!(
            uncompensated_lum < 10.0,
            "Uncompensated luminance={uncompensated_lum}"
        );
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

    #[test]
    fn reinhard_tonemapping_scales_peak_to_target_smoothly() {
        let ref_white = 203.0;
        let dest_max = 600.0;
        let content_max = 2000.0;

        assert_eq!(tonemap_reinhard(0.0, ref_white, content_max, dest_max), 0.0);

        let mapped_ref = tonemap_reinhard(ref_white, ref_white, content_max, dest_max);
        assert!(mapped_ref > 100.0 && mapped_ref < ref_white);

        let mapped_dest_max = tonemap_reinhard(dest_max, ref_white, content_max, dest_max);
        assert!(mapped_dest_max < dest_max);

        let mapped_peak = tonemap_reinhard(content_max, ref_white, content_max, dest_max);
        assert!((mapped_peak - dest_max).abs() < 1e-4);

        // Within range without excess peak -> identity/clamp
        let no_excess = tonemap_reinhard(500.0, ref_white, 600.0, 600.0);
        assert_eq!(no_excess, 500.0);
    }

    #[test]
    fn sdr_color_to_hdr_hardware_offload() {
        let white = Color32F::new(1.0, 1.0, 1.0, 1.0);
        let offloaded = sdr_color_to_hdr(white, 203.0, 2.2, 0.0, true, false);
        assert!((offloaded.r() - 1.0).abs() < 1e-5);
        assert!((offloaded.g() - 1.0).abs() < 1e-5);
        assert!((offloaded.b() - 1.0).abs() < 1e-5);
        assert_eq!(offloaded.a(), 1.0);

        let shader_pq = sdr_color_to_hdr(white, 203.0, 2.2, 0.0, false, false);
        // In PQ space, 203 nits is ~0.5807
        assert!((shader_pq.r() - 0.5807).abs() < 1e-3);

        let sdr_passthrough = sdr_color_to_hdr(white, 203.0, 2.2, 0.0, false, true);
        assert_eq!(sdr_passthrough, white);
    }

    fn bt2020_to_rec709(rgb: [f64; 3]) -> [f64; 3] {
        [
            1.6604903 * rgb[0] - 0.5876391 * rgb[1] - 0.0728516 * rgb[2],
            -0.1245500 * rgb[0] + 1.1328999 * rgb[1] - 0.0083480 * rgb[2],
            -0.0181511 * rgb[0] - 0.1005787 * rgb[1] + 1.1187299 * rgb[2],
        ]
    }

    #[test]
    fn bt2020_to_rec709_preserves_neutral_white() {
        let white = bt2020_to_rec709([1.0, 1.0, 1.0]);
        assert!(white.into_iter().all(|c| (c - 1.0).abs() < 2e-6));
    }

    #[test]
    fn sdr_tonemapping_config_sets_correct_uniforms() {
        let config = HdrOutputConfig::sdr_tonemapping();
        assert!(config.is_sdr);
        assert_eq!(config.reference_white, 203.0);
        assert_eq!(config.max_luminance, 203.0);

        let mut uniforms = vec![
            Uniform::new("hdr_reference_white", 0.0_f32),
            Uniform::new("hdr_sdr_gamma", 0.0_f32),
            Uniform::new("hdr_gamut_stretch", 0.0_f32),
            Uniform::new("hdr_input_pq", 0.0_f32),
            Uniform::new("hdr_input_hlg", 0.0_f32),
            Uniform::new("hdr_input_primaries", 0.0_f32),
            Uniform::new("hdr_content_reference", 0.0_f32),
            Uniform::new("hdr_max_content_luminance", 0.0_f32),
            Uniform::new("hdr_max_destination_luminance", 0.0_f32),
        ];

        // 1. SDR surface
        update_hdr_surface_uniforms(Some(&ImageDescription::SRGB), &mut uniforms, &config);
        let find_val = |uniforms: &[Uniform<'_>], name: &str| -> f32 {
            match uniforms.iter().find(|u| u.name.as_ref() == name).unwrap().value {
                UniformValue::_1f(v) => v,
                _ => panic!(),
            }
        };
        assert_eq!(find_val(&uniforms, "hdr_input_pq"), 0.0);
        assert_eq!(find_val(&uniforms, "hdr_input_hlg"), 0.0);
        assert_eq!(find_val(&uniforms, "hdr_max_destination_luminance"), 203.0);

        // 2. HDR PQ surface
        let hdr_desc = ImageDescription::WINDOWS_BT2100;
        update_hdr_surface_uniforms(Some(&hdr_desc), &mut uniforms, &config);
        assert_eq!(find_val(&uniforms, "hdr_input_pq"), 1.0);
        assert_eq!(find_val(&uniforms, "hdr_input_hlg"), 0.0);
        assert_eq!(find_val(&uniforms, "hdr_max_destination_luminance"), 203.0);
        assert_eq!(find_val(&uniforms, "hdr_max_content_luminance"), 10000.0);
    }
}
