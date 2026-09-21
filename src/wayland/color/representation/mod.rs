mod protocol;
use wayland_server::{Dispatch, DisplayHandle, GlobalDispatch, Weak, protocol::wl_surface::WlSurface};

use crate::wayland::compositor::{self, Cacheable, SurfaceData};

pub use self::protocol::*;
pub use self::wp_color_representation_surface_v1::{AlphaMode, ChromaLocation, Coefficients, Range};
mod dispatch;

use std::sync::Mutex;

#[derive(Debug)]
pub struct ColorRepresentationState {
    pub(crate) alpha_modes: Vec<AlphaMode>,
    pub(crate) coefficients_and_ranges: Vec<(Coefficients, Range)>,
    pub(crate) known_instances: Vec<wp_color_representation_manager_v1::WpColorRepresentationManagerV1>,
}

pub trait ColorRepresentationHandler {
    fn color_representation_state(&mut self) -> &mut ColorRepresentationState;
}

#[derive(Debug, Clone, Copy, Default)]
struct ColorRepresentationSurfaceCachedState {
    coefficients: Option<Coefficients>,
    range: Option<Range>,
    chroma_location: Option<ChromaLocation>,
    alpha_mode: Option<AlphaMode>,
}

impl Cacheable for Option<ColorRepresentationSurfaceCachedState> {
    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        *self
    }

    fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
        match (self, into) {
            (Some(this), Some(into)) => {
                into.coefficients = this.coefficients.or_else(|| into.coefficients.take());
                into.range = this.range.or_else(|| into.range.take());
                into.chroma_location = this.chroma_location.or_else(|| into.chroma_location.take());
                into.alpha_mode = this.alpha_mode.or_else(|| into.alpha_mode.take());
            }
            (this, into) => {
                *into = this;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorRepresentation {
    pub alpha_mode: AlphaMode,
    pub coefficients: Option<Coefficients>,
    pub range: Option<Range>,
    pub chroma_location: Option<ChromaLocation>,
}

pub fn get_color_representation(surface: &WlSurface) -> ColorRepresentation {
    compositor::with_states(surface, get_color_representation_from_states)
}

pub fn get_color_representation_from_states(states: &SurfaceData) -> ColorRepresentation {
    states
        .cached_state
        .get::<Option<ColorRepresentationSurfaceCachedState>>()
        .current()
        .as_ref()
        .map(|state| ColorRepresentation {
            coefficients: state.coefficients,
            range: state.range,
            chroma_location: state.chroma_location,
            alpha_mode: state.alpha_mode.unwrap_or(AlphaMode::PremultipliedElectrical),
        })
        .unwrap_or_else(|| ColorRepresentation {
            alpha_mode: AlphaMode::PremultipliedElectrical,
            coefficients: None,
            range: None,
            chroma_location: None,
        })
}

#[derive(Debug)]
struct ColorRepresentationSurfaceData {
    instance: Mutex<Option<wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1>>,
}

impl ColorRepresentationSurfaceData {
    fn new() -> Self {
        Self {
            instance: Mutex::new(None),
        }
    }

    fn is_resource_attached(&self) -> bool {
        self.instance.lock().unwrap().is_some()
    }
}

impl ColorRepresentationState {
    pub fn new<D>(
        dh: &DisplayHandle,
        alpha_modes: impl IntoIterator<Item = AlphaMode>,
        coefficients_and_ranges: impl IntoIterator<Item = (Coefficients, Range)>,
    ) -> ColorRepresentationState
    where
        D: GlobalDispatch<wp_color_representation_manager_v1::WpColorRepresentationManagerV1, ()>
            + Dispatch<wp_color_representation_manager_v1::WpColorRepresentationManagerV1, ()>
            + Dispatch<wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1, Weak<WlSurface>>
            + ColorRepresentationHandler
            + 'static,
    {
        dh.create_global::<D, wp_color_representation_manager_v1::WpColorRepresentationManagerV1, ()>(1, ());
        ColorRepresentationState {
            alpha_modes: alpha_modes.into_iter().collect(),
            coefficients_and_ranges: coefficients_and_ranges.into_iter().collect(),
            known_instances: Vec::new(),
        }
    }

    pub fn default_state<D>(dh: &DisplayHandle) -> ColorRepresentationState
    where
        D: GlobalDispatch<wp_color_representation_manager_v1::WpColorRepresentationManagerV1, ()>
            + Dispatch<wp_color_representation_manager_v1::WpColorRepresentationManagerV1, ()>
            + Dispatch<wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1, Weak<WlSurface>>
            + ColorRepresentationHandler
            + 'static,
    {
        Self::new::<D>(
            dh,
            [AlphaMode::PremultipliedElectrical],
            [
                (Coefficients::Identity, Range::Full),
                (Coefficients::Bt709, Range::Limited),
                (Coefficients::Bt709, Range::Full),
                (Coefficients::Bt601, Range::Limited),
                (Coefficients::Bt601, Range::Full),
                (Coefficients::Bt2020, Range::Limited),
                (Coefficients::Bt2020, Range::Full),
            ],
        )
    }
}
