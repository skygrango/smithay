use crate::wayland::compositor;
use crate::wayland::{Dispatch2, GlobalDispatch2};

use super::{
    ColorRepresentationHandler, ColorRepresentationSurfaceCachedState, ColorRepresentationSurfaceData,
    wp_color_representation_manager_v1, wp_color_representation_surface_v1,
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, New, Resource, Weak, protocol::wl_surface::WlSurface,
};

impl<D> GlobalDispatch2<wp_color_representation_manager_v1::WpColorRepresentationManagerV1, D> for ()
where
    D: Dispatch<wp_color_representation_manager_v1::WpColorRepresentationManagerV1, ()>
        + ColorRepresentationHandler
        + 'static,
{
    fn bind(
        &self,
        state: &mut D,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<wp_color_representation_manager_v1::WpColorRepresentationManagerV1>,
        data_init: &mut DataInit<'_, D>,
    ) {
        let state = state.color_representation_state();
        let instance = data_init.init(resource, ());

        for mode in &state.alpha_modes {
            instance.supported_alpha_mode(*mode);
        }
        for (coefficients, range) in &state.coefficients_and_ranges {
            instance.supported_coefficients_and_ranges(*coefficients, *range);
        }
        instance.done();

        state.known_instances.push(instance);
    }
}

impl<D> Dispatch2<wp_color_representation_manager_v1::WpColorRepresentationManagerV1, D> for ()
where
    D: Dispatch<wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1, Weak<WlSurface>>
        + ColorRepresentationHandler
        + 'static,
{
    fn request(
        &self,
        _state: &mut D,
        _client: &Client,
        resource: &wp_color_representation_manager_v1::WpColorRepresentationManagerV1,
        request: wp_color_representation_manager_v1::Request,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            wp_color_representation_manager_v1::Request::GetSurface { id, surface } => {
                compositor::with_states(&surface, |states| {
                    let data = states
                        .data_map
                        .get_or_insert_threadsafe(ColorRepresentationSurfaceData::new);

                    if data.is_resource_attached() {
                        resource.post_error(
                            wp_color_representation_manager_v1::Error::SurfaceExists,
                            "Surface already has a color representation object",
                        );
                        return;
                    }

                    let instance = data_init.init(id, surface.downgrade());
                    *data.instance.lock().unwrap() = Some(instance);
                });
            }
            wp_color_representation_manager_v1::Request::Destroy => {}
        }
    }

    fn destroyed(
        &self,
        state: &mut D,
        _client: wayland_backend::server::ClientId,
        resource: &wp_color_representation_manager_v1::WpColorRepresentationManagerV1,
    ) {
        let state = state.color_representation_state();
        state.known_instances.retain(|i| i != resource);
    }
}

impl<D> Dispatch2<wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1, D> for Weak<WlSurface>
where
    D: ColorRepresentationHandler + 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &Client,
        resource: &wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1,
        request: wp_color_representation_surface_v1::Request,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        let Ok(surface) = self.upgrade() else {
            resource.post_error(
                wp_color_representation_surface_v1::Error::Inert,
                "The related surface was destroyed",
            );
            return;
        };

        match request {
            wp_color_representation_surface_v1::Request::SetAlphaMode { alpha_mode } => {
                let wayland_server::WEnum::Value(alpha_mode) = alpha_mode else {
                    resource.post_error(
                        wp_color_representation_surface_v1::Error::AlphaMode,
                        "Unknown alpha mode",
                    );
                    return;
                };

                let handler_state = state.color_representation_state();
                if !handler_state.alpha_modes.contains(&alpha_mode) {
                    resource.post_error(
                        wp_color_representation_surface_v1::Error::AlphaMode,
                        "unsupported alpha mode used",
                    );
                    return;
                }

                compositor::with_states(&surface, |states| {
                    let mut guard = states
                        .cached_state
                        .get::<Option<ColorRepresentationSurfaceCachedState>>();
                    let representation = guard.pending();
                    if representation.is_none() {
                        *representation = Some(ColorRepresentationSurfaceCachedState::default());
                    }

                    representation.as_mut().unwrap().alpha_mode = Some(alpha_mode);
                });
            }
            wp_color_representation_surface_v1::Request::SetCoefficientsAndRange { coefficients, range } => {
                let (wayland_server::WEnum::Value(coefficients), wayland_server::WEnum::Value(range)) =
                    (coefficients, range)
                else {
                    resource.post_error(
                        wp_color_representation_surface_v1::Error::Coefficients,
                        "Unknown coefficients or range",
                    );
                    return;
                };

                let handler_state = state.color_representation_state();
                if !handler_state
                    .coefficients_and_ranges
                    .contains(&(coefficients, range))
                {
                    resource.post_error(
                        wp_color_representation_surface_v1::Error::Coefficients,
                        "unsupported coefficients or range used",
                    );
                    return;
                }

                if coefficients == wp_color_representation_surface_v1::Coefficients::Identity
                    && range != wp_color_representation_surface_v1::Range::Full
                {
                    resource.post_error(
                        wp_color_representation_surface_v1::Error::Coefficients,
                        "Limited range isn't supported with RGB",
                    );
                    return;
                }

                compositor::with_states(&surface, |states| {
                    let mut guard = states
                        .cached_state
                        .get::<Option<ColorRepresentationSurfaceCachedState>>();
                    let representation = guard.pending();
                    if representation.is_none() {
                        *representation = Some(ColorRepresentationSurfaceCachedState::default());
                    }

                    let r = representation.as_mut().unwrap();
                    r.coefficients = Some(coefficients);
                    r.range = Some(range);
                });
            }
            wp_color_representation_surface_v1::Request::SetChromaLocation { chroma_location } => {
                let wayland_server::WEnum::Value(chroma_location) = chroma_location else {
                    resource.post_error(
                        wp_color_representation_surface_v1::Error::ChromaLocation,
                        "Unknown chroma location",
                    );
                    return;
                };

                compositor::with_states(&surface, |states| {
                    let mut guard = states
                        .cached_state
                        .get::<Option<ColorRepresentationSurfaceCachedState>>();
                    let representation = guard.pending();
                    if representation.is_none() {
                        *representation = Some(ColorRepresentationSurfaceCachedState::default());
                    }

                    representation.as_mut().unwrap().chroma_location = Some(chroma_location);
                });
            }
            wp_color_representation_surface_v1::Request::Destroy => {}
        }
    }

    fn destroyed(
        &self,
        _state: &mut D,
        _client: wayland_backend::server::ClientId,
        _resource: &wp_color_representation_surface_v1::WpColorRepresentationSurfaceV1,
    ) {
        let Ok(surface) = self.upgrade() else {
            return;
        };

        compositor::with_states(&surface, |states| {
            if let Some(data) = states.data_map.get::<ColorRepresentationSurfaceData>() {
                data.instance.lock().unwrap().take();
            }

            *states
                .cached_state
                .get::<Option<ColorRepresentationSurfaceCachedState>>()
                .pending() = None;
        });
    }
}
