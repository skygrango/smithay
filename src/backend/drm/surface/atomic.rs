use drm::control::Device as ControlDevice;
use drm::control::atomic::AtomicModeReq;
use drm::control::connector::Interface;
use drm::control::property::ValueType;
use drm::control::{
    AtomicCommitFlags, Mode, PageFlipFlags, PlaneType, connector, crtc, dumbbuffer::DumbBuffer, framebuffer,
    plane, property,
};

use std::collections::{HashMap, HashSet};
#[cfg(debug_assertions)]
use std::fmt;
use std::ops::RangeInclusive;
use std::os::unix::io::AsRawFd;
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, Ordering},
};

use crate::backend::drm::color::{self, Colorspace, ConnectorColorState, CrtcColorState, DrmColorLut};
use crate::backend::drm::error::AccessError;
use crate::utils::{Coordinate, Rectangle, Transform};
use crate::{
    backend::{
        allocator::format::{get_bpp, get_depth},
        drm::{
            DrmDeviceFd,
            device::DrmDeviceInternal,
            device::atomic::{PropMapping, map_props},
            error::Error,
            plane_type,
        },
    },
    utils::DevPath,
};

use tracing::{debug, info, info_span, instrument, trace, warn};

use super::{PlaneConfig, PlaneState, VrrSupport};

/// Connector color state resolved against the actual properties of the surface's connectors,
/// ready to be put into an atomic request.
///
/// The `Colorspace` property is an enum property, so the requested [`Colorspace`] is resolved
/// to its raw value by name, per connector, when the state is staged. HDR metadata is carried
/// as an already-created property blob.
#[derive(Debug, Clone)]
pub struct ResolvedColorState {
    /// Raw value of the requested colorspace per connector. Connectors without a
    /// `Colorspace` property (only permitted for [`Colorspace::Default`]) are absent.
    colorspace_values: HashMap<connector::Handle, u64>,
    /// The `HDR_OUTPUT_METADATA` blob; `Blob(0)` when no metadata is signalled.
    hdr_blob: property::Value<'static>,
    /// The raw `max bpc` value, if one should be set.
    max_bpc: Option<u64>,
}

impl Default for ResolvedColorState {
    fn default() -> Self {
        ResolvedColorState {
            colorspace_values: HashMap::new(),
            hdr_blob: property::Value::Blob(0),
            max_bpc: None,
        }
    }
}

/// Resolved CRTC hardware color pipeline blobs, created when [`CrtcColorState`] is staged.
#[derive(Debug, Clone, PartialEq)]
pub struct CrtcColorBlobs {
    /// `GAMMA_LUT` blob handle (`Blob(0)` = disabled).
    pub gamma_blob: property::Value<'static>,
    /// `CTM` blob handle (`Blob(0)` = disabled).
    pub ctm_blob: property::Value<'static>,
}

impl Default for CrtcColorBlobs {
    fn default() -> Self {
        CrtcColorBlobs {
            gamma_blob: property::Value::Blob(0),
            ctm_blob: property::Value::Blob(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub active: bool,
    pub mode: Mode,
    pub blob: property::Value<'static>,
    pub vrr: bool,
    pub connectors: HashSet<connector::Handle>,
    pub color_state: ConnectorColorState,
    pub resolved_color: ResolvedColorState,
    /// Staged CRTC hardware color pipeline configuration (GAMMA_LUT, CTM).
    pub crtc_color_state: CrtcColorState,
    /// Resolved blobs for the staged CRTC color pipeline.
    pub crtc_color_blobs: CrtcColorBlobs,
    /// Whether HDR hardware CRTC offloading is active (e.g. GAMMA_LUT PQ encoding).
    pub hdr_hardware_offload: bool,
}

impl PartialEq for State {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // `resolved_color` and `crtc_color_blobs` are derived from their respective logical
        // states (and own the metadata blobs), so like the mode blob they are excluded from
        // the comparison.
        self.active == other.active
            && self.mode == other.mode
            && self.vrr == other.vrr
            && self.connectors == other.connectors
            && self.color_state == other.color_state
            && self.crtc_color_state == other.crtc_color_state
            && self.hdr_hardware_offload == other.hdr_hardware_offload
    }
}

impl State {
    fn current_state<A: DevPath + ControlDevice>(
        fd: &A,
        crtc: crtc::Handle,
        prop_mapping: &mut PropMapping,
    ) -> Result<Self, Error> {
        let crtc_info = fd.get_crtc(crtc).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading crtc info",
                dev: fd.dev_path(),
                source,
            })
        })?;

        // If we have no current mode, we create a fake one, which will not match (and thus gets overridden on the commit below).
        // A better fix would probably be making mode an `Option`, but that would mean
        // we need to be sure, we require a mode to always be set without relying on the compiler.
        // So we cheat, because it works and is easier to handle later.
        let current_mode = crtc_info.mode().unwrap_or_else(|| unsafe { std::mem::zeroed() });
        let current_blob = match crtc_info.mode() {
            Some(mode) => fd.create_property_blob(&mode).map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to create Property Blob for mode",
                    dev: fd.dev_path(),
                    source,
                })
            })?,
            None => property::Value::Unknown(0),
        };

        let res_handles = fd.resource_handles().map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading drm resources",
                dev: fd.dev_path(),
                source,
            })
        })?;

        // the current set of connectors are those, that already have the correct `CRTC_ID` set.
        // so we collect them for `current_state` and set the user-given once in `pending_state`.
        //
        // If they don't match, `commit_pending` will return true and they will be changed on the next `commit`.
        let mut current_connectors = HashSet::new();
        // make sure the mapping is up to date
        map_props(fd, res_handles.connectors(), &mut prop_mapping.connectors)?;
        for conn in res_handles.connectors() {
            let crtc_prop = prop_mapping.conn_prop_handle(*conn, "CRTC_ID")?;
            if let (Ok(crtc_prop_info), Ok(props)) = (fd.get_property(crtc_prop), fd.get_properties(*conn)) {
                let (ids, vals) = props.as_props_and_values();
                for (&id, &val) in ids.iter().zip(vals.iter()) {
                    if id == crtc_prop {
                        if let property::Value::CRTC(Some(conn_crtc)) =
                            crtc_prop_info.value_type().convert_value(val)
                        {
                            if conn_crtc == crtc {
                                current_connectors.insert(*conn);
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Get the current active (dpms) state and vrr state of the CRTC
        //
        // Changing a CRTC to active might require a modeset
        let mut active = None;
        let mut vrr = None;
        if let Ok(props) = fd.get_properties(crtc) {
            let active_prop = prop_mapping.crtcs.get(&crtc).and_then(|m| m.get("ACTIVE"));
            let vrr_prop = prop_mapping.crtcs.get(&crtc).and_then(|m| m.get("VRR_ENABLED"));
            let (ids, vals) = props.as_props_and_values();
            for (&id, &val) in ids.iter().zip(vals.iter()) {
                if Some(&id) == active_prop {
                    active = property::ValueType::Boolean.convert_value(val).as_boolean();
                    break;
                }
                if Some(&id) == vrr_prop {
                    vrr = property::ValueType::Boolean.convert_value(val).as_boolean();
                    break;
                }
            }
        }

        // Read back the current color state of the connectors, so that e.g. HDR signalling
        // left enabled by a previous KMS client is known and gets reset on our next commit.
        //
        // `max bpc` is deliberately not read back: `None` means "leave the property alone",
        // so reading back the kernel's current value would only force spurious modesets.
        let mut color_state = ConnectorColorState::default();
        for conn in &current_connectors {
            let Ok(props) = fd.get_properties(*conn) else {
                continue;
            };
            for (prop, raw) in props {
                let Ok(info) = fd.get_property(prop) else { continue };
                match info.name().to_str() {
                    Ok("Colorspace") => {
                        if color_state.colorspace != Colorspace::Default {
                            continue;
                        }
                        if let property::ValueType::Enum(values) = info.value_type() {
                            let (_, enums) = values.values();
                            // `Default` is a kernel uapi constant (DRM_MODE_COLORIMETRY_DEFAULT == 0),
                            // anything else we can't match by name is some foreign colorimetry.
                            color_state.colorspace = enums
                                .iter()
                                .find(|e| e.value() == raw)
                                .and_then(|e| e.name().to_str().ok())
                                .and_then(Colorspace::from_kernel_name)
                                .unwrap_or(if raw == 0 {
                                    Colorspace::Default
                                } else {
                                    Colorspace::Unknown
                                });
                        }
                    }
                    Ok("HDR_OUTPUT_METADATA") => {
                        if raw == 0 || color_state.hdr_metadata.is_some() {
                            continue;
                        }
                        match fd.get_property_blob(raw) {
                            Ok(data) => {
                                color_state.hdr_metadata = color::ffi::HdrOutputMetadata::parse(&data);
                                if color_state.hdr_metadata.is_none() {
                                    warn!(
                                        ?conn,
                                        "Failed to parse current HDR_OUTPUT_METADATA blob, assuming none"
                                    );
                                }
                            }
                            Err(err) => {
                                warn!(?conn, "Failed to read current HDR_OUTPUT_METADATA blob: {}", err)
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(State {
            // If we don't know the active state we just assume off.
            // This is highly unlikely, but having a false negative should do no harm.
            active: active.unwrap_or(false),
            mode: current_mode,
            blob: current_blob,
            // If we don't know the VRR state, the driver doesn't support the property
            vrr: vrr.unwrap_or(false),
            connectors: current_connectors,
            color_state,
            // We don't own the current metadata blob (if any), so don't reference it here;
            // requests are only ever built from the pending state anyway.
            resolved_color: ResolvedColorState::default(),
            // CRTC hardware color pipeline: start in passthrough (no LUT/CTM).
            crtc_color_state: CrtcColorState::default(),
            crtc_color_blobs: CrtcColorBlobs::default(),
            hdr_hardware_offload: false,
        })
    }

    fn clear(&mut self) {
        self.mode = unsafe { std::mem::zeroed() };
        self.blob = property::Value::Unknown(0);
        self.connectors.clear();
        self.active = false;
        self.vrr = false;
        self.color_state = ConnectorColorState::default();
        self.resolved_color = ResolvedColorState::default();
        self.crtc_color_state = CrtcColorState::default();
        self.crtc_color_blobs = CrtcColorBlobs::default();
        self.hdr_hardware_offload = false;
    }
}

/// Resolves the given color state against the actual properties of `connectors`.
///
/// Fails with [`Error::UnknownProperty`] if the state requests something a connector has no
/// property for (a non-default colorspace, HDR metadata or a `max bpc` value). Connectors
/// missing a property that isn't actively used are simply skipped, so plain SDR state
/// resolves successfully on any connector.
fn resolve_color_state<A: DevPath + ControlDevice>(
    fd: &A,
    prop_mapping: &PropMapping,
    color_state: &ConnectorColorState,
    hdr_blob: property::Value<'static>,
    connectors: impl IntoIterator<Item = connector::Handle>,
) -> Result<ResolvedColorState, Error> {
    let mut resolved = ResolvedColorState {
        colorspace_values: HashMap::new(),
        hdr_blob,
        max_bpc: color_state.max_bpc.map(u64::from),
    };

    // `Colorspace::Unknown` is a readback-only value and cannot be requested.
    let colorspace_name = color_state.colorspace.kernel_name();

    for conn in connectors {
        if colorspace_name.is_none() {
            return Err(Error::UnknownProperty {
                handle: conn.into(),
                name: "Colorspace",
            });
        }

        match prop_mapping.conn_prop_handle(conn, "Colorspace") {
            Ok(prop) => {
                let info = fd.get_property(prop).map_err(|source| {
                    Error::Access(AccessError {
                        errmsg: "Failed to query Colorspace property",
                        dev: fd.dev_path(),
                        source,
                    })
                })?;
                let value = if let property::ValueType::Enum(values) = info.value_type() {
                    let (_, enums) = values.values();
                    enums
                        .iter()
                        .find(|e| e.name().to_str().ok() == colorspace_name)
                        .map(|e| e.value())
                } else {
                    None
                };
                match value {
                    Some(value) => {
                        resolved.colorspace_values.insert(conn, value);
                    }
                    None if color_state.colorspace == Colorspace::Default => {}
                    None => {
                        return Err(Error::UnknownProperty {
                            handle: conn.into(),
                            name: "Colorspace",
                        });
                    }
                }
            }
            Err(_) if color_state.colorspace == Colorspace::Default => {}
            Err(_) => {
                return Err(Error::UnknownProperty {
                    handle: conn.into(),
                    name: "Colorspace",
                });
            }
        }

        if color_state.hdr_metadata.is_some()
            && prop_mapping
                .conn_prop_handle(conn, "HDR_OUTPUT_METADATA")
                .is_err()
        {
            return Err(Error::UnknownProperty {
                handle: conn.into(),
                name: "HDR_OUTPUT_METADATA",
            });
        }

        if color_state.max_bpc.is_some() && prop_mapping.conn_prop_handle(conn, "max bpc").is_err() {
            return Err(Error::UnknownProperty {
                handle: conn.into(),
                name: "max bpc",
            });
        }
    }

    Ok(resolved)
}

#[derive(Debug)]
pub struct AtomicDrmSurface {
    pub(in crate::backend::drm) fd: Arc<DrmDeviceInternal>,
    pub(super) active: Arc<AtomicBool>,
    crtc: crtc::Handle,
    plane: plane::Handle,
    used_planes: Mutex<HashSet<plane::Handle>>,
    prop_mapping: Arc<RwLock<PropMapping>>,
    state: RwLock<State>,
    pending: RwLock<State>,
    pub(super) span: tracing::Span,
    supports_async_page_flips: bool,
}

impl AtomicDrmSurface {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fd: Arc<DrmDeviceInternal>,
        active: Arc<AtomicBool>,
        crtc: crtc::Handle,
        plane: plane::Handle,
        prop_mapping: Arc<RwLock<PropMapping>>,
        mode: Mode,
        connectors: &[connector::Handle],
        supports_async_page_flips: bool,
    ) -> Result<Self, Error> {
        let span = info_span!("drm_atomic", crtc = ?crtc);
        let _guard = span.enter();
        info!(
            "Initializing drm surface ({:?}:{:?}) with mode {:?} and connectors {:?}",
            crtc, plane, mode, connectors
        );

        let state = State::current_state(&*fd, crtc, &mut prop_mapping.write().unwrap())?;
        let color_state = state.color_state;
        let blob = fd.create_property_blob(&mode).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to create Property Blob for mode",
                dev: fd.dev_path(),
                source,
            })
        })?;
        let resolved_color = resolve_color_state(
            &*fd,
            &prop_mapping.read().unwrap(),
            &color_state,
            property::Value::Blob(0),
            connectors.iter().copied(),
        )?;
        let pending = State {
            active: true,
            mode,
            blob,
            vrr: false,
            connectors: connectors.iter().copied().collect(),
            color_state,
            resolved_color,
            crtc_color_state: CrtcColorState::default(),
            crtc_color_blobs: CrtcColorBlobs::default(),
            hdr_hardware_offload: false,
        };

        drop(_guard);

        let surface = AtomicDrmSurface {
            fd,
            active,
            crtc,
            plane,
            used_planes: Mutex::new(HashSet::new()),
            prop_mapping,
            state: RwLock::new(state),
            pending: RwLock::new(pending),
            span,
            supports_async_page_flips,
        };

        Ok(surface)
    }

    // we need a framebuffer to do test commits, which we use to verify our pending state.
    // here we create a dumbbuffer for that purpose.
    #[profiling::function]
    fn create_test_buffer(
        &self,
        size: (u16, u16),
        plane: plane::Handle,
        is_hdr: bool,
    ) -> Result<TestBuffer, Error> {
        let needs_alpha = plane_type(&*self.fd, plane)? != PlaneType::Primary;
        let preferred_format = match (needs_alpha, is_hdr) {
            (true, true) => crate::backend::allocator::Fourcc::Abgr2101010,
            (false, true) => crate::backend::allocator::Fourcc::Xbgr2101010,
            (true, false) => crate::backend::allocator::Fourcc::Argb8888,
            (false, false) => crate::backend::allocator::Fourcc::Xrgb8888,
        };

        if let Ok(buf) = self.create_test_buffer_format(size, preferred_format) {
            return Ok(buf);
        }

        let fallback_format = if needs_alpha {
            crate::backend::allocator::Fourcc::Argb8888
        } else {
            crate::backend::allocator::Fourcc::Xrgb8888
        };
        self.create_test_buffer_format(size, fallback_format)
    }

    fn create_test_buffer_format(
        &self,
        size: (u16, u16),
        format: crate::backend::allocator::Fourcc,
    ) -> Result<TestBuffer, Error> {
        let (w, h) = size;
        let db = self
            .fd
            .create_dumb_buffer((w as u32, h as u32), format, get_bpp(format).unwrap() as u32)
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to create dumb buffer",
                    dev: self.fd.dev_path(),
                    source,
                })
            })?;
        let fb_result = self
            .fd
            .add_framebuffer(
                &db,
                get_depth(format).unwrap() as u32,
                get_bpp(format).unwrap() as u32,
            )
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to create framebuffer",
                    dev: self.fd.dev_path(),
                    source,
                })
            });

        match fb_result {
            Ok(fb) => Ok(TestBuffer {
                fd: self.fd.clone(),
                db,
                fb,
            }),
            Err(err) => {
                let _ = self.fd.destroy_dumb_buffer(db);
                Err(err)
            }
        }
    }

    pub fn current_connectors(&self) -> HashSet<connector::Handle> {
        self.state.read().unwrap().connectors.clone()
    }

    pub fn pending_connectors(&self) -> HashSet<connector::Handle> {
        self.pending.read().unwrap().connectors.clone()
    }

    pub fn current_mode(&self) -> Mode {
        self.state.read().unwrap().mode
    }

    pub fn pending_mode(&self) -> Mode {
        self.pending.read().unwrap().mode
    }

    fn ensure_props_known(&self, conns: &[connector::Handle]) -> Result<(), Error> {
        let mapping_exists = {
            let prop_mapping = self.prop_mapping.read().unwrap();
            conns
                .iter()
                .all(|conn| prop_mapping.connectors.contains_key(conn))
        };
        if !mapping_exists {
            map_props(
                &*self.fd,
                self.fd
                    .resource_handles()
                    .map_err(|source| {
                        Error::Access(AccessError {
                            errmsg: "Error loading connector info",
                            dev: self.fd.dev_path(),
                            source,
                        })
                    })?
                    .connectors(),
                &mut self.prop_mapping.write().unwrap().connectors,
            )?;
        }
        Ok(())
    }

    #[instrument(parent = &self.span, skip(self))]
    pub fn add_connector(&self, conn: connector::Handle) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        self.ensure_props_known(&[conn])?;
        let info = self.fd.get_connector(conn, false).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error loading connector info",
                dev: self.fd.dev_path(),
                source,
            })
        })?;

        let mut pending = self.pending.write().unwrap();

        // check if the connector can handle the current mode
        if info.modes().contains(&pending.mode) {
            let test_buffer =
                self.create_test_buffer(pending.mode.size(), self.plane, pending.color_state.is_hdr())?;

            // check if config is supported
            let prop_mapping = self.prop_mapping.read().unwrap();
            let plane_state = PlaneState {
                handle: self.plane,
                config: Some(PlaneConfig {
                    src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                    dst: Rectangle::from_size(
                        (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                    ),
                    transform: Transform::Normal,
                    alpha: 1.0,
                    damage_clips: None,
                    fb: test_buffer.fb,
                    fence: None,
                }),
            };

            let mut connectors = pending.connectors.clone();
            connectors.insert(conn);

            // resolve the pending color state for the new connector as well
            let mut resolved_color = pending.resolved_color.clone();
            resolved_color.colorspace_values.extend(
                resolve_color_state(
                    &*self.fd,
                    &prop_mapping,
                    &pending.color_state,
                    resolved_color.hdr_blob,
                    [conn],
                )?
                .colorspace_values,
            );

            let req = AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                Some(pending.blob),
                pending.vrr,
                Some(&resolved_color),
                Some(&pending.crtc_color_blobs),
                &connectors,
                [],
                [&plane_state],
            )?;
            self.fd
                .atomic_commit(
                    AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                    req.build()?,
                )
                .map_err(|_| Error::TestFailed(self.crtc))?;

            // seems to be, lets add the connector
            pending.connectors.insert(conn);
            pending.resolved_color = resolved_color;

            Ok(())
        } else {
            Err(Error::ModeNotSuitable(pending.mode))
        }
    }

    #[instrument(parent = &self.span, skip(self))]
    pub fn remove_connector(&self, conn: connector::Handle) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut pending = self.pending.write().unwrap();

        // the test would also prevent this, but the error message is far less helpful
        if pending.connectors.contains(&conn) && pending.connectors.len() == 1 {
            return Err(Error::SurfaceWithoutConnectors(self.crtc));
        }

        // check if new config is supported (should be)
        let test_buffer =
            self.create_test_buffer(pending.mode.size(), self.plane, pending.color_state.is_hdr())?;

        let prop_mapping = self.prop_mapping.read().unwrap();
        let plane_state = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                dst: Rectangle::from_size(
                    (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                ),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };

        let mut connectors = pending.connectors.clone();
        connectors.remove(&conn);
        let req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            Some(pending.blob),
            pending.vrr,
            Some(&pending.resolved_color),
            Some(&pending.crtc_color_blobs),
            &connectors,
            [&conn],
            [&plane_state],
        )?;
        self.fd
            .atomic_commit(
                AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                req.build()?,
            )
            .map_err(|_| Error::TestFailed(self.crtc))?;

        // seems to be, lets remove the connector
        pending.connectors.remove(&conn);
        pending.resolved_color.colorspace_values.remove(&conn);

        Ok(())
    }

    #[instrument(parent = &self.span, skip(self))]
    pub fn set_connectors(&self, connectors: &[connector::Handle]) -> Result<(), Error> {
        // the test would also prevent this, but the error message is far less helpful
        if connectors.is_empty() {
            return Err(Error::SurfaceWithoutConnectors(self.crtc));
        }

        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let current = self.state.read().unwrap();
        let mut pending = self.pending.write().unwrap();

        self.ensure_props_known(connectors)?;
        let conns = connectors.iter().cloned().collect::<HashSet<_>>();
        let removed = current.connectors.difference(&conns);

        let test_buffer =
            self.create_test_buffer(pending.mode.size(), self.plane, pending.color_state.is_hdr())?;

        let prop_mapping = self.prop_mapping.read().unwrap();
        let plane_state = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                dst: Rectangle::from_size(
                    (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                ),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };
        // re-resolve the pending color state for the new connector set
        let resolved_color = resolve_color_state(
            &*self.fd,
            &prop_mapping,
            &pending.color_state,
            pending.resolved_color.hdr_blob,
            conns.iter().copied(),
        )?;

        let req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            Some(pending.blob),
            pending.vrr,
            Some(&resolved_color),
            Some(&pending.crtc_color_blobs),
            &conns,
            removed,
            [&plane_state],
        )?;

        self.fd
            .atomic_commit(
                AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                req.build()?,
            )
            .map_err(|_| Error::TestFailed(self.crtc))?;

        pending.connectors = conns;
        pending.resolved_color = resolved_color;

        Ok(())
    }

    #[instrument(level = "debug", parent = &self.span, skip(self))]
    pub fn use_mode(&self, mode: Mode) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut pending = self.pending.write().unwrap();

        // check if new config is supported
        let new_blob = self.fd.create_property_blob(&mode).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to create Property Blob for mode",
                dev: self.fd.dev_path(),
                source,
            })
        })?;

        let test_buffer = self.create_test_buffer(mode.size(), self.plane, pending.color_state.is_hdr())?;

        let prop_mapping = self.prop_mapping.read().unwrap();
        let plane_state = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(mode.size().into()).to_f64(),
                dst: Rectangle::from_size((mode.size().0 as i32, mode.size().1 as i32).into()),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };
        let req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            Some(new_blob),
            pending.vrr,
            Some(&pending.resolved_color),
            Some(&pending.crtc_color_blobs),
            pending.connectors.iter(),
            [],
            [&plane_state],
        )?;
        if let Err(err) = self
            .fd
            .atomic_commit(
                AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                req.build()?,
            )
            .map_err(|_| Error::TestFailed(self.crtc))
        {
            let _ = self.fd.destroy_property_blob(new_blob.into());
            return Err(err);
        }

        // seems to be, lets change the mode
        pending.mode = mode;
        pending.blob = new_blob;

        Ok(())
    }

    pub fn vrr_supported(&self, conn: connector::Handle) -> Result<VrrSupport, Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let props = self.prop_mapping.read().unwrap();
        if self
            .prop_mapping
            .read()
            .unwrap()
            .crtc_prop_handle(self.crtc, "VRR_ENABLED")
            .is_err()
        {
            return Ok(VrrSupport::NotSupported);
        }

        if let Some(vrr_prop) = props
            .connectors
            .get(&conn)
            .and_then(|props| props.get("vrr_capable"))
        {
            for (prop, value) in self.fd.get_properties(conn).map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Error querying properties",
                    dev: self.fd.dev_path(),
                    source,
                })
            })? {
                if prop == *vrr_prop {
                    let interface = self
                        .fd
                        .get_connector(conn, false)
                        .map_err(|source| {
                            Error::Access(AccessError {
                                errmsg: "Error querying connector",
                                dev: self.fd.dev_path(),
                                source,
                            })
                        })?
                        .interface();

                    // see: https://gitlab.freedesktop.org/drm/amd/-/issues/2200#note_2159982
                    // Currently setting VRR for HDMI connectors will cause flickering despite not needing `ALLOW_MODESET`
                    // TODO: Once the kernel is fixed, do actual test commits with and without `ALLOW_MODESET`.
                    return Ok(
                        match ValueType::Boolean.convert_value(value).as_boolean().unwrap() {
                            true if interface == Interface::HDMIA || interface == Interface::HDMIB => {
                                VrrSupport::RequiresModeset
                            }
                            true => VrrSupport::Supported,
                            false => VrrSupport::NotSupported,
                        },
                    );
                }
            }
        }

        Ok(VrrSupport::NotSupported)
    }

    /// Queries the size of the CRTC's hardware `GAMMA_LUT` if supported.
    pub fn crtc_gamma_lut_size(&self) -> Result<Option<u64>, Error> {
        let prop_mapping = self.prop_mapping.read().unwrap();
        let prop_handle = match prop_mapping.crtc_prop_handle(self.crtc, "GAMMA_LUT_SIZE") {
            Ok(h) => h,
            Err(_) => return Ok(None),
        };
        for (prop, value) in self.fd.get_properties(self.crtc).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to query CRTC properties",
                dev: self.fd.dev_path(),
                source,
            })
        })? {
            if prop == prop_handle {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// Returns whether the CRTC supports hardware color transformation matrix (`CTM`).
    pub fn crtc_has_ctm(&self) -> bool {
        let prop_mapping = self.prop_mapping.read().unwrap();
        prop_mapping.crtc_prop_handle(self.crtc, "CTM").is_ok()
    }

    /// Queries the size of the CRTC's hardware `DEGAMMA_LUT` if supported.
    pub fn crtc_degamma_lut_size(&self) -> Result<Option<u64>, Error> {
        let prop_mapping = self.prop_mapping.read().unwrap();
        let prop_handle = match prop_mapping.crtc_prop_handle(self.crtc, "DEGAMMA_LUT_SIZE") {
            Ok(h) => h,
            Err(_) => return Ok(None),
        };
        for (prop, value) in self.fd.get_properties(self.crtc).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to query CRTC properties",
                dev: self.fd.dev_path(),
                source,
            })
        })? {
            if prop == prop_handle {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    pub fn vrr_enabled(&self) -> bool {
        self.pending.read().unwrap().vrr
    }

    pub fn use_vrr(&self, value: bool) -> Result<(), Error> {
        let mut current = self.state.write().unwrap();
        let mut pending = self.pending.write().unwrap();
        if pending.vrr == value {
            return Ok(());
        }
        let prop_mapping = self.prop_mapping.read().unwrap();

        if value && prop_mapping.crtc_prop_handle(self.crtc, "VRR_ENABLED").is_err() {
            return Err(Error::UnknownProperty {
                handle: self.crtc.into(),
                name: "VRR_ENABLED",
            });
        }

        let test_buffer =
            self.create_test_buffer(pending.mode.size(), self.plane, pending.color_state.is_hdr())?;
        let plane_config = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                dst: Rectangle::from_size(
                    (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                ),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };

        let req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            Some(pending.blob),
            value,
            Some(&pending.resolved_color),
            Some(&pending.crtc_color_blobs),
            &pending.connectors,
            &[],
            [&plane_config],
        )?;

        if *current == *pending {
            // Try a non modesetting commit
            let non_modeset_req = AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                None,
                value,
                None,
                None,
                [],
                [],
                [&plane_config],
            )?;

            if self
                .fd
                .atomic_commit(AtomicCommitFlags::TEST_ONLY, non_modeset_req.build()?)
                .is_ok()
            {
                pending.vrr = value;
                current.vrr = value;
                return Ok(());
            }
        }

        // Try a modeset commit
        self.fd
            .atomic_commit(
                AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                req.build()?,
            )
            .map_err(|_| Error::TestFailed(self.crtc))?;

        pending.vrr = value;
        Ok(())
    }

    /// Returns the colorspaces supported by the given connector's `Colorspace` property.
    ///
    /// Contains at least [`Colorspace::Default`], which is also the only entry if the
    /// connector has no `Colorspace` property at all.
    pub fn supported_colorspaces(&self, conn: connector::Handle) -> Result<Vec<Colorspace>, Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let prop = match self
            .prop_mapping
            .read()
            .unwrap()
            .conn_prop_handle(conn, "Colorspace")
        {
            Ok(prop) => prop,
            Err(_) => return Ok(vec![Colorspace::Default]),
        };

        let info = self.fd.get_property(prop).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to query Colorspace property",
                dev: self.fd.dev_path(),
                source,
            })
        })?;

        let mut supported = vec![Colorspace::Default];
        if let property::ValueType::Enum(values) = info.value_type() {
            let (_, enums) = values.values();
            supported.extend(
                enums
                    .iter()
                    .filter_map(|e| e.name().to_str().ok())
                    .filter_map(Colorspace::from_kernel_name)
                    .filter(|cs| *cs != Colorspace::Default),
            );
        }
        Ok(supported)
    }

    /// Returns whether the given connector supports the `HDR_OUTPUT_METADATA` property.
    pub fn hdr_metadata_supported(&self, conn: connector::Handle) -> Result<bool, Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        Ok(self
            .prop_mapping
            .read()
            .unwrap()
            .conn_prop_handle(conn, "HDR_OUTPUT_METADATA")
            .is_ok())
    }

    /// Returns the valid range of the given connector's `max bpc` property, or `None` if the
    /// connector has no such property.
    pub fn max_bpc_range(&self, conn: connector::Handle) -> Result<Option<RangeInclusive<u32>>, Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let prop = match self
            .prop_mapping
            .read()
            .unwrap()
            .conn_prop_handle(conn, "max bpc")
        {
            Ok(prop) => prop,
            Err(_) => return Ok(None),
        };

        let info = self.fd.get_property(prop).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to query max bpc property",
                dev: self.fd.dev_path(),
                source,
            })
        })?;

        Ok(match info.value_type() {
            property::ValueType::UnsignedRange(min, max) => Some(min as u32..=max as u32),
            _ => None,
        })
    }

    /// Returns the currently pending [`ConnectorColorState`].
    pub fn pending_color_state(&self) -> ConnectorColorState {
        self.pending.read().unwrap().color_state
    }

    /// Returns the currently active [`ConnectorColorState`].
    pub fn current_color_state(&self) -> ConnectorColorState {
        self.state.read().unwrap().color_state
    }

    /// Stages a new [`ConnectorColorState`] to be applied on the next commit.
    ///
    /// The state is validated with a `TEST_ONLY` commit, but only applied by the next
    /// [`commit`](Self::commit). Unlike VRR, color properties are never applied via
    /// [`page_flip`](Self::page_flip): some drivers treat e.g. a `Colorspace` change as
    /// requiring a full modeset and can misbehave badly (up to hanging the display pipe)
    /// when connector color properties are committed without complete CRTC/plane state.
    /// The kernel latches connector property values, so page flips after the commit keep
    /// the signalled state without re-emitting it.
    pub fn use_color_state(&self, color_state: ConnectorColorState) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let current = self.state.read().unwrap();
        let mut pending = self.pending.write().unwrap();
        if pending.color_state == color_state {
            return Ok(());
        }

        let destroy_blob = |blob: property::Value<'static>| {
            if let property::Value::Blob(id) = blob {
                if id != 0 {
                    if let Err(err) = self.fd.destroy_property_blob(id) {
                        warn!("Failed to destroy HDR metadata property blob: {}", err);
                    }
                }
            }
        };

        let hdr_blob = match color_state.hdr_metadata {
            Some(meta) => self
                .fd
                .create_property_blob(&color::ffi::HdrOutputMetadata::from(meta))
                .map_err(|source| {
                    Error::Access(AccessError {
                        errmsg: "Failed to create HDR metadata property blob",
                        dev: self.fd.dev_path(),
                        source,
                    })
                })?,
            None => property::Value::Blob(0),
        };

        let prop_mapping = self.prop_mapping.read().unwrap();
        let resolved = match resolve_color_state(
            &*self.fd,
            &prop_mapping,
            &color_state,
            hdr_blob,
            pending.connectors.iter().copied(),
        ) {
            Ok(resolved) => resolved,
            Err(err) => {
                destroy_blob(hdr_blob);
                return Err(err);
            }
        };

        let test_buffer = match self.create_test_buffer(pending.mode.size(), self.plane, color_state.is_hdr())
        {
            Ok(buf) => buf,
            Err(err) => {
                destroy_blob(hdr_blob);
                return Err(err);
            }
        };

        let plane_config = PlaneState {
            handle: self.plane,
            config: Some(PlaneConfig {
                src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                dst: Rectangle::from_size(
                    (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                ),
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: test_buffer.fb,
                fence: None,
            }),
        };

        let test_request = |blobs: &CrtcColorBlobs| -> Result<(), Error> {
            let req = AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                Some(pending.blob),
                pending.vrr,
                Some(&resolved),
                Some(blobs),
                &pending.connectors,
                &[],
                [&plane_config],
            )?;

            self.fd
                .atomic_commit(
                    AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                    req.build()?,
                )
                .map_err(|_| Error::TestFailed(self.crtc))
        };

        let mut final_crtc_state = CrtcColorState::default();
        let mut final_crtc_blobs = CrtcColorBlobs::default();
        let mut hardware_offload = false;

        // If HDR is active, attempt to offload PQ transfer function onto CRTC GAMMA_LUT
        // only if explicitly enabled via SMITHAY_HDR_HARDWARE_OFFLOAD=1.
        // By default, offload is disabled to keep CRTC LUT linear/identity (matching KWin),
        // preventing double-transfer-function distortions and dynamic-range clipping in UNORM framebuffers.
        let allow_offload = std::env::var("SMITHAY_HDR_HARDWARE_OFFLOAD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if color_state.is_hdr() && allow_offload {
            let lut_size = if prop_mapping.crtc_prop_handle(self.crtc, "GAMMA_LUT").is_ok() {
                if let Ok(size_handle) = prop_mapping.crtc_prop_handle(self.crtc, "GAMMA_LUT_SIZE") {
                    self.fd
                        .get_properties(self.crtc)
                        .ok()
                        .and_then(|props| props.into_iter().find(|(p, _)| *p == size_handle).map(|(_, v)| v))
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(size) = lut_size {
                if size > 0 && size <= 4096 {
                    let ref_white = color_state.reference_white.unwrap_or(203.0);
                    let pq_lut = DrmColorLut::create_pq_lut(size as usize, ref_white);
                    if let Ok(gamma_blob) = self.fd.create_property_blob(&pq_lut) {
                        let candidate_blobs = CrtcColorBlobs {
                            gamma_blob,
                            ctm_blob: property::Value::Blob(0),
                        };
                        if test_request(&candidate_blobs).is_ok() {
                            info!(
                                "staged CRTC hardware GAMMA_LUT offload for HDR (lut_size={})",
                                size
                            );
                            final_crtc_state = CrtcColorState {
                                gamma_lut: Some(pq_lut),
                                ctm: None,
                                degamma_lut: None,
                            };
                            final_crtc_blobs = candidate_blobs;
                            hardware_offload = true;
                        } else {
                            warn!("CRTC GAMMA_LUT offload rejected by driver; falling back to GPU shader");
                            destroy_blob(candidate_blobs.gamma_blob);
                        }
                    }
                }
            }
        }

        // If hardware offload was not engaged (SDR mode, no GAMMA_LUT, or offload rejected),
        // validate the state with empty CRTC color pipeline (GPU shader path).
        if !hardware_offload {
            if let Err(err) = test_request(&final_crtc_blobs) {
                destroy_blob(hdr_blob);
                return Err(err);
            }
        }

        // Destroy a previously staged blob that was never committed. A committed blob
        // (referenced by the current state) is destroyed by `commit` once it is replaced.
        let old_blob = std::mem::replace(&mut pending.resolved_color.hdr_blob, property::Value::Blob(0));
        if old_blob != current.resolved_color.hdr_blob {
            destroy_blob(old_blob);
        }
        let old_gamma = std::mem::replace(&mut pending.crtc_color_blobs.gamma_blob, property::Value::Blob(0));
        if old_gamma != current.crtc_color_blobs.gamma_blob {
            destroy_blob(old_gamma);
        }
        let old_ctm = std::mem::replace(&mut pending.crtc_color_blobs.ctm_blob, property::Value::Blob(0));
        if old_ctm != current.crtc_color_blobs.ctm_blob {
            destroy_blob(old_ctm);
        }

        pending.color_state = color_state;
        pending.resolved_color = resolved;
        pending.crtc_color_state = final_crtc_state;
        pending.crtc_color_blobs = final_crtc_blobs;
        pending.hdr_hardware_offload = hardware_offload;
        Ok(())
    }

    /// Stages a new [`CrtcColorState`] (hardware GAMMA_LUT and CTM) to be applied on the
    /// next [`commit`](Self::commit).
    ///
    /// Creates DRM property blobs from the provided LUT and CTM data, validates them with a
    /// `TEST_ONLY` atomic commit, and stages the result. DEGAMMA_LUT is intentionally not
    /// exposed; it is buggy on most drivers (Intel, AMD, NVidia) in the post-blend pipeline
    /// and is deliberately excluded from HDR rendering (matching KWin's `drm_crtc.cpp`).
    ///
    /// Returns [`Error::TestFailed`] if the driver rejects the requested pipeline,
    /// [`Error::UnknownProperty`] if the CRTC has no `GAMMA_LUT`/`CTM` property and a
    /// non-`None` value is requested for it.
    pub fn use_crtc_color_state(&self, crtc_color_state: CrtcColorState) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let current = self.state.read().unwrap();
        let mut pending = self.pending.write().unwrap();
        if pending.crtc_color_state == crtc_color_state {
            return Ok(());
        }

        let destroy_blob = |blob: property::Value<'static>| {
            if let property::Value::Blob(id) = blob {
                if id != 0 {
                    if let Err(err) = self.fd.destroy_property_blob(id) {
                        warn!("Failed to destroy CRTC color pipeline blob: {}", err);
                    }
                }
            }
        };

        let prop_mapping = self.prop_mapping.read().unwrap();

        // Create GAMMA_LUT blob if requested.
        let gamma_blob = match &crtc_color_state.gamma_lut {
            Some(lut) => {
                if prop_mapping.crtc_prop_handle(self.crtc, "GAMMA_LUT").is_err() {
                    return Err(Error::UnknownProperty {
                        handle: self.crtc.into(),
                        name: "GAMMA_LUT",
                    });
                }
                self.fd.create_property_blob(lut).map_err(|source| {
                    Error::Access(AccessError {
                        errmsg: "Failed to create GAMMA_LUT property blob",
                        dev: self.fd.dev_path(),
                        source,
                    })
                })?
            }
            None => property::Value::Blob(0),
        };

        // Create CTM blob if requested.
        let ctm_blob = match &crtc_color_state.ctm {
            Some(ctm) => {
                if prop_mapping.crtc_prop_handle(self.crtc, "CTM").is_err() {
                    destroy_blob(gamma_blob);
                    return Err(Error::UnknownProperty {
                        handle: self.crtc.into(),
                        name: "CTM",
                    });
                }
                match self.fd.create_property_blob(ctm) {
                    Ok(blob) => blob,
                    Err(source) => {
                        destroy_blob(gamma_blob);
                        return Err(Error::Access(AccessError {
                            errmsg: "Failed to create CTM property blob",
                            dev: self.fd.dev_path(),
                            source,
                        }));
                    }
                }
            }
            None => property::Value::Blob(0),
        };

        let new_blobs = CrtcColorBlobs { gamma_blob, ctm_blob };

        // TEST_ONLY validate the new pipeline.
        let res = (|| {
            let test_buffer =
                self.create_test_buffer(pending.mode.size(), self.plane, pending.color_state.is_hdr())?;
            let plane_config = PlaneState {
                handle: self.plane,
                config: Some(PlaneConfig {
                    src: Rectangle::from_size(pending.mode.size().into()).to_f64(),
                    dst: Rectangle::from_size(
                        (pending.mode.size().0 as i32, pending.mode.size().1 as i32).into(),
                    ),
                    transform: Transform::Normal,
                    alpha: 1.0,
                    damage_clips: None,
                    fb: test_buffer.fb,
                    fence: None,
                }),
            };

            let req = AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                Some(pending.blob),
                pending.vrr,
                Some(&pending.resolved_color),
                Some(&new_blobs),
                &pending.connectors,
                &[],
                [&plane_config],
            )?;

            self.fd
                .atomic_commit(
                    AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                    req.build()?,
                )
                .map_err(|_| Error::TestFailed(self.crtc))
        })();

        if let Err(err) = res {
            destroy_blob(new_blobs.gamma_blob);
            destroy_blob(new_blobs.ctm_blob);
            return Err(err);
        }

        // Destroy previously staged blobs that were never committed.
        let old_gamma = std::mem::replace(&mut pending.crtc_color_blobs.gamma_blob, property::Value::Blob(0));
        if old_gamma != current.crtc_color_blobs.gamma_blob {
            destroy_blob(old_gamma);
        }
        let old_ctm = std::mem::replace(&mut pending.crtc_color_blobs.ctm_blob, property::Value::Blob(0));
        if old_ctm != current.crtc_color_blobs.ctm_blob {
            destroy_blob(old_ctm);
        }

        pending.crtc_color_state = crtc_color_state;
        pending.crtc_color_blobs = new_blobs;
        pending.hdr_hardware_offload = pending.crtc_color_state.gamma_lut.is_some();
        Ok(())
    }

    /// Returns whether HDR hardware CRTC offloading (e.g. GAMMA_LUT) is staged for the next commit.
    pub fn pending_hdr_hardware_offload(&self) -> bool {
        self.pending.read().unwrap().hdr_hardware_offload
    }

    /// Returns whether HDR hardware CRTC offloading is currently active on the CRTC.
    pub fn current_hdr_hardware_offload(&self) -> bool {
        self.state.read().unwrap().hdr_hardware_offload
    }

    /// Returns the currently pending [`CrtcColorState`].
    pub fn pending_crtc_color_state(&self) -> CrtcColorState {
        self.pending.read().unwrap().crtc_color_state.clone()
    }

    /// Returns the currently active [`CrtcColorState`].
    pub fn current_crtc_color_state(&self) -> CrtcColorState {
        self.state.read().unwrap().crtc_color_state.clone()
    }

    pub fn commit_pending(&self) -> bool {
        *self.pending.read().unwrap() != *self.state.read().unwrap()
    }

    #[instrument(level = "trace", parent = &self.span, skip(self, planes))]
    #[profiling::function]
    pub fn test_state<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
        async_commit: bool,
    ) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let current = self.state.read().unwrap();
        let pending = self.pending.read().unwrap();

        self.test_state_internal(planes, allow_modeset, async_commit, &current, &pending)
    }

    fn test_state_internal<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        allow_modeset: bool,
        async_commit: bool,
        current: &'_ State,
        pending: &'_ State,
    ) -> Result<(), Error> {
        let planes = planes.into_iter().collect::<Vec<_>>();

        let current_conns = current.connectors.clone();
        let pending_conns = pending.connectors.clone();
        let removed = current_conns.difference(&pending_conns);
        let prop_mapping = self.prop_mapping.read().unwrap();

        let req = if allow_modeset {
            AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                Some(pending.blob),
                pending.vrr,
                Some(&pending.resolved_color),
                Some(&pending.crtc_color_blobs),
                &pending_conns,
                removed,
                &*planes,
            )?
        } else {
            AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                None,
                current.vrr,
                None,
                None,
                [],
                [],
                &*planes,
            )?
        };

        let mut flags = AtomicCommitFlags::TEST_ONLY;
        if allow_modeset {
            flags |= AtomicCommitFlags::ALLOW_MODESET;
        }

        if async_commit {
            flags |= AtomicCommitFlags::PAGE_FLIP_ASYNC;
        }

        self.fd.atomic_commit(flags, req.build()?).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Error testing state",
                dev: self.fd.dev_path(),
                source,
            })
        })
    }

    fn flip_flags(&self, value: PageFlipFlags) -> AtomicCommitFlags {
        let mut v = AtomicCommitFlags::empty();
        if value.contains(PageFlipFlags::EVENT) {
            v |= AtomicCommitFlags::PAGE_FLIP_EVENT;
        }
        if self.supports_async_page_flips && value.contains(PageFlipFlags::ASYNC) {
            v |= AtomicCommitFlags::PAGE_FLIP_ASYNC;
        }
        v
    }

    #[instrument(level = "trace", parent = &self.span, skip(self, planes))]
    #[profiling::function]
    pub fn commit<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        flip_flags: super::PageFlipFlags,
    ) -> Result<(), Error> {
        let flip_flags = self.flip_flags(flip_flags);

        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let planes = planes.into_iter().collect::<Vec<_>>();
        let mut current = self.state.write().unwrap();
        let mut used_planes = self.used_planes.lock().unwrap();
        let pending = self.pending.read().unwrap();

        debug!(current = ?*current, pending = ?*pending, ?planes, "Preparing Commit",);

        // we need the differences to know, which connectors need to change properties
        let current_conns = current.connectors.clone();
        let pending_conns = pending.connectors.clone();
        let removed = current_conns.difference(&pending_conns);

        for conn in removed.clone() {
            if let Ok(info) = self.fd.get_connector(*conn, false) {
                info!("Removing connector: {:?}", info.interface());
            } else {
                info!("Removing unknown connector");
            }
        }

        for conn in &pending_conns {
            if let Ok(info) = self.fd.get_connector(*conn, false) {
                info!("Adding connector: {:?}", info.interface());
            } else {
                info!("Adding unknown connector");
            }
        }

        if current.mode != pending.mode {
            info!("Setting new mode: {:?}", pending.mode.name());
        }

        trace!("Testing screen config");

        // test the new config and return the request if it would be accepted by the driver.
        let prop_mapping = self.prop_mapping.read().unwrap();
        let req = {
            let req = AtomicRequest::build_request(
                &prop_mapping,
                self.crtc,
                Some(pending.blob),
                pending.vrr,
                Some(&pending.resolved_color),
                Some(&pending.crtc_color_blobs),
                &pending_conns,
                removed,
                &*planes,
            )?;

            if let Err(err) = self.fd.atomic_commit(
                AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                req.build()?,
            ) {
                warn!("New screen configuration invalid!:\n\t{:?}\n\t{}\n", req, err);

                return Err(Error::TestFailed(self.crtc));
            } else {
                if current.mode != pending.mode {
                    if let Err(err) = self.fd.destroy_property_blob(current.blob.into()) {
                        warn!("Failed to destroy old mode property blob: {}", err);
                    }
                }
                // Like the mode blob: once the property moves off the old metadata blob, the
                // kernel keeps it alive as long as needed, so it's safe to release it now.
                if current.resolved_color.hdr_blob != pending.resolved_color.hdr_blob {
                    if let property::Value::Blob(id) = current.resolved_color.hdr_blob {
                        if id != 0 {
                            if let Err(err) = self.fd.destroy_property_blob(id) {
                                warn!("Failed to destroy old HDR metadata property blob: {}", err);
                            }
                        }
                    }
                }
                // Destroy old CRTC color pipeline blobs now that the commit will replace them.
                let destroy_crtc_blob = |blob: property::Value<'static>| {
                    if let property::Value::Blob(id) = blob {
                        if id != 0 {
                            if let Err(err) = self.fd.destroy_property_blob(id) {
                                warn!("Failed to destroy old CRTC color pipeline blob: {}", err);
                            }
                        }
                    }
                };
                if current.crtc_color_blobs.gamma_blob != pending.crtc_color_blobs.gamma_blob {
                    destroy_crtc_blob(current.crtc_color_blobs.gamma_blob);
                }
                if current.crtc_color_blobs.ctm_blob != pending.crtc_color_blobs.ctm_blob {
                    destroy_crtc_blob(current.crtc_color_blobs.ctm_blob);
                }

                // new config
                req
            }
        };

        debug!("Setting screen: {:?}", req);
        let result = self
            .fd
            .atomic_commit(
                if flip_flags.contains(AtomicCommitFlags::PAGE_FLIP_EVENT) {
                    // on the atomic api we can modeset and trigger a page_flip event on the same call!
                    flip_flags | AtomicCommitFlags::ALLOW_MODESET
                    // we also *should* not need to wait for completion, like with `set_crtc`,
                    // because we have tested this exact commit already, so we do not expect any errors later down the line.
                    //
                    // but there is always an exception and `amdgpu` can fail in interesting ways with this flag set...
                    // https://gitlab.freedesktop.org/drm/amd/-/issues?scope=all&utf8=%E2%9C%93&state=opened&search=drm_atomic_helper_wait_for_flip_done
                    //
                    // so we skip this flag:
                    // AtomicCommitFlags::Nonblock,
                } else {
                    AtomicCommitFlags::ALLOW_MODESET
                },
                req.build()?,
            )
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Error setting crtc",
                    dev: self.fd.dev_path(),
                    source,
                })
            });

        if result.is_ok() {
            *current = pending.clone();
            for plane in planes.iter() {
                if plane.config.is_some() {
                    used_planes.insert(plane.handle);
                } else {
                    used_planes.remove(&plane.handle);
                }
            }
        }

        result
    }

    #[instrument(level = "trace", parent = &self.span, skip(self, planes))]
    #[profiling::function]
    pub fn page_flip<'a>(
        &self,
        planes: impl IntoIterator<Item = PlaneState<'a>>,
        flip_flags: PageFlipFlags,
    ) -> Result<(), Error> {
        let flip_flags = self.flip_flags(flip_flags);

        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mut used_planes = self.used_planes.lock().unwrap();
        let planes = planes.into_iter().collect::<Vec<_>>();

        // page flips work just like commits with fewer parameters..
        let prop_mapping = self.prop_mapping.read().unwrap();
        // Connector color properties and CRTC color pipeline blobs are deliberately omitted
        // (`None`): the kernel latches them from the last full commit, and re-emitting them on
        // every flip would make the kernel re-run its modeset checks and potentially cause
        // sinks to renegotiate infoframes.
        let req = AtomicRequest::build_request(
            &prop_mapping,
            self.crtc,
            None,
            self.state.read().unwrap().vrr,
            None,
            None,
            [],
            [],
            &*planes,
        )?;

        // .. and without `AtomicCommitFlags::AllowModeset`.
        // If we would set anything here, that would require a modeset, this would fail,
        // indicating a problem in our assumptions.
        trace!(?planes, "Queueing page flip: {:?}", req);
        let res = self
            .fd
            .atomic_commit(
                if flip_flags.contains(AtomicCommitFlags::PAGE_FLIP_EVENT) {
                    flip_flags | AtomicCommitFlags::NONBLOCK
                } else {
                    AtomicCommitFlags::NONBLOCK
                },
                req.build()?,
            )
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Page flip commit failed",
                    dev: self.fd.dev_path(),
                    source,
                })
            });

        if res.is_ok() {
            for plane in planes.iter() {
                if plane.config.is_some() {
                    used_planes.insert(plane.handle);
                } else {
                    used_planes.remove(&plane.handle);
                }
            }
        }

        res
    }

    // this helper function disconnects the plane.
    // this is mostly used to remove the contents quickly, e.g. on tty switch,
    // as other compositors might not make use of other planes,
    // leaving our e.g. cursor or overlays as a relict of a better time on the screen.
    pub fn clear_plane(&self, plane: plane::Handle) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let mapping = self.prop_mapping.read().unwrap();
        let mut req = AtomicRequest::new(&mapping);
        req.reset_plane(plane)?;
        let req = req.build()?;

        let result = self
            .fd
            .atomic_commit(AtomicCommitFlags::empty(), req)
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to commit on clear_plane",
                    dev: self.fd.dev_path(),
                    source,
                })
            });

        if result.is_ok() {
            self.used_planes.lock().unwrap().remove(&plane);
        }

        result
    }

    #[profiling::function]
    fn clear_state(&self) -> Result<(), Error> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(Error::DeviceInactive);
        }

        let _guard = self.span.enter();
        let prop_mapping = self.prop_mapping.read().unwrap();
        let mut req = AtomicRequest::new(&prop_mapping);
        // reset all planes we used
        for plane in self.used_planes.lock().unwrap().iter() {
            req.reset_plane(*plane)?;
        }

        // disable connectors again
        let current = self.state.read().unwrap();
        for conn in current.connectors.iter() {
            req.reset_connector(*conn)?;
        }

        // disable crtc
        req.reset_crtc(self.crtc)?;
        std::mem::drop(current);

        let res = self
            .fd
            .atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req.build()?)
            .map_err(|source| {
                Error::Access(AccessError {
                    errmsg: "Failed to commit on clear_state",
                    dev: self.fd.dev_path(),
                    source,
                })
            });

        if res.is_ok() {
            self.used_planes.lock().unwrap().clear();
            self.state.write().unwrap().clear();
        }

        res
    }

    pub(crate) fn reset_state<B: DevPath + ControlDevice + 'static>(
        &self,
        fd: Option<&B>,
    ) -> Result<(), Error> {
        *self.state.write().unwrap() = if let Some(fd) = fd {
            State::current_state(fd, self.crtc, &mut self.prop_mapping.write().unwrap())?
        } else {
            State::current_state(&*self.fd, self.crtc, &mut self.prop_mapping.write().unwrap())?
        };

        // Re-initialize the mode blob which might got lost after suspend/resume
        let mut pending = self.pending.write().unwrap();
        let blob = self.fd.create_property_blob(&pending.mode).map_err(|source| {
            Error::Access(AccessError {
                errmsg: "Failed to create Property Blob for mode",
                dev: self.fd.dev_path(),
                source,
            })
        })?;

        let old_blob = std::mem::replace(&mut pending.blob, blob);
        let _ = self.fd.destroy_property_blob(old_blob.into());

        // Re-initialize the HDR metadata blob, which might have gotten lost after
        // suspend/resume; the pending color state then re-asserts itself on the next commit,
        // since the re-read current state reflects whatever survived.
        if let Some(meta) = pending.color_state.hdr_metadata {
            let hdr_blob = self
                .fd
                .create_property_blob(&color::ffi::HdrOutputMetadata::from(meta))
                .map_err(|source| {
                    Error::Access(AccessError {
                        errmsg: "Failed to create HDR metadata property blob",
                        dev: self.fd.dev_path(),
                        source,
                    })
                })?;
            let old_blob = std::mem::replace(&mut pending.resolved_color.hdr_blob, hdr_blob);
            if let property::Value::Blob(id) = old_blob {
                if id != 0 {
                    let _ = self.fd.destroy_property_blob(id);
                }
            }
        }

        Ok(())
    }

    pub(crate) fn device_fd(&self) -> &DrmDeviceFd {
        self.fd.device_fd()
    }

    pub fn clear(&self) -> Result<(), Error> {
        self.clear_state()
    }
}

struct TestBuffer {
    fd: Arc<DrmDeviceInternal>,
    db: DumbBuffer,
    fb: framebuffer::Handle,
}

impl AsRef<framebuffer::Handle> for TestBuffer {
    fn as_ref(&self) -> &framebuffer::Handle {
        &self.fb
    }
}

impl Drop for TestBuffer {
    fn drop(&mut self) {
        let _ = self.fd.destroy_framebuffer(self.fb);
        let _ = self.fd.destroy_dumb_buffer(self.db);
    }
}

impl Drop for AtomicDrmSurface {
    fn drop(&mut self) {
        if !self.active.load(Ordering::SeqCst) {
            // the device is gone or we are on another tty
            // old state has been restored, we shouldn't touch it.
            // if we are on another tty the connectors will get disabled
            // by the device, when switching back
            return;
        }

        let _guard = self.span.enter();
        if let Err(err) = self.clear_state() {
            warn!("Unable to clear state: {}", err);
        }
    }
}

#[inline]
fn to_fixed<N: Coordinate>(n: N) -> u32 {
    f64::round(n.to_f64() * (1 << 16) as f64) as u32
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct DrmRotation: u8 {
        const ROTATE_0      =   0b00000001;
        const ROTATE_90     =   0b00000010;
        const ROTATE_180    =   0b00000100;
        const ROTATE_270    =   0b00001000;
        const REFLECT_X     =   0b00010000;
        const REFLECT_Y     =   0b00100000;
    }
}

impl From<Transform> for DrmRotation {
    #[inline]
    fn from(transform: Transform) -> Self {
        match transform {
            Transform::Normal => DrmRotation::ROTATE_0,
            Transform::_90 => DrmRotation::ROTATE_90,
            Transform::_180 => DrmRotation::ROTATE_180,
            Transform::_270 => DrmRotation::ROTATE_270,
            Transform::Flipped => DrmRotation::REFLECT_Y,
            Transform::Flipped90 => DrmRotation::REFLECT_Y | DrmRotation::ROTATE_90,
            Transform::Flipped180 => DrmRotation::REFLECT_Y | DrmRotation::ROTATE_180,
            Transform::Flipped270 => DrmRotation::REFLECT_Y | DrmRotation::ROTATE_270,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        backend::drm::surface::atomic::to_fixed,
        utils::{Physical, Rectangle},
    };

    use super::AtomicDrmSurface;

    fn is_send<S: Send>() {}

    #[test]
    fn surface_is_send() {
        is_send::<AtomicDrmSurface>();
    }

    #[test]
    fn test_fixed_point() {
        let geometry: Rectangle<f64, Physical> = Rectangle::from_size((1920.0, 1080.0).into());
        let fixed = to_fixed(geometry.size.w) as u64;
        assert_eq!(125829120, fixed);
    }

    #[test]
    fn test_fractional_fixed_point() {
        let geometry: Rectangle<f64, Physical> = Rectangle::from_size((1920.1, 1080.0).into());
        let fixed = to_fixed(geometry.size.w) as u64;
        assert_eq!(125835674, fixed);
    }
}

#[cfg(debug_assertions)]
struct AtomicRequest<'a> {
    mapping: &'a PropMapping,
    crtc_props: HashMap<crtc::Handle, HashMap<&'static str, property::Value<'a>>>,
    connector_props: HashMap<connector::Handle, HashMap<&'static str, property::Value<'a>>>,
    plane_props: HashMap<plane::Handle, HashMap<&'static str, property::Value<'a>>>,
}

#[cfg(not(debug_assertions))]
#[cfg_attr(not(debug_assertions), derive(Debug))]
struct AtomicRequest<'a> {
    mapping: &'a PropMapping,
    request: AtomicModeReq,
}

#[cfg(debug_assertions)]
impl fmt::Debug for AtomicRequest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtomicRequest")
            .field("crtcs", &self.crtc_props)
            .field("connectors", &self.connector_props)
            .field("plane", &self.plane_props)
            .finish()
    }
}

#[cfg(debug_assertions)]
impl<'a> AtomicRequest<'a> {
    fn new(mapping: &'a PropMapping) -> AtomicRequest<'a> {
        AtomicRequest {
            mapping,
            crtc_props: HashMap::new(),
            connector_props: HashMap::new(),
            plane_props: HashMap::new(),
        }
    }

    fn set_connector(
        &mut self,
        conn: connector::Handle,
        crtc: crtc::Handle,
        color: Option<&ResolvedColorState>,
    ) -> Result<(), Error> {
        let connector_props = self.connector_props.entry(conn).or_default();
        connector_props.insert("CRTC_ID", property::Value::CRTC(Some(crtc)));
        if let Some(color) = color {
            if let Some(value) = color.colorspace_values.get(&conn) {
                connector_props.insert("Colorspace", property::Value::Unknown(*value));
            }
            if self.mapping.conn_prop_handle(conn, "HDR_OUTPUT_METADATA").is_ok() {
                connector_props.insert("HDR_OUTPUT_METADATA", color.hdr_blob);
            }
            if let Some(max_bpc) = color.max_bpc {
                if self.mapping.conn_prop_handle(conn, "max bpc").is_ok() {
                    connector_props.insert("max bpc", property::Value::UnsignedRange(max_bpc));
                }
            }
        }
        Ok(())
    }

    fn reset_connector(&mut self, conn: connector::Handle) -> Result<(), Error> {
        let connector_props = self.connector_props.entry(conn).or_default();
        connector_props.insert("CRTC_ID", property::Value::CRTC(None));
        // Reset color signalling, so the next KMS client starts from a defined SDR state.
        // `max bpc` is left alone; there is no meaningful default to restore.
        if self.mapping.conn_prop_handle(conn, "Colorspace").is_ok() {
            // Default == DRM_MODE_COLORIMETRY_DEFAULT, a kernel uapi constant
            connector_props.insert("Colorspace", property::Value::Unknown(0));
        }
        if self.mapping.conn_prop_handle(conn, "HDR_OUTPUT_METADATA").is_ok() {
            connector_props.insert("HDR_OUTPUT_METADATA", property::Value::Blob(0));
        }
        Ok(())
    }

    fn set_crtc(
        &mut self,
        crtc: crtc::Handle,
        mode: Option<property::Value<'static>>,
        vrr: bool,
        crtc_color_blobs: Option<&CrtcColorBlobs>,
    ) -> Result<(), Error> {
        let crtc_props = self.crtc_props.entry(crtc).or_default();

        crtc_props.insert("ACTIVE", property::Value::Boolean(true));
        if let Some(blob) = mode {
            crtc_props.insert("MODE_ID", blob);
        }
        if self.mapping.crtc_prop_handle(crtc, "VRR_ENABLED").is_ok() {
            crtc_props.insert("VRR_ENABLED", property::Value::Boolean(vrr));
        } else if vrr {
            return Err(Error::UnknownProperty {
                handle: crtc.into(),
                name: "VRR_ENABLED",
            });
        }

        // Insert CRTC hardware color pipeline blobs for debug visualization.
        if let Some(blobs) = crtc_color_blobs {
            if self.mapping.crtc_prop_handle(crtc, "GAMMA_LUT").is_ok() {
                crtc_props.insert("GAMMA_LUT", blobs.gamma_blob);
            }
            if self.mapping.crtc_prop_handle(crtc, "CTM").is_ok() {
                crtc_props.insert("CTM", blobs.ctm_blob);
            }
        }

        Ok(())
    }

    fn reset_crtc(&mut self, crtc: crtc::Handle) -> Result<(), Error> {
        let crtc_props = self.crtc_props.entry(crtc).or_default();

        crtc_props.insert("ACTIVE", property::Value::Boolean(false));
        crtc_props.insert("MODE_ID", property::Value::Blob(0));
        if self.mapping.crtc_prop_handle(crtc, "VRR_ENABLED").is_ok() {
            crtc_props.insert("VRR_ENABLED", property::Value::Boolean(false));
        }
        Ok(())
    }

    fn set_plane(&mut self, crtc: crtc::Handle, plane_state: &PlaneState<'a>) -> Result<(), Error> {
        let handle = plane_state.handle;
        let plane_props = self.plane_props.entry(handle).or_default();

        if let Some(config) = plane_state.config.as_ref() {
            plane_props.insert("CRTC_ID", property::Value::CRTC(Some(crtc)));
            plane_props.insert("FB_ID", property::Value::Framebuffer(Some(config.fb)));
            // these are 16.16. fixed point
            plane_props.insert(
                "SRC_X",
                property::Value::UnsignedRange(to_fixed(config.src.loc.x) as u64),
            );
            plane_props.insert(
                "SRC_Y",
                property::Value::UnsignedRange(to_fixed(config.src.loc.y) as u64),
            );
            plane_props.insert(
                "SRC_W",
                property::Value::UnsignedRange(to_fixed(config.src.size.w) as u64),
            );
            plane_props.insert(
                "SRC_H",
                property::Value::UnsignedRange(to_fixed(config.src.size.h) as u64),
            );

            plane_props.insert("CRTC_X", property::Value::SignedRange(config.dst.loc.x as i64));
            plane_props.insert("CRTC_Y", property::Value::SignedRange(config.dst.loc.y as i64));
            plane_props.insert("CRTC_W", property::Value::UnsignedRange(config.dst.size.w as u64));
            plane_props.insert("CRTC_H", property::Value::UnsignedRange(config.dst.size.h as u64));

            if self.mapping.plane_prop_handle(handle, "rotation").is_ok() {
                plane_props.insert(
                    "rotation",
                    property::Value::Bitmask(DrmRotation::from(config.transform).bits() as u64),
                );
            } else if config.transform != Transform::Normal {
                // if we are missing the rotation property we can no rely on
                // the driver to report a non working configuration and can
                // only guarantee that Transform::Normal (no rotation) will
                // work
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "rotation",
                });
            }
            if self.mapping.plane_prop_handle(handle, "alpha").is_ok() {
                plane_props.insert(
                    "alpha",
                    property::Value::UnsignedRange((config.alpha * u16::MAX as f32).round() as u64),
                );
            } else if config.alpha != 1.0 {
                // if we are missing the alpha property we can not display any transparent alpha values
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "alpha",
                });
            }
            if self.mapping.plane_prop_handle(handle, "FB_DAMAGE_CLIPS").is_ok() {
                if let Some(damage) = config.damage_clips.as_ref() {
                    plane_props.insert("FB_DAMAGE_CLIPS", *damage);
                } else {
                    plane_props.insert("FB_DAMAGE_CLIPS", property::Value::Blob(0));
                }
            }
            if self.mapping.plane_prop_handle(handle, "IN_FENCE_FD").is_ok() {
                if let Some(fence) = config.fence.as_ref().map(|f| f.as_raw_fd()) {
                    plane_props.insert("IN_FENCE_FD", property::Value::SignedRange(fence as i64));
                } else {
                    plane_props.insert("IN_FENCE_FD", property::Value::SignedRange(-1));
                }
            } else if config.fence.is_some() {
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "IN_FENCE_FD",
                });
            }
        } else {
            self.reset_plane(handle)?;
        }

        Ok(())
    }

    fn reset_plane(&mut self, plane: plane::Handle) -> Result<(), Error> {
        let plane_props = self.plane_props.entry(plane).or_default();

        plane_props.insert("CRTC_ID", property::Value::CRTC(None));
        plane_props.insert("FB_ID", property::Value::Framebuffer(None));
        // these are 16.16. fixed point
        plane_props.insert("SRC_X", property::Value::UnsignedRange(0u64));
        plane_props.insert("SRC_Y", property::Value::UnsignedRange(0u64));
        plane_props.insert("SRC_W", property::Value::UnsignedRange(0u64));
        plane_props.insert("SRC_H", property::Value::UnsignedRange(0u64));

        plane_props.insert("CRTC_X", property::Value::SignedRange(0i64));
        plane_props.insert("CRTC_Y", property::Value::SignedRange(0i64));
        plane_props.insert("CRTC_W", property::Value::UnsignedRange(0u64));
        plane_props.insert("CRTC_H", property::Value::UnsignedRange(0u64));

        if self.mapping.plane_prop_handle(plane, "rotation").is_ok() {
            plane_props.insert(
                "rotation",
                property::Value::Bitmask(DrmRotation::from(Transform::Normal).bits() as u64),
            );
        }
        if self.mapping.plane_prop_handle(plane, "alpha").is_ok() {
            plane_props.insert("alpha", property::Value::UnsignedRange(0xffff));
        }
        if self.mapping.plane_prop_handle(plane, "FB_DAMAGE_CLIPS").is_ok() {
            plane_props.insert("FB_DAMAGE_CLIPS", property::Value::Blob(0));
        }
        if self.mapping.plane_prop_handle(plane, "IN_FENCE_FD").is_ok() {
            plane_props.insert("IN_FENCE_FD", property::Value::SignedRange(-1));
        }
        Ok(())
    }

    fn build(&self) -> Result<AtomicModeReq, Error> {
        let mut req = AtomicModeReq::new();

        for (crtc, props) in &self.crtc_props {
            for (name, value) in props {
                req.add_property(*crtc, self.mapping.crtc_prop_handle(*crtc, name)?, *value);
            }
        }
        for (conn, props) in &self.connector_props {
            for (name, value) in props {
                req.add_property(*conn, self.mapping.conn_prop_handle(*conn, name)?, *value);
            }
        }
        for (plane, props) in &self.plane_props {
            for (name, value) in props {
                req.add_property(*plane, self.mapping.plane_prop_handle(*plane, name)?, *value);
            }
        }

        Ok(req)
    }
}

#[cfg(not(debug_assertions))]
impl<'a> AtomicRequest<'a> {
    fn new(mapping: &'a PropMapping) -> AtomicRequest<'a> {
        AtomicRequest {
            mapping,
            request: AtomicModeReq::new(),
        }
    }

    fn set_connector(
        &mut self,
        conn: connector::Handle,
        crtc: crtc::Handle,
        color: Option<&ResolvedColorState>,
    ) -> Result<(), Error> {
        self.request.add_property(
            conn,
            self.mapping.conn_prop_handle(conn, "CRTC_ID")?,
            property::Value::CRTC(Some(crtc)),
        );
        if let Some(color) = color {
            if let Some(value) = color.colorspace_values.get(&conn) {
                self.request.add_property(
                    conn,
                    self.mapping.conn_prop_handle(conn, "Colorspace")?,
                    property::Value::Unknown(*value),
                );
            }
            if let Ok(prop) = self.mapping.conn_prop_handle(conn, "HDR_OUTPUT_METADATA") {
                self.request.add_property(conn, prop, color.hdr_blob);
            }
            if let Some(max_bpc) = color.max_bpc {
                if let Ok(prop) = self.mapping.conn_prop_handle(conn, "max bpc") {
                    self.request
                        .add_property(conn, prop, property::Value::UnsignedRange(max_bpc));
                }
            }
        }
        Ok(())
    }

    fn reset_connector(&mut self, conn: connector::Handle) -> Result<(), Error> {
        self.request.add_property(
            conn,
            self.mapping.conn_prop_handle(conn, "CRTC_ID")?,
            property::Value::CRTC(None),
        );
        // Reset color signalling, so the next KMS client starts from a defined SDR state.
        // `max bpc` is left alone; there is no meaningful default to restore.
        if let Ok(prop) = self.mapping.conn_prop_handle(conn, "Colorspace") {
            // Default == DRM_MODE_COLORIMETRY_DEFAULT, a kernel uapi constant
            self.request.add_property(conn, prop, property::Value::Unknown(0));
        }
        if let Ok(prop) = self.mapping.conn_prop_handle(conn, "HDR_OUTPUT_METADATA") {
            self.request.add_property(conn, prop, property::Value::Blob(0));
        }
        Ok(())
    }

    fn set_crtc(
        &mut self,
        crtc: crtc::Handle,
        mode: Option<property::Value<'static>>,
        vrr: bool,
        crtc_color_blobs: Option<&CrtcColorBlobs>,
    ) -> Result<(), Error> {
        if let Some(blob) = mode {
            self.request
                .add_property(crtc, self.mapping.crtc_prop_handle(crtc, "MODE_ID")?, blob);
        }

        self.request.add_property(
            crtc,
            self.mapping.crtc_prop_handle(crtc, "ACTIVE")?,
            property::Value::Boolean(true),
        );

        if let Ok(vrr_prop) = self.mapping.crtc_prop_handle(crtc, "VRR_ENABLED") {
            self.request
                .add_property(crtc, vrr_prop, property::Value::Boolean(vrr));
        } else if vrr {
            return Err(Error::UnknownProperty {
                handle: crtc.into(),
                name: "VRR_ENABLED",
            });
        }

        // Apply CRTC hardware color pipeline blobs (GAMMA_LUT, CTM).
        // Only set properties that the CRTC actually has; silently skip missing ones so
        // that clearing (Blob(0)) on unsupported hardware doesn't fail the commit.
        if let Some(blobs) = crtc_color_blobs {
            if let Ok(prop) = self.mapping.crtc_prop_handle(crtc, "GAMMA_LUT") {
                self.request.add_property(crtc, prop, blobs.gamma_blob);
            }
            if let Ok(prop) = self.mapping.crtc_prop_handle(crtc, "CTM") {
                self.request.add_property(crtc, prop, blobs.ctm_blob);
            }
        }

        Ok(())
    }

    fn reset_crtc(&mut self, crtc: crtc::Handle) -> Result<(), Error> {
        self.request.add_property(
            crtc,
            self.mapping.crtc_prop_handle(crtc, "ACTIVE")?,
            property::Value::Boolean(false),
        );
        self.request.add_property(
            crtc,
            self.mapping.crtc_prop_handle(crtc, "MODE_ID")?,
            property::Value::Blob(0),
        );
        if let Ok(prop) = self.mapping.crtc_prop_handle(crtc, "VRR_ENABLED") {
            self.request
                .add_property(crtc, prop, property::Value::Boolean(false));
        }
        Ok(())
    }

    fn set_plane(&mut self, crtc: crtc::Handle, plane_state: &PlaneState<'_>) -> Result<(), Error> {
        let handle = plane_state.handle;
        if let Some(config) = plane_state.config.as_ref() {
            // connect the plane to the CRTC
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_ID")?,
                property::Value::CRTC(Some(crtc)),
            );

            // Set the fb for the plane
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "FB_ID")?,
                property::Value::Framebuffer(Some(config.fb)),
            );

            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_X")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.loc.x) as u64),
            );
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_Y")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.loc.y) as u64),
            );
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_W")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.size.w) as u64),
            );
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "SRC_H")?,
                // these are 16.16. fixed point
                property::Value::UnsignedRange(to_fixed(config.src.size.h) as u64),
            );

            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_X")?,
                property::Value::SignedRange(config.dst.loc.x as i64),
            );
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_Y")?,
                property::Value::SignedRange(config.dst.loc.y as i64),
            );
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_W")?,
                property::Value::UnsignedRange(config.dst.size.w as u64),
            );
            self.request.add_property(
                handle,
                self.mapping.plane_prop_handle(handle, "CRTC_H")?,
                property::Value::UnsignedRange(config.dst.size.h as u64),
            );
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "rotation") {
                self.request.add_property(
                    handle,
                    prop,
                    property::Value::Bitmask(DrmRotation::from(config.transform).bits() as u64),
                );
            } else if config.transform != Transform::Normal {
                // if we are missing the rotation property we can no rely on
                // the driver to report a non working configuration and can
                // only guarantee that Transform::Normal (no rotation) will
                // work
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "rotation",
                });
            }
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "alpha") {
                self.request.add_property(
                    handle,
                    prop,
                    property::Value::UnsignedRange((config.alpha * u16::MAX as f32).round() as u64),
                );
            } else if config.alpha != 1.0 {
                // if we are missing the alpha property we can not display any transparent alpha values
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "alpha",
                });
            }
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "FB_DAMAGE_CLIPS") {
                if let Some(damage) = config.damage_clips.as_ref() {
                    self.request.add_property(handle, prop, *damage);
                } else {
                    self.request.add_property(handle, prop, property::Value::Blob(0));
                }
            }
            if let Ok(prop) = self.mapping.plane_prop_handle(handle, "IN_FENCE_FD") {
                if let Some(fence) = config.fence.as_ref().map(|f| f.as_raw_fd()) {
                    self.request
                        .add_property(handle, prop, property::Value::SignedRange(fence as i64));
                } else {
                    self.request
                        .add_property(handle, prop, property::Value::SignedRange(-1));
                }
            } else if config.fence.is_some() {
                return Err(Error::UnknownProperty {
                    handle: handle.into(),
                    name: "IN_FENCE_FD",
                });
            }
        } else {
            self.reset_plane(handle)?;
        }

        Ok(())
    }

    fn reset_plane(&mut self, plane: plane::Handle) -> Result<(), Error> {
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_ID")?,
            property::Value::CRTC(None),
        );

        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "FB_ID")?,
            property::Value::Framebuffer(None),
        );

        // reset the plane properties
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_X")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        );
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_Y")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        );
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_W")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        );
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "SRC_H")?,
            // these are 16.16. fixed point
            property::Value::UnsignedRange(0u64),
        );

        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_X")?,
            property::Value::SignedRange(0i64),
        );
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_Y")?,
            property::Value::SignedRange(0i64),
        );
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_W")?,
            property::Value::UnsignedRange(0u64),
        );
        self.request.add_property(
            plane,
            self.mapping.plane_prop_handle(plane, "CRTC_H")?,
            property::Value::UnsignedRange(0u64),
        );
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "rotation") {
            self.request.add_property(
                plane,
                prop,
                property::Value::Bitmask(DrmRotation::from(Transform::Normal).bits() as u64),
            );
        }
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "alpha") {
            self.request
                .add_property(plane, prop, property::Value::UnsignedRange(0xffff));
        }
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "FB_DAMAGE_CLIPS") {
            self.request.add_property(plane, prop, property::Value::Blob(0));
        }
        if let Ok(prop) = self.mapping.plane_prop_handle(plane, "IN_FENCE_FD") {
            self.request
                .add_property(plane, prop, property::Value::SignedRange(-1));
        }
        Ok(())
    }

    fn build(&self) -> Result<AtomicModeReq, Error> {
        Ok(self.request.clone())
    }
}

impl<'a> AtomicRequest<'a> {
    #[allow(clippy::too_many_arguments)]
    fn build_request(
        mapping: &'a PropMapping,
        crtc: crtc::Handle,
        blob: Option<property::Value<'static>>,
        vrr: bool,
        color: Option<&ResolvedColorState>,
        crtc_color_blobs: Option<&CrtcColorBlobs>,
        connectors: impl IntoIterator<Item = &'a connector::Handle>,
        removed_connectors: impl IntoIterator<Item = &'a connector::Handle>,
        planes: impl IntoIterator<Item = &'a PlaneState<'a>>,
    ) -> Result<AtomicRequest<'a>, Error> {
        let mut req = AtomicRequest::new(mapping);

        // requests consist out of a set of properties and their new values
        // for different drm objects (crtc, plane, connector, ...).

        // for every connector that is new, we need to set our crtc_id
        // (and the color state, if this is a full commit rather than a page flip)
        for conn in connectors {
            req.set_connector(*conn, crtc, color)?;
        }

        // for every connector that got removed, we need to set no crtc_id.
        // (this is a bit problematic, because this means we need to remove, commit, add, commit
        // in the right order to move a connector to another surface. otherwise we disable the
        // the connector here again...)
        for conn in removed_connectors {
            req.reset_connector(*conn)?;
        }

        // Set the crtc properties (active, mode_id, vrr_enabled, and hardware color pipeline).
        req.set_crtc(crtc, blob, vrr, crtc_color_blobs)?;

        for plane_state in planes.into_iter() {
            req.set_plane(crtc, plane_state)?;
        }

        Ok(req)
    }
}
