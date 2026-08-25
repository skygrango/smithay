//! Protocol for confining the pointer.
//!
//! This provides a way for the client to request that the pointer is confined to a region or
//! locked in place.
use std::{
    ops,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use papaya::{HashMap as LockfreeHashMap, Operation};

use wayland_protocols::wp::pointer_constraints::zv1::server::{
    zwp_confined_pointer_v1::{self, ZwpConfinedPointerV1},
    zwp_locked_pointer_v1::{self, ZwpLockedPointerV1},
    zwp_pointer_constraints_v1::{self, Lifetime, ZwpPointerConstraintsV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, WEnum, backend::GlobalId,
    protocol::wl_surface::WlSurface,
};

use super::compositor::{self, RegionAttributes};
use crate::{
    input::{SeatHandler, pointer::PointerHandle},
    utils::{Logical, Point},
    wayland::{Dispatch2, GlobalData, GlobalDispatch2, seat::PointerUserData},
};

const VERSION: u32 = 1;

/// Handler for pointer constraints
pub trait PointerConstraintsHandler: SeatHandler {
    /// Pointer lock or confinement constraint created for `pointer` on `surface`
    ///
    /// Use [`with_pointer_constraint`] to access the constraint.
    fn new_constraint(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {}

    /// Pointer constraint removed for `pointer` on `surface`
    fn remove_constraint(
        &mut self,
        _surface: &WlSurface,
        _pointer: &PointerHandle<Self>,
        _constraint: Option<&PointerConstraint>,
    ) {
    }

    /// The client holding a LockedPointer has committed a cursor position hint.
    ///
    /// This is emitted upon a surface commit if the cursor position hint has been updated.
    ///
    /// Use [`with_pointer_constraint`] to access the constraint and check if it is active.
    fn cursor_position_hint(
        &mut self,
        _surface: &WlSurface,
        _pointer: &PointerHandle<Self>,
        _location: Point<f64, Logical>,
    ) {
    }
}

/// Constraint confining pointer to a region of the surface
#[derive(Debug, Clone)]
pub struct ConfinedPointer {
    handle: zwp_confined_pointer_v1::ZwpConfinedPointerV1,
    region: Option<RegionAttributes>,
    pending_region: Option<RegionAttributes>,
    lifetime: WEnum<Lifetime>,
    active: Arc<AtomicBool>,
}

impl ConfinedPointer {
    /// Region in which to confine the pointer
    pub fn region(&self) -> Option<&RegionAttributes> {
        self.region.as_ref()
    }
}

/// Constraint locking pointer in place
#[derive(Debug, Clone)]
pub struct LockedPointer {
    handle: zwp_locked_pointer_v1::ZwpLockedPointerV1,
    region: Option<RegionAttributes>,
    pending_region: Option<RegionAttributes>,
    lifetime: WEnum<Lifetime>,
    cursor_position_hint: Option<Point<f64, Logical>>,
    pending_cursor_position_hint: Option<Point<f64, Logical>>,
    active: Arc<AtomicBool>,
}

impl LockedPointer {
    /// Region in which to activate the lock
    pub fn region(&self) -> Option<&RegionAttributes> {
        self.region.as_ref()
    }

    /// Position the client is rendering a cursor, if any
    pub fn cursor_position_hint(&self) -> Option<Point<f64, Logical>> {
        self.cursor_position_hint
    }
}

/// A constraint imposed on the pointer instance
#[derive(Debug, Clone)]
pub enum PointerConstraint {
    /// Pointer is confined to a region of the surface
    Confined(ConfinedPointer),
    /// Pointer is locked in place
    Locked(LockedPointer),
}

/// A reference to a pointer constraint that can be activated or deactivated.
///
/// The derefs to `[PointerConstraint]`.
#[derive(Debug)]
pub struct PointerConstraintRef<'a, D: SeatHandler + 'static> {
    constraint: &'a PointerConstraint,
    constraints: &'a LockfreeHashMap<PointerHandle<D>, PointerConstraint>,
}

impl<D: SeatHandler + 'static> ops::Deref for PointerConstraintRef<'_, D> {
    type Target = PointerConstraint;

    fn deref(&self) -> &Self::Target {
        self.constraint
    }
}

impl<D: SeatHandler + PointerConstraintsHandler + 'static> PointerConstraintRef<'_, D> {
    /// Send `locked`/`unlocked`
    ///
    /// This is not sent automatically since compositors may have different
    /// policies about when to allow and activate constraints.
    pub fn activate(&self) {
        match self.constraint {
            PointerConstraint::Confined(confined) => {
                if !confined.active.swap(true, Ordering::SeqCst) {
                    confined.handle.confined();
                }
            }
            PointerConstraint::Locked(locked) => {
                if !locked.active.swap(true, Ordering::SeqCst) {
                    locked.handle.locked();
                }
            }
        }
    }

    /// Send `unlocked`/`unconfined`
    ///
    /// For oneshot constraints, will destroy the constraint.
    ///
    /// This is sent automatically when the surface loses pointer focus, but
    /// may also be invoked while the surface is focused.
    ///
    /// Returns the deactivated constraint if deactivation occurred.
    /// The caller is responsible for invoking
    /// [`PointerConstraintsHandler::remove_constraint`] outside of the
    /// [`with_pointer_constraint`] closure to avoid deadlocks.
    pub fn deactivate(self, pointer: &PointerHandle<D>) -> Option<PointerConstraint> {
        let deactivated = match self.constraint {
            PointerConstraint::Confined(confined) => {
                if confined.active.swap(false, Ordering::SeqCst) {
                    confined.handle.unconfined();
                    true
                } else {
                    false
                }
            }
            PointerConstraint::Locked(locked) => {
                if locked.active.swap(false, Ordering::SeqCst) {
                    locked.handle.unlocked();
                    true
                } else {
                    false
                }
            }
        };

        if deactivated {
            if self.lifetime() == WEnum::Value(Lifetime::Oneshot) {
                self.constraints.pin().remove(pointer);
            }
            Some(self.constraint.clone())
        } else {
            None
        }
    }
}

impl PointerConstraint {
    /// Constraint is active
    pub fn is_active(&self) -> bool {
        match self {
            PointerConstraint::Confined(confined) => &confined.active,
            PointerConstraint::Locked(locked) => &locked.active,
        }
        .load(Ordering::SeqCst)
    }

    /// Region in which to lock or confine the pointer
    pub fn region(&self) -> Option<&RegionAttributes> {
        match self {
            PointerConstraint::Confined(confined) => confined.region(),
            PointerConstraint::Locked(locked) => locked.region(),
        }
    }

    fn lifetime(&self) -> WEnum<Lifetime> {
        match self {
            PointerConstraint::Confined(confined) => confined.lifetime,
            PointerConstraint::Locked(locked) => locked.lifetime,
        }
    }

    /// Commits the pending state of the constraint, and returns the cursor position hint if it has changed.
    fn commit(&mut self) -> Option<Point<f64, Logical>> {
        match self {
            Self::Confined(confined) => {
                confined.region.clone_from(&confined.pending_region);
                None
            }
            Self::Locked(locked) => {
                locked.region.clone_from(&locked.pending_region);
                locked.pending_cursor_position_hint.take().inspect(|hint| {
                    locked.cursor_position_hint = Some(*hint);
                })
            }
        }
    }
}

/// Pointer constraints state.
#[derive(Debug)]
pub struct PointerConstraintsState {
    global: GlobalId,
}

impl PointerConstraintsState {
    /// Create a new pointer constraints global
    pub fn new<D>(display: &DisplayHandle) -> Self
    where
        D: GlobalDispatch<ZwpPointerConstraintsV1, GlobalData>,
        D: Dispatch<ZwpPointerConstraintsV1, GlobalData>,
        D: Dispatch<ZwpConfinedPointerV1, PointerConstraintUserData<D>>,
        D: Dispatch<ZwpLockedPointerV1, PointerConstraintUserData<D>>,
        D: SeatHandler,
        D: 'static,
    {
        let global = display.create_global::<D, ZwpPointerConstraintsV1, _>(VERSION, GlobalData);

        Self { global }
    }

    /// Get the id of ZwpPointerConstraintsV1 global
    pub fn global(&self) -> GlobalId {
        self.global.clone()
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct PointerConstraintUserData<D: SeatHandler> {
    surface: WlSurface,
    pointer: Option<PointerHandle<D>>,
}

struct PointerConstraintData<D: SeatHandler + 'static> {
    constraints: LockfreeHashMap<PointerHandle<D>, PointerConstraint>,
}

// TODO Public method to get current constraints for surface/seat
/// Get constraint for surface and pointer, if any
pub fn with_pointer_constraint<
    D: SeatHandler + 'static,
    T,
    F: FnOnce(Option<PointerConstraintRef<'_, D>>) -> T,
>(
    surface: &WlSurface,
    pointer: &PointerHandle<D>,
    f: F,
) -> T {
    with_constraint_data::<D, _, _>(surface, |data| match data {
        Some(data) => {
            let map = data.constraints.pin();
            let constraint = map.get(pointer).map(|constraint| PointerConstraintRef {
                constraint,
                constraints: &data.constraints,
            });
            f(constraint)
        }
        None => f(None),
    })
}

fn update_constraint<D: SeatHandler + 'static>(
    surface: &WlSurface,
    pointer: &PointerHandle<D>,
    mut f: impl FnMut(&mut PointerConstraint),
) {
    with_constraint_data::<D, _, _>(surface, |data| {
        if let Some(data) = data {
            data.constraints.pin().compute(pointer.clone(), |entry| {
                if let Some((_key, constraint)) = entry {
                    let mut constraint = constraint.clone();
                    f(&mut constraint);
                    Operation::Insert(constraint)
                } else {
                    Operation::Abort(())
                }
            });
        }
    });
}

fn commit_hook<D: SeatHandler + PointerConstraintsHandler + 'static>(
    state: &mut D,
    _dh: &DisplayHandle,
    surface: &WlSurface,
) {
    let position_hints = with_constraint_data::<D, _, _>(surface, |data| {
        let Some(data) = data else { return Vec::new() };
        let map = data.constraints.pin();
        let mut position_hints = Vec::new();

        let keys: Vec<_> = map.keys().cloned().collect();
        for pointer in keys {
            let mut hint = None;
            map.compute(pointer.clone(), |entry| {
                if let Some((_key, constraint)) = entry {
                    let mut constraint = constraint.clone();
                    hint = constraint.commit();
                    Operation::Insert(constraint)
                } else {
                    Operation::Abort(())
                }
            });
            if let Some(hint) = hint {
                position_hints.push((pointer, hint));
            }
        }

        position_hints
    });

    for (pointer, hint) in position_hints {
        state.cursor_position_hint(surface, &pointer, hint);
    }
}

/// Get `PointerConstraintData` associated with a surface, if any.
fn with_constraint_data<D: SeatHandler + 'static, T, F: FnOnce(Option<&PointerConstraintData<D>>) -> T>(
    surface: &WlSurface,
    f: F,
) -> T {
    compositor::with_states(surface, |states| {
        let data = states.data_map.get::<PointerConstraintData<D>>();
        f(data)
    })
}

/// Add constraint for surface, or raise protocol error if one exists
fn add_constraint<D: SeatHandler + PointerConstraintsHandler + 'static>(
    pointer_constraints: &ZwpPointerConstraintsV1,
    surface: &WlSurface,
    pointer: &PointerHandle<D>,
    constraint: PointerConstraint,
) {
    let mut added = false;
    compositor::with_states(surface, |states| {
        added = states
            .data_map
            .insert_if_missing_threadsafe(|| PointerConstraintData::<D> {
                constraints: LockfreeHashMap::new(),
            });
        let data = states.data_map.get::<PointerConstraintData<D>>().unwrap();
        let map = data.constraints.pin();

        if map.contains_key(pointer) {
            pointer_constraints.post_error(
                zwp_pointer_constraints_v1::Error::AlreadyConstrained,
                "pointer constraint already exists for surface and seat",
            );
        } else {
            map.insert(pointer.clone(), constraint);
        }
    });

    if added {
        compositor::add_post_commit_hook(surface, commit_hook::<D>);
    }
}

fn remove_constraint<D: SeatHandler + PointerConstraintsHandler + 'static>(
    state: &mut D,
    surface: &WlSurface,
    pointer: &PointerHandle<D>,
) {
    let constraint = remove_constraint_from_surface::<D>(surface, pointer);
    if let Some(constraint) = constraint {
        state.remove_constraint(surface, pointer, Some(&constraint));
    }
}

fn remove_constraint_from_surface<D: SeatHandler + 'static>(
    surface: &WlSurface,
    pointer: &PointerHandle<D>,
) -> Option<PointerConstraint> {
    with_constraint_data::<D, _, _>(surface, |data| {
        data.and_then(|data| data.constraints.pin().remove(pointer).cloned())
    })
}

impl<D> Dispatch2<ZwpPointerConstraintsV1, D> for GlobalData
where
    D: Dispatch<ZwpConfinedPointerV1, PointerConstraintUserData<D>>,
    D: Dispatch<ZwpLockedPointerV1, PointerConstraintUserData<D>>,
    D: SeatHandler,
    D: PointerConstraintsHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &wayland_server::Client,
        pointer_constraints: &ZwpPointerConstraintsV1,
        request: zwp_pointer_constraints_v1::Request,
        _dh: &DisplayHandle,
        data_init: &mut wayland_server::DataInit<'_, D>,
    ) {
        match request {
            zwp_pointer_constraints_v1::Request::LockPointer {
                id,
                surface,
                pointer,
                region,
                lifetime,
            } => {
                let region = region.as_ref().map(compositor::get_region_attributes);
                let pointer = pointer.data::<PointerUserData<D>>().unwrap().handle.clone();
                let handle = data_init.init(
                    id,
                    PointerConstraintUserData {
                        surface: surface.clone(),
                        pointer: pointer.clone(),
                    },
                );
                if let Some(pointer) = pointer {
                    add_constraint(
                        pointer_constraints,
                        &surface,
                        &pointer,
                        PointerConstraint::Locked(LockedPointer {
                            handle,
                            region: region.clone(),
                            pending_region: region,
                            lifetime,
                            cursor_position_hint: None,
                            pending_cursor_position_hint: None,
                            active: Arc::new(AtomicBool::new(false)),
                        }),
                    );
                    state.new_constraint(&surface, &pointer);
                }
            }
            zwp_pointer_constraints_v1::Request::ConfinePointer {
                id,
                surface,
                pointer,
                region,
                lifetime,
            } => {
                let region = region.as_ref().map(compositor::get_region_attributes);
                let pointer = pointer.data::<PointerUserData<D>>().unwrap().handle.clone();
                let handle = data_init.init(
                    id,
                    PointerConstraintUserData {
                        surface: surface.clone(),
                        pointer: pointer.clone(),
                    },
                );
                if let Some(pointer) = pointer {
                    add_constraint(
                        pointer_constraints,
                        &surface,
                        &pointer,
                        PointerConstraint::Confined(ConfinedPointer {
                            handle,
                            region: region.clone(),
                            pending_region: region,
                            lifetime,
                            active: Arc::new(AtomicBool::new(false)),
                        }),
                    );
                    state.new_constraint(&surface, &pointer);
                }
            }
            zwp_pointer_constraints_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }
}

impl<D> GlobalDispatch2<ZwpPointerConstraintsV1, D> for GlobalData
where
    D: Dispatch<ZwpPointerConstraintsV1, GlobalData> + SeatHandler + 'static,
{
    fn bind(
        &self,
        _state: &mut D,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpPointerConstraintsV1>,
        data_init: &mut DataInit<'_, D>,
    ) {
        data_init.init(resource, GlobalData);
    }
}

impl<D> Dispatch2<ZwpConfinedPointerV1, D> for PointerConstraintUserData<D>
where
    D: SeatHandler,
    D: PointerConstraintsHandler,
    D: 'static,
{
    fn request(
        &self,
        _state: &mut D,
        _client: &wayland_server::Client,
        _confined_pointer: &ZwpConfinedPointerV1,
        request: zwp_confined_pointer_v1::Request,
        _dh: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, D>,
    ) {
        let Some(pointer) = &self.pointer else {
            return;
        };

        match request {
            zwp_confined_pointer_v1::Request::SetRegion { region } => {
                update_constraint(&self.surface, pointer, |constraint| {
                    if let PointerConstraint::Confined(confined) = constraint {
                        confined.pending_region = region.as_ref().map(compositor::get_region_attributes);
                    }
                });
            }
            zwp_confined_pointer_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }

    fn destroyed(
        &self,
        state: &mut D,
        _client: wayland_server::backend::ClientId,
        _resource: &ZwpConfinedPointerV1,
    ) {
        let Some(pointer) = &self.pointer else {
            return;
        };

        remove_constraint(state, &self.surface, pointer);
    }
}

impl<D> Dispatch2<ZwpLockedPointerV1, D> for PointerConstraintUserData<D>
where
    D: SeatHandler,
    D: PointerConstraintsHandler,
    D: 'static,
{
    fn request(
        &self,
        _state: &mut D,
        _client: &wayland_server::Client,
        _locked_pointer: &ZwpLockedPointerV1,
        request: zwp_locked_pointer_v1::Request,
        _dh: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, D>,
    ) {
        let Some(pointer) = &self.pointer else {
            return;
        };

        match request {
            zwp_locked_pointer_v1::Request::SetCursorPositionHint { surface_x, surface_y } => {
                update_constraint(&self.surface, pointer, |constraint| {
                    if let PointerConstraint::Locked(locked) = constraint {
                        locked.pending_cursor_position_hint = Some((surface_x, surface_y).into());
                    }
                });
            }
            zwp_locked_pointer_v1::Request::SetRegion { region } => {
                update_constraint(&self.surface, pointer, |constraint| {
                    if let PointerConstraint::Locked(locked) = constraint {
                        locked.pending_region = region.as_ref().map(compositor::get_region_attributes);
                    }
                });
            }
            zwp_locked_pointer_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }

    fn destroyed(
        &self,
        state: &mut D,
        _client: wayland_server::backend::ClientId,
        _resource: &ZwpLockedPointerV1,
    ) {
        let Some(pointer) = &self.pointer else {
            return;
        };

        remove_constraint(state, &self.surface, pointer);
    }
}
