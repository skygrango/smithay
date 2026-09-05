use crate::{
    utils::{
        Serial,
        hook::{Hook, HookId},
    },
    wayland::compositor::{SUBSURFACE_ROLE, handlers::SubsurfaceState},
};

use super::{
    BufferAssignment, CompositorHandler, SurfaceAttributes, SurfaceData,
    cache::MultiCache,
    handlers::{SurfaceUserData, is_effectively_sync},
    transaction::{Barrier, Blocker, PendingTransaction, TransactionQueue},
};
use std::{
    any::Any,
    collections::VecDeque,
    fmt,
    sync::{Arc, Mutex, MutexGuard, atomic::Ordering},
};
use wayland_server::{
    DisplayHandle, Resource,
    protocol::{wl_output::Transform, wl_surface::WlSurface},
};

type CommitHook = dyn Fn(&mut dyn Any, &DisplayHandle, &WlSurface) + Send + Sync;
type DestructionHook = dyn Fn(&mut dyn Any, &WlSurface) + Send + Sync;

/// Node of a subsurface tree, holding some user specified data type U
/// at each node
///
/// This type is internal to Smithay, and should not appear in the
/// public API
///
/// It is a bidirectional tree, meaning we can move along it in both
/// direction (top-bottom or bottom-up). We are taking advantage of the
/// fact that lifetime of objects are decided by Wayland-server to ensure
/// the cleanup will be done properly, and we won't leak anything.
///
/// Each node also appears within its children list, to allow relative placement
/// between them.
pub(crate) struct PrivateSurfaceData {
    parent: Option<WlSurface>,
    children: Vec<WlSurface>,
    public_data: SurfaceData,
    pending_transaction: PendingTransaction,
    current_txid: Serial,
    pre_commit_hooks: Vec<Hook<CommitHook>>,
    post_commit_hooks: Vec<Hook<CommitHook>>,
    destruction_hooks: Vec<Hook<DestructionHook>>,
    pub(crate) timeline_queue: Arc<Mutex<TransactionQueue>>,
}

impl fmt::Debug for PrivateSurfaceData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrivateSurfaceData")
            .field("parent", &self.parent)
            .field("children", &self.children)
            .field("public_data", &self.public_data)
            .field("pending_transaction", &"...")
            .field("current_txid", &self.current_txid)
            .field("commit_hooks", &"...")
            .field("pre_commit_hooks.len", &self.pre_commit_hooks.len())
            .field("post_commit_hooks.len", &self.post_commit_hooks.len())
            .field("destruction_hooks.len", &self.destruction_hooks.len())
            .field("timeline_queue", &self.timeline_queue)
            .finish()
    }
}

/// An error type signifying that the surface already has a role and
/// cannot be assigned an other
///
/// Generated if you attempt a role operation on a surface that does
/// not have the role you asked for.
#[derive(Debug)]
pub struct AlreadyHasRole;

impl std::fmt::Display for AlreadyHasRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Surface already has a role.")
    }
}

impl std::error::Error for AlreadyHasRole {}

pub enum Location {
    Before,
    After,
}

/// Possible actions to do after handling a node during tree traversal
#[derive(Debug)]
pub enum TraversalAction<T> {
    /// Traverse its children as well, providing them the data T
    DoChildren(T),
    /// Skip its children
    SkipChildren,
    /// Stop traversal completely
    Break,
}

impl PrivateSurfaceData {
    pub fn new() -> Mutex<PrivateSurfaceData> {
        Mutex::new(PrivateSurfaceData {
            parent: None,
            children: vec![],
            public_data: SurfaceData {
                role: Default::default(),
                data_map: Default::default(),
                cached_state: MultiCache::new(),
            },
            pending_transaction: Default::default(),
            current_txid: Serial(0),
            pre_commit_hooks: Vec::new(),
            post_commit_hooks: Vec::new(),
            destruction_hooks: Vec::new(),
            timeline_queue: Arc::new(Mutex::new(TransactionQueue::default())),
        })
    }

    /// Initializes the surface, must be called at creation for state coherence
    pub fn init(surface: &WlSurface) {
        let mut my_data = Self::lock_user_data(surface);
        debug_assert!(my_data.children.is_empty());
        my_data.children.push(surface.clone());
    }

    /// Cleans the `as_ref().user_data` of that surface, must be called when it is destroyed
    pub fn cleanup<D: 'static>(state: &mut D, surface_data: &SurfaceUserData, surface: &WlSurface) {
        // Don't hold the lock on `surface_data.inner`, while also locking the parent
        let old_parent = surface_data.inner.lock().unwrap().parent.take();
        if let Some(old_parent) = old_parent {
            // We had a parent, lets unregister ourselves from it
            let old_parent_mutex = &old_parent.data::<SurfaceUserData>().unwrap().inner;
            let mut old_parent_guard = old_parent_mutex.lock().unwrap();
            old_parent_guard.children.retain(|c| c.id() != surface.id());
        }

        let my_data_mutex = &surface_data.inner;
        let mut my_data = my_data_mutex.lock().unwrap();
        // orphan all our children
        for child in my_data.children.drain(..) {
            let child_mutex = &child.data::<SurfaceUserData>().unwrap().inner;
            if std::ptr::eq(child_mutex, my_data_mutex) {
                // This child is ourselves, don't do anything.
                continue;
            }

            let mut child_guard = child_mutex.lock().unwrap();
            child_guard.parent = None;
        }
        let mut guard = my_data.public_data.cached_state.get::<SurfaceAttributes>();
        if let Some(BufferAssignment::NewBuffer(buffer)) = guard.current().buffer.take() {
            buffer.release();
        };
        if let Some(BufferAssignment::NewBuffer(buffer)) = guard.pending().buffer.take() {
            buffer.release();
        };

        let hooks = my_data.destruction_hooks.clone();
        // don't hold the mutex while the hooks are invoked
        drop(guard);
        drop(my_data);
        for hook in hooks {
            (hook.cb)(state, surface)
        }
    }

    pub fn lock_user_data(surface: &WlSurface) -> MutexGuard<'_, PrivateSurfaceData> {
        surface.data::<SurfaceUserData>().unwrap().inner.lock().unwrap()
    }

    pub fn set_role(surface: &WlSurface, role: &'static str) -> Result<(), AlreadyHasRole> {
        let mut my_data = Self::lock_user_data(surface);
        if my_data.public_data.role.is_some() && my_data.public_data.role != Some(role) {
            return Err(AlreadyHasRole);
        }
        my_data.public_data.role = Some(role);
        Ok(())
    }

    pub fn get_role(surface: &WlSurface) -> Option<&'static str> {
        Self::lock_user_data(surface).public_data.role
    }

    pub fn with_states<T, F: FnOnce(&SurfaceData) -> T>(surface: &WlSurface, f: F) -> T {
        let guard = Self::lock_user_data(surface);
        f(&guard.public_data)
    }

    pub fn add_blocker(surface: &WlSurface, blocker: impl Blocker + Send + 'static) {
        Self::lock_user_data(surface)
            .pending_transaction
            .add_blocker(blocker)
    }

    pub fn remove_pre_commit_hook(surface: &WlSurface, hook_id: &HookId) {
        Self::lock_user_data(surface)
            .pre_commit_hooks
            .retain(|hook| hook.id != *hook_id);
    }

    pub fn remove_post_commit_hook(surface: &WlSurface, hook_id: &HookId) {
        Self::lock_user_data(surface)
            .post_commit_hooks
            .retain(|hook| hook.id != *hook_id);
    }

    pub fn remove_destruction_hook(surface: &WlSurface, hook_id: &HookId) {
        Self::lock_user_data(surface)
            .destruction_hooks
            .retain(|hook| hook.id != *hook_id);
    }

    pub fn add_pre_commit_hook(
        surface: &WlSurface,
        hook: impl Fn(&mut dyn Any, &DisplayHandle, &WlSurface) + Send + Sync + 'static,
    ) -> HookId {
        let hook: Hook<CommitHook> = Hook::new(Arc::new(hook));
        let id = hook.id.clone();
        Self::lock_user_data(surface).pre_commit_hooks.push(hook);
        id
    }

    pub fn add_post_commit_hook(
        surface: &WlSurface,
        hook: impl Fn(&mut dyn Any, &DisplayHandle, &WlSurface) + Send + Sync + 'static,
    ) -> HookId {
        let hook: Hook<CommitHook> = Hook::new(Arc::new(hook));
        let id = hook.id.clone();
        Self::lock_user_data(surface).post_commit_hooks.push(hook);
        id
    }

    pub fn add_destruction_hook(
        surface: &WlSurface,
        hook: impl Fn(&mut dyn Any, &WlSurface) + Send + Sync + 'static,
    ) -> HookId {
        let hook: Hook<DestructionHook> = Hook::new(Arc::new(hook));
        let id = hook.id.clone();
        Self::lock_user_data(surface).destruction_hooks.push(hook);
        id
    }

    pub fn invoke_pre_commit_hooks<D: 'static>(state: &mut D, dh: &DisplayHandle, surface: &WlSurface) {
        // don't hold the mutex while the hooks are invoked
        let hooks = Self::lock_user_data(surface).pre_commit_hooks.clone();
        for hook in hooks {
            (hook.cb)(state, dh, surface);
        }
    }

    pub fn invoke_post_commit_hooks<D: 'static>(state: &mut D, dh: &DisplayHandle, surface: &WlSurface) {
        // don't hold the mutex while the hooks are invoked
        let hooks = Self::lock_user_data(surface).post_commit_hooks.clone();
        for hook in hooks {
            (hook.cb)(state, dh, surface);
        }
    }

    fn commit_sync_surface_tree(
        surface: &WlSurface,
        parent_transaction: &PendingTransaction,
        dh: &DisplayHandle,
    ) {
        let children = PrivateSurfaceData::get_children(surface);
        let mut my_data = Self::lock_user_data(surface);

        for child in children {
            Self::commit_sync_surface_tree(&child, &my_data.pending_transaction, dh);
        }

        let current_txid = my_data.current_txid;
        my_data.public_data.cached_state.commit(Some(current_txid), dh);
        my_data
            .pending_transaction
            .insert_state(surface.clone(), current_txid);

        let child_tx = std::mem::take(&mut my_data.pending_transaction);
        child_tx.merge_into(parent_transaction);
        my_data.current_txid.0 = my_data.current_txid.0.wrapping_add(1);
    }

    /// Returns the timeline queue for this surface
    pub fn timeline_queue(surface: &WlSurface) -> Arc<Mutex<TransactionQueue>> {
        Self::lock_user_data(surface).timeline_queue.clone()
    }

    /// Updates the timeline queue for all direct and indirect descendants that are effectively
    /// synchronized to the given surface.
    pub(crate) fn update_sync_subtree_queues(
        surface: &WlSurface,
        target_queue: &Arc<Mutex<TransactionQueue>>,
    ) {
        let children = Self::get_children(surface);
        for child in children {
            let is_child_sync = Self::with_states(&child, |state| {
                state
                    .data_map
                    .get::<SubsurfaceState>()
                    .map(|s| s.sync.load(Ordering::Acquire))
                    .unwrap_or(false)
            });

            if is_child_sync {
                let mut child_guard = Self::lock_user_data(&child);
                child_guard.timeline_queue = target_queue.clone();
                drop(child_guard);
                Self::update_sync_subtree_queues(&child, target_queue);
            }
        }
    }

    /// Collects the surface and all direct and indirect descendants that are synchronized to it.
    pub(crate) fn collect_sync_subtree(surface: &WlSurface, out: &mut Vec<WlSurface>) {
        out.push(surface.clone());
        let children = Self::get_children(surface);
        for child in children {
            let is_child_sync = Self::with_states(&child, |state| {
                state
                    .data_map
                    .get::<SubsurfaceState>()
                    .map(|s| s.sync.load(Ordering::Acquire))
                    .unwrap_or(false)
            });
            if is_child_sync {
                Self::collect_sync_subtree(&child, out);
            }
        }
    }

    /// Links a subsurface's timeline queue to its parent queue upon becoming synchronized
    pub fn sync_subsurface_queue(surface: &WlSurface) {
        if let Some(parent) = Self::get_parent(surface) {
            let parent_queue = Self::timeline_queue(&parent);
            let mut child_guard = Self::lock_user_data(surface);
            let old_queue = child_guard.timeline_queue.clone();
            child_guard.timeline_queue = parent_queue.clone();
            drop(child_guard);

            if !Arc::ptr_eq(&old_queue, &parent_queue) {
                let mut old_guard = old_queue.lock().unwrap();
                let mut parent_guard = parent_queue.lock().unwrap();
                while let Some(tx) = old_guard.transactions.pop_front() {
                    parent_guard.transactions.push_back(tx);
                }
            }

            Self::update_sync_subtree_queues(surface, &parent_queue);
        }
    }

    /// Assigns a new independent timeline queue to a subsurface upon becoming desynchronized
    pub fn desync_subsurface_queue<C: CompositorHandler + 'static>(
        surface: &WlSurface,
        dh: &DisplayHandle,
        state: &mut C,
    ) -> Arc<Mutex<TransactionQueue>> {
        let new_queue = Arc::new(Mutex::new(TransactionQueue::default()));
        let mut subtree_surfaces = Vec::new();
        Self::collect_sync_subtree(surface, &mut subtree_surfaces);

        let (old_queue, parent) = {
            let mut child_guard = Self::lock_user_data(surface);
            let old_queue = child_guard.timeline_queue.clone();
            child_guard.timeline_queue = new_queue.clone();
            let parent = child_guard.parent.clone();
            (old_queue, parent)
        };

        let mut last_merged_barrier: Option<Barrier> = None;
        if let Some(ref parent) = parent {
            let parent_queue = Self::timeline_queue(parent);
            if Arc::ptr_eq(&old_queue, &parent_queue) {
                let mut parent_guard = parent_queue.lock().unwrap();
                let mut new_guard = new_queue.lock().unwrap();

                let mut remaining_parent_txs = VecDeque::new();

                while let Some(mut tx) = parent_guard.transactions.pop_front() {
                    if tx.contains_any_surface(&subtree_surfaces) {
                        if tx.only_contains_surfaces(&subtree_surfaces) {
                            // Belongs exclusively to this subtree. Move to new_queue.
                            if let Some(ref barrier) = last_merged_barrier {
                                tx.add_blocker(barrier.clone());
                            }
                            new_guard.append(tx);
                        } else {
                            // Merged transaction containing both parent/outside surfaces and this subtree.
                            // Must remain in parent_queue.
                            let barrier = Barrier::new(false);
                            tx.completion_barriers.push(barrier.clone());
                            last_merged_barrier = Some(barrier);
                            remaining_parent_txs.push_back(tx);
                        }
                    } else {
                        // Unrelated transaction, keep in parent_queue.
                        remaining_parent_txs.push_back(tx);
                    }
                }
                parent_guard.transactions = remaining_parent_txs;
            }
        }

        if let Some(barrier) = last_merged_barrier {
            Self::lock_user_data(surface)
                .pending_transaction
                .add_blocker(barrier);
        }

        // Propagate the new independent queue to all synchronized descendants in this subtree.
        // Safe to call now because lock_user_data(surface) was already dropped.
        Self::update_sync_subtree_queues(surface, &new_queue);

        // According to Wayland protocol specification:
        // "If a surface's parent surface behaves as desynchronized, then the cached state is applied on set_desync."
        let parent_effectively_sync = parent.as_ref().is_some_and(is_effectively_sync);

        if !parent_effectively_sync {
            let pending_tx = {
                let mut guard = Self::lock_user_data(surface);
                if !guard.pending_transaction.is_empty() {
                    Some(std::mem::take(&mut guard.pending_transaction))
                } else {
                    None
                }
            };
            if let Some(tx) = pending_tx {
                new_queue.lock().unwrap().append(tx.finalize());
            }
        }

        Self::drain_surface_tree(surface, dh, state);

        new_queue
    }

    /// Drains and applies ready transactions across the entire window surface tree containing `surface`.
    ///
    /// Evaluates queues in topological order (root first, then descendants) to allow completion
    /// barriers from parent transactions to unblock dependent transactions in child queues
    /// within the same event dispatch.
    pub fn drain_surface_tree<C: CompositorHandler + 'static>(
        surface: &WlSurface,
        dh: &DisplayHandle,
        state: &mut C,
    ) {
        let mut root = surface.clone();
        while let Some(parent) = Self::get_parent(&root) {
            root = parent;
        }

        let mut queues: Vec<Arc<Mutex<TransactionQueue>>> = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(current) = queue.pop_front() {
            let q = Self::timeline_queue(&current);
            if !queues.iter().any(|existing| Arc::ptr_eq(existing, &q)) {
                queues.push(q);
            }
            for child in PrivateSurfaceData::get_children(&current) {
                queue.push_back(child);
            }
        }

        loop {
            let mut transactions = Vec::new();
            for queue in &queues {
                let ready = {
                    let mut guard = queue.lock().unwrap();
                    guard.take_ready()
                };
                transactions.extend(ready);
            }

            if transactions.is_empty() {
                break;
            }

            for tx in transactions {
                tx.apply(dh, state);
            }
        }
    }

    pub fn commit<C: CompositorHandler + 'static>(surface: &WlSurface, dh: &DisplayHandle, state: &mut C) {
        let is_sync = is_effectively_sync(surface);
        let children = PrivateSurfaceData::get_children(surface);
        let mut my_data = Self::lock_user_data(surface);
        // commit our state
        let current_txid = my_data.current_txid;
        my_data.public_data.cached_state.commit(Some(current_txid), dh);
        // take all our children state into our pending transaction
        for child in children {
            // if the child is effectively sync, take its state
            // this is the case if either we are effectively sync, or the child is explicitly sync
            let mut child_data = Self::lock_user_data(&child);
            let is_child_sync = child_data
                .public_data
                .data_map
                .get::<SubsurfaceState>()
                .map(|s| s.sync.load(Ordering::Acquire))
                .unwrap_or(false);

            // if we are not sync, but the child is we also have to commit the complete child surface tree
            if !is_sync && is_child_sync {
                std::mem::drop(child_data);
                Self::commit_sync_surface_tree(&child, &my_data.pending_transaction, dh);
            } else if is_sync || is_child_sync {
                let child_tx = std::mem::take(&mut child_data.pending_transaction);
                child_tx.merge_into(&my_data.pending_transaction);
                child_data.current_txid.0 = child_data.current_txid.0.wrapping_add(1);
            }
        }
        my_data
            .pending_transaction
            .insert_state(surface.clone(), current_txid);
        my_data.current_txid.0 = my_data.current_txid.0.wrapping_add(1);
        if !is_sync {
            let _client = match surface.client() {
                Some(client) => client,
                None => return,
            };
            // if we are not sync, add the transaction directly to our surface's timeline queue
            let tx = std::mem::take(&mut my_data.pending_transaction);
            let queue = my_data.timeline_queue.clone();
            // release the mutex, as applying the transaction will try to lock it
            std::mem::drop(my_data);

            let mut queue_guard = queue.lock().unwrap();
            queue_guard.append(tx.finalize());
            let transactions = queue_guard.take_ready();
            std::mem::drop(queue_guard);

            for tx in transactions {
                let had_barriers = !tx.completion_barriers.is_empty();
                tx.apply(dh, state);
                if had_barriers {
                    Self::drain_surface_tree(surface, dh, state);
                }
            }
        }
    }

    /// Checks if the first surface is an ancestor of the second
    pub fn is_ancestor(a: &WlSurface, b: &WlSurface) -> bool {
        let b_guard = Self::lock_user_data(b);
        if let Some(ref parent) = b_guard.parent {
            if parent.id() == a.id() {
                true
            } else {
                let parent = parent.clone();
                std::mem::drop(b_guard);
                Self::is_ancestor(a, &parent)
            }
        } else {
            false
        }
    }

    /// Sets the parent of a surface
    ///
    /// if this surface already has a role, does nothing and fails, otherwise
    /// its role is now to be a subsurface
    pub fn set_parent(child: &WlSurface, parent: &WlSurface) -> Result<(), AlreadyHasRole> {
        // debug_assert!(child.as_ref().is_alive());
        // debug_assert!(parent.as_ref().is_alive());

        // ensure the child is not the parent itself or its ancestor
        if child == parent || Self::is_ancestor(child, parent) {
            return Err(AlreadyHasRole);
        }

        let parent_queue = Self::timeline_queue(parent);

        // change child's parent
        {
            let mut child_guard = Self::lock_user_data(child);
            // if surface already has a role, it cannot become a subsurface
            if child_guard.public_data.role.is_some() && child_guard.public_data.role != Some(SUBSURFACE_ROLE)
            {
                return Err(AlreadyHasRole);
            }
            // ensure the child doesn't have a parent already set by a previous
            // wl_subcompositor.get_subsurface request
            if child_guard.parent.is_some() {
                return Err(AlreadyHasRole);
            }
            child_guard.public_data.role = Some(SUBSURFACE_ROLE);
            child_guard.parent = Some(parent.clone());
            child_guard.timeline_queue = parent_queue.clone();
        }
        Self::update_sync_subtree_queues(child, &parent_queue);
        // register child to new parent
        Self::lock_user_data(parent).children.push(child.clone());

        Ok(())
    }

    /// Remove a pre-existing parent of this child
    ///
    /// Does nothing if it has no parent. If the child was synchronized and sharing its parent's queue,
    /// an independent timeline queue is created, exclusive transactions are migrated, and the new queue
    /// is returned so it can be registered with the client's compositor state.
    pub fn unset_parent(child: &WlSurface) -> Option<Arc<Mutex<TransactionQueue>>> {
        let (old_parent, old_queue) = {
            let mut child_guard = Self::lock_user_data(child);
            let old_parent = child_guard.parent.take();
            let old_queue = child_guard.timeline_queue.clone();
            (old_parent, old_queue)
        };

        let old_parent = old_parent?;

        Self::lock_user_data(&old_parent)
            .children
            .retain(|c| c.id() != child.id());

        let parent_queue = Self::timeline_queue(&old_parent);
        if Arc::ptr_eq(&old_queue, &parent_queue) {
            let new_queue = Arc::new(Mutex::new(TransactionQueue::default()));
            Self::lock_user_data(child).timeline_queue = new_queue.clone();

            let mut subtree_surfaces = Vec::new();
            Self::collect_sync_subtree(child, &mut subtree_surfaces);

            {
                let mut parent_guard = parent_queue.lock().unwrap();
                let mut new_guard = new_queue.lock().unwrap();
                let mut remaining_parent_txs = VecDeque::new();

                while let Some(tx) = parent_guard.transactions.pop_front() {
                    if tx.contains_any_surface(&subtree_surfaces) {
                        if tx.only_contains_surfaces(&subtree_surfaces) {
                            new_guard.append(tx);
                        } else {
                            remaining_parent_txs.push_back(tx);
                        }
                    } else {
                        remaining_parent_txs.push_back(tx);
                    }
                }
                parent_guard.transactions = remaining_parent_txs;
            }

            Self::update_sync_subtree_queues(child, &new_queue);

            Some(new_queue)
        } else {
            None
        }
    }

    /// Retrieve the parent surface (if any) of this surface
    pub fn get_parent(child: &WlSurface) -> Option<WlSurface> {
        Self::lock_user_data(child).parent.clone()
    }

    /// Retrieve the children surface (if any) of this surface
    pub fn get_children(parent: &WlSurface) -> Vec<WlSurface> {
        Self::lock_user_data(parent)
            .children
            .iter()
            .filter(|s| s.id() != parent.id())
            .cloned()
            .collect()
    }

    /// Reorders a surface relative to one of its sibling
    ///
    /// Fails if `relative_to` is not a sibling or parent of `surface`.
    pub fn reorder(surface: &WlSurface, to: Location, relative_to: &WlSurface) -> Result<(), ()> {
        let parent = Self::get_parent(surface).ok_or(())?;

        fn index_of(surface: &WlSurface, slice: &[WlSurface]) -> Option<usize> {
            for (i, s) in slice.iter().enumerate() {
                if s.id() == surface.id() {
                    return Some(i);
                }
            }
            None
        }

        let mut parent_guard = Self::lock_user_data(&parent);
        let my_index = index_of(surface, &parent_guard.children).unwrap();
        let mut other_index = match index_of(relative_to, &parent_guard.children) {
            Some(idx) => idx,
            None => return Err(()),
        };
        let me = parent_guard.children.remove(my_index);
        if my_index < other_index {
            other_index -= 1;
        }
        let new_index = match to {
            Location::Before => other_index,
            Location::After => other_index + 1,
        };
        parent_guard.children.insert(new_index, me);

        Ok(())
    }
}

impl PrivateSurfaceData {
    /// Access sequentially the attributes associated with a surface tree,
    /// in a depth-first order.
    ///
    /// Note that an internal lock is taken during access of this data,
    /// so the tree cannot be manipulated at the same time.
    ///
    /// The first callback determines if this node children should be processed or not.
    ///
    /// The second actually does the processing, being called on children in display depth
    /// order.
    ///
    /// The third is called once all the children of a node has been processed (including itself), only if the first
    /// returned `DoChildren`, and gives an opportunity to early stop
    pub fn map_tree<F1, F2, F3, T>(
        surface: &WlSurface,
        initial: &T,
        mut filter: F1,
        mut processor: F2,
        mut post_filter: F3,
        reverse: bool,
    ) where
        F1: FnMut(&WlSurface, &SurfaceData, &T) -> TraversalAction<T>,
        F2: FnMut(&WlSurface, &SurfaceData, &T),
        F3: FnMut(&WlSurface, &SurfaceData, &T) -> bool,
    {
        Self::map(
            surface,
            initial,
            &mut filter,
            &mut processor,
            &mut post_filter,
            reverse,
        );
    }

    // helper function for map_tree
    fn map<F1, F2, F3, T>(
        surface: &WlSurface,
        initial: &T,
        filter: &mut F1,
        processor: &mut F2,
        post_filter: &mut F3,
        reverse: bool,
    ) -> bool
    where
        F1: FnMut(&WlSurface, &SurfaceData, &T) -> TraversalAction<T>,
        F2: FnMut(&WlSurface, &SurfaceData, &T),
        F3: FnMut(&WlSurface, &SurfaceData, &T) -> bool,
    {
        let data_guard = &mut *Self::lock_user_data(surface);
        // call the filter on ourselves
        match filter(surface, &data_guard.public_data, initial) {
            TraversalAction::DoChildren(t) => {
                // loop over children
                if reverse {
                    for c in data_guard.children.iter().rev() {
                        if c.id() == surface.id() {
                            processor(surface, &data_guard.public_data, initial);
                        } else if !Self::map(c, &t, filter, processor, post_filter, true) {
                            return false;
                        }
                    }
                } else {
                    for c in &data_guard.children {
                        if c.id() == surface.id() {
                            processor(surface, &data_guard.public_data, initial);
                        } else if !Self::map(c, &t, filter, processor, post_filter, false) {
                            return false;
                        }
                    }
                }
                post_filter(surface, &data_guard.public_data, initial)
            }
            TraversalAction::SkipChildren => {
                // still process ourselves
                processor(surface, &data_guard.public_data, initial);
                true
            }
            TraversalAction::Break => false,
        }
    }
}

/// The latest surface state suggest by wl_compositor `v6` events.
#[derive(Debug)]
pub struct SuggestedSurfaceState {
    /// Latest scale sent via `wl_surface::preferred_buffer_scale`.
    pub scale: i32,
    /// Latest transform sent via `wl_surface::preferred_buffer_transform`.
    pub transform: Transform,
}

impl Default for SuggestedSurfaceState {
    fn default() -> Self {
        Self {
            scale: 1,
            transform: Transform::Normal,
        }
    }
}
