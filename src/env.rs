use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use wasmer::{Memory, Table, TypedFunction};

use crate::snapi::{SnapiEnv, snapi_bridge_unofficial_release_env};

/// Process-global monotonic counters for guest-visible `napi_env` and
/// N-API scope handles (firebox#684).
///
/// One instance is owned by the process-global `NapiCtx` and cloned (by
/// `Arc`) into every per-thread [`NapiEnv`].  Because WASIX guest threads
/// share linear memory, the small-integer handles the bridge writes back
/// must be unique across threads; minting them from a single shared
/// counter guarantees that.  See the `handle_ids` doc on `NapiCtxInner`
/// for the full collision story.
#[derive(Debug, Default)]
pub(crate) struct NapiHandleIds {
    next_env_id: AtomicU32,
    next_scope_id: AtomicU32,
}

impl NapiHandleIds {
    /// Returns the next never-before-used `napi_env` handle (>= 1).
    fn next_env_id(&self) -> u32 {
        // `fetch_add` returns the previous value; bias the sequence so the
        // first handle is 1 (0 is reserved as the "no env" sentinel that
        // `resolve_napi_env` treats as null).
        self.next_env_id
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
    }

    /// Returns the next never-before-used N-API scope handle (>= 1).
    fn next_scope_id(&self) -> u32 {
        self.next_scope_id
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
    }
}

pub(crate) struct HostBufferCopy {
    pub(crate) handle_id: u32,
    pub(crate) backing_store_token: u64,
    pub(crate) guest_ptr: u32,
    pub(crate) byte_len: usize,
}

pub(crate) struct GuestBackingStoreMapping {
    pub(crate) host_addr: u64,
    pub(crate) guest_ptr: u32,
    pub(crate) byte_len: usize,
}

#[derive(Default)]
pub(crate) struct NapiEnv {
    pub(crate) memory: Option<Memory>,
    pub(crate) malloc_fn: Option<TypedFunction<i32, i32>>,
    pub(crate) table: Option<Table>,
    /// Maps value handle IDs to their guest-memory data pointers.
    /// Used for buffers/arraybuffers backed by guest linear memory.
    pub(crate) guest_data_ptrs: HashMap<u32, u32>,
    /// Maps stable host backing-store tokens to guest-memory data pointers.
    /// This keeps external Buffer/ArrayBuffer aliases stable even when V8/N-API
    /// surfaces the same backing store through a different value handle.
    pub(crate) guest_data_backing_stores: HashMap<u64, GuestBackingStoreMapping>,
    /// Host-owned buffer/arraybuffer mappings copied into guest memory for the
    /// duration of an active callback. These are written back on callback exit.
    pub(crate) host_buffer_copies: Vec<HostBufferCopy>,
    pub(crate) host_buffer_copy_frames: Vec<usize>,
    /// Host-owned buffer copies created while servicing a single guest-side
    /// native binding invocation (typically bracketed by napi_get_cb_info and a
    /// return-value creation call).
    pub(crate) host_buffer_method_frames: Vec<usize>,
    pub(crate) default_napi_env_id: Option<u32>,
    /// Process-global handle-id source, shared (by `Arc`) across every
    /// per-thread `NapiEnv` so guest-visible `napi_env` / scope handles
    /// never collide across WASIX threads that share linear memory
    /// (firebox#684).
    pub(crate) handle_ids: Arc<NapiHandleIds>,
    pub(crate) napi_envs: HashMap<u32, usize>,
    pub(crate) napi_state_to_guest_env: HashMap<usize, u32>,
    pub(crate) napi_scopes: HashMap<u32, u32>,
}

impl NapiEnv {
    pub(crate) fn register_napi_env(&mut self, env: SnapiEnv) -> (u32, u32) {
        // firebox#684: mint handles from the process-global counter so the
        // guest-visible ids are unique across every WASIX thread (each of
        // which has its own NapiEnv but shares guest linear memory).
        let env_id = self.handle_ids.next_env_id();
        let scope_id = self.handle_ids.next_scope_id();

        self.napi_envs.insert(env_id, env as usize);
        self.napi_state_to_guest_env.insert(env as usize, env_id);
        self.napi_scopes.insert(scope_id, env_id);
        (env_id, scope_id)
    }

    pub(crate) fn unregister_napi_scope(&mut self, scope_id: u32) -> Option<SnapiEnv> {
        let env_id = self.napi_scopes.remove(&scope_id)?;
        if self.default_napi_env_id == Some(env_id) {
            self.default_napi_env_id = None;
        }
        let env = self.napi_envs.remove(&env_id)?;
        self.napi_state_to_guest_env.remove(&env);
        Some(env as SnapiEnv)
    }

    pub(crate) fn resolve_napi_env(&self, guest_env: i32) -> SnapiEnv {
        let env_id = if guest_env > 0 {
            guest_env as u32
        } else {
            return std::ptr::null_mut();
        };
        self.napi_envs
            .get(&env_id)
            .map(|env| *env as SnapiEnv)
            .unwrap_or(std::ptr::null_mut())
    }
}

impl Drop for NapiEnv {
    fn drop(&mut self) {
        let scope_ids: Vec<u32> = self.napi_scopes.keys().copied().collect();
        for scope_id in scope_ids {
            if let Some(env) = self.unregister_napi_scope(scope_id) {
                unsafe {
                    let _ = snapi_bridge_unofficial_release_env(env);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `NapiEnv` whose handle ids come from `ids`, mimicking the
    /// per-thread `NapiEnv` that `NapiSession::create_imports` seeds from
    /// the process-global `NapiCtx` counter. (`NapiEnv` implements `Drop`,
    /// so it cannot be built with struct-update syntax — default first,
    /// then assign the shared counter, exactly as `create_imports` does.)
    fn env_sharing(ids: &Arc<NapiHandleIds>) -> NapiEnv {
        let mut env = NapiEnv::default();
        env.handle_ids = Arc::clone(ids);
        env
    }

    /// Registers `env` and immediately unregisters its scope so the
    /// `NapiEnv::drop` FFI release path is never reached in tests.
    fn register_then_unregister(env: &mut NapiEnv, dummy: SnapiEnv) -> (u32, u32) {
        let (env_id, scope_id) = env.register_napi_env(dummy);
        let _ = env.unregister_napi_scope(scope_id);
        (env_id, scope_id)
    }

    #[test]
    fn handle_ids_start_at_one() {
        let ids = NapiHandleIds::default();
        assert_eq!(ids.next_env_id(), 1, "first env handle must be 1");
        assert_eq!(ids.next_scope_id(), 1, "first scope handle must be 1");
    }

    #[test]
    fn handle_ids_are_monotonic() {
        let ids = NapiHandleIds::default();
        assert_eq!(ids.next_env_id(), 1);
        assert_eq!(ids.next_env_id(), 2);
        assert_eq!(ids.next_env_id(), 3);
        // env and scope counters advance independently.
        assert_eq!(ids.next_scope_id(), 1);
        assert_eq!(ids.next_scope_id(), 2);
    }

    /// firebox#684 regression guard: two distinct `NapiEnv`s (as a parent
    /// and a worker thread would have) that share one `NapiHandleIds`
    /// must never mint the same guest-visible `napi_env` handle. Before
    /// the fix, each per-thread `NapiEnv` minted `1`, so the parent and a
    /// `worker_threads` worker collided on env handle `1`.
    #[test]
    fn shared_counter_yields_unique_env_handles_across_envs() {
        let ids = Arc::new(NapiHandleIds::default());
        let dummy = 0x1usize as SnapiEnv;

        let mut parent = env_sharing(&ids);
        let mut worker = env_sharing(&ids);

        let (parent_env_id, _) = register_then_unregister(&mut parent, dummy);
        let (worker_env_id, _) = register_then_unregister(&mut worker, dummy);

        assert_eq!(parent_env_id, 1, "parent env handle");
        assert_eq!(worker_env_id, 2, "worker env handle");
        assert_ne!(
            parent_env_id, worker_env_id,
            "parent and worker must not collide on the same napi_env handle"
        );
    }

    /// A single `NapiEnv` still resolves the handle it just registered —
    /// the per-thread `napi_envs` map keeps the local id→SnapiEnv binding
    /// even though the id space is now process-global.
    #[test]
    fn registered_env_resolves_locally() {
        let ids = Arc::new(NapiHandleIds::default());
        let dummy = 0x42usize as SnapiEnv;

        let mut env = env_sharing(&ids);
        let (env_id, scope_id) = env.register_napi_env(dummy);

        assert_eq!(
            env.resolve_napi_env(env_id as i32),
            dummy,
            "the registering thread must resolve its own env handle"
        );
        // A foreign handle the local map never registered resolves to null.
        assert!(
            env.resolve_napi_env((env_id + 1) as i32).is_null(),
            "an unregistered handle resolves to null on this thread"
        );

        // Drain so Drop's FFI release path is not exercised in the test.
        let _ = env.unregister_napi_scope(scope_id);
    }

    #[test]
    fn non_positive_handles_resolve_to_null() {
        let env = NapiEnv::default();
        assert!(env.resolve_napi_env(0).is_null());
        assert!(env.resolve_napi_env(-1).is_null());
    }
}
