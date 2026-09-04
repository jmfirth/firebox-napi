use anyhow::{Context, Result, bail};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use wasmer::{
    ExternType, Function, FunctionEnv, Imports, Instance, Module, StoreMut, Table, Value,
};

use crate::{
    NAPI_EXTENSION_WASMER_MODULE_NAME, NAPI_EXTENSION_WASMER_MODULE_PREFIX, NAPI_MODULE_NAME,
    NapiEnv, NapiVersion, NapiWasmerExtensionVersion,
    guest::napi::{is_known_napi_import, register_env_imports, register_napi_imports},
};

#[derive(Debug, Clone, Default)]
pub struct NapiLimits {
    pub max_sessions: Option<usize>,
    pub max_envs: Option<usize>,
    pub max_total_external_memory: Option<u64>,
    pub max_total_heap_bytes: Option<u64>,
}

#[derive(Debug, Default)]
pub struct NapiCtxBuilder {
    limits: NapiLimits,
}

#[derive(Clone, Debug)]
pub struct NapiCtx {
    inner: Arc<NapiCtxInner>,
}

#[derive(Clone)]
pub struct NapiSession {
    inner: Arc<NapiSessionInner>,
}

#[derive(Clone, Debug)]
pub struct NapiRuntimeHooks {
    ctx: NapiCtx,
    sessions: Arc<Mutex<HashMap<usize, VecDeque<NapiSession>>>>,
}

#[derive(Debug)]
struct NapiCtxInner {
    limits: NapiLimits,
    active_sessions: AtomicUsize,
    /// Process-global monotonic counters for guest-visible `napi_env` /
    /// N-API scope handles (firebox#684).
    ///
    /// Each WASIX guest thread instantiates the Edge.js module with its
    /// own per-instance [`NapiSession`] and therefore its own
    /// per-thread [`NapiEnv`] host state.  WASIX threads share the same
    /// guest *linear memory*, so the small-integer `napi_env` / scope
    /// handles that the bridge writes back into guest memory are visible
    /// across threads.  Edge.js keys per-environment state (its
    /// `g_environments` map, the worker registry, the platform-task
    /// `owning_thread`) on that integer.  If two threads each minted
    /// handle `1` from independent per-`NapiEnv` counters, the parent
    /// env and a `worker_threads` worker env collided on the same key —
    /// the worker thread then drove the *parent's* platform-task state,
    /// tripping `AssertOwningThread` and hard-aborting the runtime.
    ///
    /// Hoisting the counters to the process-global `NapiCtx` (one per
    /// firebox process; see `napi_v8::ctx()`) makes every handle unique
    /// across all threads that share guest memory, so the guest-side
    /// per-env keying stays unambiguous.
    handle_ids: Arc<crate::env::NapiHandleIds>,
}

struct NapiSessionInner {
    ctx: Arc<NapiCtxInner>,
    imported_memory_type: Option<wasmer::MemoryType>,
    imported_table_type: Option<wasmer::TableType>,
    func_env: Mutex<Option<FunctionEnv<NapiEnv>>>,
}

impl std::fmt::Debug for NapiSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NapiSession").finish_non_exhaustive()
    }
}

impl Drop for NapiSessionInner {
    fn drop(&mut self) {
        self.ctx.active_sessions.fetch_sub(1, Ordering::AcqRel);
    }
}

impl NapiCtxBuilder {
    pub fn max_sessions(mut self, max_sessions: usize) -> Self {
        self.limits.max_sessions = Some(max_sessions);
        self
    }

    pub fn max_envs(mut self, max_envs: usize) -> Self {
        self.limits.max_envs = Some(max_envs);
        self
    }

    pub fn max_total_external_memory(mut self, bytes: u64) -> Self {
        self.limits.max_total_external_memory = Some(bytes);
        self
    }

    pub fn max_total_heap_bytes(mut self, bytes: u64) -> Self {
        self.limits.max_total_heap_bytes = Some(bytes);
        self
    }

    pub fn build(self) -> NapiCtx {
        NapiCtx {
            inner: Arc::new(NapiCtxInner {
                limits: self.limits,
                active_sessions: AtomicUsize::new(0),
                handle_ids: Arc::new(crate::env::NapiHandleIds::default()),
            }),
        }
    }
}

impl Default for NapiCtx {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl NapiCtx {
    pub fn builder() -> NapiCtxBuilder {
        NapiCtxBuilder::default()
    }

    pub fn limits(&self) -> &NapiLimits {
        &self.inner.limits
    }

    pub fn active_sessions(&self) -> usize {
        self.inner.active_sessions.load(Ordering::Acquire)
    }

    pub fn prepare_module(&self, module: &Module) -> Result<NapiSession> {
        self.new_session(module)
    }

    pub fn module_needs_napi(
        module: &Module,
    ) -> (Option<NapiVersion>, Option<NapiWasmerExtensionVersion>) {
        let mut napi_version = None;
        let mut napi_extension_version = None;

        for import in module.imports() {
            if import.module() == NAPI_MODULE_NAME {
                napi_version = Some(match napi_version {
                    Some(NapiVersion::Unknown) => NapiVersion::Unknown,
                    _ if is_known_napi_import(import.name()) => NapiVersion::V10,
                    _ => NapiVersion::Unknown,
                });
                continue;
            }

            let Some(detected_extension_version) =
                napi_wasmer_extension_version_from_namespace(import.module())
            else {
                continue;
            };

            napi_extension_version = Some(match napi_extension_version {
                None => detected_extension_version,
                Some(existing) if existing == detected_extension_version => existing,
                Some(_) => NapiWasmerExtensionVersion::Unknown,
            });
        }

        (napi_version, napi_extension_version)
    }

    pub fn runtime_hooks(&self) -> NapiRuntimeHooks {
        NapiRuntimeHooks {
            ctx: self.clone(),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn new_session(&self, module: &Module) -> Result<NapiSession> {
        let previous = self.inner.active_sessions.fetch_add(1, Ordering::AcqRel);
        if let Some(max_sessions) = self.inner.limits.max_sessions
            && previous >= max_sessions
        {
            self.inner.active_sessions.fetch_sub(1, Ordering::AcqRel);
            bail!("refusing to create more than {max_sessions} active N-API sessions");
        }

        let imported_memory_type = module.imports().find_map(|import| {
            if import.module() == "env"
                && import.name() == "memory"
                && let ExternType::Memory(ty) = import.ty()
            {
                return Some(*ty);
            }
            None
        });

        let imported_table_type = module.imports().find_map(|import| {
            if import.module() == "env"
                && import.name() == "__indirect_function_table"
                && let ExternType::Table(ty) = import.ty()
            {
                return Some(*ty);
            }
            None
        });

        Ok(NapiSession {
            inner: Arc::new(NapiSessionInner {
                ctx: Arc::clone(&self.inner),
                imported_memory_type,
                imported_table_type,
                func_env: Mutex::new(None),
            }),
        })
    }
}

impl NapiRuntimeHooks {
    fn module_key(module: &Module) -> usize {
        module as *const Module as usize
    }

    pub fn additional_imports(&self, module: &Module, store: &mut StoreMut<'_>) -> Result<Imports> {
        let (napi_version, napi_extension_version) = NapiCtx::module_needs_napi(module);
        if napi_version.is_none() && napi_extension_version.is_none() {
            return Ok(Imports::new());
        }

        if let Some(version) = napi_version
            && !NapiVersion::V10.is_compatible_with(version)
        {
            bail!("unsupported N-API import version: {version:?}");
        }

        if let Some(version) = napi_extension_version
            && !NapiWasmerExtensionVersion::V0.is_compatible_with(version)
        {
            bail!("unsupported Wasmer N-API extension version: {version:?}");
        }

        let session = self.ctx.prepare_module(module)?;
        let imports = session.create_imports(store)?;
        let mut sessions = self
            .sessions
            .lock()
            .expect("poisoned NapiRuntimeHooks session queue");
        sessions
            .entry(Self::module_key(module))
            .or_default()
            .push_back(session);
        Ok(imports)
    }

    pub fn configure_instance(
        &self,
        module: &Module,
        store: &mut StoreMut<'_>,
        instance: &Instance,
        imported_memory: Option<&wasmer::Memory>,
        imported_table: Option<&wasmer::Table>,
        imported_malloc: Option<&Function>,
    ) -> Result<()> {
        let (napi_version, napi_extension_version) = NapiCtx::module_needs_napi(module);
        if napi_version.is_none() && napi_extension_version.is_none() {
            return Ok(());
        }

        let session = {
            let mut sessions = self
                .sessions
                .lock()
                .expect("poisoned NapiRuntimeHooks session queue");
            let key = Self::module_key(module);
            let Some(queue) = sessions.get_mut(&key) else {
                bail!("missing pending N-API session for module instance setup");
            };
            let session = queue
                .pop_front()
                .context("missing queued N-API session for module instance setup")?;
            if queue.is_empty() {
                sessions.remove(&key);
            }
            session
        };

        session.configure_instance(
            store,
            instance,
            imported_memory,
            imported_table,
            imported_malloc,
        )
    }
}

impl NapiSession {
    pub fn create_imports(&self, store: &mut StoreMut<'_>) -> Result<Imports> {
        let mut import_object = Imports::new();
        register_env_imports(store, &mut import_object);

        // firebox#684: seed this per-thread NapiEnv with the process-global
        // handle-id counters so guest-visible napi_env / scope handles are
        // unique across every WASIX thread that shares guest memory.
        let mut napi_env = NapiEnv::default();
        napi_env.handle_ids = Arc::clone(&self.inner.ctx.handle_ids);
        let func_env = FunctionEnv::new(store, napi_env);
        {
            let mut guard = self
                .inner
                .func_env
                .lock()
                .expect("poisoned NapiSession mutex");
            *guard = Some(func_env.clone());
        }
        register_napi_imports(store, &func_env, &mut import_object);

        if let Some(memory_type) = self.inner.imported_memory_type {
            let memory = wasmer::Memory::new(&mut *store, memory_type)?;
            import_object.define("env", "memory", memory.clone());
            func_env.as_mut(&mut *store).memory = Some(memory);
        }

        if let Some(table_type) = self.inner.imported_table_type {
            let table = Table::new(&mut *store, table_type, Value::FuncRef(None))?;
            import_object.define("env", "__indirect_function_table", table.clone());
            func_env.as_mut(&mut *store).table = Some(table);
        }

        Ok(import_object)
    }

    pub fn configure_instance(
        &self,
        store: &mut StoreMut<'_>,
        instance: &Instance,
        imported_memory: Option<&wasmer::Memory>,
        imported_table: Option<&wasmer::Table>,
        imported_malloc: Option<&Function>,
    ) -> Result<()> {
        let func_env = {
            let guard = self
                .inner
                .func_env
                .lock()
                .expect("poisoned NapiSession mutex");
            guard
                .clone()
                .context("missing runtime function env during instance setup")?
        };

        if let Some(memory) = imported_memory {
            func_env.as_mut(&mut *store).memory = Some(memory.clone());
        }

        // firebox#MDA: bind the guest allocator, preferring the one the DL
        // linker resolved over the main instance's own exports.
        //
        // Three instantiation shapes reach this, and only the first two are
        // served by an export lookup:
        //   * NON-DL (static main) and FAT PIC main: libc is linked into the
        //     main module, so the instance *exports* `malloc` (and, for an
        //     Edge.js-style build, the `unofficial_napi_guest_malloc` wrapper).
        //     The export lookup below binds the real allocator.
        //   * DL THIN main (Route C): libc is a *side module* named in
        //     `NEEDED`. The main *imports* `env.malloc` and exports NEITHER
        //     name, so the export lookup finds nothing and `malloc_fn` stays
        //     `None`. Every host-side guest allocation
        //     (`napi_create_arraybuffer`, `napi_create_buffer`,
        //     `napi_create_buffer_copy`, `napi_get_node_version`) then takes
        //     its host-memory fallback and cannot hand the guest back an
        //     addressable `void** data` — Edge.js's `InstallShouldAbortToggle`
        //     reads `data == nullptr`, `internalBinding('util')` comes back
        //     `undefined`, and Node's bootstrap dies destructuring
        //     `privateSymbols`.
        //
        // The linker supplies `imported_malloc` by resolving the symbol across
        // the whole link graph — the same resolution the guest's own
        // `env.malloc` import went through — so the provider calls the very
        // allocator the guest calls. This is the third member of the
        // firebox#714 (`env.memory`) / firebox#717
        // (`env.__indirect_function_table`) family: a host-provider resource a
        // DL main imports rather than exports.
        let linker_malloc =
            imported_malloc.and_then(|malloc| malloc.typed::<i32, i32>(&store).ok());
        if let Some(malloc) = linker_malloc {
            func_env.as_mut(&mut *store).malloc_fn = Some(malloc);
        } else {
            for export_name in ["unofficial_napi_guest_malloc", "malloc"] {
                if let Ok(malloc) = instance
                    .exports
                    .get_typed_function::<i32, i32>(&store, export_name)
                {
                    func_env.as_mut(&mut *store).malloc_fn = Some(malloc);
                    break;
                }
            }
        }

        // firebox#717: re-point the provider's `NapiEnv.table` onto the
        // authoritative indirect function table — the symmetric partner of the
        // firebox#714 `imported_memory` re-point above, for the function-table
        // resource.
        //
        // Two instantiation shapes reach this:
        //   * NON-DL (static main): the instance *exports*
        //     `__indirect_function_table`, so the export lookup below succeeds
        //     and binds the real table.
        //   * DL (PIC dynamic-main): the instance *imports*
        //     `env.__indirect_function_table` from the WASIX linker — it does
        //     NOT export it. The export lookup fails, so without the linker
        //     supplying `imported_table` the provider would keep the empty
        //     placeholder `Table::new(.., FuncRef(None))` minted in
        //     `create_imports`. Every host→guest N-API callback dispatch
        //     (`call_guest_callback` -> `table.get(wasm_fn_ptr)`) would then
        //     read `FuncRef(None)` and silently return 0, i.e. JS `undefined`
        //     (the firebox#717 `internalBinding('builtins')` is undefined
        //     symptom). The linker's authoritative table arrives via
        //     `imported_table`; prefer it, then fall back to the export for the
        //     static path.
        if let Some(table) = imported_table {
            func_env.as_mut(&mut *store).table = Some(table.clone());
        } else if let Ok(table) = instance.exports.get_table("__indirect_function_table") {
            func_env.as_mut(&mut *store).table = Some(table.clone());
        }
        Ok(())
    }
}

fn napi_wasmer_extension_version_from_namespace(
    namespace: &str,
) -> Option<NapiWasmerExtensionVersion> {
    if namespace == NAPI_EXTENSION_WASMER_MODULE_NAME {
        return Some(NapiWasmerExtensionVersion::V0);
    }

    let suffix = namespace.strip_prefix(NAPI_EXTENSION_WASMER_MODULE_PREFIX)?;
    Some(match suffix {
        "0" => NapiWasmerExtensionVersion::V0,
        _ => NapiWasmerExtensionVersion::Unknown,
    })
}

#[cfg(test)]
mod tests {
    use super::NapiCtx;
    use crate::{NapiVersion, NapiWasmerExtensionVersion};
    use wasmer::{Module, Store};
    use wat::parse_str;

    const EMPTY_WASM_MODULE: &[u8] = b"\0asm\x01\0\0\0";

    #[test]
    fn max_sessions_limit_is_enforced() {
        let store = Store::default();
        let module = Module::new(&store, EMPTY_WASM_MODULE).expect("empty wasm module compiles");
        let ctx = NapiCtx::builder().max_sessions(1).build();

        let first = ctx
            .prepare_module(&module)
            .expect("first session should be created");
        assert_eq!(ctx.active_sessions(), 1);
        assert!(ctx.prepare_module(&module).is_err());

        drop(first);
        assert_eq!(ctx.active_sessions(), 0);

        let _second = ctx
            .prepare_module(&module)
            .expect("session slot should be released after drop");
        assert_eq!(ctx.active_sessions(), 1);
    }

    fn compile_wat(store: &Store, wat: &str) -> Module {
        let wasm = parse_str(wat).expect("wat module parses");
        Module::new(store, wasm).expect("wat module compiles")
    }

    #[test]
    fn module_needs_napi_detects_none() {
        let store = Store::default();
        let module = Module::new(&store, EMPTY_WASM_MODULE).expect("empty wasm module compiles");

        assert_eq!(NapiCtx::module_needs_napi(&module), (None, None));
    }

    #[test]
    fn module_needs_napi_detects_core_napi_v10() {
        let store = Store::default();
        let module = compile_wat(
            &store,
            r#"(module
                (import "napi" "napi_get_undefined" (func))
            )"#,
        );

        assert_eq!(
            NapiCtx::module_needs_napi(&module),
            (Some(NapiVersion::V10), None)
        );
    }

    #[test]
    fn module_needs_napi_detects_unknown_core_napi() {
        let store = Store::default();
        let module = compile_wat(
            &store,
            r#"(module
                (import "napi" "napi_future_function" (func))
            )"#,
        );

        assert_eq!(
            NapiCtx::module_needs_napi(&module),
            (Some(NapiVersion::Unknown), None)
        );
    }

    #[test]
    fn module_needs_napi_detects_extension_v0() {
        let store = Store::default();
        let module = compile_wat(
            &store,
            r#"(module
                (import "napi_extension_wasmer_v0" "unofficial_napi_get_hash_seed" (func))
            )"#,
        );

        assert_eq!(
            NapiCtx::module_needs_napi(&module),
            (None, Some(NapiWasmerExtensionVersion::V0))
        );
    }

    #[test]
    fn module_needs_napi_detects_unknown_extension_version() {
        let store = Store::default();
        let module = compile_wat(
            &store,
            r#"(module
                (import "napi_extension_wasmer_v1" "unofficial_napi_get_hash_seed" (func))
            )"#,
        );

        assert_eq!(
            NapiCtx::module_needs_napi(&module),
            (None, Some(NapiWasmerExtensionVersion::Unknown))
        );
    }

    #[test]
    fn module_needs_napi_detects_mixed_namespaces() {
        let store = Store::default();
        let module = compile_wat(
            &store,
            r#"(module
                (import "napi" "napi_get_undefined" (func))
                (import "napi_extension_wasmer_v0" "unofficial_napi_get_hash_seed" (func))
            )"#,
        );

        assert_eq!(
            NapiCtx::module_needs_napi(&module),
            (Some(NapiVersion::V10), Some(NapiWasmerExtensionVersion::V0))
        );
    }

    #[test]
    fn napi_version_compatibility_is_additive() {
        assert!(NapiVersion::V10.is_compatible_with(NapiVersion::V10));
        assert!(!NapiVersion::V10.is_compatible_with(NapiVersion::Unknown));
        assert!(NapiVersion::Unknown.is_compatible_with(NapiVersion::V10));
        assert!(!NapiVersion::Unknown.is_compatible_with(NapiVersion::Unknown));
    }

    #[test]
    fn napi_wasmer_extension_version_compatibility_is_strict() {
        assert!(NapiWasmerExtensionVersion::V0.is_compatible_with(NapiWasmerExtensionVersion::V0));
        assert!(
            !NapiWasmerExtensionVersion::V0.is_compatible_with(NapiWasmerExtensionVersion::Unknown)
        );
        assert!(
            !NapiWasmerExtensionVersion::Unknown.is_compatible_with(NapiWasmerExtensionVersion::V0)
        );
        assert!(
            !NapiWasmerExtensionVersion::Unknown
                .is_compatible_with(NapiWasmerExtensionVersion::Unknown)
        );
    }
}
