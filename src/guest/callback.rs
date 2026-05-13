use std::ffi::c_void;
use std::panic::AssertUnwindSafe;

use wasmer::{FunctionEnvMut, Table, Value};

use crate::{NapiEnv, snapi::SnapiEnv};

use super::util::read_guest_bytes;

type RawFunctionEnvMut = FunctionEnvMut<'static, NapiEnv>;

#[repr(C)]
struct CallbackInvocationCtx {
    env: *mut RawFunctionEnvMut,
}

fn call_guest_callback(
    env: &mut FunctionEnvMut<NapiEnv>,
    table: &Table,
    guest_env: i32,
    wasm_fn_ptr: u32,
    callback_arg: u32,
) -> u32 {
    let Some(elem) = table.get(env, wasm_fn_ptr) else {
        return 0;
    };
    let func = match elem {
        Value::FuncRef(Some(func)) => func,
        Value::FuncRef(None) => return 0,
        _ => return 0,
    };
    match func.call(
        env,
        &[Value::I32(guest_env), Value::I32(callback_arg as i32)],
    ) {
        Ok(ret_vals) => match ret_vals.first() {
            Some(Value::I32(v)) => *v as u32,
            Some(Value::I64(v)) => *v as u32,
            _ => 0,
        },
        Err(err) => {
            eprintln!("[callback trampoline] error calling function: {err}");
            0
        }
    }
}

fn flush_host_buffer_copies(
    env: &mut FunctionEnvMut<NapiEnv>,
    snapi_env: SnapiEnv,
    frame_start: usize,
) {
    flush_host_buffer_copies_since(env, snapi_env, frame_start);
    env.data_mut().host_buffer_copy_frames.pop();
}

pub fn flush_pending_host_buffer_copies(env: &mut FunctionEnvMut<NapiEnv>, snapi_env: SnapiEnv) {
    if snapi_env.is_null() || env.data().host_buffer_copies.is_empty() {
        return;
    }

    let drained = {
        let state = env.data_mut();
        state
            .host_buffer_copy_frames
            .iter_mut()
            .for_each(|start| *start = 0);
        std::mem::take(&mut state.host_buffer_copies)
    };

    for mapping in drained {
        if mapping.byte_len > 0
            && let Some(bytes) = read_guest_bytes(env, mapping.guest_ptr as i32, mapping.byte_len)
        {
            unsafe {
                crate::snapi::snapi_bridge_overwrite_value_bytes(
                    snapi_env,
                    mapping.handle_id,
                    bytes.as_ptr().cast(),
                    mapping.byte_len as u32,
                );
            }
        }

        let state = env.data_mut();
        state.guest_data_ptrs.remove(&mapping.handle_id);
        if mapping.backing_store_token != 0 {
            state
                .guest_data_backing_stores
                .remove(&mapping.backing_store_token);
        }
    }
}

pub fn flush_host_buffer_copies_since(
    env: &mut FunctionEnvMut<NapiEnv>,
    snapi_env: SnapiEnv,
    frame_start: usize,
) {
    let start = frame_start.min(env.data().host_buffer_copies.len());
    let drained = {
        let state = env.data_mut();
        state.host_buffer_copies.split_off(start)
    };

    for mapping in drained {
        if mapping.byte_len > 0
            && let Some(bytes) = read_guest_bytes(env, mapping.guest_ptr as i32, mapping.byte_len)
        {
            unsafe {
                crate::snapi::snapi_bridge_overwrite_value_bytes(
                    snapi_env,
                    mapping.handle_id,
                    bytes.as_ptr().cast(),
                    mapping.byte_len as u32,
                );
            }
        }

        let state = env.data_mut();
        state.guest_data_ptrs.remove(&mapping.handle_id);
        if mapping.backing_store_token != 0 {
            state
                .guest_data_backing_stores
                .remove(&mapping.backing_store_token);
        }
    }
}

pub fn with_callback_state<R>(
    env: &mut FunctionEnvMut<NapiEnv>,
    snapi_env: SnapiEnv,
    f: impl FnOnce() -> R,
) -> R {
    if snapi_env.is_null() {
        return f();
    }

    let mut ctx = CallbackInvocationCtx {
        env: (env as *mut FunctionEnvMut<'_, NapiEnv>).cast::<RawFunctionEnvMut>(),
    };
    let frame_start = env.data().host_buffer_copies.len();
    let method_frame_depth = env.data().host_buffer_method_frames.len();
    env.data_mut().host_buffer_copy_frames.push(frame_start);
    let prev = unsafe {
        crate::snapi::snapi_bridge_swap_active_callback_ctx(
            snapi_env,
            (&mut ctx as *mut CallbackInvocationCtx).cast::<c_void>(),
        )
    };
    struct CallbackStateGuard {
        snapi_env: SnapiEnv,
        prev: *mut c_void,
        env: *mut RawFunctionEnvMut,
        frame_start: usize,
        method_frame_depth: usize,
    }
    impl Drop for CallbackStateGuard {
        fn drop(&mut self) {
            if !self.env.is_null() {
                let env = unsafe { &mut *self.env.cast::<FunctionEnvMut<'_, NapiEnv>>() };
                flush_host_buffer_copies(env, self.snapi_env, self.frame_start);
                env.data_mut()
                    .host_buffer_method_frames
                    .truncate(self.method_frame_depth);
                if self.frame_start > 0 {
                    flush_pending_host_buffer_copies(env, self.snapi_env);
                }
            }
            unsafe {
                crate::snapi::snapi_bridge_swap_active_callback_ctx(self.snapi_env, self.prev);
            }
        }
    }
    let _guard = CallbackStateGuard {
        snapi_env,
        prev,
        env: ctx.env,
        frame_start,
        method_frame_depth,
    };
    f()
}

/// Rust trampoline called from C++ when a V8 callback fires.
/// Re-enters the active guest callback scope and dispatches into the WASM guest.
///
/// # Panic safety (firebox#352 cascade 3)
///
/// The function is `extern "C"` so it is implicitly `nounwind` under the
/// Rust ABI — any panic propagating out of this frame triggers an immediate
/// `process::abort()` (SIGABRT), which terminates the host process before
/// stdio drains.  The body therefore runs inside `catch_unwind`; a caught
/// panic is logged and the trampoline returns `0` to V8 (the convention
/// `call_guest_callback` already uses for "could not call").  This keeps a
/// trapping wasm callback (including `WasiError::Exit(127)` propagating
/// through V8's callback chain during `npm install` sub-process spawns)
/// from SIGABRT-ing the parent harness; the wasm trap surfaces as a clean
/// 0 return and the calling V8 frame observes a no-op callback result.
///
/// Note: a returning trap from `func.call(...)` is delivered as
/// `Result::Err` and handled in `call_guest_callback`'s error arm without
/// unwinding.  `catch_unwind` here is the safety net for any *other* panic
/// path (memory-view access, table.clone(), wasmer-internal asserts) that
/// could fire under reentrant V8→wasm callback chains where the wasm-side
/// stack frame holding `FunctionEnvMut` may have been unwound by an
/// earlier trap.  See [`call_guest_callback`] for the trap-handled path.
#[unsafe(no_mangle)]
pub extern "C" fn snapi_host_invoke_wasm_callback(
    callback_ctx: *mut c_void,
    guest_env: u32,
    wasm_fn_ptr: u32,
    callback_arg: u32,
) -> u32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if callback_ctx.is_null() {
            eprintln!("[callback trampoline] no active callback scope available");
            return 0;
        }
        let ctx = unsafe { &mut *(callback_ctx as *mut CallbackInvocationCtx) };
        if ctx.env.is_null() {
            eprintln!("[callback trampoline] callback scope env cleared");
            return 0;
        }
        let env = unsafe { &mut *ctx.env.cast::<FunctionEnvMut<'_, NapiEnv>>() };
        let Some(table) = env.data().table.clone() else {
            return 0;
        };
        call_guest_callback(env, &table, guest_env as i32, wasm_fn_ptr, callback_arg)
    }));

    match result {
        Ok(ret) => ret,
        Err(payload) => {
            let msg = panic_message(&payload);
            // Best-effort stderr — if stderr is itself a wasm pipe, this
            // may not surface, but we have already prevented the abort.
            eprintln!(
                "[callback trampoline] panic in wasm callback dispatch \
                 (returning 0 to V8 to preserve harness): {msg}"
            );
            0
        }
    }
}

/// Extract a printable message from a `catch_unwind` payload.
///
/// Rust panic payloads are either `&'static str`, `String`, or some other
/// `Any`.  We surface the first two and fall back to a generic marker.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the panic-safe `extern "C"` callback trampoline.
    //!
    //! We can't easily fabricate a real `CallbackInvocationCtx` (it carries
    //! a live `FunctionEnvMut<NapiEnv>` which requires a Store + Module +
    //! Instance), so these tests exercise:
    //!
    //! 1. The null-ctx early-return path (no FFI panic, returns 0).
    //! 2. The `panic_message` helper for the three payload shapes
    //!    `catch_unwind` may surface.
    //!
    //! The "panic-caught" property of `catch_unwind` itself is a `std`
    //! invariant we rely on; testing it here would just test stdlib.
    //! What this test guards against is the trampoline ever growing a
    //! panic site OUTSIDE the `catch_unwind` block.  The function body is
    //! short and the closure boundary is visible — keep it that way.
    use super::*;

    /// A null `callback_ctx` must short-circuit cleanly to `0` without
    /// even attempting to dereference and without panicking.  The actual
    /// extern fn is called via its plain Rust signature; the `extern "C"`
    /// ABI is exercised at link time when V8 calls it from C++.
    #[test]
    fn null_callback_ctx_returns_zero() {
        let result = snapi_host_invoke_wasm_callback(std::ptr::null_mut(), 0, 0, 0);
        assert_eq!(result, 0, "null ctx must short-circuit to 0");
    }

    /// A ctx with a null `env` pointer must short-circuit cleanly to `0`
    /// (the previous-context-cleared case — happens when a guard's `Drop`
    /// has set the env back to null but C++ still calls in).
    #[test]
    fn null_env_pointer_returns_zero() {
        let mut ctx = CallbackInvocationCtx {
            env: std::ptr::null_mut(),
        };
        let ctx_ptr = (&mut ctx as *mut CallbackInvocationCtx).cast::<c_void>();
        let result = snapi_host_invoke_wasm_callback(ctx_ptr, 0, 0, 0);
        assert_eq!(result, 0, "ctx.env = null must short-circuit to 0");
    }

    /// `panic_message` recovers `&'static str` payloads.
    #[test]
    fn panic_message_static_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("static-message");
        let msg = panic_message(&payload);
        assert_eq!(msg, "static-message");
    }

    /// `panic_message` recovers `String` payloads (the common case for
    /// `panic!("formatted: {}", x)`).
    #[test]
    fn panic_message_owned_string() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("owned-message"));
        let msg = panic_message(&payload);
        assert_eq!(msg, "owned-message");
    }

    /// `panic_message` falls back to a marker for non-string payloads.
    #[test]
    fn panic_message_non_string_fallback() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42i32);
        let msg = panic_message(&payload);
        assert_eq!(msg, "<non-string panic payload>");
    }
}
