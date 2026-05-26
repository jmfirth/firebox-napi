#include <stdio.h>
#include <string.h>

#include "napi_test_helpers.h"

int main(void) {
  napi_env env = napi_wasm_init_env();
  CHECK_OR_FAIL(env != NULL, "napi_wasm_init_env returned NULL");

  // ---- Test napi_create_string_utf8 with explicit length ----
  {
    napi_value str;
    NAPI_CALL(env,
              napi_create_string_utf8(env, "Hello, World!", 5, &str));
    char buf[256];
    size_t len;
    NAPI_CALL(env,
              napi_get_value_string_utf8(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 5, "explicit length: expected len 5");
    CHECK_OR_FAIL(strcmp(buf, "Hello") == 0,
                  "explicit length: expected 'Hello'");
  }

  // ---- Test napi_create_string_utf8 with NAPI_AUTO_LENGTH ----
  {
    napi_value str;
    NAPI_CALL(env,
              napi_create_string_utf8(env, "auto length test",
                                      NAPI_AUTO_LENGTH, &str));
    char buf[256];
    size_t len;
    NAPI_CALL(env,
              napi_get_value_string_utf8(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 16, "auto length: expected len 16");
    CHECK_OR_FAIL(strcmp(buf, "auto length test") == 0,
                  "auto length: string mismatch");
  }

  // ---- Test napi_create_string_utf8 with zero length (empty string) ----
  {
    napi_value str;
    NAPI_CALL(env, napi_create_string_utf8(env, "", NAPI_AUTO_LENGTH, &str));
    char buf[256];
    size_t len;
    NAPI_CALL(env,
              napi_get_value_string_utf8(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 0, "empty string: expected len 0");
    CHECK_OR_FAIL(strcmp(buf, "") == 0, "empty string: expected ''");
  }

  // ---- Test napi_get_value_string_utf8 with NULL buffer (length query) ----
  {
    napi_value str;
    NAPI_CALL(env,
              napi_create_string_utf8(env, "measure me", NAPI_AUTO_LENGTH,
                                      &str));
    size_t len;
    NAPI_CALL(env,
              napi_get_value_string_utf8(env, str, NULL, 0, &len));
    CHECK_OR_FAIL(len == 10, "length query: expected len 10");
  }

  // ---- Test buffer-too-small truncation behavior ----
  {
    napi_value str;
    NAPI_CALL(env,
              napi_create_string_utf8(env, "truncate me", NAPI_AUTO_LENGTH,
                                      &str));
    // Buffer of 6 bytes: room for 5 chars + null terminator
    char buf[6];
    size_t len;
    NAPI_CALL(env,
              napi_get_value_string_utf8(env, str, buf, sizeof(buf), &len));
    // Should write 5 chars + null terminator, len reports chars copied
    CHECK_OR_FAIL(len == 5, "truncation: expected copied len 5");
    CHECK_OR_FAIL(strcmp(buf, "trunc") == 0,
                  "truncation: expected 'trunc'");
  }

  // ---- Test napi_create_string_latin1 and napi_get_value_string_latin1 ----
  {
    napi_value str;
    NAPI_CALL(env,
              napi_create_string_latin1(env, "Latin1 test", NAPI_AUTO_LENGTH,
                                        &str));
    char buf[256];
    size_t len;
    NAPI_CALL(env,
              napi_get_value_string_latin1(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 11, "latin1: expected len 11");
    CHECK_OR_FAIL(strcmp(buf, "Latin1 test") == 0,
                  "latin1: string mismatch");
  }

  // ---- Test latin1 with explicit length ----
  {
    napi_value str;
    NAPI_CALL(env, napi_create_string_latin1(env, "abcdef", 3, &str));
    char buf[256];
    size_t len;
    NAPI_CALL(env,
              napi_get_value_string_latin1(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 3, "latin1 explicit: expected len 3");
    CHECK_OR_FAIL(strcmp(buf, "abc") == 0,
                  "latin1 explicit: expected 'abc'");
  }

  // ---- Test typeof string value ----
  {
    napi_value str;
    NAPI_CALL(env,
              napi_create_string_utf8(env, "typeof test", NAPI_AUTO_LENGTH,
                                      &str));
    napi_valuetype vtype;
    NAPI_CALL(env, napi_typeof(env, str, &vtype));
    CHECK_OR_FAIL(vtype == napi_string, "typeof: expected napi_string");
  }

  // ---- firebox#614 regression: embedded NUL in utf8 explicit-length ----
  //
  // Pre-fix the guest-side `guest_napi_create_string_utf8` shim wrapped the
  // input bytes in `CString::new(...).unwrap_or_default()`, which errored
  // for any input containing an embedded NUL byte and substituted an empty
  // CString. The bridge then called V8's `String::NewFromUtf8(empty_ptr,
  // length=5)`, which read past the empty CString and produced an all-NUL
  // string of length 5. This broke every Buffer.toString('utf8') / tar
  // header parse path and surfaced as the cascade-9 §G6 wedge that #610
  // worked around at the Edge.js layer.
  //
  // After the fix the raw bytes (including embedded NULs) round-trip
  // through napi_create_string_utf8 → V8 → napi_get_value_string_utf8
  // byte-identically.
  {
    const char input[5] = {0x41, 0x42, 0x00, 0x43, 0x44}; // "AB\0CD"
    napi_value str;
    NAPI_CALL(env, napi_create_string_utf8(env, input, 5, &str));
    char buf[16] = {0};
    size_t len = 0;
    NAPI_CALL(env,
              napi_get_value_string_utf8(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 5, "embedded-NUL utf8: expected len 5");
    CHECK_OR_FAIL(buf[0] == 0x41, "embedded-NUL utf8: buf[0] expected 'A'");
    CHECK_OR_FAIL(buf[1] == 0x42, "embedded-NUL utf8: buf[1] expected 'B'");
    CHECK_OR_FAIL(buf[2] == 0x00, "embedded-NUL utf8: buf[2] expected NUL");
    CHECK_OR_FAIL(buf[3] == 0x43, "embedded-NUL utf8: buf[3] expected 'C'");
    CHECK_OR_FAIL(buf[4] == 0x44, "embedded-NUL utf8: buf[4] expected 'D'");
  }

  // ---- firebox#614 regression: embedded NUL in latin1 explicit-length ----
  //
  // Same class as the utf8 case above — the latin1 guest shim also wrapped
  // the input in `CString::new(...).unwrap_or_default()` and exhibited the
  // same all-NUL substitution for embedded-NUL inputs.
  {
    const char input[5] = {0x41, 0x42, 0x00, 0x43, 0x44}; // "AB\0CD"
    napi_value str;
    NAPI_CALL(env, napi_create_string_latin1(env, input, 5, &str));
    char buf[16] = {0};
    size_t len = 0;
    NAPI_CALL(env,
              napi_get_value_string_latin1(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 5, "embedded-NUL latin1: expected len 5");
    CHECK_OR_FAIL(buf[0] == 0x41, "embedded-NUL latin1: buf[0] expected 'A'");
    CHECK_OR_FAIL(buf[1] == 0x42, "embedded-NUL latin1: buf[1] expected 'B'");
    CHECK_OR_FAIL(buf[2] == 0x00, "embedded-NUL latin1: buf[2] expected NUL");
    CHECK_OR_FAIL(buf[3] == 0x43, "embedded-NUL latin1: buf[3] expected 'C'");
    CHECK_OR_FAIL(buf[4] == 0x44, "embedded-NUL latin1: buf[4] expected 'D'");
  }

  // ---- firebox#614 regression: tar-header-shape utf8 (16-byte field with
  //      trailing NUL padding, the literal #610 §G6 shape) ----
  {
    // Mimics a tar header "name" field: ASCII prefix + NUL padding.
    char input[16];
    memset(input, 0, sizeof(input));
    memcpy(input, "package/", 8);
    napi_value str;
    NAPI_CALL(env, napi_create_string_utf8(env, input, 16, &str));
    char buf[32] = {0};
    size_t len = 0;
    NAPI_CALL(env,
              napi_get_value_string_utf8(env, str, buf, sizeof(buf), &len));
    CHECK_OR_FAIL(len == 16, "tar-shape utf8: expected len 16");
    CHECK_OR_FAIL(memcmp(buf, "package/", 8) == 0,
                  "tar-shape utf8: expected 'package/' prefix");
    CHECK_OR_FAIL(buf[8] == 0x00 && buf[15] == 0x00,
                  "tar-shape utf8: expected NUL padding");
  }

  return PrintSuccess("TEST_STRING");
}
