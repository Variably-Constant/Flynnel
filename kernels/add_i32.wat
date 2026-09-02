;; WebAssembly text (WAT) reference kernel for the Flynnel WASM backend.
;;
;; Registered with `WasmBackend::register_kernel` at
;; `src/backend/wasm.rs`, which JITs the `.wasm` binary via
;; wasmtime + cranelift. Exports a single function `add(i32, i32)
;; -> i32` that returns the integer sum.
;;
;; Verified against the WASM backend via
;; `examples/wasm_dispatch_demo.rs`:
;;   cargo run --release --example wasm_dispatch_demo \
;;             --features wasm-reference
;;
;; This WAT file is the human-readable source; the equivalent
;; binary bytes are inlined in `examples/wasm_dispatch_demo.rs` as
;; the `ADD_WASM` constant. To rebuild the binary bytes from this
;; source install `wabt` (WebAssembly Binary Toolkit) and run:
;;   wat2wasm kernels/add_i32.wat -o kernels/add_i32.wasm
;;
;; Kept intentionally minimal so the demo focuses on the routing
;; surface rather than the WASM semantics.
(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add))
