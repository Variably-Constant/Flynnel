#!/usr/bin/env python3
"""
Flynnel TPU JAX bridge.

Spawned as a child process by `flynnel::backend::tpu_jax::TpuJaxBackend`.
Reads line-oriented JSON requests from stdin, writes line-oriented JSON
responses to stdout. One request -> one response, in order.

Wire protocol (every message is a single JSON object on one line):

  Request                                                | Response
  -------------------------------------------------------+-----------------------------------
  {"op":"ping"}                                          | {"ok":true,"devices":[...], "jax_version":"..."}
  {"op":"register","name":"foo","source":"def foo(c,*a):..."} | {"ok":true,"handle":<u64>}
  {"op":"dispatch","handle":<u64>,"count":<u32>,"args":[...]} | {"ok":true}
  {"op":"shutdown"}                                      | {"ok":true,"goodbye":true}
  (any unknown op)                                       | {"ok":false,"error":"..."}

KernelArg encoding (the "args" array elements):

  {"i32":<int>}  {"i64":<int>}  {"u32":<int>}  {"u64":<int>}
  {"f32":<float>}  {"f64":<float>}  {"device_ptr":<int>}

`register` exec()s the supplied source into a private namespace, picks
up the function bound to `name`, jax.jit()s it, and stores it by handle.
`dispatch` looks the function up by handle, unpacks args into positional
parameters (count first, then the arg values in caller-supplied order),
calls the function, and blocks until the result device-array
materialises (so the caller can rely on the launch having completed
when the response arrives).
"""

import json
import sys
import traceback


def _send(obj):
    sys.stdout.write(json.dumps(obj))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _try_import_jax():
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp, None
    except Exception as e:
        return None, None, str(e)


def _unpack_arg(arg):
    """Convert a single wire-protocol arg object to its Python value."""
    if not isinstance(arg, dict) or len(arg) != 1:
        raise ValueError(f"arg must be a single-key object, got {arg!r}")
    key, value = next(iter(arg.items()))
    if key in ("i32", "i64", "u32", "u64", "device_ptr"):
        return int(value)
    if key in ("f32", "f64"):
        return float(value)
    raise ValueError(f"unknown arg type {key!r}")


def main():
    jax, jnp, import_err = _try_import_jax()
    next_handle = 1
    kernels = {}  # handle -> (jit_function, original_callable)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")

            if op == "ping":
                if jax is None:
                    _send({"ok": False, "error": f"jax import failed: {import_err}"})
                    continue
                try:
                    devices = [str(d) for d in jax.devices()]
                except Exception as e:
                    devices = []
                    err = f"jax.devices() failed: {e}"
                    _send({"ok": False, "error": err, "jax_version": jax.__version__})
                    continue
                _send({
                    "ok": True,
                    "devices": devices,
                    "jax_version": getattr(jax, "__version__", "unknown"),
                })

            elif op == "register":
                if jax is None:
                    _send({"ok": False, "error": f"jax not importable: {import_err}"})
                    continue
                name = req.get("name")
                source = req.get("source")
                if not isinstance(name, str) or not isinstance(source, str):
                    _send({"ok": False, "error": "register requires 'name' and 'source' strings"})
                    continue
                ns = {"jax": jax, "jnp": jnp}
                exec(source, ns)
                if name not in ns or not callable(ns[name]):
                    _send({"ok": False, "error": f"source must define a callable named {name!r}"})
                    continue
                original = ns[name]
                # count (first positional arg) is the dispatch dimension /
                # output-shape parameter by contract (see module docstring,
                # `def foo(c, *a)`). Mark it static so JAX can use it as a
                # concrete Python int at trace time - kernels routinely use
                # count to size jnp.arange / jnp.zeros / etc, which require
                # a concrete shape. Without static_argnums the kernel sees
                # an abstract tracer and any `int(count)` or shape-using
                # call raises ConcretizationTypeError at jit-compile time.
                # The trade-off: each distinct count value triggers a fresh
                # jit compilation. That matches consumer expectations - a
                # different count IS a different output shape.
                jit_fn = jax.jit(original, static_argnums=(0,))
                handle = next_handle
                next_handle += 1
                kernels[handle] = (jit_fn, original)
                _send({"ok": True, "handle": handle})

            elif op == "dispatch":
                if jax is None:
                    _send({"ok": False, "error": f"jax not importable: {import_err}"})
                    continue
                handle = req.get("handle")
                count = req.get("count")
                args = req.get("args", [])
                if handle not in kernels:
                    _send({"ok": False, "error": f"unknown handle {handle}"})
                    continue
                try:
                    py_args = [_unpack_arg(a) for a in args]
                except ValueError as e:
                    _send({"ok": False, "error": str(e)})
                    continue
                jit_fn, _original = kernels[handle]
                try:
                    result = jit_fn(int(count), *py_args)
                    # Block until materialised so the caller can rely on
                    # launch completion when this response arrives.
                    if hasattr(result, "block_until_ready"):
                        result.block_until_ready()
                except Exception as e:
                    _send({"ok": False, "error": f"jit call failed: {e}"})
                    continue
                _send({"ok": True})

            elif op == "shutdown":
                _send({"ok": True, "goodbye": True})
                return

            else:
                _send({"ok": False, "error": f"unknown op {op!r}"})

        except json.JSONDecodeError as e:
            _send({"ok": False, "error": f"json decode: {e}"})
        except Exception:
            _send({"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
