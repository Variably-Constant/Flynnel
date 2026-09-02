# JAX reference kernel for the Flynnel TPU backend.
#
# Registered with `TpuJaxBackend::register_kernel` at
# `src/backend/tpu_jax.rs`. The Rust side ships this source text as
# bytes over the bridge; the Python side runs it under
# `jax.jit(static_argnums=(0,))` so `count` is compile-time-baked
# and every other argument is a JAX traced array.
#
# Verified against a live TPU host via
# `examples/tpu_jax_demo.rs`:
#   cargo run --example tpu_jax_demo \
#             --features tpu-jax-reference --release
#
# The kernel doubles `count` values then adds a scalar; the caller
# can inspect the returned sum for correctness. Kept intentionally
# minimal so the demo focuses on the routing surface rather than
# the JAX numerics.

def double_then_sum(count, scalar):
    arr = jnp.arange(count) * 2
    return jnp.sum(arr) + scalar
