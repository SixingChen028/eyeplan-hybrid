#!/usr/bin/env python3
import jax
import jax.numpy as jnp

print(f"jax.__version__={jax.__version__}")
print(f"default_backend={jax.default_backend()}")

devices = jax.devices()
print(f"devices={devices}")
print(f"device_count={len(devices)}")

x = jnp.arange(8.0)
y = jnp.sin(x).sum()
print(f"sanity_sum={float(y)}")

has_gpu = any(d.platform == "gpu" for d in devices)
print(f"has_gpu={has_gpu}")
if not has_gpu:
    raise SystemExit("JAX GPU backend not detected")
