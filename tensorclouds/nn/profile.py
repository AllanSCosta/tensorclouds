from functools import partial
from typing import Any, Callable, Optional

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np

from ..tensor_cloud import TensorCloud 

def breakpoint_if_nonfinite(x):
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x)


@jax.custom_vjp
def print_grad(x):
    return x


def print_grad_fwd(x):
    return x, None


def print_grad_bwd(_, x_grad):
    jax.debug.print("x_grad: {}", x_grad)
    return (x_grad,)


print_grad.defvjp(print_grad_fwd, print_grad_bwd)


def print_and_return_zero(
    message,
    shapes,
    dtypes,
    mean,
    amplitude,
    minval,
    maxval,
    hasnan,
    hasnan_masked,
    hasinf,
    hasinf_masked,
    mask_t,
    mask_f,
):
    t = None
    if t is None:
        t = " " + "*" * 9
    elif t > 1.0:
        t = f"  {t: 5.1f}s  "
    elif t > 1e-3:
        t = f"  {1000 * t: 5.1f}ms "
    else:
        t = f"  {1e6 * t: 6.1f}us"

    flags = []
    if hasnan.any():
        flags += ["NaN 🤬"]
    if hasnan_masked.any():
        flags += ["NaN (in masked part) 🤬"]
    if hasinf.any():
        flags += ["Inf 🤯"]
    if hasinf_masked.any():
        flags += ["Inf (in masked part) 🤯"]
    if any(d == np.float16 for d in dtypes):
        flags += ["f16"]
    if any(d == np.float32 for d in dtypes):
        flags += ["f32"]
    if any(d == np.float64 for d in dtypes):
        flags += ["f64"]
    if any(d == np.int8 for d in dtypes):
        flags += ["i8"]
    if any(d == np.int16 for d in dtypes):
        flags += ["i16"]
    if any(d == np.int32 for d in dtypes):
        flags += ["i32"]
    if any(d == np.int64 for d in dtypes):
        flags += ["i64"]
    if any(d == np.uint8 for d in dtypes):
        flags += ["u8"]
    if any(d == np.uint16 for d in dtypes):
        flags += ["u16"]
    if any(d == np.uint32 for d in dtypes):
        flags += ["u32"]
    if any(d == np.uint64 for d in dtypes):
        flags += ["u64"]
    if any(d == np.bool_ for d in dtypes):
        flags += ["bool"]
    if any(d == np.complex64 for d in dtypes):
        flags += ["c64"]
    if any(d == np.complex128 for d in dtypes):
        flags += ["c128"]

    if len(shapes) == 1:
        s = f"{shapes[0]}"
    else:
        s = f"{shapes}"

    if len(s) > 20:
        s = s[:17] + "..."

    total_len = 40 + 10 - len(s)
    i = total_len - len(message)

    mask_txt = ""
    if mask_f.sum() > 0:
        mask_txt = (
            f" {mask_f.sum() / (mask_t.sum() + mask_f.sum()) * 100: 2.0f}% masked"
        )

    v = max(-minval.min(), maxval.max())
    if v < 1.0:
        emoji = "🤏"
    elif v < 10.0:
        emoji = "👌"
    elif v < 100.0:
        emoji = "👀"
    elif hasnan.any() or hasinf.any():
        emoji = "🔴"
    else:
        emoji = "🔥"

    msg = (
        f"{'-' * (i//2)} {message[:total_len]} {'-' * (i - i//2)}{s}{t} "
        f"{mean.mean(): 8.1e} ±{amplitude.max(): 8.1e} [{minval.min(): 7.1e},{maxval.max(): 7.1e}]{emoji} {','.join(flags)}{mask_txt}"
    )
    print(msg)

    return np.zeros(mean.shape, dtype=np.int32)  # vmap support


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def profile(message: str, tree_array: Any, tree_mask: Optional[Any] = None):
    if isinstance(tree_array, e3nn.IrrepsArray):
        leaves = [tree_array.array]
    else:
        leaves = jax.tree_util.tree_leaves(tree_array)

    if tree_mask is None:
        mask = [jnp.ones(e.shape, dtype=jnp.bool_) for e in leaves]
    else:
        mask = jax.tree_util.tree_leaves(tree_mask)
        if len(mask) == 1:
            mask = len(leaves) * mask

    mask = [jnp.broadcast_to(m, e.shape) for e, m in zip(leaves, mask)]

    if hasattr(tree_array, "shape"):
        shapes = [tree_array.shape]
    else:
        shapes = [e.shape for e in leaves]

    dtypes = [e.dtype for e in leaves]

    mean = jnp.mean(
        jnp.array([jnp.where(m, e, 0.0).sum() / m.sum() for e, m in zip(leaves, mask)])
    )
    amplitude = (
        jnp.mean(
            jnp.array(
                [
                    jnp.where(m, e**2, 0.0).sum() / m.sum()
                    for e, m in zip(leaves, mask)
                ]
            )
        )
        ** 0.5
    )
    minval = jnp.min(
        jnp.array([jnp.where(m, e, e.max()).min() for e, m in zip(leaves, mask)])
    )
    maxval = jnp.max(
        jnp.array([jnp.where(m, e, e.min()).max() for e, m in zip(leaves, mask)])
    )
    hasnan = jnp.any(
        jnp.array([jnp.isnan(jnp.where(m, e, 0.0)).any() for e, m in zip(leaves, mask)])
    )
    hasnan_masked = jnp.any(
        jnp.array([jnp.isnan(jnp.where(m, 0.0, e)).any() for e, m in zip(leaves, mask)])
    )
    hasinf = jnp.any(
        jnp.array([jnp.isinf(jnp.where(m, e, 0.0)).any() for e, m in zip(leaves, mask)])
    )
    hasinf_masked = jnp.any(
        jnp.array([jnp.isinf(jnp.where(m, 0.0, e)).any() for e, m in zip(leaves, mask)])
    )
    mask_t = jnp.array([m.sum() for m in mask]).sum()
    mask_f = jnp.array([(1 - m).sum() for m in mask]).sum()

    zero = jax.pure_callback(
        callback=partial(print_and_return_zero, message, shapes, dtypes),
        result_shape_dtypes=jnp.array(0, dtype=jnp.int32),
        vectorized=True,
        mean=mean,
        amplitude=amplitude,
        minval=minval,
        maxval=maxval,
        hasnan=hasnan,
        hasnan_masked=hasnan_masked,
        hasinf=hasinf,
        hasinf_masked=hasinf_masked,
        mask_t=mask_t,
        mask_f=mask_f,
    )

    return jax.tree_util.tree_map(lambda e: e + zero, tree_array)


@profile.defjvp
def profile_jvp(message, primals, tangents):
    (x, m) = primals
    (dx, dm) = tangents
    return profile(f"(jvp){message}", x, m), dx


def dummyfy(func: Callable) -> Callable:
    def dummy(*args, **kwargs):
        # Make sure the outputs still depends on the inputs
        s = sum(
            x.flatten()[0]
            for x in jax.tree_util.tree_leaves((args, kwargs))
            if hasattr(x, "flatten")
        )

        # Create dummy outputs with the same shape and dtype as the original outputs
        return jax.tree_util.tree_map(
            lambda x: s.astype(x.dtype) + jnp.zeros(x.shape, x.dtype),
            func(*args, **kwargs),
        )

    return dummy
