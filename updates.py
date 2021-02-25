import jax
import haiku as hk

@jax.jit
def ema_update(
    avg_params: hk.Params,
    new_params: hk.Params,
    epsilon: float = 0.001) -> hk.Params:
    return jax.tree_multimap(lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2,
         avg_params, new_params)