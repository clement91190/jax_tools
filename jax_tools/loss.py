import jax
import jax.numpy as jnp


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -1.0 * jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

