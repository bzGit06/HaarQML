import numpy as np
import jax
import jax.numpy as jnp
from opt_einsum import contract
from functools import partial
from itertools import product, permutations
import qutip as qt


@partial(jax.jit, static_argnums=(2, ))
def rhoMMT(states, probs=None, K=1):
    # evluate K-th moment density operator in parallel
    rho1 = contract('mi, mj->mij', states, states.conj())
    rhok = rho1
    for _ in range(K-1):
        rhok = contract('mij, mkl->mikjl', rhok, rho1)
        rhok = rhok.reshape(states.shape[0], rhok.shape[1]*rhok.shape[2],
                            rhok.shape[3]*rhok.shape[4])
    if probs is None:
        return jnp.mean(rhok, axis=0)
    else:
        return np.sum(probs[:, np.newaxis, np.newaxis]*rhok, axis=0)


@partial(jax.jit, static_argnums=(2, 3))
def rhoMMT_seq(states, probs=None, K=1, slice=1):
    # evluate K-th moment density operator in sequential
    N = len(states)
    rho_K = 0
    for i in range(N//slice):
        each = states[slice*i: slice*(i+1)]
        if probs is None:
            rho_K += rhoMMT(each, K=K)
        else:
            rho_K += rhoMMT(each, probs[slice*i: slice*(i+1)], K=K)
    return rho_K


@partial(jax.jit, static_argnums=(2, ))
def framePot(states, probs=None, K=1):
    inners = jnp.abs(contract('mi, ni->mn', states.conj(), states))
    if probs is None:
        F_K = jnp.mean(inners**(2*K))
    else:
        F_K = jnp.mean(contract('m,n->mn', probs, probs) * inners**(2*K))
    return F_K


@partial(jax.jit, static_argnums=(2, 3))
def framePot_seq(states, probs=None, K=1, slice=1):
    # evaluate frame potential in sequential
    N = len(states)
    F_K = 0
    for i in range(N//slice):
        each = jnp.abs(contract('mi, ni->mn', states.conj(),
                       states[slice*i: slice*(i+1)]))**(2*K)
        if not (probs is None):
            each = contract('m, n->mn', probs,
                            probs[slice*i: slice*(i+1)])*each
        F_K += jnp.sum(each)
    return F_K / N**2

@jax.jit
def fp(x, y, K=1):
    return jnp.abs(jnp.dot(x.conj(), y)) ** (2 * K)


@partial(jax.jit, static_argnums=(1, 2))
def framePot_batched(states, K=1, batch_sizes=None):
    def inner_map(x): return jnp.mean(jax.vmap(lambda y: fp(x, y, K=K))(states))

    if batch_sizes is None:
        return jnp.mean(jax.lax.map(inner_map, states))
    else:
        f = []
        for k in range(len(states)//batch_sizes):
            f.append(jnp.mean(jax.lax.map(inner_map, states[batch_sizes*k: batch_sizes*(k+1)])))
        return jnp.mean(jnp.array(f))
