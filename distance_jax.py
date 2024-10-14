import jax
from jax import numpy as jnp

from ott.geometry import pointcloud
from ott.solvers.linear import solve
from ott.geometry.costs import CostFn

from opt_einsum import contract
from functools import partial

@jax.jit
def avgStateFid_pure(states, psi0):
    rho = jnp.mean(contract('bi, bj->bij', states, states.conj(), backend='jax'), axis=0)
    return jnp.real(psi0.conj() @ rho @ psi0)

@jax.jit
def avgStateSupFid_id(states):
    d = states.shape[1]
    rho = jnp.mean(jnp.einsum('bi, bj->bij', states, states.conj()), axis=0)
    purity_rho = jnp.real(jnp.einsum('ij, ji->', rho, rho))
    supF = 1./d + jnp.sqrt((1. - purity_rho) * (1. - 1./d))
    return supF

@jax.jit
def avgStateFid_id(states):
    d = states.shape[1]
    rho = jnp.mean(jnp.einsum('bi, bj->bij', states, states.conj()), axis=0)
    val = jnp.linalg.eigvalsh(rho)
    F = jnp.sum(jnp.sqrt(jnp.abs(val)))**2 / d
    return jnp.real(F)

@jax.jit
def avgStateSubFid_id(states):
    d = states.shape[1]
    rho = jnp.mean(jnp.einsum('bi, bj->bij', states, states.conj()), axis=0)
    purity_rho = jnp.real(jnp.einsum('ij, ji->', rho, rho))
    subF = 1./d + jnp.sqrt(2.) * jnp.sqrt(1. - purity_rho)/d
    return subF


@jax.tree_util.register_pytree_node_class
class Trace(CostFn):
    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return 1. - jnp.abs(jnp.conj(x) @ y.T) ** 2.


@partial(jax.jit, static_argnums=(4, 5, 6, ))
def sinkhornDistance(Set1, Set2, prob1=None, prob2=None, reg=0.01, threshold=0.001, lse_mode=True):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    geom = pointcloud.PointCloud(Set1, Set2, cost_fn=Trace(), epsilon=reg)
    ot = solve(geom, a=prob1, b=prob2, lse_mode=lse_mode, threshold=threshold)
    
    return ot.reg_ot_cost


