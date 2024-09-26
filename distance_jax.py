import jax
from jax import numpy as jnp

from ott.geometry import pointcloud
from ott.solvers.linear import solve
from ott.geometry.costs import CostFn

from opt_einsum import contract
from functools import partial

# @jax.jit
# def avgStateSupFid(states, sigma):
#     rho = jnp.mean(contract('bi, bj->bij', states, states.conj(), backend='jax'), axis=0)
#     esf2_prod = (1. - jnp.trace(rho @ rho)) * (1. - jnp.trace(sigma @ sigma))
#     supF = jnp.trace(rho @ sigma) + jnp.sqrt(esf2_prod)

#     return jnp.real(supF)

@jax.jit
def avgStateSupFid(states, sigma):
    rho = jnp.mean(jnp.einsum('bi, bj->bij', states, states.conj()), axis=0)
    trace_rs = jnp.einsum('ij, ji->', rho, sigma)
    trace_rr = jnp.einsum('ij, ji->', rho, rho)
    trace_ss = jnp.einsum('ij, ji->', sigma, sigma)
    supF = trace_rs + jnp.sqrt((1. - trace_rr) * (1. - trace_ss))

    return jnp.real(supF)

@jax.jit
def avgStateSupFid_pure(states, psi0):
    rho = jnp.mean(contract('bi, bj->bij', states, states.conj(), backend='jax'), axis=0)
    return jnp.real(psi0.conj() @ rho @ psi0)


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


