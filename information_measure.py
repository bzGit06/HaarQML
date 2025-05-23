import numpy as np
from scipy.stats import unitary_group

import random
from random import choices

from functools import partial
from itertools import product, combinations
from opt_einsum import contract

import tensorcircuit as tc
import qutip as qt

import jax
import jax.numpy as jnp


def psiR_purity(psi_RA, Na):
    '''
    given bipartite pure state, evaluate the purity of the reduced state of subsystem
    '''
    rho_RA = jnp.einsum('i,j->ij', psi_RA, psi_RA.conj())
    rho_R = jnp.einsum('ijkj->ik', rho_RA.reshape(2**Na, 2**Na, 2**Na, 2**Na))
    
    return jnp.real(jnp.trace(rho_R @ rho_R))
    
psiR_purity_vec = jax.jit(jax.vmap(psiR_purity, in_axes=(0, None)), static_argnums=(1, ))


def psiR_vNEntropy(psi_RA, Na):
    '''
    given bipartite pure state, evaluate the von-Neumann entropy of the reduced state of subsystem
    '''
    rho_RA = jnp.einsum('i,j->ij', psi_RA, psi_RA.conj())
    rho_R = jnp.einsum('ijkj->ik', rho_RA.reshape(2**Na, 2**Na, 2**Na, 2**Na))
    
    vals = jnp.linalg.eigvalsh(rho_R)
    vals = jnp.where(vals > 0, vals, 1e-14)
    return -jnp.sum(vals * jnp.log2(vals))

psiR_vNEntropy_vec = jax.jit(jax.vmap(psiR_vNEntropy, in_axes=(0, None)), static_argnums=1)
