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

import datetime

K = tc.set_backend('jax')
tc.set_dtype('complex128')


def maxEntangle_state(N):
    '''
    create maximally entangled state
    N: number of qubit in one party
    '''
    d = 2**N
    Psi = 0
    for i in range(d):
        Psi += qt.tensor(qt.basis(d, i), qt.basis(d, i))
    return Psi.unit()

@partial(jax.jit, static_argnums=(1, 2))
def randomMeasure(inputs, Na, Nb, key):
    '''
    given samples of input pure states, perform projective measurement on
    the ancillary system, and collect post measurement pure states
    '''
    num = len(inputs)
    psis = inputs.reshape(num, 4**Na, 2**Nb)
    probs = jnp.linalg.norm(psis, axis=1)**2.
    res = jax.random.categorical(key, jnp.log(probs))
    post_states = psis[jnp.arange(num), :, res]
    post_states /= jnp.sqrt(probs[jnp.arange(num), res])[:, jnp.newaxis]
    
    return post_states


def seqModel_RA(z, Us, psi_RA, Na, Nb, T):
    '''
    sequential model with random unitary, output the corresponding probability 
    of projective measurement trajectories, and corresponding state
    '''
    zero_B = qt.basis(2**Nb, 0).full().squeeze()
    
    prob_z = []
    for t in range(T):
        inputs = jnp.kron(psi_RA, zero_B)
        c = tc.Circuit(2*Na + Nb, inputs=inputs)
        c.any(*list(range(Na, 2*Na+Nb)), unitary=Us[t])
        # post-selection on measure
        psi = c.state().reshape(4**Na, 2**Nb)
        post_psi = psi[:, z[t]]
        p_zt = jnp.linalg.norm(post_psi)**2
        psi_RA = post_psi / jnp.sqrt(p_zt)
        prob_z.append(p_zt)
    return jnp.array(prob_z), psi_RA

seqModel_RA_vec = K.jit(K.vmap(seqModel_RA, vectorized_argnums=0), static_argnums=(3, 4, 5))


def globalCircuit(inputs, U, Na, Nb):
    '''
    apply a global unitary to the system
    '''
    c = tc.Circuit(2*Na + Nb, inputs=inputs)
    c.any(*list(range(Na, 2*Na+Nb)), unitary=U)
    return c.state()

globalCircuit_vec = K.jit(K.vmap(globalCircuit, vectorized_argnums=0), static_argnums=(2, 3))


def seqModel_RAmc(Us, psi_RA, Na, Nb, T, num):
    '''
    sequential model with random unitary, output post-measurement state 
    through monte-carlo sampling
    '''
    psis_RA = jnp.stack([psi_RA]*num)
    zero_B = qt.basis(2**Nb, 0).full().squeeze()

    seed = int(1e6 * datetime.datetime.now().timestamp())
    key = jax.random.PRNGKey(seed)
    for t in range(T):
        inputs = jnp.einsum('bi, j->bij', psis_RA, zero_B).reshape(num, 4**Na*2**Nb)
        psis = globalCircuit_vec(inputs, Us[t], Na, Nb)

        key, subkey = jax.random.split(key)
        psis_RA = randomMeasure(psis, Na, Nb, subkey)
    
    return psis_RA

