from functools import partial
from itertools import combinations
from opt_einsum import contract

import numpy as np
from scipy.stats import unitary_group

import jax
from jax import numpy as jnp
from jax import random

import tensorcircuit as tc

import qutip as qt

import datetime

K = tc.set_backend("jax")
tc.set_dtype('complex64')


def globalHaarCirc(input, U, n_tot):
    c = tc.Circuit(n_tot, inputs=input)
    c.any(*range(n_tot), unitary=U)
    return c.state()


class haarSeqModel():
    def __init__(self, n, na, T, L=0):
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.circ_global = K.jit(K.vmap(partial(globalHaarCirc, n_tot=self.n_tot),
                                        vectorized_argnums=0))

    def setUnitary(self, Us):
        self.Us = Us

    @partial(jax.jit, static_argnums=(0, ))
    def randomMeasure(self, inputs, key):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        '''
        n_batch = inputs.shape[0]
        m_probs = jnp.abs(jnp.reshape(inputs, [n_batch, 2 ** self.na, 2 ** self.n])) ** 2.0
        m_probs = jnp.log(jnp.sum(m_probs, axis=2))
        m_res = jax.random.categorical(key, m_probs)
        indices = 2 ** self.n * jnp.reshape(m_res, [-1, 1]) + jnp.arange(2 ** self.n)
        post_state = jnp.take_along_axis(inputs, indices, axis=1)
        post_state /= jnp.linalg.norm(post_state, axis=1)[:, jnp.newaxis]
        
        return post_state

    def dataGeneration_global(self, ndata):
        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        inputs_t = np.zeros((ndata, 2**self.n), dtype=complex)
        inputs_t[:, 0] = 1.
        states = [jnp.array(inputs_t)]
        
        for t in range(self.T):
            inputs_t = jnp.concatenate([states[t], jnp.zeros(shape=(ndata, 2**self.n_tot-2**self.n), 
                                                              dtype=jnp.complex64)], axis=1)
            outputs_full = self.circ_global(inputs_t, self.Us[t])
            key, subkey = jax.random.split(key)
            outputs_t = self.randomMeasure(outputs_full, subkey)
            states.append(outputs_t)

        states = jnp.stack(states)        
        return states

 