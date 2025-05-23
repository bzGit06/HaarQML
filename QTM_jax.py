from functools import partial
from itertools import combinations

import numpy as np
from scipy.stats import unitary_group

import jax
from jax import numpy as jnp
from jax import random

import tensorcircuit as tc

import datetime

K = tc.set_backend("jax")
tc.set_dtype('complex64')


def HaarSampleGeneration(Ndata, n, seed):
    '''
    generate random haar states,
    used as inputs in the t=T step for backward denoise
    Args:
    Ndata: number of samples in dataset
    '''
    np.random.seed(seed)
    states_T = unitary_group.rvs(dim=2**n, size=Ndata)[:,:,0]

    return jnp.array(states_T)
    

def paramQC(input, params, n_tot, L):
    '''
    parameteric circuit following hardware efficient ansatz
    Args:
    input: input state
    params: variational parameters
    n_tot: total number of qubits
    L = layers of circuit
    '''
    c = tc.Circuit(n_tot, inputs=input)

    for l in range(L):
        for i in range(n_tot):
            c.rx(i, theta=params[2* n_tot * l + i])
            c.ry(i, theta=params[2* n_tot* l + n_tot + i])

        for i in range(n_tot // 2):
            c.cz(2 * i, 2 * i + 1)

        for i in range((n_tot-1) // 2):
            c.cz(2 * i + 1, 2 * i + 2)

    return c.state()


class QTM():
    def __init__(self, n, na, T, L):
        '''
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        T: number of diffusion steps
        L: layers of circuit before each measurement
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        # embed the circuit to a vectorized pytorch neural network layer
        self.pQC_vmap = K.jit(K.vmap(partial(paramQC, n_tot=self.n_tot, L=self.L), 
                                     vectorized_argnums=0))
    

    # @partial(jax.jit, static_argnums=(0, ))
    # def randomMeasure(self, inputs, key):
    #     '''
    #     Given the inputs on both data & ancilla qubits before measurmenets,
    #     calculate the post-measurement state.
    #     The measurement and state output are calculated in parallel for data samples
    #     Args:
    #     inputs: states to be measured, first na qubit is ancilla
    #     '''
    #     n_batch = inputs.shape[0]
    #     m_probs = jnp.abs(jnp.reshape(inputs, [n_batch, 2 ** self.na, 2 ** self.n])) ** 2.0
    #     m_probs = jnp.log(jnp.sum(m_probs, axis=2))
    #     m_res = jax.random.categorical(key, m_probs)
    #     indices = 2 ** self.n * jnp.reshape(m_res, [-1, 1]) + jnp.arange(2 ** self.n)
    #     post_state = jnp.take_along_axis(inputs, indices, axis=1)
    #     post_state /= jnp.linalg.norm(post_state, axis=1)[:, jnp.newaxis]
        
    #     return post_state

    @partial(jax.jit, static_argnums=(0, ))
    def randomMeasure(self, inputs, key):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        '''
        psi = jnp.reshape(inputs, [-1, 2 ** self.na, 2 ** self.n]).transpose(1, 2, 0)
        m_probs = jnp.log(jnp.sum(jnp.abs(psi) ** 2.0, axis=1)).T
        m_res = jax.random.categorical(key, m_probs)
        
        post_state = jnp.choose(m_res, psi, mode='wrap').T
        post_state /= jnp.linalg.norm(post_state, axis=1)[:, jnp.newaxis]
        
        return post_state
    
    @partial(jax.jit, static_argnums=(0, ))
    def pQCoutput(self, inputs, params, key):
        output_full = self.pQC_vmap(inputs, params)
        output_postM = self.randomMeasure(output_full, key)
        return output_postM
    
    def prepareInput_t(self, inputs_0, params_tot, t):
        '''
        prepare the input samples for step t including zero ancillas
        Args:
        inputs_0: the input state at the beginning
        params_tot: all circuit parameters till step t+1
        '''
        Ndata = len(inputs_0)

        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        zero_tensor = jnp.zeros(shape=(Ndata, 2**self.n_tot - 2**self.n), dtype=jnp.complex64)
        inputs_t = jnp.concatenate([inputs_0, zero_tensor], axis=1)
        for tt in range(t):
            key, subkey = jax.random.split(key)
            outputs = self.pQCoutput(inputs_t, params_tot[tt], subkey)
            inputs_t = jnp.concatenate([outputs, zero_tensor], axis=1)

        return inputs_t

    def dataGeneration(self, inputs_0, params_tot):
        '''
        generate data from t=1 to t=T
        '''
        Ndata = len(inputs_0)
        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        states = [inputs_0]

        zero_tensor = jnp.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), dtype=jnp.complex64)
        inputs_t = jnp.concatenate([inputs_0, zero_tensor], axis=1)

        for tt in range(self.T):
            key, subkey = jax.random.split(key)
            outputs_t = self.pQCoutput(inputs_t, params_tot[tt], subkey)
            inputs_t = jnp.concatenate([outputs_t, jnp.zeros(shape=(Ndata, 2**self.n_tot-2**self.n), 
                                                              dtype=jnp.complex64)], axis=1)
            states.append(outputs_t)

        states = jnp.stack(states)
        return states





