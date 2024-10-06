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

def localHaarCirc(input, Us, n_tot, L):
    '''
    random unitary circuit in brickwall
    '''
    c = tc.Circuit(n_tot, inputs=input)
    for l in range(L):
        c.any(*range(n_tot), unitary=Us[2*l])
        c.any(*range(n_tot), unitary=Us[2*l + 1])
    return c.state()

def layerU(Us, n_tot, odd=True):
    '''
    '''
    U = Us[0]
    for k in range(1, len(Us)):
        U = contract('ij, kl->ikjl', U, Us[k]).reshape(4**(k+1), 4**(k+1))

    if odd:
        U = contract('ij, kl->ikjl', U, np.eye(2**(n_tot % 2))).reshape(2**n_tot, 2**n_tot)
    else:
        U = contract('ij, kl->ikjl', np.eye(2), U).reshape(4**len(Us)*2, 4**len(Us)*2)
        U = contract('ij, kl->ikjl', U, np.eye(2**(1 - n_tot % 2))).reshape(2**n_tot, 2**n_tot)
    
    return U
    
def localUs(n_tot, T, L, seed):
    np.random.seed(seed)
    Us = []
    for _ in range(T):
        Us_layer = []
        for l in range(L):
            Us_odd = unitary_group.rvs(dim=4, size=n_tot//2).reshape(n_tot//2, 4, 4)
            if (n_tot - 1)//2:
                Us_even = unitary_group.rvs(dim=4, size=(n_tot - 1)//2).reshape((n_tot - 1)//2, 4, 4)
                Us_layer.extend([layerU(Us_odd, n_tot, True), layerU(Us_even, n_tot, False)])
            else:
                Us_layer.extend([layerU(Us_odd, n_tot, True), np.eye(2**n_tot)])
        Us_layer = np.stack(Us_layer)
        Us.append(Us_layer)
    
    return np.stack(Us)

class haarSeqModel():
    def __init__(self, n, na, T, L=0):
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.circ_global = K.jit(K.vmap(partial(globalHaarCirc, n_tot=self.n_tot),
                                        vectorized_argnums=0))
        self.circ_local = K.jit(K.vmap(partial(localHaarCirc, n_tot=self.n_tot, L=L), 
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

    def dataGeneration_local(self, ndata):
        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        inputs_t = np.zeros((ndata, 2**self.n), dtype=complex)
        inputs_t[:, 0] = 1.
        states = [jnp.array(inputs_t)]
        
        for t in range(self.T):
            inputs_t = jnp.concatenate([states[t], jnp.zeros(shape=(ndata, 2**self.n_tot-2**self.n), 
                                                              dtype=jnp.complex64)], axis=1)
            outputs_full = self.circ_local(inputs_t, self.Us[t])
            key, subkey = jax.random.split(key)
            outputs_t = self.randomMeasure(outputs_full, subkey)
            states.append(outputs_t)
        states = jnp.stack(states)
        return states
    
    

class MFIMSeqEvo():
    def __init__(self, n, na, T):
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        
        H = self.MFIM(n + na, 0.8090, 0.9045)
        Es, vecs = H.eigenstates()
        self.Es = jnp.array(Es, dtype=jnp.float32)
        self.vecs = jnp.stack([v.full() for v in vecs]).squeeze().astype(jnp.complex64)


    def MFIM(self, n, hx, hy):
        # mixing-field Ising model
        Xs = [qt.tensor([qt.sigmax() if i==j else qt.qeye(2) for j in range(n)]) for i in range(n)]
        Ys = [qt.tensor([qt.sigmay() if i==j else qt.qeye(2) for j in range(n)]) for i in range(n)]

        H = 0
        for i in range(n):
            H += hx * Xs[i]
            H += hy * Ys[i]
        # open-boundary
        for i in range(n-1):
            H += Xs[i]*Xs[i+1]
        return H
    
    #@partial(jax.jit, static_argnums=(0, ))
    def mfimEvo(self, inputs, t):
        cs = contract('bi, mi->mb', self.vecs.conj(), inputs)
        evo_phases = np.exp(-1j*self.Es*t)
        amplitudes = cs * evo_phases
        state_t = jnp.sum(contract('mb, bi->mbi', amplitudes, self.vecs), axis=1)
        return state_t

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
    
    def dataGeneration(self, ndata, ts):
        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        inputs_t = np.zeros((ndata, 2**self.n), dtype=jnp.complex64)
        inputs_t[:, 0] = 1.
        states = [jnp.array(inputs_t)]
        
        for t in range(self.T):
            inputs_t = jnp.concatenate([states[t], jnp.zeros(shape=(ndata, 2**self.n_tot-2**self.n), 
                                                              dtype=jnp.complex64)], axis=1)
            outputs_full = self.mfimEvo(inputs_t, ts[t])
            key, subkey = jax.random.split(key)
            outputs_t = self.randomMeasure(outputs_full, subkey)
            states.append(outputs_t)

        states = jnp.stack(states)
        return states