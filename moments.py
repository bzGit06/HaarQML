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

@partial(jax.jit, static_argnums=(2, ))
def rhoMMT_seq(states, probs=None, K=1, slice=1):
    # evluate K-th moment density operator in sequential
    N = len(probs)
    rho_K = 0
    rhoMMT_jit = jax.jit(partial(rhoMMT, K=K))
    for i in range(N//slice):
        each = states[slice*i: slice*(i+1)]
        if probs is None:
            rho_K += rhoMMT_jit(each)
        else:
            rho_K += rhoMMT_jit(each, probs[slice*i: slice*(i+1)])
    return rho_K

@partial(jax.jit, static_argnums=(2, ))
def framePot(states, probs=None, K=1):
    inners = jnp.abs(contract('mi, ni->mn', states.conj(), states))
    if probs is None:
        F_K = jnp.mean(inners**(2*K))
    else:
        F_K = jnp.mean(contract('m,n->mn', probs, probs) * inners**(2*K))
    return F_K


def mmtDist_p(rho1, rho2, p):
    # p-th order moment distance
    if p==1:
        ord = 'nuc'
    elif p==2:
        ord = 'fro'
    return jnp.linalg.norm(rho1 - rho2, ord=ord)


def permGroup(K):
    sigmas = np.stack(list(permutations(range(K))))
    return sigmas

def permOp(sigma, n, K, basis_full):
    '''
    given a permutation sigma, generate the operator for n-qubit state
    '''
    d = 2**n
    basis_perm_full = basis_full[:, sigma] # permute basis following sigma
    op = np.zeros((d ** K, d ** K))
    pos_perm_full = np.sum(np.stack([basis_perm_full[:, i] * (d**(K-1-i)) for i in range(K)]), 
                            axis=0) # find position of permutated basis
    op[pos_perm_full, np.arange(d ** K)] = 1 # generate the permutation operator

    return op

def haarEnsembleMMT(n, K):
    basis_full = np.stack(product(range(2**n), repeat=K))
    sigmas = permGroup(K)
    rho = 0
    for k in range(len(sigmas)):
        rho += permOp(sigmas[k], n, K, basis_full)
    denom = np.prod(np.arange(K) + 2**n)
    
    return jnp.array(rho / denom)

def pauliTwirl_K2(n, rho=None, Cs=None):
    paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    rho2 = 0
    for i, s in enumerate(product(range(4), repeat=n)):
        P = qt.tensor([paulis[x] for x in s])
        if Cs is None:
            c_P = qt.expect(P, rho)
        else:
            c_P = Cs[i]
        rho2 += c_P**2 * qt.tensor(P, P)
    rho2 /= 4**n
    return rho2

def pauliTwirlEnsemble(n, Psi0):
    paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    states = []
    for s in product(range(4), repeat=n):
        P = qt.tensor([paulis[x] for x in s])
        states.append(P * Psi0)
    return states