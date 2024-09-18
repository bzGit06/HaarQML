import numpy as np
from scipy.stats import unitary_group, kstest, uniform
from opt_einsum import contract
from itertools import product

from distance_jax import naturalDistance, sinkhornDistance, avgStateSupFid

import jax
import jax.numpy as jnp
from jax import random
import optax

import qutip as qt
from qutip import Bloch

import time
import datetime
import os

def Training_t(model, t, tau, inputs_0, X0, XT, p0=None, pT=None, params_tot=[], 
        epochs=1001, dis_measure='sd'):
    '''
    Training for Quantum Wasserstein transformation
    Args:
    t: interpolation step
    X0: beginning state distribution
    XT: target state distribution
    params_tot: collection of PQC parameters for steps < t 
    epochs: number of iterations
    dis_measure: the distance measure to compare two distributions of quantum states
    '''
    Ndata = len(inputs_0)

    loss_hist = [] # record of training history
    params_hist = [] # record of training parameters

    # initialize parameters
    key = random.PRNGKey(42)
    params_t = random.normal(key, shape=(2*model.n_tot*model.L,))

    # set optimizer and learning rate decay
    optimizer = optax.adam(learning_rate = 0.0005)
    opt_state = optimizer.init(params_t)

    if dis_measure == 'sd':
        def loss_func(params_t, data_0, data_T, prob_0, prob_T):
            seed = int(1e6 * datetime.datetime.now().timestamp())
            key = jax.random.PRNGKey(seed)

            key, subkey = jax.random.split(key)
            input_t_1 = model.prepareInput_t(inputs_0, params_tot, t)
            output_1 = model.pQCoutput(input_t_1, params_t, subkey)
            loss1 = sinkhornDistance(output_1, data_0, prob2=prob_0, reg=0.005)

            _, subkey = jax.random.split(key)
            input_t_2 = model.prepareInput_t(inputs_0, params_tot, t)
            output_2 = model.pQCoutput(input_t_2, params_t, subkey)
            loss2 = sinkhornDistance(output_2, data_T, prob2=prob_T, reg=0.005)

            return (1-tau) * loss1 + tau * loss2
        
    elif dis_measure == 'mmd':
        def loss_func(params_t, data_0, data_T, prob_0=None, prob_T=None):
            seed = int(1e6 * datetime.datetime.now().timestamp())
            key = jax.random.PRNGKey(seed)

            key, subkey = jax.random.split(key)
            input_t_1 = model.prepareInput_t(inputs_0, params_tot, t)
            output_1 = model.pQCoutput(input_t_1, params_t, subkey)
            loss1 = naturalDistance(output_1, data_0)

            _, subkey = jax.random.split(key)
            input_t_2 = model.prepareInput_t(inputs_0, params_tot, t)
            output_2 = model.pQCoutput(input_t_2, params_t, subkey)
            loss2 = naturalDistance(output_2, data_T)

            return (1-tau) * loss1 + tau * loss2

    loss_func_vg = jax.jit(jax.value_and_grad(loss_func))
    #@partial(jax.jit, static_argnums=(2, ))
    def update(params_t, data_0, data_T, prob_0, prob_T, opt_state):
        loss_value, grads = loss_func_vg(params_t, data_0, data_T, prob_0, prob_T)

        updates, new_opt_state = optimizer.update(grads, opt_state, params_t)
        new_params_t = optax.apply_updates(params_t, updates)

        return new_params_t, new_opt_state, loss_value

    t0 = time.time()
    for step in range(epochs):
        np.random.seed()
        idx1 = np.random.choice(len(X0), size=Ndata, replace=False)
        idx2 = np.random.choice(len(XT), size=Ndata, replace=False)
        data_0 = X0[idx1]
        data_T = XT[idx2]
        if p0 is None:
            prob_0 = None
        else:
            prob_0 = p0[idx1] * len(X0)/Ndata
        if pT is None:
            prob_T = None
        else:
            prob_T = pT[idx2] * len(X0)/Ndata

        if step % (epochs//50) == 0:
            params_hist.append(params_t)

        params_t, opt_state, loss_value = update(params_t, data_0, data_T, prob_0, prob_T, opt_state)
        loss_hist.append(loss_value) # record the current loss
        if step % 1000 == 0:
            print("Step {}, loss: {:.7f}, time elapsed: {:.4f} seconds".format(step, loss_value, time.time() - t0))
        
    return jnp.stack(params_hist), loss_hist

def TrainingAvgState_t(model, t, tau, inputs_0, sigma0, sigmaT, params_tot=[], epochs=1001):
    Ndata = len(inputs_0)

    loss_hist = [] # record of training history
    params_hist = [] # record of training parameters
    states_hist = []
    
    # initialize parameters
    key = random.PRNGKey(42)
    params_t = random.normal(key, shape=(2*model.n_tot*model.L,))

    # set optimizer and learning rate decay
    optimizer = optax.adam(learning_rate = 0.0005)
    opt_state = optimizer.init(params_t)

    def loss_func(params_t, sigma1, sigma2):
        seed = int(1e6 * datetime.datetime.now().timestamp())
        key = jax.random.PRNGKey(seed)

        key, subkey = jax.random.split(key)
        input_t_1 = model.prepareInput_t(inputs_0, params_tot, t)
        output_1 = model.pQCoutput(input_t_1, params_t, subkey)
        loss1 = 1. - avgStateSupFid(output_1, sigma1)
        
        _, subkey = jax.random.split(key)
        input_t_2 = model.prepareInput_t(inputs_0, params_tot, t)
        output_2 = model.pQCoutput(input_t_2, params_t, subkey)
        model.current_states = output_2
        loss2 = 1. - avgStateSupFid(output_2, sigma2)

        return (1-tau) * loss1 + tau * loss2, output_2

    loss_func_vg = jax.jit(jax.value_and_grad(loss_func, has_aux=True))
    #@partial(jax.jit, static_argnums=(2, ))
    def update(params_t, sigma1, sigma2, opt_state):
        (loss_value, new_states), grads = loss_func_vg(params_t, sigma1, sigma2)

        updates, new_opt_state = optimizer.update(grads, opt_state, params_t)
        new_params_t = optax.apply_updates(params_t, updates)

        return new_params_t, new_opt_state, loss_value, new_states

    t0 = time.time()
    for step in range(epochs):
        if step % (epochs//50) == 0:
            params_hist.append(params_t)

        params_t, opt_state, loss_value, states_t = update(params_t, sigma0, sigmaT, opt_state)
        loss_hist.append(loss_value) # record the current loss
        
        if step % 1000 == 0:
            states_hist.append(states_t)
            print("Step {}, loss: {:.7f}, time elapsed: {:.4f} seconds".format(step, loss_value, time.time() - t0))
        
    return jnp.stack(params_hist), loss_hist, np.stack(states_hist)


def localRevTraining_t(model, model_ref, t, inputs_T, X0, params, params_ref, epochs, dis_measure='sd'):
    '''
    training a circuit as a replacement on random weak scrambling circuits
    Args:
    model: local reverse model
    model_ref: the model to be reversed for reference
    t: interpolation step
    inputs_T: input training data for reverse model
    X0: prior data distribution
    params: reverse model parameters
    params_ref: reference model parameters
    epochs: number of iterations
    dis_measure: the distance measure to compare two distributions of quantum states
    '''

    loss_hist = [] # record of training history
    params_hist = [] # record of training parameters

    # initialize parameters
    key = random.PRNGKey(42)
    params_t = random.normal(key, shape=(2 * model.n_tot * model.L, ))

    # set optimizer and learning rate decay
    optimizer = optax.adam(learning_rate = 0.0005)
    opt_state = optimizer.init(params_t)

    if dis_measure == 'sd':
        def loss_func(params_t, data_0):
            seed = int(1e6 * datetime.datetime.now().timestamp())
            key = jax.random.PRNGKey(seed)

            key, subkey = jax.random.split(key)
            input_t = model.prepareInput_t(inputs_T, params, t)
            output_t = model.pQCoutput(input_t, params_t, subkey)

            data_t = model_ref.prepareInput_t(data_0, params_ref, model_ref.T-1-t)[:, :2**model_ref.n]
            loss = sinkhornDistance(output_t, data_t, reg=0.005)
        
            return loss
    
    loss_func_vg = jax.jit(jax.value_and_grad(loss_func))
    #@partial(jax.jit, static_argnums=(2, ))
    def update(params_t, data_0, opt_state):
        loss_value, grads = loss_func_vg(params_t, data_0)

        updates, new_opt_state = optimizer.update(grads, opt_state, params_t)
        new_params_t = optax.apply_updates(params_t, updates)

        return new_params_t, new_opt_state, loss_value

    t0 = time.time()
    for step in range(epochs):
        np.random.seed()
        idx = np.random.choice(len(X0), size=len(inputs_T), replace=False)
        data_0 = X0[idx]
        if step % (epochs//50) == 0:
            params_hist.append(params_t)

        params_t, opt_state, loss_value = update(params_t, data_0, opt_state)
        if step % 1000 == 0:
            print("Step {}, loss: {:.7f}, time elapsed: {:.4f} seconds".format(step, loss_value, time.time() - t0))
        
        loss_hist.append(loss_value) # record the current loss

    return jnp.stack(params_hist), loss_hist