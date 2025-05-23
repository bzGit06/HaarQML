import numpy as np
from distance_jax import sinkhornDistance, avgStateFid_pure, \
    avgStateSupFid_id, avgStateFid_id, avgStateSubFid_id

import jax
import jax.numpy as jnp
from jax import random
import optax

import time
import datetime
import os


def TrainingAvgState_t(model, t, tau, inputs_0, psi0, params_tot=[], epochs=1001, dist_name='sup'):
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

    if dist_name == 'sup':
        def loss_func(params_t, psi):
            seed = int(1e6 * datetime.datetime.now().timestamp())
            key = jax.random.PRNGKey(seed)

            key, subkey = jax.random.split(key)
            input_t_1 = model.prepareInput_t(inputs_0, params_tot, t)
            output_1 = model.pQCoutput(input_t_1, params_t, subkey)
            loss1 = 1. - avgStateFid_pure(output_1, psi)
            
            _, subkey = jax.random.split(key)
            input_t_2 = model.prepareInput_t(inputs_0, params_tot, t)
            output_2 = model.pQCoutput(input_t_2, params_t, subkey)
            model.current_states = output_2
            loss2 = 1. - avgStateSupFid_id(output_2)

            return (1. - tau) * loss1 + tau * loss2, output_2
        
    elif dist_name == 'fid':
        def loss_func(params_t, psi):
            seed = int(1e6 * datetime.datetime.now().timestamp())
            key = jax.random.PRNGKey(seed)

            key, subkey = jax.random.split(key)
            input_t_1 = model.prepareInput_t(inputs_0, params_tot, t)
            output_1 = model.pQCoutput(input_t_1, params_t, subkey)
            loss1 = 1. - avgStateFid_pure(output_1, psi)
            
            _, subkey = jax.random.split(key)
            input_t_2 = model.prepareInput_t(inputs_0, params_tot, t)
            output_2 = model.pQCoutput(input_t_2, params_t, subkey)
            model.current_states = output_2
            loss2 = 1. - avgStateFid_id(output_2)

            return (1. - tau) * loss1 + tau * loss2, output_2
        
    else:
        def loss_func(params_t, psi):
            seed = int(1e6 * datetime.datetime.now().timestamp())
            key = jax.random.PRNGKey(seed)

            key, subkey = jax.random.split(key)
            input_t_1 = model.prepareInput_t(inputs_0, params_tot, t)
            output_1 = model.pQCoutput(input_t_1, params_t, subkey)
            loss1 = 1. - avgStateFid_pure(output_1, psi)
            
            _, subkey = jax.random.split(key)
            input_t_2 = model.prepareInput_t(inputs_0, params_tot, t)
            output_2 = model.pQCoutput(input_t_2, params_t, subkey)
            model.current_states = output_2
            loss2 = 1. - avgStateSubFid_id(output_2)

            return (1. - tau) * loss1 + tau * loss2, output_2

    loss_func_vg = jax.jit(jax.value_and_grad(loss_func, has_aux=True))
    #@partial(jax.jit, static_argnums=(2, ))
    def update(params_t, psi, opt_state):
        (loss_value, new_states), grads = loss_func_vg(params_t, psi)

        updates, new_opt_state = optimizer.update(grads, opt_state, params_t)
        new_params_t = optax.apply_updates(params_t, updates)

        return new_params_t, new_opt_state, loss_value, new_states

    t0 = time.time()
    for step in range(epochs):
        if step % (epochs//50) == 0:
            params_hist.append(params_t)

        params_t, opt_state, loss_value, states_t = update(params_t, psi0, opt_state)
        loss_hist.append(loss_value) # record the current loss
        
        if step % 1000 == 0:
            states_hist.append(states_t)
            print("Step {}, loss: {:.7f}, time elapsed: {:.4f} seconds".format(step, loss_value, time.time() - t0))
        
    return jnp.stack(params_hist), loss_hist, np.stack(states_hist)


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
                
