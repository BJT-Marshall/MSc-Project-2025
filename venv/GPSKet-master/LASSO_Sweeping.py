import numpy as np
import netket as nk

import GPSKet

from GPSKet.models import qGPS

import jax.numpy as jnp

from GPSKet.supervised.supervised_qgps import QGPSLogSpaceFit

import jax


import optax

import GPSKet.operator.hamiltonian.J1J2 as j1j2

import sklearn
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

from functools import partial
from random import randint


def initialise_system(L,M,seed):
    """Initialises the system Hamiltonian and generates a variational quantum state using the qGPS model."""
    ha = j1j2.get_J1_J2_Hamiltonian(Lx=L, J2=0.0, sign_rule=True, on_the_fly_en=True)
    hi = ha.hilbert

    model = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)

    sa = nk.sampler.MetropolisExchange(hi, graph=ha.graph, n_chains_per_rank=1, d_max=L)
    vs = nk.vqs.MCState(sa, model, n_samples=2000, chunk_size=1, seed=seed)

    return vs, ha

def generate_test_data(ha):
    """Generate the exact ground state energy, amplitudes, log amplitudes and the basis state configurations for the inputted Hamiltonian."""
    e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
    configs = jnp.array(ha.hilbert.states_to_local_indices(ha.hilbert.all_states()))
    amps = jnp.array(state.flatten())
    log_amps = jnp.log(amps * jnp.sign(amps[0]))
    #--------------------------------------------------------
    log_amps -= jnp.mean(log_amps)
    amps = jnp.exp(log_amps)
    #--------------------------------------------------------
    return e, configs, amps, log_amps

def lossfun(
    log_amps: list,
    configs: list,
    epsilon, 
    indices: list, 
    vs, 
    weight_psi_sq
    ):
    """Least Square Error loss function on the log amplitudes."""
    indices = jnp.atleast_1d(
        indices
    )  # Another reformatting step to make sure the indices are in the correct format.
    estimated_amps = jnp.exp(vs._apply_fun({"params": {"epsilon": epsilon}}, configs))
    estimated_amps = estimated_amps/jnp.linalg.norm(estimated_amps)
    sampled_log_amps = jnp.log(estimated_amps)[indices]
    # The log amplitudes, of the state parameterised by epsilon and the qGPS model, of the configurations refferred to by the indices passed into lossfun.
    # Weightings are an additional step that weight the Loss Function to be affected more by the error of prominent log amplitudes of the target state.
    # i.e. if the state was |psi> = 1/sqrt(6)|0> + sqrt(5/6)|1>, the error on the |1> term would produce a higher (worse) loss function than the same error on the |0> term. In this setup, by a factor of 5
    if weight_psi_sq:
        weightings = abs(jnp.exp(log_amps[indices])) ** 2
    # Else, the weightings are all one and have no effect
    else:
        weightings = jnp.ones(len(indices))
    # Return the loss function defined as sum_{indices sampled}(weighting(index) * |psi_(qGPS)(index) - psi_(exact)(index)|^2)
    return jnp.sum(
        weightings * jnp.abs(sampled_log_amps - (log_amps[indices])) ** 2
    )

def lasso_linear_sweeping(iterations: int, indices: list, configs: list, amps:list, log_amps: list, alpha: float, vs, scaling, n_models):
    
    epsilon = np.array(vs.parameters["epsilon"])  # reset the epsilon tensor
    learning = QGPSLogSpaceFit(
        epsilon
    )  # The way of interfacing the learning model with the qGPS state
    
    #Initial set of estimated log wavefunction amplitudes to be used for initial scaling
    estimated_log_amps = vs._apply_fun({"params": {"epsilon": learning.epsilon}}, configs)
    
    #Define a LASSO Learning model for each site.
    if n_models:
        lasso_models = []
        for s in np.arange(epsilon.shape[-1]):
            temp_model = Lasso(alpha=alpha, fit_intercept=False, warm_start=True, max_iter=5000)
            lasso_models.append(temp_model)
    else:
        lasso_models = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)

    #Fitting Loop
    for i in range(iterations):
        for site in np.arange(epsilon.shape[-1]):

            learning.reset()
            
            learning.ref_sites = site
            
            if n_models:
                model = lasso_models[site]
            else:
                model = lasso_models    
            
            #Guarenteeing a regularization of zero for the first iteration
            if i==0:
                model.alpha = 0 
                increment = 0.01
            else:
                increment = 0.1*model.get_params()["alpha"]*jnp.exp(-i/iterations) #scales with current alpha and also decreases with iterations to converge
           
            #model.alpha = float((1-jnp.exp(-i/iterations))*alpha)
            
            prior_mean = 1.0 if site != 0 else 0.0

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:

                log_scalings = estimated_log_amps/jnp.linalg.norm(estimated_log_amps)
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
                
                K=learning.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector = scalings[indices]*K

                fit_data = scalings[indices].flatten()*(log_amps[indices]) -(prior_mean*np.sum(feature_vector, axis=1))
                fit_data -=jnp.mean(fit_data)
            
            else:
                K=learning.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector = K
                fit_data = log_amps[indices] - prior_mean* np.sum(feature_vector, axis=1)
                

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights = model.fit(X=feature_vector, y=fit_data).coef_

            #revert the prior mean adjustment above
            learning.weights = optimal_weights + prior_mean

            #Update the weights and the epsilon tensor held in the learning object.
            learning.valid_kern = abs(np.diag(K.conj().T.dot(K))) > learning.kern_cutoff
            learning.update_epsilon_with_weights()
        
            #Convert the learnt qGPS model into log wavefunction amplitudes.
    
            estimated_log_amps = vs._apply_fun({"params": {"epsilon": learning.epsilon}}, configs)

            #Compute metric to automise the regularization strength 
            model.alpha = adjust_regularization(model.get_params()["alpha"], increment) #Using the getter method of the LASSO estimator to pass the current alpha into the function
            print(model.alpha)
            input()
            #THREE ELEMENT CHECK
            

            
    return estimated_log_amps, log_amps


def adjust_regularization(current_alpha: float, increment: float):
    if current_alpha - increment >0:
        low_alpha = current_alpha-increment
    else:
        low_alpha = 0
    alpha_set = [low_alpha, current_alpha, current_alpha+increment]
    metric_set = [overlap(alpha_prime) for alpha_prime in alpha_set] #inpuot metric set

    #Fit a polynomial to the pairs (alpha_i, metric(alpha_i))
    if metric_set[1] == max(metric_set):    
        if metric_set[0] == metric_set[1]:
            poly_coefs = jnp.polyfit(alpha_set, metric_set, 2)
        else:
            poly_coefs = jnp.polyfit(alpha_set, metric_set, 3)
    else:
        poly_coefs = jnp.ployfit(alpha_set, metric_set, 2)
            
    #Compute the roots of the derivative of the polynomial
    poly_derivative = jnp.polyder(poly_coefs, 1)
    if min(jnp.roots(poly_derivative))>=0:
        new_alpha = min(jnp.roots(poly_derivative))
    else:
        new_alpha = alpha_set[metric_set.index(min(metric_set))]
    if new_alpha ==0:
        new_alpha = increment

    #Return the alpha corresponding to the minimum metric from the interpolated polynomial
    return float(new_alpha)

def overlap(log_amps_1, log_amps_2):
    """
    Computes the overlap metric, |<psi_1|psi_2>|, between two sets of inputed log wavefunction amplitudes. Note that both inputed sets of data
    must be defined using the same basis and indexing convention.
    
    Inputs:
        log_amps_1: (list) A list of indexed log wavefunction amplitudes.
        log_amps_2: (list) A list of indexed log wavefunction amplitudes.

    Returns:
        Overlap: (float) The magnitude of the dot product between the two sets of wavefunction amplitudes.
    """

    #Normalises the wavefunction amplitudes.
    
    amps_1  = jnp.exp(log_amps_1)
    amps_1  = amps_1/jnp.linalg.norm(amps_1)
    amps_2 = jnp.exp(log_amps_2)
    amps_2 = amps_2/jnp.linalg.norm(amps_2)
            
    #Computes the overlap of wavefunction data provided.
    
    return abs(amps_1.T.dot(amps_2))  


def temp_metric():
    """Must be at a minimum for the best fir"""
    return 1


vs, ha = initialise_system(L=14, M = 12, seed = 1)
_, configs, amps, log_amps = generate_test_data(ha)
estimated_log_amps, o = lasso_linear_sweeping(
    50,
    jnp.atleast_1d([randint(0,len(ha.hilbert.all_states())) for x in range(0,int(0.3*len(ha.hilbert.all_states())))]), #jnp.atleast_1d(jnp.arange(180)),  #len(ha.hilbert.all_states())
    configs,
    amps, 
    log_amps, 
    3*10**-3,
    vs, 
    True,
    False
    )
