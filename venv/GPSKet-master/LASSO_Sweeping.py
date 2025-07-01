import numpy as np
import netket as nk
import scipy as spy

import GPSKet

from GPSKet.models import qGPS

import jax.numpy as jnp

from GPSKet.supervised.supervised_qgps import QGPSLogSpaceFit

import jax


import optax

import GPSKet.operator.hamiltonian.J1J2 as j1j2

import sklearn
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

from functools import partial
from random import randint
from random import sample


def initialise_test_system(L,M,seed,j2):
    """Initialises the system Hamiltonian and generates a variational quantum state using the qGPS model."""
    ha = j1j2.get_J1_J2_Hamiltonian(Lx=L, J2=j2, sign_rule=[True,False], on_the_fly_en=True)
    hi = ha.hilbert

    model_R = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)
    model_I = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)

    sa_R = nk.sampler.MetropolisExchange(hi, graph=ha.graph, n_chains_per_rank=1, d_max=L)
    sa_I = nk.sampler.MetropolisExchange(hi, graph=ha.graph, n_chains_per_rank=1, d_max=L)

    vs_R = nk.vqs.MCState(sa_R, model_R, n_samples=5000, chunk_size=1, seed=seed)
    vs_I = nk.vqs.MCState(sa_I, model_I, n_samples=5000, chunk_size=1, seed=seed)

    return vs_R, vs_I, ha

def generate_test_data(ha):
    """Generate the exact ground state energy, amplitudes, log amplitudes and the basis state configurations for the inputted Hamiltonian."""
    e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
    configs = jnp.array(ha.hilbert.states_to_local_indices(ha.hilbert.all_states()))
    amps = jnp.array(state.flatten())
    #ln(z) = ln(|z|) + iArg(z)
    log_amps_R = jnp.log(jnp.abs(amps))
    log_amps_I = jnp.angle(amps)

    return configs, log_amps_R, log_amps_I

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

def lasso_linear_sweeping(iterations: int, indices: list, hamiltonian, vs_R, vs_I, scaling, n_models):

    configs, log_amps_R, log_amps_I = generate_test_data(hamiltonian)    
    epsilon_R = np.array(vs_R.parameters["epsilon"])  # reset the epsilon tensor
    learning_R = QGPSLogSpaceFit(
        epsilon_R
    )  # The way of interfacing the learning model with the qGPS state
    epsilon_I = np.array(vs_I.parameters["epsilon"])  # reset the epsilon tensor
    learning_I = QGPSLogSpaceFit(
        epsilon_I
    )  # The way of interfacing the learning model with the qGPS state
    
    #Initial set of estimated log wavefunction amplitudes to be used for initial scaling
    estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)

    #Define a LASSO Learning model for each site.
    if n_models:
        lasso_models_R = []
        lasso_models_I = []
        for s in np.arange(epsilon_R.shape[-1]):
            lasso_models_R.append(LassoCV(eps=0.001, fit_intercept=False, cv=3, max_iter=5000))
            lasso_models_I.append(LinearRegression(fit_intercept=False))
    
    else:
        alpha = 10**(-4)
        lasso_models_R = Ridge(alpha= alpha, max_iter = 5000, fit_intercept= False)
        #lasso_models_R = Lasso(alpha = 10**(-7), fit_intercept=False, max_iter=5000)
        #lasso_models_R = LassoCV(alphas = [0], eps=0.00001, fit_intercept=False, cv=3, max_iter=5000)
        lasso_models_I = LinearRegression(fit_intercept=False)

    #Testing
    overlaps = []
    overlaps_R = []
    overlaps_I = []

    if type(indices) == int:
        indices = jnp.array(sample([x for x in range(len(ha.hilbert.all_states()))], indices))
    elif type(indices) == list:
        indices = jnp.array(indices)

    #Fitting Loop
    for i in range(iterations):
        
        #Running the sweeping for each different alpha to calibrate regularization
        
        for site in np.arange(epsilon_R.shape[-1]):

            learning_R.reset()
            learning_I.reset()
                
            learning_R.ref_sites = site
            learning_I.ref_sites = site
                
            if n_models:
                model_R = lasso_models_R[site]
                model_I = lasso_models_I[site]
            else:
                model_R = lasso_models_R
                model_I = lasso_models_I    

            #current_alpha = 1-jnp.exp(-i/iterations)
            #lasso_models_R.alpha = float(current_alpha*alpha)
                
                
            prior_mean = 1.0 if site != 0 else 0.0

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:
                estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)
                print(estimated_log_amps_R)
                #input()
                log_scalings = estimated_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(estimated_log_amps_R)))
                
                #- jnp.mean(estimated_log_amps_R)
                #scalings = jnp.expand_dims(jnp.exp(log_scalings)/jnp.sqrt(jnp.mean((abs(jnp.exp(log_scalings))**2))), -1)
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
                print(scalings)
                #input() #THIS ONE

                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = scalings[indices]*K_R 

                print("Overlap:" +str(overlap(estimated_log_amps_R, log_amps_R)))
                fit_data_R = log_amps_R[indices]
                #- jnp.mean(log_amps_R[indices])
                #- jnp.max(log_amps_R[indices])
                #print(fit_data_R)
                #input()

                fit_data_R = scalings[indices].flatten()*(fit_data_R)-(prior_mean*np.sum(feature_vector_R, axis=1))
                print(fit_data_R)
                #input()
                print(feature_vector_R)
                #input() #THIS ONE

                #K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices])
                #feature_vector_I = scalings[indices]*K_I
                #fit_data_I = scalings[indices].flatten()*(log_amps_I[indices]) -(prior_mean*np.sum(feature_vector_I, axis=1))
                #fit_data_I -=jnp.mean(fit_data_I)

                #TEST
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices])
                feature_vector_I = K_I
                fit_data_I = log_amps_I[indices] - prior_mean* np.sum(feature_vector_I, axis=1)
                #fit_data_I -=jnp.mean(fit_data_I)
                fit_data_I -= jnp.pi/2
            
            else:
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) 
                feature_vector_R = K_R
                feature_vector_I = K_I
                fit_data_R = log_amps_R[indices] - prior_mean* np.sum(feature_vector_R, axis=1)
                fit_data_I = log_amps_I[indices] - prior_mean* np.sum(feature_vector_I, axis=1)
                    

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights_R = model_R.fit(X=feature_vector_R, y=fit_data_R).coef_
            optimal_weights_I = model_I.fit(X=feature_vector_I, y=fit_data_I).coef_

            print(optimal_weights_R)
            #input() #THIS ONE

            #revert the prior mean adjustment above
            learning_R.weights = optimal_weights_R + prior_mean
            learning_I.weights = optimal_weights_I + prior_mean

            #Update the weights and the epsilon tensor held in the learning object.
            learning_R.valid_kern = abs(np.diag(K_R.conj().T.dot(K_R))) > learning_R.kern_cutoff
            learning_R.update_epsilon_with_weights()

            learning_I.valid_kern = abs(np.diag(K_I.conj().T.dot(K_I))) > learning_I.kern_cutoff
            learning_I.update_epsilon_with_weights()
            
            #Convert the learnt qGPS model into log wavefunction amplitudes.
        
            estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs) #Real Valued
            estimated_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs) #Real Valued


            #Recombination of the real and imaginary components of the log amplitudes.
            estimated_log_amps = estimated_log_amps_R + estimated_log_amps_I*1j #Complex Valued


        #Testing
        #estimated_log_amps = vs._apply_fun({"params": {"epsilon": learning.epsilon}}, configs)
        overlaps_R.append(overlap(estimated_log_amps_R, log_amps_R)) #Overlap of Real valued against Real valued
        overlaps_I.append(overlap(jnp.imag(estimated_log_amps), log_amps_I))

        print("estimated log amps:" +str(estimated_log_amps_R))

        for i in range(len(log_amps_R)):
            print(log_amps_R[i]-jnp.mean(log_amps_R), fit_data_R[i] - jnp.mean(fit_data_R) ,estimated_log_amps_R[i] - jnp.mean(estimated_log_amps_R))
        print("Overlap:" +str(overlap(estimated_log_amps_R, log_amps_R)))
        overlaps.append(overlap(estimated_log_amps, log_amps_R + log_amps_I))
        #input() #THIS ONE
        #print(overlap(estimated_log_amps_R, log_amps_R))
        #print(overlap(estimated_log_amps, log_amps_R + log_amps_I))
        #input()
        #alphas.append(model.alpha_)
        #features.append(n_features_removed(learning.epsilon))
        
            
    return estimated_log_amps, overlaps, overlaps_R, overlaps_I, vs_R, vs_I, log_amps_R


def n_features_removed(epsilon_tensor):
    """
    Utility function used to determine if features have been pruned by the LASSO regularization.

    Inputs:
        epsilon_tensor: (nd_array) Parametric tensor of the qGPS model taken as input.

    Returns:
        n: (int) Number of parameters within the epsilon tensor pruned by the LASSO regularization (i.e. set to null weights, 0 or 1 depending on reference site)
    """
    n=0
    for element in epsilon_tensor.flatten():
        if element==1:
            n+=1
        elif element==0:
            n+=1
    return n

def fit_polynomial(x, alpha_set: list, metric_set: list):
    """
    Interpolates a 2nd or 3rd degree polynomial fit from the data points (alpha_i, metric(alpha_i)) 
    for alpha_i in alpha_set and metric(alpha_i) in metric_set. Evaluates this polynomial for input x and returns the result.
    The choice of 2nd or 3rd degree polynomial is made to guarentee a minimum exists for x in the range [0, inf).

    Inputs:
        x: (float) A positive float, passed into the interpolated polynomial to be evaluated.
        alpha_set: (list) A list of x-coordinates used for interpolation of a 2nd or 3rd degree polynomial.
        metric_set: (list) A list of y-coordinates used for interpolation of a 2nd or 3rd degree polynomial.
    
    Returns:
        poly_x: (float) The value of the interpolated polynomial, evaluated at input x.
    
    """
    alpha_array = jnp.array(alpha_set)
    metric_array = jnp.array(metric_set)

    #Fit a polynomial to the pairs (alpha_i, metric(alpha_i))
    if metric_set[1] == max(metric_set):    
        if metric_set[0] == metric_set[1]:
            poly_coefs = jnp.polyfit(alpha_array, metric_array, 2)
        else:
            poly_coefs = jnp.polyfit(alpha_array, metric_array, 3)
    else:
        poly_coefs = jnp.polyfit(alpha_array, metric_array, 2)
    
    #Compute the polynomial for value "x".
    poly_x = 0
    for c in range(1,len(poly_coefs)+1):
        poly_x += poly_coefs[-c]*(x**(c-1))

    #Return the polynomial value, metric(alpha), at value alpha = x for use in finding the optimal alpha to update the model.
    return poly_x

def adjust_regularization(alpha_set: list, metric_set: list):
    """
    Computes the location of the minimum of the polynomial interpolated from the data points (alpha_i, metric(alpha_i)) 
    for alpha_i in alpha_set and metric(alpha_i) in metric_set. Restricted to only return positive valued results.
    If the computed minimum is identically zero, returns the difference alpha_set[-1] - alpha_set[-2], 
    or "increment" in the context of the LASSO linear sweeping algorithm.

    Inputs:
        alpha_set: (list) A list of x-coordinates used for interpolation of a 2nd or 3rd degree polynomial.
        metric_set: (list) A list of y-coordinates used for interpolation of a 2nd or 3rd degree polynomial.

    Outputs:
        new_alpha: (float) Minimum of the interpolated polynomial, subject to the constraint new_alpha>0.
    """
    #Find the minimum of the interpolated polynomial.
    min_alpha = spy.optimize.minimize(fun = fit_polynomial, x0 = alpha_set[-1], args = (alpha_set, metric_set), method='L-BFGS-B', bounds = ((0,2*alpha_set[-1]),))
    new_alpha = min(min_alpha.x)
    
    #If the minimum corresponds to zero regularization, choose to increase it by the increment anyway.
    if new_alpha ==0:
        new_alpha = alpha_set[-1] - alpha_set[-2] #The increment change

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
    
    return abs(jnp.conjugate(amps_1).dot(amps_2))

def reformat_epsilon(epsilon):
    epsilon_ = []
    for m in range(epsilon.shape[-2]):
        temp_epsilon = []
        for d in range(2):
            temp_epsilon.append([epsilon[d][m][l] for l in range(epsilon.shape[-1])])
        epsilon_.append(temp_epsilon)

    return epsilon_ 

def export_epsilon(epsilon, title):
    """
    Exported epsilon has first line (M,D,L), then all entries of the epsilon tensor. 
    When reading from file, read the first line and set up loops to read data back out into the correct lists
    """
    epsilon_ref = reformat_epsilon(epsilon)
    shape_data = epsilon_ref.shape
    with open(str(title)+'.txt', 'w') as f:
        f.write(f"{shape_data}\n")
        for element in epsilon_ref.flatten():
            f.write(f"{element}\n")

    return None

vs_R, vs_I, ha = initialise_test_system(L=12, M = 25, seed = 1, j2 = 0.8)
"""configs, log_amps_R, log_amps_I = generate_test_data(ha)
with open("AA.txt", "w") as f:
    for i in range(924):
        f.write(f"{log_amps_R[i] + 1j*log_amps_I[i]}\n")
print(len(ha.hilbert.all_states()))
print(log_amps_I)
input()"""
estimated_log_amps, o,oR,oI, vs_R, vs_I, lR = lasso_linear_sweeping(
    200,
    int(len(ha.hilbert.all_states())),
    ha,
    vs_R,
    vs_I, 
    True,
    False,
    )

with open("AR.txt", "w") as f:
    for element in estimated_log_amps:
        f.write(f"{jnp.real(element)}\n")
with open("AI.txt", "w") as f:
    for element in estimated_log_amps:
        f.write(f"{jnp.imag(element)}\n")

print(oR)
print(jnp.real(estimated_log_amps))
print(jnp.exp(lR)/jnp.linalg.norm(jnp.exp(lR))-jnp.exp(jnp.real(estimated_log_amps))/jnp.linalg.norm(jnp.exp(jnp.real(estimated_log_amps))))

#print("Real Overlap" + str(oR[-1]))
#print("Imag Overlap" +str(oI[-1]))
#print("Total Overlap" + str(o[-1]))
#print("Imag E Amps" +str(jnp.imag(estimated_log_amps)))
#print("Imag Overlap Manual" + str(overlap(jnp.imag(estimated_log_amps), lI)))
#print("Real E Amps" + str(jnp.real(estimated_log_amps)))
#print("Real Amps" + str(lR))

"""configs, lR, lI = generate_test_data(ha)
lA = lR + lI*1j
print(overlap(lR,lR))
print(overlap(lI,lI))
print(overlap(lA,lA))"""

