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
from sklearn.linear_model import RidgeCV

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

def ridge_sweeping(iterations: int, indices: list, alpha = 0.001 ,hamiltonian = None, vs_R = None, vs_I = None, scaling = True, n_models = False):

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
    
    estimated_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs)
    estimated_log_amps = estimated_log_amps_R + estimated_log_amps_I*1j

    if type(alpha) == int:
        alpha =[alpha,alpha]

    #Define a LASSO Learning model for each site.
    if n_models:
        ridge_models_R = []
        ridge_models_I = []
        for s in np.arange(epsilon_R.shape[-1]):
            ridge_models_R.append(Ridge(alpha= alpha[0], max_iter = 5000, fit_intercept= False))
            ridge_models_I.append(Ridge(alpha= alpha[1], max_iter = 5000, fit_intercept= False))
    
    else:
        ridge_models_R = Ridge(alpha= alpha[0], max_iter = 5000, fit_intercept= False)
        ridge_models_I = Ridge(alpha = alpha[1], max_iter = 5000, fit_intercept= False)

    #Testing
    overlaps = []
    overlaps_R = []
    overlaps_I = []

    if type(indices) == int:
        indices = jnp.array(sample([x for x in range(len(ha.hilbert.all_states()))], indices))
    elif type(indices) == list:
        indices = jnp.array(indices)

    if type(iterations) == int:
        iterations = [iterations, iterations]
    

    #Real Estimator Fitting Loop
    for i in range(iterations[0]):
        
        #Running the sweeping for each different alpha to calibrate regularization
        
        for site in np.arange(epsilon_R.shape[-1]):

            learning_R.reset()
                
            learning_R.ref_sites = site
                
            if n_models:
                model_R = ridge_models_R[site]
            else:
                model_R = ridge_models_R

                
                
            prior_mean = 1.0 if site != 0 else 0.0

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:
                estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)
                log_scalings = estimated_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(estimated_log_amps_R)))
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
            
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = scalings[indices]*K_R 
                print(feature_vector_R)
                print("feature vector")
                #input()
                
                fit_data_R = log_amps_R[indices]
                fit_data_R = scalings[indices].flatten()*(fit_data_R)-(prior_mean*np.sum(feature_vector_R, axis=1))
                print(fit_data_R)
                print("fit_data")
                #input()
            
            else:
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = K_R
                fit_data_R = log_amps_R[indices] - prior_mean* np.sum(feature_vector_R, axis=1)
                    

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights_R = model_R.fit(X=feature_vector_R, y=fit_data_R).coef_

            print(optimal_weights_R)
            print("weights")
            #input()

            #revert the prior mean adjustment above
            learning_R.weights = optimal_weights_R + prior_mean

            #Update the weights and the epsilon tensor held in the learning object.
            learning_R.valid_kern = abs(np.diag(K_R.conj().T.dot(K_R))) > learning_R.kern_cutoff
            learning_R.update_epsilon_with_weights()
            
            #Convert the learnt qGPS model into log wavefunction amplitudes.
        
            estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs) #Real Valued
            print(estimated_log_amps_R)
            print("e log amps")
            #input()
        estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs) #Real Valued
        print(estimated_log_amps_R)
        print("e log amps")
        #input()
        
        overlaps_R.append(float(overlap(estimated_log_amps_R, log_amps_R))) #Overlap of Real valued against Real valued

    #Imaginary Estimator Loop -------------------------------------------------------------------------------------------------------------------

    for i in range(iterations[1]):
        
        #Running the sweeping for each different alpha to calibrate regularization
        
        for site in np.arange(epsilon_R.shape[-1]):

            learning_I.reset()
            
            learning_I.ref_sites = site
                
            if n_models:
                model_I = ridge_models_I[site]
            else:
                model_I = ridge_models_I    

                
                
            prior_mean = 1.0 if site != 0 else 0.0

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:
                estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)
                log_scalings = estimated_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(estimated_log_amps_R)))
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
            
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method 
                feature_vector_I = scalings[indices]*K_I 

                
                fit_data_I = log_amps_I[indices]
                phase_shift = jnp.mean(fit_data_I)
                fit_data_I -=phase_shift
                fit_data_I = scalings[indices].flatten()*(fit_data_I)-(prior_mean*np.sum(feature_vector_I, axis=1))

                """#TEST
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices])
                feature_vector_I = K_I
                fit_data_I = log_amps_I[indices] - prior_mean* np.sum(feature_vector_I, axis=1)
                #fit_data_I -=jnp.mean(fit_data_I)
                fit_data_I -= jnp.pi/2"""
            
            else:
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_I = K_I
                fit_data_I = log_amps_I[indices] - prior_mean* np.sum(feature_vector_I, axis=1)
                    

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights_I = model_I.fit(X=feature_vector_I, y=fit_data_I).coef_

            print(optimal_weights_I)
            #input()

            #revert the prior mean adjustment above
            learning_I.weights = optimal_weights_I + prior_mean

            #Update the weights and the epsilon tensor held in the learning object.

            learning_I.valid_kern = abs(np.diag(K_I.conj().T.dot(K_I))) > learning_I.kern_cutoff
            learning_I.update_epsilon_with_weights()
            
            #Convert the learnt qGPS model into log wavefunction amplitudes.
            estimated_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs) #Real Valued
            estimated_log_amps_I +=phase_shift

            #Recombination of the real and imaginary components of the log amplitudes.
            estimated_log_amps = estimated_log_amps_R + estimated_log_amps_I*1j #Complex Valued


        #Testing
        #estimated_log_amps = vs._apply_fun({"params": {"epsilon": learning.epsilon}}, configs)
        overlaps_I.append(float(overlap(estimated_log_amps_I, log_amps_I)))
        overlaps.append(float(overlap(estimated_log_amps, log_amps_R + log_amps_I*1j)))
            
    return estimated_log_amps, overlaps, overlaps_R, overlaps_I, vs_R, vs_I, learning_R.epsilon, learning_I.epsilon, log_amps_R + log_amps_I*1j, configs


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

def lasso_sweeping(iterations, indices, ha, scaling, vs_R, vs_I, epsilon_R, epsilon_I, log_amps, configs): 

    learning_R = QGPSLogSpaceFit(
        epsilon_R
    )  # The way of interfacing the learning model with the qGPS state

    learning_I = QGPSLogSpaceFit(
        epsilon_I
    ) # The way of interfacing the learning model with the qGPS state
    
    #Sets of real and imaginary training data
    log_amps_R = jnp.real(log_amps)
    log_amps_I = jnp.imag(log_amps)   

    #Estimated log amplitudes from the current qGPS model state
    e_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)
    e_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs)
    #print(e_log_amps)
    #input()
    
    #LASSO learning models
    lasso_model_I = Lasso(alpha = 0.00000001, fit_intercept=False, max_iter=5000)
    lasso_model_R = Lasso(alpha = 0.000001, fit_intercept=False, max_iter=5000)
    
    #Handling the indices input to be used as training data
    if type(indices) == int:
        indices = jnp.array(sample([x for x in range(len(ha.hilbert.all_states()))], indices))
    elif type(indices) == list:
        indices = jnp.array(indices)

    if type(iterations) == int:
        iterations = [iterations, iterations]

    ov = []
    ov_R=[]
    ov_I=[]
    feat = []
    alphas =[]

    #Real Fitting Loop
    for i in range(iterations[0]):
        
        #Running the sweeping for each different alpha to calibrate regularization
        
        for site in np.arange(epsilon_R.shape[-1]):

            learning_R.reset()
                
            learning_R.ref_sites = site
                
            model_R = lasso_model_R
                
                
            prior_mean = 1.0 if site != 0 else 0.0

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:
                e_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)

                #Scalings that dictate the relative importance of fitting each amplitude based on its amplitude 
                log_scalings = e_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(e_log_amps_R)))
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)

                #Scaling of the kernel matrix and fit data vector for both sets of training data
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = scalings[indices]*K_R 

                fit_data_R = log_amps_R[indices]

                fit_data_R = scalings[indices].flatten()*(fit_data_R)-(prior_mean*np.sum(feature_vector_R, axis=1))
            
            else:
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = K_R
                fit_data_R = log_amps_R[indices]
                    

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights_R = model_R.fit(X=feature_vector_R, y=fit_data_R).coef_

            #revert the prior mean adjustment above
            learning_R.weights = optimal_weights_R + prior_mean
            #global_weight = jnp.max(abs(optimal_weights_R))
            #print(global_weight)
            #input()

            #if global_weight == 0:
                #learning_R.weights = jnp.array([1 for x in range(len(optimal_weights_R))])
            #else:
                #learning_R.weights = optimal_weights_R/global_weight

            #print(learning_R.weights)
            #input()

            #Update the weights and the epsilon tensor held in the learning object.
            learning_R.valid_kern = abs(np.diag(K_R.conj().T.dot(K_R))) > learning_R.kern_cutoff
            learning_R.update_epsilon_with_weights()

            
            #Convert the learnt qGPS model into log wavefunction amplitudes.
            e_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs) #Real Valued


    for i in range(iterations[1]):
            #Running the sweeping for each different alpha to calibrate regularization
        
        for site in np.arange(epsilon_R.shape[-1]):

            learning_I.reset()
                
            learning_I.ref_sites = site
                
            model_I = lasso_model_I
                
                
            prior_mean = 1.0 if site != 0 else 0.0

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:

                #Scalings that dictate the relative importance of fitting each amplitude based on its amplitude 
                log_scalings = e_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(e_log_amps_R)))
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)

                #Scaling of the kernel matrix and fit data vector for both sets of training data
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_I = scalings[indices]*K_I 
                
                print(log_amps_I)
                #input()
                fit_data_I = log_amps_I[indices]
                print(fit_data_I)
                #input()


                fit_data_I = scalings[indices].flatten()*(fit_data_I)-(prior_mean*np.sum(feature_vector_I, axis=1))
                #fit_data_I = fit_data_I*jnp.pi/jnp.max(fit_data_I)
                print(fit_data_I)
                #input()
            
            else: 
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_I = K_I
                fit_data_I = log_amps_I[indices]- jnp.pi/2 
                    

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights_I = model_I.fit(X=feature_vector_I, y=fit_data_I).coef_


            #revert the prior mean adjustment above
            learning_I.weights = optimal_weights_I + prior_mean
            #global_weight = jnp.max(abs(optimal_weights_R))
            #print(global_weight)
            #input()

            #if global_weight == 0:
                #learning_R.weights = jnp.array([1 for x in range(len(optimal_weights_R))])
            #else:
                #learning_R.weights = optimal_weights_R/global_weight

            #print(learning_R.weights)
            #input()

            #Update the weights and the epsilon tensor held in the learning object.
            learning_I.valid_kern = abs(np.diag(K_I.conj().T.dot(K_I))) > learning_I.kern_cutoff
            learning_I.update_epsilon_with_weights()

            
            #Convert the learnt qGPS model into log wavefunction amplitudes.

            e_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs) #Real Valued

        #for i in range(len(log_amps_R)):
            #print(log_amps_R[i]-jnp.mean(log_amps_R), fit_data_R[i] - jnp.mean(fit_data_R) ,e_log_amps_R[i] - jnp.mean(e_log_amps_R))
        e_log_amps = e_log_amps_R + e_log_amps_I*1j
        #print("Overlap:" +str(overlap(e_log_amps, log_amps)))
        #input()
        ov.append(overlap(e_log_amps, log_amps))
        ov_R.append(overlap(e_log_amps_R, log_amps_R))
        ov_I.append(overlap(e_log_amps_I, fit_data_I))
        feat.append(n_features_removed(learning_R.epsilon))
        alphas.append(model_R.alpha)
        print(e_log_amps_I)
        #input()
        

    return e_log_amps, ov, ov_R, ov_I, feat, alphas, fit_data_I



vs_R, vs_I, ha = initialise_test_system(L=8, M = 25, seed = 1, j2 = 0.6)
"""configs, log_amps_R, log_amps_I = generate_test_data(ha)
with open("AA.txt", "w") as f:
    for i in range(len(ha.hilbert.all_states())):
        f.write(f"{log_amps_R[i] + 1j*log_amps_I[i]}\n")
print(len(ha.hilbert.all_states()))
print(log_amps_I)
print(log_amps_R)
input()"""

estimated_log_amps, o,oR,oI, vs_R, vs_I, epsilon_R, epsilon_I, log_amps, configs= ridge_sweeping(
        [50,50],
        int(len(ha.hilbert.all_states())),
        [0.1,0],
        ha,
        vs_R,
        vs_I, 
        True,
        False,
        )
with open("AA.txt", "w") as f:
    for i in range(len(ha.hilbert.all_states())):
        f.write(f"{complex(log_amps[i])}\n")
with open("FEALFIT.txt", "w") as f:
    for element in log_amps:
        f.write(f"{float(jnp.real(element))}\n")
with open("REALE.txt", "w") as f:
    for element in estimated_log_amps:
        f.write(f"{jnp.real(complex(element))}\n")
with open("IMAGFIT.txt", "w") as f:
    for element in log_amps:
        f.write(f"{float(jnp.imag(element))}\n")
with open("IMAGE.txt", "w") as f:
    for element in estimated_log_amps:
        f.write(f"{jnp.imag(complex(element))}\n")

print(oR)
print(oI)
print(o)
print(len(ha.hilbert.all_states()))
input()
print(jnp.imag(log_amps))
input()

e_log_amps, ov, ov_R, ov_I, feat, alphas, f_I = lasso_sweeping(
        [50,100],
        int(len(ha.hilbert.all_states())),
        ha,
        True,
        vs_R,
        vs_I,
        epsilon_R,
        epsilon_I,
        log_amps,
        configs
        )

print(ov)
print(ov_R)
print(ov_I)

with open("IMAGFIT.txt", "w") as f:
    for element in f_I:
        f.write(f"{float(element)}\n")
with open("IMAGE.txt", "w") as f:
    for element in e_log_amps:
        f.write(f"{jnp.imag(complex(element))}\n")

input()

overlaps = []
#for iteration_set in [[10,[50,50]],[10,[100,100]],[20,[50,50]],[20,[100,100]],[30,[50,50]],[30,[100,100]]]:
#for iteration_set in [[0,[50,50]],[0,[100,100]]]:
for iteration_set in [[10,[50,100]], [100,[50,200]]]:

    estimated_log_amps, o,oR,oI, vs_R, vs_I, epsilon_R, epsilon_I, log_amps, configs= ridge_sweeping(
        iteration_set[0],
        int(len(ha.hilbert.all_states())),
        0.001,
        ha,
        vs_R,
        vs_I, 
        True,
        False,
        )

    e_log_amps, ov, ov_R, ov_I, feat, alphas = lasso_sweeping(
        iteration_set[1],
        int(len(ha.hilbert.all_states())),
        ha,
        True,
        vs_R,
        vs_I,
        epsilon_R,
        epsilon_I,
        log_amps,
        configs
        )
    overlaps.append([overlap(estimated_log_amps,log_amps),overlap(e_log_amps,log_amps),ov_R[-1], ov_I[-1]])
    with open("SIGNAMPS"+str(iteration_set)+".txt", "w") as f:
        for element in e_log_amps:
            f.write(f"{float(jnp.imag(element))}\n")
    with open("TESTROSIGN1"+str(iteration_set)+".txt", "w") as f:
        for element in o:
            f.write(f"{float(element)}\n")
    with open("TESTLOSIGN1"+str(iteration_set)+".txt", "w") as f:
        for element in ov:
            f.write(f"{float(element)}\n")


print([[10,50],[10,100],[20,50],[20,100],[30,50],[30,100]])
print(overlaps)
#input()

"""with open("AR.txt", "w") as f:
    for element in estimated_log_amps:
        f.write(f"{jnp.real(element)}\n")
with open("AI.txt", "w") as f:
    for element in estimated_log_amps:
        f.write(f"{jnp.imag(element)}\n")

print(oR)
print(jnp.real(estimated_log_amps))
#print(jnp.exp(lR)/jnp.linalg.norm(jnp.exp(lR))-jnp.exp(jnp.real(estimated_log_amps))/jnp.linalg.norm(jnp.exp(jnp.real(estimated_log_amps))))

print(ov)
print(feat)
print(alphas)
with open("ARLASSO.txt", "w") as f:
    for element in e_log_amps:
        f.write(f"{jnp.real(element)}\n")"""
