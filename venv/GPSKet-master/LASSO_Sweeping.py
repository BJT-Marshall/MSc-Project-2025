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


def initialise_test_system(L,j2):
    """Initialises the system Hamiltonian and generates a variational quantum state using the qGPS model."""
    ha = j1j2.get_J1_J2_Hamiltonian(Lx=L, J2=j2, sign_rule=[True,False], on_the_fly_en=True)

    return ha

def generate_test_data(ha):
    """Generate the exact ground state energy, amplitudes, log amplitudes and the basis state configurations for the inputted Hamiltonian."""
    e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
    amps = jnp.array(state.flatten())
    #ln(z) = ln(|z|) + iArg(z)
    log_amps_R = jnp.log(jnp.abs(amps))
    log_amps_I = jnp.angle(amps)

    return log_amps_R, log_amps_I

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
                
                fit_data_R = log_amps_R[indices]
                mean_data_R = ((jnp.exp(log_amps_R)/jnp.linalg.norm(jnp.exp(log_amps_R)))**2).flatten()*fit_data_R
                fit_data_R -=jnp.sum(mean_data_R)
                fit_data_R = scalings[indices].flatten()*(fit_data_R)
                fit_data_R -=(prior_mean*np.sum(feature_vector_R, axis=1))
            
            else:
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = K_R
                fit_data_R = log_amps_R[indices]
                mean_data_R = ((jnp.exp(log_amps_R)/jnp.linalg.norm(jnp.exp(log_amps_R)))**2).flatten()*fit_data_R
                fit_data_R -=jnp.sum(mean_data_R)- prior_mean* np.sum(feature_vector_R, axis=1)
                    

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
        print(float(overlap(estimated_log_amps_R, log_amps_R)))
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
                phase_shift = jnp.sum(((jnp.exp(log_amps_I)/jnp.linalg.norm(jnp.exp(log_amps_I)))**2).flatten()*fit_data_I)
                fit_data_I -=phase_shift
                fit_data_I = scalings[indices].flatten()*(fit_data_I)-(prior_mean*np.sum(feature_vector_I, axis=1))

            
            else:
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_I = K_I
                phase_shift = jnp.sum(((jnp.exp(log_amps_I)/jnp.linalg.norm(jnp.exp(log_amps_I)))**2).flatten()*fit_data_I)
                fit_data_I = log_amps_I[indices] 
                fit_data_I -= phase_shift
                fit_data_I -= prior_mean* np.sum(feature_vector_I, axis=1)
                    

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
        if element==0:
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
        for l in range(epsilon.shape[-1]):
            temp_epsilon.append([epsilon[d][m][l] for d in range(epsilon.shape[0])])
        epsilon_.append(temp_epsilon)

    return epsilon_ 

def undo_reformat_epsilon(epsilon):
    epsilon_ = []
    epsilon =jnp.array(epsilon)
    for d in range(epsilon.shape[-1]):
        temp_epsilon = []
        for m in range(epsilon.shape[0]):
            temp_epsilon.append([epsilon[m][l][d] for l in range(epsilon.shape[-2])])
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

def lasso_sweeping(iterations, alpha, indices, ha, M, scaling, log_amps, seed):

    model_R = qGPS(ha.hilbert, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)
    model_I = qGPS(ha.hilbert, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)

    sa_R = nk.sampler.MetropolisExchange(ha.hilbert, graph=ha.graph, n_chains_per_rank = 50)
    sa_I = nk.sampler.MetropolisExchange(ha.hilbert, graph=ha.graph, n_chains_per_rank = 50)

    vs_R = nk.vqs.MCState(sa_R, model_R, n_samples=5000, seed=seed)
    vs_I = nk.vqs.MCState(sa_I, model_I, n_samples=5000, seed=seed)

    #Local state configurations
    configs = jnp.array(ha.hilbert.states_to_local_indices(ha.hilbert.all_states()))
    
    epsilon_R = np.array(vs_R.parameters["epsilon"])  # reset the epsilon tensor
    learning_R = QGPSLogSpaceFit(
        epsilon_R
    )  # The way of interfacing the learning model with the qGPS state

    epsilon_I = np.array(vs_I.parameters["epsilon"])  # reset the epsilon tensor
    learning_I = QGPSLogSpaceFit(
        epsilon_I
    ) # The way of interfacing the learning model with the qGPS state
    
    #Dividing training data into real and imaginary sets
    log_amps_R = jnp.real(log_amps)
    log_amps_I = jnp.imag(log_amps)   

    #Estimated log magnitudes from the current qGPS model state used for initial scaling
    e_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)

    #Handling the indices input to be used as training data
    if type(indices) == int:
        indices = jnp.array(sample([x for x in range(len(ha.hilbert.all_states()))], indices))
    elif type(indices) == list:
        indices = jnp.array(indices)

    #If alpha is given, both models are LASSO models with the inputted alphas. If alpha is "CV", both models are LASSOCV models with automatic cross validation.
    if type(alpha) == float:
        alpha = [alpha,alpha]
        #LASSO learning models
        lasso_model_R = Lasso(alpha = alpha[0], fit_intercept=False, max_iter=5000)
        lasso_model_I = Lasso(alpha = alpha[1], fit_intercept=False, max_iter=5000)
    elif type(alpha) == list:
        #LASSO learning models
        lasso_model_R = Lasso(alpha = alpha[0], fit_intercept=False, max_iter=5000)
        lasso_model_I = Lasso(alpha = alpha[1], fit_intercept=False, max_iter=5000)
    elif alpha == "CV":
        lasso_model_R = LassoCV(fit_intercept=False, max_iter=5000)
        lasso_model_I = LassoCV(fit_intercept=False, max_iter=5000)
    elif alpha == None:
        lasso_model_R = Lasso(alpha = 1/(((M)**4)), fit_intercept=False, max_iter=5000) #*(len(indices)**2)
        lasso_model_I = Lasso(alpha = 1/(((M)**4)), fit_intercept=False, max_iter=5000)
    

    if type(iterations) == int:
        iterations = [iterations, iterations]

    #Testing Data
    ov = []
    ov_R=[]
    ov_I=[]
    feat_R = []
    feat_I = []
    alphas =[]

    #Real Fitting Loop
    for i in range(iterations[0]):
        
        #Running the sweeping for each different alpha to calibrate regularization
        
        for site in np.arange(epsilon_R.shape[-1]):

            learning_R.reset()
                
            learning_R.ref_sites = site
                
            model_R = lasso_model_R

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:
                estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs[indices])
                log_scalings = estimated_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(estimated_log_amps_R)))
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
            
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = scalings*K_R
                
                fit_data_R = log_amps_R[indices]
                mean_data_R = ((jnp.exp(log_amps_R[indices])/jnp.linalg.norm(jnp.exp(log_amps_R[indices])))**2).flatten()*fit_data_R
                fit_data_R -=jnp.sum(mean_data_R)
                fit_data_R = scalings.flatten()*(fit_data_R)
            
            else:
                K_R=learning_R.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_R = K_R
                fit_data_R = log_amps_R[indices]
                mean_data_R = ((jnp.exp(log_amps_R[indices])/jnp.linalg.norm(jnp.exp(log_amps_R[indices])))**2).flatten()*fit_data_R
                fit_data_R -=jnp.sum(mean_data_R)
                    
            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights_R = model_R.fit(X=feature_vector_R, y=fit_data_R).coef_

            #Calculate Rescalings to pull out of the epsilon tensor
            #Rescalings to adjust the scalings of the epsilon parameters
            m_ = int(len(optimal_weights_R)/2)
            odd_indices = [2*n for n in range(m_)]
            rescalings_R = [max([abs(optimal_weights_R[i]), abs(optimal_weights_R[i+1])]) for i in odd_indices]
            for index in odd_indices:
                if rescalings_R[int(index/2)] == 0:
                    optimal_weights_R[index] = 1
                    optimal_weights_R[index+1] = 1
                else:
                    optimal_weights_R[index] = optimal_weights_R[index]/rescalings_R[int(index/2)]
                    optimal_weights_R[index+1] = optimal_weights_R[index+1]/rescalings_R[int(index/2)]
            learning_R.weights = jnp.array(optimal_weights_R)
            

            #Update the weights and the epsilon tensor held in the learning object.
            learning_R.valid_kern = abs(np.diag(K_R.conj().T.dot(K_R))) > learning_R.kern_cutoff
            learning_R.update_epsilon_with_weights()

            #Convert the learnt qGPS model into log wavefunction amplitudes.
            #Re-extract these weights for predicting log amplitudes
            
            rescaled_epsilon_R = jnp.array([element for element in jnp.array(learning_R.epsilon).flatten()]).reshape(learning_R.epsilon.shape[0],learning_R.epsilon.shape[1],learning_R.epsilon.shape[2])
            for m in range(rescaled_epsilon_R.shape[1]):
                for d in range(rescaled_epsilon_R.shape[0]):
                    #For the current reference site multiply back in the scalings
                    rescaled_epsilon_R = rescaled_epsilon_R.at[d,m,site].multiply(rescalings_R[m])

            
            #Seperate learning object just to hold the rescaled epsilon tnesor for predicting log amplitudes
            learning_R_pred = QGPSLogSpaceFit(
                jnp.array(rescaled_epsilon_R)
                )  # The way of interfacing the learning model with the qGPS state

            #Predict log amplitudes from rescaled epislon
            e_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R_pred.epsilon}}, configs) #Real Valued

            print(optimal_weights_R)
            print(e_log_amps_R)
            #input()
        
        feat_R.append(n_features_removed(learning_R_pred.epsilon))
        ov_R.append(overlap(e_log_amps_R, log_amps_R))
        print("overlap:")
        e_pred_R = vs_R._apply_fun({"params": {"epsilon": learning_R_pred.epsilon}}, configs[indices]) #Real Valued
        print(overlap(e_pred_R, log_amps_R[indices]))
        #input()

    for i in range(iterations[1]):
        
        for site in np.arange(epsilon_R.shape[-1]):

            learning_I.reset()
                
            learning_I.ref_sites = site
                
            model_I = lasso_model_I
                
                

            #if flag: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:
                estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs)
                log_scalings = estimated_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(estimated_log_amps_R)))
                scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
            
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method 
                feature_vector_I = scalings[indices]*K_I 

                
                fit_data_I = log_amps_I[indices]
                phase_shift = jnp.sum(((jnp.exp(log_amps_I[indices])/jnp.linalg.norm(jnp.exp(log_amps_I[indices])))**2).flatten()*fit_data_I)
                fit_data_I -=phase_shift
                fit_data_I = scalings[indices].flatten()*(fit_data_I) #-(prior_mean*np.sum(feature_vector_I, axis=1))

            
            else:
                K_I=learning_I.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector_I = K_I
                phase_shift = jnp.sum(((jnp.exp(log_amps_I[indices])/jnp.linalg.norm(jnp.exp(log_amps_I[indices])))**2).flatten()*fit_data_I)
                fit_data_I = log_amps_I[indices] 
                fit_data_I -= phase_shift
                    

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            optimal_weights_I = model_I.fit(X=feature_vector_I, y=fit_data_I).coef_


            #Calculate Rescalings to pull out of the epsilon tensor
            #Rescalings to adjust the scalings of the epsilon parameters
            m_ = int(len(optimal_weights_I)/2)
            odd_indices = [2*n for n in range(m_)]
            rescalings_I = [max([abs(optimal_weights_I[i]), abs(optimal_weights_I[i+1])]) for i in odd_indices]
            for index in odd_indices:
                if rescalings_I[int(index/2)] == 0:
                    optimal_weights_I[index] = 1
                    optimal_weights_I[index+1] = 1
                else:
                    optimal_weights_I[index] = optimal_weights_I[index]/rescalings_I[int(index/2)]
                    optimal_weights_I[index+1] = optimal_weights_I[index+1]/rescalings_I[int(index/2)]
            learning_I.weights = jnp.array(optimal_weights_I)
    
            #Update the weights and the epsilon tensor held in the learning object.
            learning_I.valid_kern = abs(np.diag(K_I.conj().T.dot(K_I))) > learning_I.kern_cutoff
            learning_I.update_epsilon_with_weights()

            
            #Convert the learnt qGPS model into log wavefunction amplitudes.
            #Re-extract these weights for predicting log amplitudes
            
            rescaled_epsilon_I = jnp.array([element for element in jnp.array(learning_I.epsilon).flatten()]).reshape(learning_I.epsilon.shape[0],learning_I.epsilon.shape[1],learning_I.epsilon.shape[2])
            for m in range(rescaled_epsilon_I.shape[1]):
                for d in range(rescaled_epsilon_I.shape[0]):
                    #For the current reference site multiply back in the scalings
                    rescaled_epsilon_I = rescaled_epsilon_I.at[d,m,site].multiply(rescalings_I[m])


            #Seperate learning object just to hold the rescaled epsilon tnesor for predicting log amplitudes
            learning_I_pred = QGPSLogSpaceFit(
                jnp.array(rescaled_epsilon_I)
                )  # The way of interfacing the learning model with the qGPS state

            #Predict log amplitudes from rescaled epislon
            e_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I_pred.epsilon}}, configs) #Real Valued
            #e_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs) #Real Valued
            e_log_amps_I +=phase_shift

        e_log_amps = e_log_amps_R + e_log_amps_I*1j
        
        #Testing Data
        ov.append(overlap(e_log_amps, log_amps))
        ov_I.append(overlap(e_log_amps_I, log_amps_I))
        feat_I.append(n_features_removed(learning_I_pred.epsilon))
        #alphas.append(model_R.alpha_)
        

    return e_log_amps, ov, ov_R, ov_I, feat_R, feat_I, alphas, learning_R_pred.epsilon, learning_I_pred.epsilon, indices


#Jastrow Data Test:---------------------------------------------------------------------------------------------------------------
"""overlaps_test = []
for a in [-5]:
    m=100
    ha = initialise_test_system(L=12, j2 = 0.8)
    lR, lI = generate_test_data(ha)
    
    log_amps = []
    for i in range(len(lR)):
        log_amps.append(lR[i] + lI[i]*1j)

    log_amps = []
    with open("Jastrow.txt") as f:
        for i in range(len(ha.hilbert.all_states())):
            log_amps.append(complex(f.readline()))

    log_amps = jnp.array(log_amps)

    e_log_amps, ov, ov_R, ov_I, feat_R, feat_I, alphas, e_R, e_I = lasso_sweeping(
            [10,10],
            [10**(a),10**(a)], #[10**(a),10**(a)]
            int(len(ha.hilbert.all_states())),
            ha,
            m,
            True,
            log_amps,
            1
            )
    overlaps_test.append(ov[-1])

with open("TESTING2.txt", "w") as f:
    for element in overlaps_test:
        f.write(f"{float(element)}\n")

print(ov_R)
#print(ov_I)
print(ov)
print("Real Epsilon")
print(e_R)
print("Imaginary Epsilon")
print(e_I)
print(feat_R)
print(feat_I)

with open("AA.txt", "w") as f:
    for i in range(len(ha.hilbert.all_states())):
        f.write(f"{complex(log_amps[i])}\n")
with open("FEALFIT.txt", "w") as f:
    for element in log_amps:
        f.write(f"{float(jnp.real(element))}\n")
with open("REALE.txt", "w") as f:
    for element in e_log_amps:
        f.write(f"{jnp.real(complex(element))}\n")
with open("IMAGFIT.txt", "w") as f:
    for element in log_amps:
        f.write(f"{float(jnp.imag(element))}\n")
with open("IMAGE.txt", "w") as f:
    for element in e_log_amps:
        f.write(f"{jnp.imag(complex(element))}\n")
with open("FEATURES_R.txt", "w") as f:
    for element in feat_R:
        f.write(f"{int(element)}\n")
with open("FEATURES_I.txt", "w") as f:
    for element in feat_I:
        f.write(f"{int(element)}\n")

#First L Elements are D=0, Second L Elements are D=1 in each of the M files created

for m_temp in range(m):
    with open("e_R"+str(m_temp)+".txt", "w") as f:
        for element in jnp.array(reformat_epsilon(e_R)[m_temp]).flatten():
            f.write(f"{float(element)}\n")
    with open("e_I"+str(m_temp)+".txt", "w") as f:
        for element in jnp.array(reformat_epsilon(e_I)[m_temp]).flatten():
            f.write(f"{float(element)}\n")"""


