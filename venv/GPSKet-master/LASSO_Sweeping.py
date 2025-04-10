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

import matplotlib.pyplot as plt

from functools import partial


def initialise_system(L,M,seed):
    """Initialises the system Hamiltonian and generates a variational quantum state using the qGPS model."""
    ha = j1j2.get_J1_J2_Hamiltonian(Lx=L, J2=0.0, sign_rule=True, on_the_fly_en=True)
    hi = ha.hilbert

    model = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)

    sa = nk.sampler.MetropolisExchange(hi, graph=ha.graph, n_chains_per_rank=1, d_max=L)
    vs = nk.vqs.MCState(sa, model, n_samples=10*L*M**2, chunk_size=1, seed=seed)

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

def lasso_linear_sweeping(tol: float, iterations: int, indices: list, configs: list, amps:list, log_amps: list, alpha: float, vs, ha, weighted_according_to_psi_squared):
    """Lasso linear sweeping model for qGPS model to the ground state computed at the top of this code. 
    THIS MUST BE USED IN CONJUCTION WITH initialise_system AND generate_test_data TO MAKE SURE YOURE FITTING THE CORRECT DATA TO THE CORRECT DATA"""
    epsilon = np.array(vs.parameters["epsilon"])  # reset the epsilon tensor
    learning = QGPSLogSpaceFit(
        epsilon
    )  # The way of interfacing the learning model with the qGPS state

    #Testing Scaling the tolerane flag of the LASSO model to the mean kernel value. tol ~ mean E-4
    """learning.ref_sites = 1
    K=learning.set_kernel_mat(update_K=True, confs=configs[indices])
    mean_K = jnp.mean(K)"""

    #Define LASSO Model
    model = Lasso(alpha=alpha, fit_intercept=False, tol = tol) #alpha is the 'lambda' parameter, defines the L1 penalty the model uses. REGULARISATION

    #Adjust the log_amps data set
    overlaps_iter = []
    for i in range(iterations):
    #ALTER THIS TO RAMP UP THE REGULARISATION AS THE ITERATIONS GO ON #Test Exponential or Linear
        #if i == 0:
            #model.alpha = 0.0
        #else:
            #model.alpha = alpha
        
        #Ramp up the regularization through the iterations

        if i == 0:
            model.alpha = 0.0
        else:
          model.alpha = float((1-jnp.exp(-i/iterations))*alpha) #rampupdata test 1
          #model.alpha = float(jnp.exp(jnp.log(alpha+1)*i/iterations)-1) #rampupdata test 2
          #model.alpha = float(alpha*i/iterations) #rampupdata test 3
          #model.alpha = float(1/2*((1-jnp.exp(-i/iterations))*alpha + jnp.exp(jnp.log(alpha+1)*i/iterations)-1)) #rampupdata test 4
        
        """if i<=int(iterations/10):
            model.alpha = 0.0
        else:
            model.alpha = ((10*i)/(9*iterations)-1/9)*alpha"""
        

        for site in np.arange(ha.hilbert.size):  # For each single site perform an optimisation cycle
            
            learning.reset()
            
            learning.ref_sites = site

            #Setting the feature vector to either be the kernel function generated by the 'set_kernel_mat' method, or the kernel function wieghted by 
            #the square amplitude of the exact fit data. Determined by the 'weighted_according_to_psi_squared' flag.
            
            prior_mean = 1.0 if site != 0 else 0.0

            #target data and feature vector both individually scaled by |psi|
            if weighted_according_to_psi_squared:
                weightings = jnp.expand_dims(jnp.abs(amps), -1)
                #weightings = weightings/max(weightings)
                
                K=learning.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                
                feature_vector = weightings[indices]*K
                temp_log_amps = log_amps

                fit_data = weightings[indices].flatten()*(temp_log_amps[indices] - prior_mean*np.sum(feature_vector, axis=1))
            else:
                K=learning.set_kernel_mat(update_K=True, confs=configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector = K
                fit_data = log_amps[indices] - prior_mean* np.sum(feature_vector, axis=1)
                

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector 'K' to the fit data)
            optimal_weights = model.fit(X=feature_vector, y=fit_data).coef_

            #revert the prior mean adjustment above
            learning.weights = optimal_weights + prior_mean

            
            #print(learning.weights)
            #print(model.alpha)
            #input()

            #Update the weights and the epsilon tensor held in the learning object.

            learning.valid_kern = abs(np.diag(K.conj().T.dot(K))) > learning.kern_cutoff

            learning.update_epsilon_with_weights()
        
            # Convert the learnt epsilon tensor into log wavefunction amplitudes.
            estimated_log_amps = vs._apply_fun({"params": {"epsilon": learning.epsilon}}, configs)
            estimated_amps  = jnp.exp(estimated_log_amps)/jnp.sqrt(jnp.sum(abs(jnp.exp(estimated_log_amps))**2))
            #-------------------------------------------------------------
            norm_amps = jnp.exp(log_amps)/jnp.sqrt(jnp.sum(abs(jnp.exp(log_amps))**2))
            #-------------------------------------------------------------
            overlap_temp = abs(estimated_amps.T.dot(norm_amps)) 
          
            #print(overlap_temp)
            #input()
        overlaps_iter.append(overlap_temp)
    iters_list = [x for x in range(iterations)]
    plt.plot(iters_list, overlaps_iter)
    plt.show()

    return estimated_log_amps, learning.epsilon



def overlap_error(configs, amps, log_amps, tol, iterations, indices_to_fit, alpha, vs, ha, weighted_bool):
    """Calculates the overlap of the """
    estimated_log_amps ,_= lasso_linear_sweeping(tol, iterations, indices_to_fit, configs, amps, log_amps, alpha, vs, ha, weighted_bool)
    estimated_amps  = jnp.exp(estimated_log_amps)/jnp.sqrt(jnp.sum(abs(jnp.exp(estimated_log_amps))**2))
    #-------------------------------------------------------------
    norm_amps = jnp.exp(log_amps)/jnp.sqrt(jnp.sum(abs(jnp.exp(log_amps))**2))
    #-------------------------------------------------------------
    overlap = abs(estimated_amps.T.dot(norm_amps))
    return overlap


#---------------------------------------------------Plotting, Testing etc etc-------------------------------------------------------------------------------

def generate_meeting_plots_25_02(L,M,alpha,iters,indices_sets,n_plots=1,extra_string=""):
    vs, ha = initialise_system(L=L,M=M, seed = 1)
    e, configs, amps, log_amps = generate_test_data(ha)
    x_ticks = [indices_sets[i] for i in range(len(indices_sets))]
    weight = [True, False]
    for i in weight:
        overlaps = []
        errors = []
        for set in indices_sets:
            overlaps.append(overlap_error(configs, amps, log_amps, iters, jnp.atleast_1d(jnp.arange(set)), alpha, vs, ha, weight[i]))
            _, error = lasso_linear_sweeping(iters, jnp.atleast_1d(jnp.arange(set)), configs, amps, log_amps, alpha, vs, ha, weight[i])
            errors.append(error[-1])
            
        #Plotting Overlaps
        plt.plot(x_ticks, overlaps)
        plt.title("<qGPS|gs> of LASSO Estimator with size of training set, (alpha=" + str(alpha) + ", iters=" +str(iters) +", M="+str(M)+", weights="+str(weight[i])+")", fontsize = 8)
        plt.xlabel("Size of Training Data")
        plt.xticks(x_ticks)
        plt.ylabel("Overlap, <qGPS|gs>")
        plt.savefig("Overlaps: Weights="+str(weight[i])+str(extra_string)+".png")  
        plt.clf()

        #Plotting Errors
        plt.plot(x_ticks, errors)
        plt.title("LSE of LASSO Estimator with size of training set, (alpha=" + str(alpha) + ", iters=" +str(iters) +", M="+str(M)+", weights="+str(weight[i])+")", fontsize = 8)
        plt.xlabel("Size of Training Data")
        plt.xticks(x_ticks)
        plt.ylabel("LSE")
        plt.savefig("Errors: Weights="+str(weight[i])+str(extra_string)+".png")
        plt.clf()

    return None

def plot_overlaps(L,M,alpha,iters,indices_sets,weights):
    vs, ha = initialise_system(L,M, seed = 1)
    _, configs, amps, log_amps = generate_test_data(ha)
    overlaps = []
    for set in indices_sets:
        overlaps.append(overlap_error(
        configs=configs,
        amps = amps, 
        log_amps = log_amps, 
        iterations = iters, 
        indices_to_fit = jnp.atleast_1d(jnp.arange(set)), 
        alpha = alpha, 
        vs = vs, 
        ha = ha, 
        weighted_bool=weights
        ))
    plt.plot(indices_sets, overlaps)
    plt.title("<qGPS|gs> of LASSO Estimator vs. size of training set, (alpha=" + str(alpha) + ", iters=" +str(iters) +", M="+str(M)+", weights="+str(weights)+")", fontsize = 8)
    plt.xlabel("Size of Training Data")
    plt.ylabel("Overlap, <qGPS|gs>")
    plt.savefig("TEST: Overlaps: L="+str(L)+", Weights ="+str(weights)+".png")

    return overlaps, len(ha.hilbert.all_states())

def calibrate_alpha(L,M,iters,weights):
    """Finds the optimal order of magnitude for alpha"""
    vs,ha = initialise_system(L,M, seed = 1)
    _, configs, amps, log_amps = generate_test_data(ha)
    overlap = 0
    alpha = 10
    while overlap<0.99:
        alpha = alpha/10
        overlap = overlap_error(configs,amps,log_amps,iters,jnp.atleast_1d(jnp.arange(len(ha.hilbert.all_states()))),alpha,vs,ha,weights)
    
    return alpha


#Seeds 1-5, L=10, M=20, tol = 10^{-4}, iters = 200, all states


#Varying M test

data = []
#M_list = [10,12,14,16,18,20,22,24,26,28,30]
#M_list = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]
M_list = [-9]
vs_,ha_ = initialise_system(L=10,M=18, seed = 1)
_, configs, amps, log_amps = generate_test_data(ha_)
for m_i in M_list:
  overlaps = []
  counters = []
  counters0 = []
  for i in range(1,6):
    counter = 0
    counter0 = 0
    vs, ha = initialise_system(L=10, M = 10, seed = i)
    estimated_log_amps, epsilon = lasso_linear_sweeping(
      10**(-4),
      200,
      jnp.atleast_1d(jnp.arange(len(ha.hilbert.all_states()))),  #len(ha.hilbert.all_states())
      configs,
      amps, 
      log_amps, 
      10**(m_i), 
      vs, 
      ha, 
      True
      )
    for element in epsilon.flatten():
        if element == 1:
          counter +=1
        elif element ==0:
            counter0 +=1
    counters.append(counter) 
    counters0.append(counter0) 
    estimated_amps  = jnp.exp(estimated_log_amps)/jnp.linalg.norm(jnp.exp(estimated_log_amps))
    overlaps.append(float(abs(estimated_amps.T.dot(amps))))
  data.append(float(jnp.mean(jnp.array(overlaps))))
  data.append(int(jnp.mean(jnp.array(counters))))
  data.append(int(jnp.mean(jnp.array(counters0))))

print(epsilon)
#print(data)
"""with open('O(a)M10L10A150R.txt', 'w') as f:
    for set in data:
        f.write(f"{set}\n")"""