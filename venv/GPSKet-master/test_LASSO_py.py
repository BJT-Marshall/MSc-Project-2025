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


# Can be used to activate or deactivate a weighting of terms according to Psi^2 in the squared error loss function
weight_according_to_amplitude_squared = True


# -----------------Set Up (Hamiltonian, Hilbert Space, qGPS Model, Sampler, qGPS Variational Quantum State)-------------------------------
def initialise_system(L,M):

    ha = j1j2.get_J1_J2_Hamiltonian(Lx=L, J2=0.0, sign_rule=True, on_the_fly_en=True)
    hi = ha.hilbert

    model = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)

    sa = nk.sampler.MetropolisExchange(hi, graph=ha.graph, n_chains_per_rank=1, d_max=L)
    vs = nk.vqs.MCState(sa, model, n_samples=50, chunk_size=1, seed=1)

    return vs, ha

#Default used in the script
vs, ha = initialise_system(L=10, M=5)

# generate exact ground state data
e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)

energy = e[0]

dataset_configs = jnp.array(ha.hilbert.states_to_local_indices(ha.hilbert.all_states()))

dataset_amplitudes = jnp.array(state.flatten())
dataset_log_amplitudes = jnp.log(dataset_amplitudes * jnp.sign(dataset_amplitudes[0]))
dataset_log_amplitudes -= jnp.mean(
    dataset_log_amplitudes
)  # Scale target dataset_log_amplitudes to be centered around zero
dataset_log_amplitudes = dataset_log_amplitudes / jnp.std(
    dataset_log_amplitudes
)  # Constrain the variance of the target data set to be 1



# All the bellow are 252 elements long.

# print(state) # state: list of 252 single element lists which is why flatten is used to make 1 252 element long list. Exact ground state amplitudes.

# print(dataset_amplitudes) #dataset_amplitudes: The flattened state (just reformatted)

# print(dataset_log_amplitudes) #dataset_log_amplitudes: Log of dataset_amplitudes, as dataset_amplitudes are all negative there is also an adjustment to make them positive and therefore "log-able"

# print(dataset_configs) #dataset_configs: Binary arrays of all possible configurations in the Hilbert Space. Total spin is constrained to zero. i.e. equal 0's and 1's.

# -----------------------------------------------------Test Garbage----------------------------------------------------------------

# print(jnp.arange(252))
# print(jnp.atleast_1d(jnp.arange(252)))
# print(jnp.arange(252)==jnp.atleast_1d(jnp.arange(252))) #In this case they are the same

"""epsilon = np.array(vs.parameters["epsilon"])
inds = jnp.atleast_1d(jnp.atleast_1d(jnp.arange(200)))
sampled_log_amps = vs._apply_fun({"params": {"epsilon": epsilon}}, dataset_configs[inds])
print(sampled_log_amps)
#print(dataset_log_amplitudes)
print(len(sampled_log_amps)) #Length of the samples passed as the second argument to the _apply_fun method of the VQS. (vs._apply_fun)
#print(sampled_log_amps == dataset_log_amplitudes) #False, sampled_log_amplitudes are the log amplitudes computed from the current qGPS model state (i.e. epsilon converted to log amplitudes.)"""

# ---------------------------------------------------------------------------------------------------------------------------------

from functools import partial
# Fit model with Gradient descent type scheme

@partial(jax.jit, static_argnames="vs")
def lossfun(epsilon, indices, vs):
    inds = jnp.atleast_1d(
        indices
    )  # Another reformatting step to make sure the indices are in the correct format.
    sampled_log_amps = vs._apply_fun(
        {"params": {"epsilon": epsilon}}, dataset_configs[inds]
    )  # The log amplitudes, of the state parameterised by epsilon and the qGPS model, of the configurations refferred to by the indices passed into lossfun.
    # Weightings are an additional step that weight the Loss Function to be affected more by the error of prominent log amplitudes of the target state.
    # i.e. if the state was |psi> = 1/sqrt(6)|0> + sqrt(5/6)|1>, the error on the |1> term would produce a higher (worse) loss function than the same error on the |0> term. In this setup, by a factor of 5
    if weight_according_to_amplitude_squared:
        weightings = abs(jnp.exp(dataset_log_amplitudes[inds])) ** 2
    # Else, the weightings are all one and have no effect
    else:
        weightings = jnp.ones(len(inds))
    # Return the loss function defined as sum_{indices sampled}(weighting(index) * |psi_(qGPS)(index) - psi_(exact)(index)|^2)
    return jnp.sum(
        weightings * jnp.abs(sampled_log_amps - (dataset_log_amplitudes[inds])) ** 2
    )  # testing undoing the stuff


# ------------------------------------------------Optax Adam Gradient Decent Scheme-------------------------------------------------------------------------------------------


def optax_adam_gradient_decent(iterations: int):
    epsilon = vs.parameters["epsilon"]

    optim = optax.adam(1.0e-2)
    opt_state = optim.init(epsilon)

    # The initial loss function value
    # print(lossfun(epsilon, jnp.arange(len(dataset_log_amplitudes))))

    @jax.jit
    def fit_step(epsilon, opt_state, indices):
        grad = jax.grad(lossfun)(epsilon, indices)
        updates, opt_state = optim.update(grad.conj(), opt_state, epsilon)
        return optax.apply_updates(epsilon, updates), opt_state

    @jax.jit
    def evaluate_full(epsilon):
        return vs._apply_fun({"params": {"epsilon": epsilon}}, dataset_configs)

    m_sq_error_full_list = []
    for i in range(iterations):
        m_sq_error_full = lossfun(
            epsilon, jnp.arange(len(dataset_log_amplitudes))
        )  # jnp.arange(len(dataset_log_amplitudes)) is just [0,1,2,...,251]        m_sq_error_full_list.append(m_sq_error_full)
        m_sq_error_full_list.append(m_sq_error_full)
        epsilon, opt_state = fit_step(
            epsilon, opt_state, jnp.arange(len(dataset_log_amplitudes))
        )
    return m_sq_error_full_list


# --------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------Yannic Linear Sweeping Model------------------------------------------------------------------


def yannic_linear_sweeping(iterations: int):
    # Fit model with sweeping
    epsilon = vs.parameters[
        "epsilon"
    ]  # revert epsilon tensor in case previous optimisation scheme has been run
    learning = QGPSLogSpaceFit(epsilon)

    m_sq_error_full_list = []
    for i in range(iterations):
        for reference_site in range(ha.hilbert.size):
            learning.ref_sites = reference_site
            m_sq_error_full = lossfun(
                learning.epsilon, jnp.arange(len(dataset_log_amplitudes)), vs
            )  # loss function of the learning objects current epsilon tensor against the exact state
            # (in this case the ground state of the J1_J2 hamiltonian), considering all indices (hence m_sq_error_FULL)

            # So for a SAMPLE of log amplitudes
            m_sq_error_full_list.append(m_sq_error_full)
            K = learning.set_kernel_mat(dataset_configs)

            if weight_according_to_amplitude_squared:  # The same process as the else clause just with a weightings matrix instead of the identity matrix used.
                # fit a weighted linear least squares model
                weightings = abs(jnp.exp(dataset_log_amplitudes)) ** 2
                optimal_weights = np.linalg.pinv(
                    K.T.dot(np.diag(weightings).dot(K)) + 1.0e-10 * np.eye(K.shape[-1])
                ).dot(K.T.dot(weightings * dataset_log_amplitudes))
            else:
                # fit a linear model
                optimal_weights = np.linalg.pinv(
                    K.T.dot(K) + 1.0e-10 * np.eye(K.shape[-1])  # .T is just transpose
                ).dot(K.T.dot(dataset_log_amplitudes))

            learning.weights = optimal_weights

            learning.valid_kern = True

            learning.update_epsilon_with_weights()
    return m_sq_error_full_list


# --------------------------------------------------------------------------------------------------------------------------------------


# Test of effect of number of optimisation steps per site for optax gradient decent model and yannics linear sweeping (FULL SET OF LOG AMPS PASSED IN)


def test_iterations():
    list = [i for i in range(1, 11)]

    for element in list:
        print("Linear Sweeping: ", yannic_linear_sweeping(element)[-1])
        print("Optax Gradient Decent: ", optax_adam_gradient_decent(element)[-1])
        test_amp, test_error = lasso_linear_sweeping(
            iterations=element,
            indices=jnp.arange(len(dataset_log_amplitudes)),
            alpha=0.001,
        )
        print("LASSO: ", np.array(test_error)[-1])
    return None


# ----------------------------------------------------------------------------------------------------------------------------------------------
def lasso_linear_sweeping(iterations: int, indices: list, alpha: float, vs, ha, weighted_according_to_psi_squared=False,):
    """Lasso linear sweeping model for qGPS model to the ground state computed at the top of this code"""
    epsilon = np.array(vs.parameters["epsilon"])  # reset the epsilon tensor
    learning = QGPSLogSpaceFit(
        epsilon
    )  # The way of interfacing the learning model with the qGPS state

    m_sq_error_full_list = []

    #Define LASSO Model
    model = Lasso(alpha=alpha, fit_intercept=False) #alpha is the 'lambda' parameter, defines the L1 penalty the model uses. REGULARISATION

    for i in range(iterations):

        if i == 0:
            model.alpha = 0.0
        else:
            model.alpha = alpha
        
        for site in np.arange(ha.hilbert.size):  # For each single site perform an optimisation cycle
            learning.ref_sites = site

            learning.reset()

            #Setting the feature vector to either be the kernel function generated by the 'set_kernel_mat' method, or the kernel function wieghted by 
            #the square amplitude of the exact fit data. Determined by the 'weighted_according_to_psi_squared' flag.

            prior_mean = 1.0 if site != 0 else 0.0

            #scaled by sqrt{|psi|^2}k
            if weighted_according_to_psi_squared:
                weightings = jnp.expand_dims(jnp.sqrt(abs(jnp.exp(dataset_log_amplitudes[indices]))**2), -1)
                K=learning.set_kernel_mat(update_K=True, confs=dataset_configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                #shape K = (n_samples, MD)
                feature_vector = weightings*K
                fit_data = weightings.flatten()*((dataset_log_amplitudes[indices]))
                fit_data -= jnp.mean(fit_data)  # Scale target fit_data to be centered around zero
                fit_data = fit_data / jnp.std(fit_data)  # Constrain the variance of the target data set to be 1
                fit_data -=weightings.flatten()*np.sum(prior_mean * feature_vector, axis=1)
            else:
                K=learning.set_kernel_mat(update_K=True, confs=dataset_configs[indices]) #sampled amplitudes converted to configs as demanded by the 'set_kernel_mat' method
                feature_vector = K 
                fit_data = dataset_log_amplitudes[indices] - np.sum(prior_mean * feature_vector, axis=1)

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector 'K' to the exact amplitudes)
            optimal_weights = model.fit(X=feature_vector, y=fit_data).coef_

            #Update the weights and the epsilon tensor held in the learning object.

            #learning.weights = (optimal_weights + old_weights) / 2 + prior_mean
            learning.weights = optimal_weights + prior_mean

            learning.valid_kern = abs(np.diag(K.conj().T.dot(K))) > learning.kern_cutoff

            learning.update_epsilon_with_weights()
        # Convert the learnt epsilon tensor into log wavefunction amplitudes.
        estimated_log_amplitudes = vs._apply_fun({"params": {"epsilon": learning.epsilon}}, dataset_configs)

        # Calculate Error
        m_sq_error_full = lossfun(learning.epsilon, indices, vs)
        m_sq_error_full_list.append(m_sq_error_full)

    return estimated_log_amplitudes, m_sq_error_full_list

#-------------------------------------------------------Testing Convergence v.s. Variational Parameters-----------------------------------------------------------

def regularization_strength(indices, iterations, max_regularization):
    """Plots the least squares error as a function of the regularisation from 0 to 'max_regularization' in steps of 0.01, for fixed iterations and indices."""
    error_list = []
    alpha_list = []
    for alpha in range(int(100*max_regularization+1)):
        _, single_error_list = lasso_linear_sweeping(iterations = iterations,indices = indices, vs=vs, ha = ha, alpha=alpha/100)
        alpha_list.append(alpha/1000)
        error_list.append(single_error_list[-1])

    plt.plot(alpha_list, error_list)
    plt.show()
    return None

#indices = jnp.atleast_1d(jnp.arange(150))
#regularization_strength(indices, 10, 0.1)



#_, error =lasso_linear_sweeping(iterations=10,indices = indices, alpha = 0.0, vs =vs)
#_, error1 =lasso_linear_sweeping(iterations=10,indices = indices, alpha = 0.0, vs = vs, weighted_according_to_psi_squared=True)


#print(error[-1])
#print(error1[-1])

def plot_support_dim(indices, iterations, max_M, min_M = 1):
    error_list = []
    M_list = [M for M in range(min_M, max_M+1)]
    for M in M_list:
        vs, _ = initialise_system(L=10, M=M)
        _, error = lasso_linear_sweeping(iterations, indices, alpha = 0.01, vs=vs, ha = ha)
        error_list.append(error[-1])
    
    #Plotting
    plt.plot(M_list, error_list)
    plt.title("Least-Squares Error of LASSO Estimator with varied support dimension, M", fontsize = 10)
    plt.xlabel("M")
    plt.xticks(M_list)
    plt.ylabel("Least-Squares Error")
    plt.show()

    return None

#indices = jnp.atleast_1d(jnp.arange(150))
#plot_support_dim(indices, iterations = 50, max_M = 15)

def overlap_error(iterations, indices, alpha, vs, ha, weighted_bool):
    """Calculates the overlap of the """
    exact_amps = dataset_amplitudes/jnp.linalg.norm(dataset_amplitudes)
    estimated_log_amps, _ = lasso_linear_sweeping(iterations, indices, alpha, vs, ha, weighted_bool)
    estimated_amps  = jnp.exp(estimated_log_amps)/jnp.linalg.norm(jnp.exp(estimated_log_amps))
    overlap = estimated_amps.T.dot(exact_amps)
    return overlap

#print(overlap_error(iterations = 100, indices = jnp.atleast_1d(jnp.arange(252)), alpha = 0.1, vs =vs, weighted_bool = False))


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_overlap_varying_alpha(L, M, iters, min_alpha, max_alpha, step_size, indices_sets):
    vs, ha = initialise_system(L,M)
    cmap = get_cmap(len(indices_sets))
    overlaps = []
    min_alpha_ref = min_alpha
    for i in range(len(indices_sets)):
        min_alpha = min_alpha_ref
        alpha = []
        overlaps.append([])
        while min_alpha < max_alpha:
            alpha.append(min_alpha)
            overlaps[i].append(jnp.abs(overlap_error(iterations = iters, indices = indices_sets[i],alpha = min_alpha, vs = vs, ha = ha, weighted_bool = False)))
            min_alpha += step_size
        plt.plot(alpha, overlaps[i], color = cmap(i), label = str(len(indices_sets[i])) +' Samples')
    
    #Plotting
    plt.title("<qGPS|gs> of LASSO Estimator with varied regularization parameter, alpha (M=" + str(M) + ", iters=" +str(iters) +")", fontsize = 10)
    plt.xlabel("alpha")
    plt.xticks(alpha)
    plt.ylabel("Overlap, <qGPS|gs>")
    plt.legend()
    plt.show()

plot_overlap_varying_alpha(L=10, M=6, iters = 10, min_alpha=0, max_alpha = 0.6, step_size= 0.05, indices_sets = [jnp.atleast_1d(jnp.arange(150)),jnp.atleast_1d(jnp.arange(175)),jnp.atleast_1d(jnp.arange(200))])


#lasso_linear_sweeping(indices = jnp.atleast_1d(jnp.arange(50)), iterations = 10, alpha = 0.2, vs=vs, weighted_according_to_psi_squared=False)

