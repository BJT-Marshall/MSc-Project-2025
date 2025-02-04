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


L = 10
M = 5

# Can be used to activate or deactivate a weighting of terms according to Psi^2 in the squared error loss function
weight_according_to_amplitude_squared = False


# -----------------Set Up (Hamiltonian, Hilbert Space, qGPS Model, Sampler, qGPS Variational Quantum State)-------------------------------
ha = j1j2.get_J1_J2_Hamiltonian(Lx=L, J2=0.0, sign_rule=True, on_the_fly_en=True)
hi = ha.hilbert

model = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)

sa = nk.sampler.MetropolisExchange(hi, graph=ha.graph, n_chains_per_rank=1, d_max=L)
vs = nk.vqs.MCState(sa, model, n_samples=50, chunk_size=1, seed=1)

# -----------------------------------------------------------------------------------------------------------------------------------------

# epsilon = np.array(vs.parameters["epsilon"])


# generate exact ground state data
e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)

energy = e[0]

dataset_configs = jnp.array(hi.states_to_local_indices(hi.all_states()))

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


# Fit model with Gradient descent type scheme
@jax.jit
def lossfun(epsilon, indices):
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
        for reference_site in np.arange(L):
            learning.ref_sites = reference_site
            m_sq_error_full = lossfun(
                learning.epsilon, jnp.arange(len(dataset_log_amplitudes))
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
def lasso_linear_sweeping(iterations: int, indices: list, alpha: float):
    """Lasso linear sweeping model for qGPS model to the ground state computed at the top of this code"""
    epsilon = np.array(vs.parameters["epsilon"])  # reset the epsilon tensor
    learning = QGPSLogSpaceFit(
        epsilon
    )  # The way of interfacing the learning model with the qGPS state

    m_sq_error_full_list = []

    for i in range(iterations):
        # Define LASSO Model
        model = Lasso(
            alpha=alpha, fit_intercept=False
        )  # alpha is the 'lambda' parameter, defines the L1 penalty the model uses. REGULARISATION
        if i == 0:
            model.alpha = 0.0
        else:
            model.alpha = alpha
        
        for site in np.arange(L):  # For each single site perform an optimisation cycle
            learning.ref_sites = site

            learning.reset()

            K = learning.set_kernel_mat(
                update_K=True, confs=dataset_configs[indices]
            )  # Setting the kernel function (feature vector) excluding the current reference site using the sampled amplitudes
            # (converted into configs because thats the way the function wants them)

            # Fitting the model (Computes the optimal weights 'w' that fits the feature vector 'K' to the exact amplitudes)

            prior_mean = 1.0 if site != 0 else 0.0

            fit_data = dataset_log_amplitudes[indices] - np.sum(prior_mean * K, axis=1)

            optimal_weights = model.fit(X=K, y=fit_data).coef_

            # Update the weights and the epsilon tensor held in the learning object.

            # learning.weights = (optimal_weights + old_weights) / 2 + prior_mean
            learning.weights = optimal_weights + prior_mean

            learning.valid_kern = abs(np.diag(K.conj().T.dot(K))) > learning.kern_cutoff

            learning.update_epsilon_with_weights()
            # print("Epsilon: ", learning.epsilon)

        # Convert the partially learnt epsilon tensor into log wavefunction amplitudes reffered to by the sample indices.
        sampled_log_amplitudes = vs._apply_fun(
            {"params": {"epsilon": learning.epsilon}}, dataset_configs[indices]
        )

        # Calculate Error
        m_sq_error_full = lossfun(
            learning.epsilon, jnp.arange(len(dataset_log_amplitudes))
        )
        m_sq_error_full_list.append(m_sq_error_full)

    return sampled_log_amplitudes, m_sq_error_full_list

#-------------------------------------------------------Testing Convergence v.s. Variational Parameters-----------------------------------------------------------

def regularization_strength(indices, iterations, max_regularization):
    """Plots the least squares error as a function of the regularisation from 0 to 'max_regularization' in steps of 0.01, for fixed iterations and indices."""
    error_list = []
    alpha_list = []
    for alpha in range(int(100*max_regularization+1)):
        _, single_error_list = lasso_linear_sweeping(iterations = iterations,indices = indices, alpha=alpha/100)
        alpha_list.append(alpha/1000)
        error_list.append(single_error_list[-1])

    plt.plot(alpha_list, error_list)
    plt.show()
    return None

indices = jnp.atleast_1d(jnp.arange(252))
regularization_strength(indices, 10, 0.1)

