import netket as nk
import numpy as np
import math

#Sections 1 and 2: Hamiltonian and Exact Diagonalisation.

#Checking that the 'NetKet' package has successfully installed.
print("NetKet version:", nk.__version__)

#Defining the 2D Lattice that our Hamiltonian will act on.

def lattice(L):
    """Generates a 2D square graph object with side length 'L' nodes.
    Takes in argument 'L'.
    Returns 'nk.graph' object 'graph'."""
    graph = nk.graph.Hypercube(length = L, n_dim = 2, pbc = True)    
    return graph

#Functions to create shorthand Pauli operators
def sx(hilbert_space, site):
    sx_site = nk.operator.spin.sigmax(hilbert_space,site)
    return sx_site
def sy(hilbert_space,site):
    sy_site = nk.operator.spin.sigmay(hilbert_space,site)
    return sy_site
def sz(hilbert_space,site):
    sz_site= nk.operator.spin.sigmaz(hilbert_space,site)
    return sz_site

def hammiltonian(hilbert_space, l:int,h:float,J:float):
    """Defining the Hamiltonain on the hilbert space where n = l**2 should be the number of nodes on the lattice of interest."""
    graph = lattice(l)
    
    #Example operators from tutorial.
    #sx_1 = nk.operator.spin.sigmax(hi,1) #Pauli-x operator acting on site 1
    #sy_2 = nk.operator.spin.sigmay(hi,2) #Pauli-y operator acting on site 2
    #sz_2 = nk.operator.spin.sigmaz(hi,2) #Pauli-z operator acting on site 2

    H = nk.operator.LocalOperator(hilbert_space) #H is a local operator on the Hilbert space
    #Adding all the terms acting on a single site. (i.e. first term sum in the Hamiltonain)
    nodes = [node for node in graph.nodes()]
    for site in nodes:
        H -= h*sx(hilbert_space,site)
    
    #Adding all the terms acting on nieghboring sites. (i.e. second sum in the Hamiltionian)
    edges = [edge for edge in graph.edges()]
    for (i,j) in edges:
        H += J*sz(hilbert_space,i)*sz(hilbert_space,j)
    
    return H

def h_to_jax_operator(H):
    """Converts the Hamiltonian to the 'jax-friendly' operator."""
    H_jax = H.to_jax_operator()
    return H_jax


from scipy.sparse.linalg import eigsh
def exact_diagonalisation(H):
    """Using the scipy.linalg.eigsh method to compute the groundstate eigen energy for Hamiltonian H"""
    e_gs, psi_gs = eigsh(H.to_sparse(), k=1, which = "SA")

    e_gs = e_gs[0]
    psi_gs = psi_gs.reshape(-1)

    return e_gs, psi_gs

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#Section 3: Variational Ansatz and Variational Energy

import jax
#Numerical operations in the model should always use jax.numpy because jax.numpy supports computing derivatives whereas numpy does not
import jax.numpy as jnp

#Flax is the framework used to define models using jax
import flax
#flax.linen is a repository of layers, initializers and nonlinear functions.
import flax.linen as nn

#A flax model must be a CLASS subclassing flax.linen.Module, or equivelantly nn.Module as defined above.
class MF(nn.Module):

    #The most compact way to define the model is:
    # - Use the __call__(self,x) function
    # - __call__ takes in a batch of input states x, where x.shape = (n_samples, N)
    # - __call__ returns a vector of n_samples log-amplitudes
    @nn.compact
    def __call__(self,input_states):
        
        #A tensor of variational parameters is defined by calling the method self.param where the arguments will be:
        # - and arbitrary name used to refer to this set of parameters
        # - an initializer used to provide the initial values
        # - the shape of the tensor
        # - the dtype of the tensor
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)

        #computing the probabilities using the sigmoid model
        p = nn.log_sigmoid(lam*input_states)

        return 0.5*jnp.sum(p, axis=-1)

class Jastrow(nn.Module):
    #<sigma_1^z, ... ,sigma_N^z|psi_jas> = e^(sum_ij sigma_i J_ij sigma_j)
    #-> the Model is defined as:
    #Model(sigma_1,...,sigma_N, theta) = sum_ij sigma_i J_ij sigma_j
    #Where the variational parameters theta is just the matrix J_ij in this model.

    #Model= (sigma_1, ..., sigma_N, J_ij) = sum_ij sigma_i J_ij sigma_j

    #Input a batch of states input_x.shape = (N_samples, N) and should return a vector of n_samples log-amplitudes
    @nn.compact
    def __call__(self, input_x):

        n_sites = input_x.shape[-1]

        #Define a tensor of variational parameters by calling the method 'self.param' where the arguments will be:
        # - arbitrary name used to refer to this set of parameters
        # - an initializer used to provide the initial values (could be random, could be chosen etc etc)
        # - The shape of the tensor
        # - the data type of the tensor elements
        # Define the two variational parameters J1 and J2
        J = self.param("J", nn.initializers.normal(), (n_sites,n_sites), float) #shape is a a square matrix of size nxn where n is the length of the input state vectors.

        #ensure the state vector elements and the Jastrow matrix elements are of the same data type
        
        dtype = jax.numpy.promote_types(J.dtype, input_x.dtype)
        
        #setting them to the same data type
        J = J.astype(dtype)
        input_x = input_x.astype(dtype)

        #Note that J_ij is not symettric, could have been initialised with any old values not nescasarrily symetrric
        #So symmetrize it by hand

        J_symm = J.T + J #J^T + J -> J_symm_ij = J_ji + J_ij -> J_symm_ji = J_ij + J_ji = J_symm_ij

        #The model is the sum defined above as Model given some Jastrow matrix J_symm. Why does it have to be symmetric?
        #einsum: Einstein Summation, i.e. summation over repeated indices. So computes the sum of the inner products <input_x|J|input_x>
        result = jnp.einsum("...i,ij,...j", input_x, J_symm,input_x)
        return result

def initialise_params_Jastrow(hilbert_space, random):
    """Initialises a set of random parameters for an input hilbert space object using the Jastrow model is the second argument is true, otherwise initialises a set of parameters all set to one."""
    #create and instance of the model
    model = Jastrow()

    #create a RNG key to initialise the random parameters
    key = jax.random.key(0)

    #initialise the weights
    if random is True:
        parameters = model.init(key, hilbert_space.random_state(jax.random.key(0)))
    else:
        parameters = model.init(key, np.ones(hilbert_space.size))
        
    return model, parameters

    


    
def initialise_params_MF(hilbert_space,random):
    """Initialises a set of random parameters for an input hilbert space object using the MF model is the second argument is true, otherwise initialises a set of parameters all set to one."""
    #create and instance of the model
    model = MF()

    #create a RNG key to initialise the random parameters
    key = jax.random.key(0)

    #initialise the weights
    if random is True:
        parameters = model.init(key, np.random.rand(hilbert_space.size))
    else:
        parameters = model.init(key, np.ones(hilbert_space.size))
        
    return model, parameters


#3.2 Returning the exponentiated wavefunction, properly normalised.

def to_array(hilbert_space,model,params):
    #model, params = initialise_params(hilbert_space,random)
    #Generating all configurations in the hilbert space:
    all_configs = hilbert_space.all_states() #a list of lists where the internal lists are state vectors of all possible states in the hilbert space

    log_wavefunction = model.apply(params, all_configs) #evaluate the model using params as passed from initilaise_params and all hilbert space configuartions as inputs
    #shape of log_wavefunction should be (len(all_configs),) i.e. the log_wavefunction should have as many terms as there are configurations of the hilbert space

    #loop through the wavefunction and exponentiate each element
    wavefunction = []
    for element in log_wavefunction:
        wavefunction.append(jnp.exp(element)) #using jax.numpy as much as possible

    #Normalisation
    #normalisation_factor = jnp.sum(log_wavefunction)
    #for element in wavefunction:
        #element = element/jnp.sqrt(normalisation_factor) #|psi>_n = 1/\sqrt(N)|psi>
    
    #convert to jax.numpy array
    wavefunction = jnp.array(wavefunction)

    #Normalisation (faster)
    wavefunction = wavefunction/jnp.linalg.norm(wavefunction)

    return(wavefunction)

import timeit

def to_array_exec_times(hilbert_space):
    time = timeit.timeit(lambda: to_array(hilbert_space), number=1)
    return time

#Taking forever to run...
def to_array_jit(hilbert_space):
    """Converts the to_array function tp a jit compiled version, to_array_jit"""
    to_array_jit = jax.jit(to_array, static_argnames="hilbert_space")
    wavefunction = to_array_jit(hilbert_space)
    return wavefunction
#Taking forever to run because of above...
def to_array_jit_exec_time(hilbert_space):
    time = timeit.timeit(lambda: to_array_jit(hilbert_space), number=1)
    return time

#--------------------------------------------------------------------------------------------------------------------------------------------------------
#Section 3.3: Energy

#Computes the energy of the mean field state for the given parameters.
def compute_energy(hilbert_space, model, params, hamiltonian_sparse):
    wavefunction_gs = to_array(hilbert_space,model,params)
    energy = wavefunction_gs.conj().T@(hamiltonian_sparse@wavefunction_gs) # <psi|H|psi> but i dont understnad the syntax at all
    return energy

def compute_energy_jit(hilbert_space):

    compute_energy_jit = jax.jit(compute_energy, static_argnames="hilbert_space")
    return compute_energy_jit

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#Section 3.4: Gradient of the energy:

from functools import partial
@partial(jax.jit, static_argnames = ["hilbert_space","model"])
def compute_energy_and_gradient(hilbert_space, model, params,hamiltonian_sparse):
    """Returns two values, energy and grad"""
    grad_fun = jax.value_and_grad(compute_energy, argnums=2)
    result = grad_fun(hilbert_space, model, params,hamiltonian_sparse)
    return result

#Test: Does seem to work, but outputs a gradient of zero which is strange.
"""hi = nk.hilbert.Spin(s=1/2, N=9)
model, params = initialise_params_MF(hi)
H_dense = h_to_jax_operator(hammiltonian(hi,3,1,1)).to_sparse()
energy, grad = compute_energy_and_gradient(hi, model, params,H_dense)
print(energy)
print(grad)"""


#Doing monte carlo a littel bir different, we want to update the complexity of the model on the fly, at each step tweak the expressiveness of our model to display our system for

#Read in amplitudes with netket, fit those amplitudes to a different model.
#ONce done VMC from scratch tutorial
#At each step of the optimisation process, 3.5, output wavefunction amplitdues. These will be used as example states to fit the model.
#GPS is essentially
#For the GPS model log(|psi>) = sum_x'=1^M prod_i=1^N epsilon_x',i,x_i 
#This tensor holds MxNx2 parameters.
#Dump out the amplitudes at a given step, fitting the wavefunction to this GPS model
#log(|psi>) = sum_x' (epsilon_x',0,up delta_x_i,up + epsilon_x',0,down delta_x_i,down) TIMES the product over the rest
#Now this takes the form of a linear model: sum_tilda(x') W_tilda(x') phi(x') where phi is a feature function containing all the rest 
#x' is the artificail dimension, i is the lattice site, x_i being the configuration we want to evaulate things for
#Ussually we set M by hand. This is what we want to be able to automatically determine

#FITTING THE GPS MODEL TO THE WAVEFUNCTIONS SPAT OUT BY NETKET IS THE WHOLE POINT
#USING LASSO TO DO THIS FITTING STEP IS THE WHOLE POINT

#SET UP THE GPS MODEL AS A FLAX MODEL (DO A LEAST SQUARES FIT WITH THIS MODEL) 
#SUM OVER X OF (LOG AMPLITUDESX OF EXACT WAVEFUNCTION CO-EFFICIENT - PREDICETED LOG AMPLITUDE OF WAVEFUNCTION)^2

#SETTING UP GPS MODEL AS A FLAX MODEL TAKE AN AFTERNOON WITH YANNIC.
#DUMP OUT THE DATA AT EVERY OPTIMISATION STEP FOR THE VMC

#Jastrow Flax Initilisation Model is a good one to do nowwwwww
#Look into LASSO Regresion python libraries
#REALLY NEED TO GET THE WAVEFUNCTION AMPLITUDES OUT BY 26TH MEETING WHEN WE SIT DOWN AND TALK ABOUT AND SET UP GPS MODEL.

#--------------------------------------------------------------------------------------------------------------------------------------------------
#Section 3.5: Optimise and look for the ground state

from tqdm import tqdm

def optimise_params(hilbert_space,model,hamiltonian_matrix,iterations):
    """Runs an optimisation algorithm in hopes of optimising the variational parameters of the model to find the ground state energy of the system.
    Arguments are: hilbert_space, hamiltonian_matrix, number of iterations."""
    
    if model == "MF":
        model, params = initialise_params_MF(hilbert_space,False)
    elif model == "Jastrow":
        model, params = initialise_params_Jastrow(hilbert_space,False)
   
    #logging: theis is a netket logger which accumulates data you throw at them?
    logger = nk.logging.RuntimeLog()

    for i in tqdm(range(iterations)):
        current_energy, current_grad = compute_energy_and_gradient(hilbert_space,model,params,hamiltonian_matrix)
        #update parameters. lower energy is good i guess? Also lower gradient is good. Want min energy and zero grad ideally
        params = jax.tree.map(lambda x,y:x-0.01*y, params, current_grad)
        #log energy AND parameters. The plan is to hopefully log the parameters at each optimisation step aswell and then use them to work backwards to the wavefunction amplitudes
        logger(step = i, item = {"Energy": current_energy}) #,"Parameters": params}) params doesnt work it can only log scalars, check documentation for a way to log arrays

    return logger

import matplotlib.pyplot as plt
def plot_data(hilbert_space,model,hamiltonian_matrix,iterations):
    logger = optimise_params(hilbert_space,model,hamiltonian_matrix,iterations)
    #plotting the data from the logger (only plotting the energy data of course.)
    plot = plt.plot(logger.data["Energy"]["iters"], logger.data["Energy"]["value"]) #guessing that iters and value are defined in the logger object
    return(plot)


"""hi = nk.hilbert.Spin(s=1/2, N=9)
H = hammiltonian(hi,3,1,1)
exact_gs_energy, exact_gs_wavefunction = exact_diagonalisation(H)
H_sparse = h_to_jax_operator(H).to_sparse()
plot_data(hi,"MF",H_sparse,100)
plt.show()"""

def plot_relative_error_to_exact_gs(hilbert_space,model,hammiltonian,iterations):
    exact_gs_energy, exact_gs_wavefunction = exact_diagonalisation(hammiltonian)
    data_logger = optimise_params(hilbert_space,model,h_to_jax_operator(hammiltonian).to_sparse(),iterations)
    relative_error_plot = plt.semilogy(data_logger.data["Energy"]["iters"], np.abs(data_logger.data["Energy"]["value"]-exact_gs_energy))
    return relative_error_plot

"""hi = nk.hilbert.Spin(s=1/2, N=9)
H = hammiltonian(hi,3,1,1)
plot_relative_error_to_exact_gs(hi,"MF",H,100)
plt.show()"""





#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Up until now we did everythin by summing over the whole hilbert space. But for larger problems that wont be possible. SO we need Monte Carlo sampling.

#Consider a simple MCMC smapleer that proposes new states by flipping individual spins.

def MC_initialise_samples(hilbert_space,chain_length,model,params):
    sampler = nk.sampler.MetropolisSampler(hilbert=hilbert_space,rule=nk.sampler.rules.LocalRule(),n_chains=20) #hilbert_space is the hilbert space to be sampled, the second argument is the transition rule and the third is the number of chains sampled.
    sampler_state = sampler.init_state(model,params,seed=1)#initilises samples
    sampler_state = sampler.reset(model,params,sampler_state)#resets samples
    samples, sampler_state = sampler.sample(model,params, state = sampler_state, chain_length =chain_length) #samples
    #To generate more samples, just call sampler.sample again or with a differnet chain length for. If you change the parameters you should call sampler.reset
    return samples, sampler, sampler_state

"""hi = nk.hilbert.Spin(s=1/2, N=9)
model, params = initialise_params_MF(hi, True)
samples, sampler, sampler_state = MC_initialise_samples(hi,20,model,params)
print(samples.shape)"""

def sigma_eta_identification(H,sigma):
    
    eta, H_sigma_eta = H.get_conn_padded(sigma)
    return eta, H_sigma_eta



def sigma_eta_examples():
    hi = nk.hilbert.Spin(s=1/2, N=9)
    H = hammiltonian(hi,3,1,1).to_jax_operator()
    sigma = hi.random_state(jax.random.key(1))

    eta, H_sigma_eta = sigma_eta_identification(H,sigma)

    print("So for one sample sigma ", sigma.shape)
    print("We have 10 connected samples eta, each composed of 9 spins", eta.shape)
    print("and 10 matrix elements", H_sigma_eta.shape)
    print("This also applies for batches of configurations")

    sigma_batch = hi.random_state(jax.random.key(1), (4,5))

    eta_batch, H_sigma_eta_batch = sigma_eta_identification(H,sigma_batch)
    print("So for each of the (4,5) samples sigma ", sigma_batch.shape)
    print("We have 10 connected samples eta, each composed of 9 spins, in a tensor shape of (4,5,10,9)", eta_batch.shape)
    print("and 10 matrix elements", H_sigma_eta_batch.shape)

#sigma_eta_examples()



def compute_local_energies(model, parameters, H, sigma):
    eta, H_sigma_eta = sigma_eta_identification(H, sigma)
    #print(sigma.shape, "sigma")
    #print(eta.shape, "eta")
    logpsi_sigma = model.apply(parameters, sigma) #computes log(psi(sigma))
    #print(logpsi_sigma.shape)
    logpsi_eta = model.apply(parameters, eta) #computes log(psi(eta))
    #print(logpsi_eta.shape)

    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)

    #E_loc = sum_(eta s.t. <sigma|H|eta> != 0) <sigma|H|eta> e^(log(psi(eta)) - log(psi(sigma))) = sum_(eta s.t. <sigma|H|eta> != 0) <sigma|H|eta> (psi(eta)/psi(sigma))
    E_loc = jnp.sum(H_sigma_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis = -1) #along the last axis of the tensor

    return logpsi_sigma, E_loc


"""hi = nk.hilbert.Spin(s=1/2, N=9)
model, params = initialise_params_MF(hi, True)
H = hammiltonian(hi,3,1,1).to_jax_operator()
samples, sampler, sampler_state = MC_initialise_samples(hi,100,model,params)
print(compute_local_energies(model,params,H,samples[0]))"""

#Sampling the energy. Now write a function that computes the energy and estimates its error. Error given by: error = sqrt(Var(E_loc)/N_samples)

@partial(jax.jit, static_argnames = "model")
def estimate_energy(model,params, H, sigma):

    #Local energies
    notUsed, E_loc = compute_local_energies(model,params,H,sigma)

    E_average = jnp.mean(E_loc) #sum of local energues over number of local energies computed
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance/E_loc.size)

    stats_object = nk.stats.Stats(mean = E_average, error_of_mean = E_error, variance = E_variance)

    return stats_object

"""hi = nk.hilbert.Spin(s=1/2, N=9)
model, params = initialise_params_MF(hi, True)
H = hammiltonian(hi,3,1,1).to_jax_operator()
samples, sampler, sampler_state = MC_initialise_samples(hi,100,model, params)
logpsi, E_loc = compute_local_energies(model,params,H,samples)

isinstance(estimate_energy(model,params,H,samples), nk.stats.Stats)
print(estimate_energy(model,params,H,samples))


#Increase number of sample to check it agrees better with exact calculation as expected

samples_many,sample_state = sampler.sample(model,params,state = sampler_state, chain_length = 5000)

print("exact energy: ", compute_energy(hi,model,params, H.to_sparse()))
print("estimated energy: ", estimate_energy(model,params, H, samples_many))"""

#Sampling the gradient of the energy

#grad E = 1/N_s sum_i^N_s (grad logpsi((sigma_i)))*(E_loc(sigma_i) - <E>) where <E> approx 1/N_s sum_i E_loc(sigma_i)


"""hi = nk.hilbert.Spin(s=1/2, N=9)
model, params = initialise_params_Jastrow(hi, True)
H = hammiltonian(hi,3,1,1).to_jax_operator()
samples, sampler, sampler_state = MC_initialise_samples(hi,100,model, params)


sigma_vector = samples.reshape(-1,hi.size)

#creating the function logpsi_sigma_fun #JASTROW MODEL
logpsi_sigma_fun = lambda pars : model.apply(params, sigma_vector)

#inputs to the function are only the parameters, output is a vector of size N_samples (N_s)

jacobian = jax.jacrev(logpsi_sigma_fun)(params)

print("The parameters of jastrow have shape: ", jax.tree.map(lambda x: x.shape, params))
print("The jacobian of jastrow has shape: ", jax.tree.map(lambda x: x.shape, jacobian))"""

@partial(jax.jit, static_argnames = "model")
def estimate_energy_and_gradient(model,params,H_jax,sigma):

    sigma = sigma.reshape(-1, sigma.shape[-1])
    logpsi_sigma, E_loc = compute_local_energies(model,params,H_jax,sigma)

    E_average = jnp.mean(E_loc) #sum of local energues over number of local energies computed
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance/E_loc.size)

    stats_object = nk.stats.Stats(mean = E_average, error_of_mean = E_error, variance = E_variance)

    #Compute the gradient
    #First define the function to be differentiated

    logpsi_sigma_fun = lambda params : model.apply(params,sigma)

    #use jax.vjp to differentiate #LOOK AT THAT DOCUMENTATION

    _, vjpfun = jax.vjp(logpsi_sigma_fun, params)
    E_grad = vjpfun((E_loc - E_average)/E_loc.size)

    return stats_object, E_grad[0]


#Wrapping everything up

from tqdm import tqdm

def ground_state(hilbert_space,model_type):
    if model_type =="MF":
        model = MF()
    elif model_type == "Jastrow":
        model = Jastrow()

    sampler = nk.sampler.MetropolisSampler(hilbert = hilbert_space, rule =nk.sampler.rules.LocalRule(), n_chains = 20)
    n_iterations = 300
    chain_length = 1000//sampler.n_chains #50 in this case


    #initialise params and sampler
    if model_type =="MF":
        model, params = initialise_params_MF(hilbert_space,True)
    elif model_type =="Jastrow":
        model,params = initialise_params_Jastrow(hilbert_space,False)

    sampler_state = sampler.init_state(model,params, seed =1)
    #create data logger for the purpose of plotting

    logger = nk.logging.RuntimeLog()

    for i in tqdm(range(n_iterations)):
        #sample from the sampler
        sampler_state = sampler.reset(model,params, state = sampler_state) #need to reset the sampler every interation because we are adjusting params every iteration
        samples, sampler_state = sampler.sample(model,params, state = sampler_state,chain_length=chain_length) #taking new samples, 1000 samples, 20 chains, 50 samples per chain

        #compute the energy and gradient estimates
        H = hammiltonian(hilbert_space, int(np.sqrt(hilbert_space.size)),1,1)
        H_jax = H.to_jax_operator()
        E,E_grad = estimate_energy_and_gradient(model,params, H_jax, samples)
        #print(E_grad)
        logpsi = model.apply(params, hilbert_space.all_states()) #compute the log of the wavefunction given the current parameters

        #Compute the normalised wavefunction from the log of the wavefunction given by the model
        psi = jnp.exp(logpsi)
        psi = psi/jnp.linalg.norm(psi)

        #update the parameters
        params = jax.tree.map(lambda x,y:x-0.01*y, params, E_grad) #learning rate of 0.01

        #log the energy and wavefunction at each iteration step
        logger(step = i, item = {"Energy": E, "Wavefunction": psi})

    return logger

def plot_ground_state(logger_object):
    plt.plot(logger_object.data["Energy"]["iters"], logger_object.data["Energy"]["Mean"])
    plt.show()

def spit_out_wavefunction(logger_object):
    print(logger_object.data["Wavefunction"]["value"])
    gs_wavefunction = logger_object.data["Wavefunction"]["value"]
    return gs_wavefunction


"""#Big test!
hi = nk.hilbert.Spin(s=1/2, N=9)
#plot_ground_state(ground_state(hi,"Jastrow"))
gs_wavefunction = spit_out_wavefunction(ground_state(hi,"Jastrow"))
#print(gs_wavefunction)
print(type(gs_wavefunction)) #is a numpy.ndarray
#Works for Jastrow ansatz, Produced nonsense for Mean-Field ansatz"""

"""def format_wavefunction(log_wavefunction):
    #Type of wavefunction should be numpy.ndarray
    log_wavefunction_array = [amplitudes[0] for amplitudes in log_wavefunction] #should be a 1D list
    log_wavefunction_array = jnp.array(log_wavefunction_array)
    wavefunction_array = jnp.exp(log_wavefunction_array)
    normalised_wavefunction = wavefunction_array/jnp.linalg.norm(wavefunction_array)
    return normalised_wavefunction"""

"""gs_wavefunction_array = format_wavefunction(gs_wavefunction)
print(type(gs_wavefunction_array))
print(gs_wavefunction_array)
energy = gs_wavefunction_array.conj().T@(hammiltonian(hi,3,1,1).to_sparse()@gs_wavefunction_array)"""

#Lets try and do

"""hi = nk.hilbert.Spin(s=1/2, N=9)
model, params = initialise_params_MF(hi, True)
H = hammiltonian(hi,3,1,1).to_jax_operator()
samples, sampler, sampler_state = MC_initialise_samples(hi,100,model, params)
logpsi, E_loc = compute_local_energies(model,params,H,samples)

print(logpsi)
print(logpsi[1][1].size)"""



"""Testing model.apply for the Jastrow ansatz
#model.apply creates the logpsi wavefunction. Takes in inputs of parameters and states?

#model.apply(params, inputs)
#Feed in a list of N inputs and get an output list of length N. If each element of the input is a random state-vector, 
#then each element of the output will be the parameters corresponding to that state-vector.
#
#i.e. for the Mean Field model, the output will be a list of lambdas [lambda_1, lambda_2, ..., lambda_N] where lambda_i  


#Lets see what happens when you apply the jastrow model to the set of all states in the hilbert space
hi = nk.hilbert.Spin(s=1/2, N=4)
model, params = initialise_params_Jastrow(hi,True)

#I expect 16 4x4 Jastrow matrices to be the output of model.apply(params, hi.all_states()), 
#i.e. the Jastrow matrices parameterising each configuration of the hilbert space.

logpsi_jastrow = model.apply(params, hi.all_states())

print(logpsi_jastrow)
print(hi.all_states())
print(logpsi_jastrow.shape)
print(logpsi_jastrow[0])

N = len(hi.all_states())
psi_jastrow = jnp.exp(logpsi_jastrow)
normalisation = jnp.linalg.norm(psi_jastrow)
psi_jastrow = psi_jastrow/normalisation

psi_string = "|psi> = "
for i in range(N-1):
    psi_string = psi_string + str(psi_jastrow[i])+str(hi.all_states()[i])+" + "
psi_string = psi_string + str(psi_jastrow[N]) + str(hi.all_states()[N])

print(psi_string)"""



"""Testing model.apply where we DONT input all the states of the hilbert space as inputs, but instead som samples, sigma

hi = nk.hilbert.Spin(s=1/2, N =4)
model,params = initialise_params_Jastrow(hi,True)

#Sampler object
sampler = nk.sampler.MetropolisSampler(
    hilbert = hi, #hilbert space to sample from 
    rule = nk.sampler.rules.LocalRule(), #transition rule. I dont know what that is
    n_chains = 1)

#intialise the sampler object by creating a sampler_state
sampler_state = sampler.init_state(model,params,seed =1)

#reset sampler state (good practice)
sampler_state = sampler.reset(model,params,sampler_state)

#Generate some samples

samples, sampler_state = sampler.sample(model,params, state=sampler_state, chain_length =5)

print(samples)
print(samples.shape) #expecting one 4 element long list

sample_logpsi = model.apply(params, samples)

print(sample_logpsi)"""

hi = nk.hilbert.Spin(s=1/2, N=9)
"""wavefunctions = spit_out_wavefunction(ground_state(hi,"Jastrow"))
#print(wavefunctions)
#print(wavefunctions.shape)
print(wavefunctions[-1])

gs = wavefunctions[-1]

H = hammiltonian(hi,2,1,1).to_sparse()

energy = gs.conj().T@(H@gs)
print(energy)"""

plot_ground_state(ground_state(hi,"Jastrow"))
#print(ground_state(hi,"Jastrow")["Energy"]["Value"])






