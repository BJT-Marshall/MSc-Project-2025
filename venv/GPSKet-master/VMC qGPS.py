#VMC with the qGPS model ansatz.

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax
import GPSKet
from GPSKet.models import qGPS
import matplotlib.pyplot as plt
from tqdm import tqdm

#-------------------------------------------------------------------------Hilbert Space/Hamiltonian Setup-----------------------------------------------------------------------
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

def ising_hamiltonian(N,h=1,J=1):
    graph = nk.graph.Hypercube(length = int(jnp.sqrt(N)), n_dim = 2, pbc = True)
    hi = nk.hilbert.Spin(s=1/2, N=N)
    
    H = nk.operator.LocalOperator(hi) #H is a local operator on the Hilbert space
    
    #Adding all the terms acting on a single site. (i.e. first term sum in the Hamiltonain)
    nodes = [node for node in graph.nodes()]
    for site in nodes:
        H -= h*sx(hi,site)
    
    #Adding all the terms acting on nieghboring sites. (i.e. second sum in the Hamiltionian)
    edges = [edge for edge in graph.edges()]
    for (i,j) in edges:
        H += J*sz(hi,i)*sz(hi,j)
    
    return graph, hi, H

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------qGPS Setup------------------------------------------------------------------------------
#Support dimension parameter M
M=3

graph, hi, H = ising_hamiltonian(N=9)

model = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)

sa = nk.sampler.MetropolisLocal(hi)
vs = nk.vqs.MCState(sa, model, n_samples=1, chunk_size=1, seed=1) #Most VMC textbooks suggest number of samples = 5-10*number of parameters
#Most VMC textbooks suggest number of samples = 5-10*number of parameters
vs.n_samples =10*vs.n_parameters

#Array containing all configurations of the hilbert space
dataset_configs = jnp.array(hi.states_to_local_indices(hi.all_states()))

#-------------------------------------------------------------------------Quantum Geometric Tensor---------------------------------------------------------------------------

qgt = vs.quantum_geometric_tensor()
_, grad = vs.expect_and_grad(H)
qgt_times_grad = qgt@grad
#print(qgt)
#print(qgt_times_grad)


sr = nk.optimizer.SR()
gs = nk.VMC(H, optimizer=nk.optimizer.Adam(learning_rate = 0.01), variational_state=vs, preconditioner=sr)
#print(gs.state)
#print(gs.energy)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def estimate_energy_and_gradient(params,samples: list):

    #Computing the local energies using the built in method of the netket.vqs.MCState object.
    E_loc = vs.local_estimators(H)

    E_average = jnp.mean(E_loc) #sum of local energues over number of local energies computed
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance/E_loc.size)

    stats_object = nk.stats.Stats(mean = E_average, error_of_mean = E_error, variance = E_variance)

    #Compute the gradient
    
    #First define the function to be differentiated
    logpsi_sigma_fun = lambda params : vs._apply_fun({"params": params}, samples[0])

    #use jax.vjp to differentiate
    _, vjpfun = jax.vjp(logpsi_sigma_fun, params)
    E_grad = vjpfun((E_loc[0] - E_average)/E_loc.size)
    return stats_object, E_grad[0]

def ground_state(n_iterations):

    #vs.parameters is the epsilon tensor in the form {"epsilon":[[[...]]]}
    #create data logger for the purpose of plotting
    logger = nk.logging.RuntimeLog()

    for i in tqdm(range(n_iterations)):
        
        #generate new samples using new parameters (reseting samples is called automatically when the parameters are updated)
        samples = vs.sample()

        #compute the energy and gradient estimates
        E,E_grad = estimate_energy_and_gradient(vs.parameters,samples)

        #apply the inverse of the quantum geometric tensor to weight the gradient and correctly update the parameters
        qgt = vs.quantum_geometric_tensor()
        grad_updates = qgt@E_grad

        
        #------------------------------------------------Wavefunction Logging--------------------------------------------------------------------
        #compute the log of the whole wavefunction given the current parameters
        logpsi = vs._apply_fun({"params": vs.parameters}, dataset_configs[hi.n_states])

        #Compute the normalised wavefunction from the log of the wavefunction given by the model
        psi = jnp.exp(logpsi)
        psi = psi/jnp.linalg.norm(psi)
        #-----------------------------------------------------------------------------------------------------------------------------------------

        #update the parameters
        vs.parameters = jax.tree.map(lambda x,y: x-0.001*y, vs.parameters, E_grad) #learning rate of 0.01
        #For every entry in epsilon its reducing it by 0.005 times the corresponding entry in E_grad.
        
        #log the energy and wavefunction at each iteration step
        logger(step = i, item = {"Energy": E, "Wavefunction": psi})

    return logger


logger = ground_state(n_iterations=100)
#print(logger["Energy"]["Mean"]) #Print energy values after each optimisation step for inspection
#plotting the energy data from the logger
plot = plt.plot(logger.data["Energy"]["iters"], logger.data["Energy"]["Mean"])
plt.show()

"""Expected ground state energy is ~ -12.0"""