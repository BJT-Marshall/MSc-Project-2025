import plotly as ply
from LASSO_Sweeping import lasso_sweeping, overlap
import netket as nk
import jax.numpy as jnp
import matplotlib.pyplot as plt
import quspin as qs
import GPSKet.operator.hamiltonian.J1J2 as j1j2
import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.basis import spin_basis_general
from QuSpin2D import square_lattice_2D_J1_J2_Hamiltonian, ED_for_and_save_psi_0, basis_to_configs, to_log_amps, QuSpin_inds
import GPSKet
from GPSKet.models import qGPS
from GPSKet.supervised.supervised_qgps import QGPSLogSpaceFit
from random import sample
from Chunk_Calling_qGPS import chunk_qGPS_call, apply_qGPS, final_wavefunction, read_configs, read_configs_indices
from handle_epsilon import export_epsilon, import_epsilon, check_epsilons, all_amps_to_file, chunk_amps_to_file


ha = j1j2.get_J1_J2_Hamiltonian(Lx=4, Ly = 6, J2=0, sign_rule=[True,False], on_the_fly_en=True)
#e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
#amps = jnp.array(state.flatten())

#2704156

#Manual sampling----------------------------------------------------------

percentage = 0.001

random_inds = sorted(sample(range(ha.hilbert.n_states), int(percentage*ha.hilbert.n_states)))
print("random inds selected")

random_states = ha.hilbert.numbers_to_states(random_inds)
#random_states = [ha.hilbert.numbers_to_states([x]) for x in random_inds]
print("random states in list")

random_configs = jnp.array([ha.hilbert.states_to_local_indices(x) for x in random_states])
print("random configs in array")

#random_log_amps = jnp.array([log_amps[x] for x in random_inds])
#print("random log amps in array")

#-------------------------------------------------------------------------

amps = []
with open("4X6J20Psi0.txt") as f:
    i=0
    for i in range(2704156):
        if i in random_inds:
            amps.append(float(f.readline()))
        else:
            f.readline()
        i+=1

amps = jnp.array(amps)
print("random amps in array")


g = nk.graph.Grid([4, 6], max_neighbor_order=1, pbc=True)
log_amps_R = jnp.log(jnp.abs(amps))
log_amps_I = jnp.angle(amps)
log_amps = []

for i in range(len(log_amps_R)):
    log_amps.append(log_amps_R[i]+log_amps_I[i]*1j)

random_log_amps = jnp.array(log_amps)
print("random logamps in array")



H = square_lattice_2D_J1_J2_Hamiltonian(6,2,1,0)

ovs = []
for a in [-9]:
    vs_R, vs_I, learning_R, learning_I, ov, phase_shift = lasso_sweeping(
            iterations = [25,25],
            alpha_ = [10**(a),10**(a)],
            configs = random_configs, #ha.hilbert.states_to_local_indices(ha.hilbert.all_states()),
            hilbert = ha.hilbert,
            graph = g,
            M = [100,125],
            scaling = True, #FALSE SCALING TO GET AROUND COMPUTATIONAL ISSUE
            log_amps = random_log_amps, #log_amps,
            seed = 1
            )
    
    ovs.append(ov[-1])

print(ovs)
print(phase_shift)

export_epsilon(learning_R, filename = "e_R_chunked2")
export_epsilon(learning_I, filename = "e_I_chunked2")
chunk_amps_to_file(ha, vs_R, vs_I, learning_R, learning_I, "e_WF_Chunk2", 1000)
#all_amps_to_file(ha, vs_R, vs_I, learning_R, learning_I, "e_WF")
#e = import_epsilon("e", [2,100,24])
#print(check_epsilons(learning_R.epsilon, e))


#Final FITTED amps.
"""e_log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, ha.hilbert.all_states())
    
#plt.plot(jnp.arange(len(ha.hilbert.all_states())),log_amps_R, color = 'b', label = "data") #dont have all the amps loaded so cant plot this rn
plt.plot(jnp.arange(ha.hilbert.n_states),e_log_amps_R, color = 'r', label = 'fit')
plt.legend()
plt.savefig("0LogFit.png") 
plt.clf()
    
e_amps_R = jnp.exp(jnp.array(e_log_amps_R))
e_amps_R = e_amps_R/jnp.linalg.norm(e_amps_R)
e_log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, ha.hilbert.all_states())
e_log_amps_I += phase_shift
e_log_amps = jnp.array([e_log_amps_R[x] + e_log_amps_I[x]*1j for x in range(len(e_log_amps_R))])
e_amps = jnp.array([e_amps_R[i]*jnp.exp(1j*e_log_amps_I[i]) for i in range(len(e_amps_R))])
e_amps = e_amps/jnp.linalg.norm(e_amps)
amps = jnp.exp(jnp.array(log_amps))
amps = amps/jnp.linalg.norm(amps)
    
    
#plt.plot(jnp.arange(len(ha.hilbert.all_states())),amps, color = 'b', label = "data")
plt.plot(jnp.arange(ha.hilbert.n_states),e_amps_R, color = 'r', label = 'fit')
plt.legend()
plt.savefig("0AmpsFit.png") 
plt.clf()"""

#print("Full Fit "+ str(overlap(e_log_amps, log_amps)))

#Goal is to get it perfectly fitting the DATA it is passed for this 4x6 system. Then can read in the full lists of amplitudes and compare the predicitons.