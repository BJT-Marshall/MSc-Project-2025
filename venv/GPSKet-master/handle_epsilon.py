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


def single_chunk_amps(ha, vs_R, vs_I, learning_R, learning_I, inds, write_to_file = False, filename = None):
    states = ha.hilbert.numbers_to_states(inds)
    configs = jnp.array([ha.hilbert.states_to_local_indices(x) for x in states])
    log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs) #again in some chunks
    log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs)
    log_amps = jnp.array([log_amps_R[i]+1j*log_amps_I[i] for i in range(len(inds))])
    amps = jnp.exp(log_amps)

    if write_to_file:
        with open(str(filename)+".txt", "w") as f:
            for amp in amps:
                f.write(f"{str(amp)}\n")

    return amps

def all_amps_to_file(ha, vs_R, vs_I, learning_R, learning_I, filename):
    with open(str(filename)+".txt", "w") as f:
        for i in range(ha.hilbert.n_states):
            state = ha.hilbert.numbers_to_states([i])
            config = ha.hilbert.states_to_local_indices(state)        
            log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, config) #again in some chunks
            log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, config)
            amp = jnp.exp(log_amps_R[i]+1j*log_amps_I[i])
            f.write(f"{str(amp)}\n")
    f.close()

    return None


def chunk_amps_to_file(ha, vs_R, vs_I, learning_R, learning_I, filename, n_chunks):
    #split into n chunks
    chunk_size = int(ha.hilbert.n_states/n_chunks) #floor
    last_chunk = ha.hilbert.n_states - chunk_size*n_chunks
    if last_chunk == 0:
        spill = False
    else:
        spill = True
    marker = 0
    with open(str(filename)+".txt", "w") as f:
        for i in range(n_chunks):
            inds = [x for x in range(marker,marker+chunk_size)]
            states = ha.hilbert.numbers_to_states(inds)
            configs = jnp.array([ha.hilbert.states_to_local_indices(x) for x in states])
            log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs) #again in some chunks
            log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs)
            log_amps = jnp.array([log_amps_R[i]+1j*log_amps_I[i] for i in range(len(inds))])
            amps = jnp.exp(log_amps)
            for amp in amps:
                f.write(f"{str(amp)}\n")
            marker +=chunk_size
            print("Chunk "+str(i)+" computed.")
        #final chunk if needed
        if spill:
            inds = [x for x in range(chunk_size*n_chunks,ha.hilbert.n_states)]
            states = ha.hilbert.numbers_to_states(inds)
            configs = jnp.array([ha.hilbert.states_to_local_indices(x) for x in states])
            log_amps_R = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, configs) #again in some chunks
            log_amps_I = vs_I._apply_fun({"params": {"epsilon": learning_I.epsilon}}, configs)
            log_amps = jnp.array([log_amps_R[i]+1j*log_amps_I[i] for i in range(len(inds))])
            amps = jnp.exp(log_amps)
            for amp in amps:
                f.write(f"{str(amp)}\n")
            print("Last chunk completed.")

    return None


#Export and deal with epsilon tensor for a fit to then handle generating the full configs for it

def export_epsilon(learning_obj, filename = None):
    e = learning_obj.epsilon
    with open(str(filename)+".txt", 'w') as f:
        for i in range(e.shape[0]):
            for j in range(e.shape[1]):
                for k in range(e.shape[2]):
                    f.write(f"{str(e[i][j][k])}\n")
    #first L entries are D=0,M=0,L
    f.close()

    return None

def import_epsilon(filename, shape):
    """shape[0] = D, shape[1] = M, shape[2] = L, i.e. shape = [M,D,L]"""
    with open(str(filename)+".txt") as f:
        e = []
        for i in range(shape[0]):
            Djk = []
            for j in range(shape[1]):
                DMk = []
                for k in range(shape[2]):
                    DMk.append(float(f.readline()))
                Djk.append(DMk)
            e.append(Djk)
    f.close()

    return(jnp.array(e))

def check_epsilons(e,e_):
    """Just for checking if the above two functions work as intended"""
    same = False
    for i in range(e.shape[0]):
        for j in range(e.shape[1]):
            for k in range(e.shape[2]):
                if e[i][j][k] == e_[i][j][k]:
                    same = True
                else:
                    same = False

    return same 
    


