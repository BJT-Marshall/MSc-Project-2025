# VMC with the qGPS model ansatz.


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


# -------------------------------------------------------------------------Hilbert Space/Hamiltonian Setup-----------------------------------------------------------------------

# Functions to create shorthand Pauli operators


def sx(hilbert_space, site):
    sx_site = nk.operator.spin.sigmax(hilbert_space, site)

    return sx_site


def sy(hilbert_space, site):
    sy_site = nk.operator.spin.sigmay(hilbert_space, site)

    return sy_site


def sz(hilbert_space, site):
    sz_site = nk.operator.spin.sigmaz(hilbert_space, site)

    return sz_site


def ising_hamiltonian(N, h=1, J=1):
    graph = nk.graph.Hypercube(length=int(jnp.sqrt(N)), n_dim=2, pbc=True)

    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    H = nk.operator.LocalOperator(hi)  # H is a local operator on the Hilbert space

    # Adding all the terms acting on a single site. (i.e. first term sum in the Hamiltonain)

    nodes = [node for node in graph.nodes()]

    for site in nodes:
        H -= h * sx(hi, site)

    # Adding all the terms acting on nieghboring sites. (i.e. second sum in the Hamiltionian)

    edges = [edge for edge in graph.edges()]

    for i, j in edges:
        H += J * sz(hi, i) * sz(hi, j)

    return graph, hi, H


# Support dimension parameter M

M = 5


graph, hi, H = ising_hamiltonian(N=9)


model = qGPS(hi, M, init_fun=GPSKet.nn.initializers.normal(1.0e-3), dtype=float)


sa = nk.sampler.MetropolisLocal(hi)

vs = nk.vqs.MCState(sa, model, n_samples=500, chunk_size=1, seed=1)


op = nk.optimizer.Sgd(learning_rate=0.001)  # the optimizer
qgt = nk.optimizer.qgt.QGTJacobianDense(
    mode="real"
)  # the quantum geometric tensor object, this is the object used to adjust the update directions
# in the stochastic reconfiguration optimizer set the mode to "holomorphic" when using a complex valued GPS

sr = nk.optimizer.SR(
    qgt=qgt,
)  # this is the stochastic reconfiguration preconditioner


gs = nk.driver.VMC(
    H,
    op,
    variational_state=vs,
    preconditioner=sr,
)

# Run optimization
ens = []
for it in gs.iter(500, 1):
    en = gs.energy.mean
    print(en)
    ens.append(en)

nk.exact.lanczos_ed(H)


plt.plot(ens)
plt.show()
