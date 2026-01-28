import matplotlib.pyplot as plt
import GPSKet.operator.hamiltonian.J1J2 as j1j2
import jax.numpy as jnp


ha = j1j2.get_J1_J2_Hamiltonian(Lx=4, Ly = 6, J2=0, sign_rule=[True,False], on_the_fly_en=True)

inds = [x for x in range(ha.hilbert.n_states)]

e_amps = []
with open("e_WF_Chunk2.txt") as f:
    e_amps = [complex(f.readline().strip()) for i in range(len(inds))]
    #for i in inds:
        #e_amps.append(f.readline())
f.close()
e_amps = jnp.array(e_amps)
e_amps = e_amps/jnp.linalg.norm(e_amps)

amps = []
with open("4X6J20Psi0.txt") as f:
    amps = [complex(f.readline().strip()) for i in range(len(inds))]
    #for i in inds:
        #amps.append(f.readline())
f.close()

plt.plot(inds,e_amps, label = 'fit', color = 'r')
plt.plot(inds,amps, label = 'exact', color = 'b')
plt.legend()
plt.savefig("FULLWFPREDICTChunk2.png")

