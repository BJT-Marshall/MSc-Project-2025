import matplotlib.pyplot as plt
import netket as nk
import jax.numpy as jnp
import GPSKet.operator.hamiltonian.J1J2 as j1j2


ha = j1j2.get_J1_J2_Hamiltonian(Lx=12,  J2=0, sign_rule=[True,False], on_the_fly_en=True)
e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
amps = jnp.array(state.flatten())

inds = [i for i in range(len(ha.hilbert.all_states()))]

data_inds = []
with open("indices.txt") as f:
    for j in range(60):
        data_inds.append(f.readline().strip())

e_amps = []
with open ("amps-indices.txt") as f:
    for k in inds:
        e_amps.append(jnp.exp(complex(f.readline().strip())))

e_amps = jnp.array(e_amps)/jnp.linalg.norm(jnp.array(e_amps))

plt.plot(inds, amps, color = 'b', label = 'data')
plt.plot(inds, e_amps, color = 'r', label = 'estimate')
plt.plot(data_inds, e_amps, color = 'g', label = 'fit')
plt.legend()
plt.xlabel("Spin Configuration")
plt.ylabel("Wavefunction Amplitude")
plt.show()

i = []
with open("indices.txt") as f:
    for j in range(56):
        i.append(f.readline().strip())


indices = [x for x in range(70)]
e_amps = []
with open ("amps-indices.txt") as f:
    for k in indices:
        e_amps.append(jnp.exp(complex(f.readline().strip())))

e_amps = jnp.array(e_amps)/jnp.linalg.norm(jnp.array(e_amps))

sampled_amps = []
sampled_e_amps = []
for element in i:
    sampled_amps.append(amps[int(element)])
    sampled_e_amps.append(e_amps[int(element)])

z = [u for u in range(56)]
plt.plot(z,sampled_amps, label ='data', color = 'b')
plt.plot(z,sampled_e_amps, label=  'estimate', color = 'r')
plt.show()


indices = [x for x in range(70)]
e_amps = []
with open ("2D-Data-X4-Y2-J2-0-A--0-M-100-I-10-T-30.txt") as f:
    for i in indices:
        e_amps.append(jnp.exp(complex(f.readline().strip())))

e_amps = jnp.array(e_amps)/jnp.linalg.norm(jnp.array(e_amps))


plt.plot(indices, amps, color = 'b', label = 'Exact State')
plt.plot(indices, e_amps, color = 'r', label = 'Predicted State')
plt.xlabel("Spin Configuration Indices")
plt.ylabel("Wavefunction Amplitudes")
plt.title("LASSO Fitting of 4x4 Spin-1/2 Lattice Isling Model Ground State on 80% Data Set. M = 100, I = 10.")
plt.legend()
#plt.savefig("Plots/2D-Test-Plot-2")
#plt.show()
plt.clf()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(24)
c=0
x = [y for y in range(200)]
for a_R in [1e-9]: #[0,1e-9,1e-8,1e-7]
    for m in [100,200]:
        e_amps = []
        with open ("2D-Data-X4-Y2-J2-0-A-"+str(a_R)+"-0-M-"+str(m)+"-I-10-T-30.txt") as f:
            for i in indices:
                e_amps.append(jnp.exp(complex(f.readline().strip())))

        e_amps = jnp.array(e_amps)/jnp.linalg.norm(jnp.array(e_amps))


        plt.plot(indices, amps, color = 'b', label = 'Exact State')
        plt.plot(indices, e_amps, color = 'r', label = 'Predicted State')
        plt.xlabel("Spin Configuration Indices")
        plt.ylabel("Wavefunction Amplitudes")
        plt.title("LASSO Fitting of 4x4 Spin-1/2 Lattice Isling Model Ground State on 80% Data Set. M = 100, I = 10.")
        plt.legend()
        #plt.savefig("Plots/2D-Test-Plot-2")
        plt.show()
        plt.clf()
    overlaps = []
    c +=1
    with open("Training_Data_80_"+str(a_R)+".txt") as f:
                for i in range(200):
                    overlaps.append(float(f.readline().strip()))
    plt.plot(x,overlaps, label = str(a_R), color = cmap(3*c))

plt.xlabel("Iterations")
plt.ylabel("Overlap")
plt.title("4x2 Lattice, M = 100, T=0.8")
plt.legend()
plt.savefig("Plots/Training_Data_80_Test4")
plt.show()
