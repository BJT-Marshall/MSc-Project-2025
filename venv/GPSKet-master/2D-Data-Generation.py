from LASSO_Sweeping import lasso_sweeping
import netket as nk
import jax.numpy as jnp
import GPSKet.operator.hamiltonian.J1J2 as j1j2
import matplotlib.pyplot as plt
#import time


ha = j1j2.get_J1_J2_Hamiltonian(Lx=4, Ly=2, J2=0, sign_rule=[True,False], on_the_fly_en=True)
e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
amps = jnp.array(state.flatten())
#ln(z) = ln(|z|) + iArg(z)
log_amps_R = jnp.log(jnp.abs(amps))
log_amps_I = jnp.angle(amps)
log_amps = []

for i in range(len(log_amps_R)):
    log_amps.append(log_amps_R[i]+log_amps_I[i]*1j)

log_amps = jnp.array(log_amps)
print(len(log_amps))
#input()

times = {"Real": [], "Imag": [], "Time": []}
overlaps = {"Real": [], "Full": []}
features = {"Real": [], "Imag": []}
f_left = []

for a_R in [10**(-9)]: #[-9,-7,-5,-3]  [0,10**(-9),10**(-8), 10**(-7)]
    overlap_limit = 0
    #m=1
    #while overlap_limit<0.999:
    for m in [15]:

        #st = time.time()

        e_log_amps, ov, ov_R, ov_I, f_R, f_I, a, e_R, e_I, indices = lasso_sweeping(
            [30,1],
            [a_R,0],
            60, #int(len(ha.hilbert.all_states()))
            ha,
            m,
            True,
            log_amps,
            1
        )
        overlap_limit = ov[-1]
        #m+=1

        with open("indices.txt", "w") as f:
            for element in indices:
                f.write(f"{element}\n")

        """with open("Training_Data_80_"+str(a_R)+str(m[0])+".txt", "w") as f:
            for element in ov_R:
                f.write(f"{element}\n")"""

        #et = time.time()
        times["Real"].append(a_R)
        times["Imag"].append(0)
        #times["Time"].append(et-st)
        overlaps["Real"].append(ov_R[-1])
        overlaps["Full"].append(ov[-1])
        features["Real"].append(f_R[-1])
        features["Imag"].append(f_I[-1])
        print(times)
        print(overlaps)
        print(features)
        with open("amps-indices.txt" , "w") as f:
            for element in e_log_amps:
                f.write(f"{element}\n")
        f_left.append((m*16)-f_R[-1])

    print(m)

with open("data10.txt", "w") as f:
    for i in range(len(times["Real"])):
        f.write(f"{str(times["Real"][i])+", "+str(times["Imag"][i])}\n")
    for i in range(len(overlaps["Real"])):
        f.write(f"{str(overlaps["Real"][i])+", "+str(overlaps["Full"][i])}\n")
    for i in range(len(features["Real"])):
        f.write(f"{str(features["Real"][i])+", "+str(features["Imag"][i])+", "+str(f_left[i])}\n")



"""print(ov_R)
print(ov)
print(e_R)
print(e_I)
print(f_R)
print(f_I)
print("Excecution Time: ", str(et-st))

amps = jnp.array([jnp.exp(log_amps_R[i])*jnp.exp(log_amps_I[i]*1j) for i in range(len(log_amps_R))])
amps = amps/jnp.linalg.norm(amps)

e_amps = jnp.array([jnp.exp(e_log_amps[i]) for i in range(len(e_log_amps))])
e_amps = e_amps/jnp.linalg.norm(e_amps)

with open("2D-Data-X4-Y4-J2-0-A-00001-0-M-15" , "w") as f:
    for element in e_log_amps:
        f.write(f"{element}\n")

indices = [x for x in range(len(amps))]

plt.plot(indices,amps, label = "data", color = "b")
plt.plot(indices,e_amps, label = "estimate", color = "r")
plt.legend()
plt.show()"""