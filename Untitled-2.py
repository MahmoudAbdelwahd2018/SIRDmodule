Spyder Editor

This is a temporary script file.



%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rc('font',size=16)




i0 = 1e-8
r0 = 0
s0 = 1 - i0
R1 = 3./100.
d0 = 0
num_times = 180 # simulate for half a year


times = np.arange(num_times)
Us = np.empty((num_times,4))
Us[0] = np.array([s0,i0,r0,d0])


def rhs(t, U, beta, gamma, sigma):
    s,i,r,d = U[0],U[1],U[2],U[3]
    return np.array([-beta*s*i,
                        beta*s*i-(gamma+sigma)*i,
                        gamma*i,
                        sigma*i
                        ])

def get_beta_gamma(R=5,recovery_time=14):

    gamma = 1./recovery_time
    beta = R*gamma
    return beta,gamma

from scipy.integrate import ode
integrator = ode(rhs)
# set initial conditions and parameters
beta,gamma = get_beta_gamma(R=3.4,recovery_time=14)
sigma = R1*gamma
integrator.set_initial_value(Us[0],0)
integrator.set_f_params(beta,gamma,sigma)

for i in range(len(times)-1):
    t = times[i+1]
    integrator.integrate(t)
    Us[i+1] = integrator.y
    
    

s,i,r,d = Us[:,0],Us[:,1],Us[:,2],Us[:,3]


plt.plot(times,s,label='susceptible')
plt.plot(times,i,label='infected')
plt.plot(times,r,label='recovered')
plt.plot(times,d,label='dead')
plt.legend(bbox_to_anchor=[0.985,0.7])
plt.xlabel('time (days)')
plt.ylabel('fraction of total population')
#Text(0, 0.5, 'fraction of total population')

plt.semilogy(times,s,label='susceptible')
plt.plot(times,i,label='infected')
plt.plot(times,r,label='recovered')
plt.plot(times,d,label='dead')
plt.legend(bbox_to_anchor=[0.985,0.7])
plt.xlabel('time (days)')
plt.ylabel('fraction of total population')