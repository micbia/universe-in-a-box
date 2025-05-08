import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as cst

from astropy.cosmology import Planck18 as cosmo

# constants
gamma = 5./3     # adiabatic index for ideal gas
alph_therm = 3.35e-4 *u.m**2/u.s # 111. * u.mm**2/u.s # thermal diffusivity

# Simulation parameters
N = 100 # resolution
boxsize = 1. * u.m
dx = boxsize/N
vol = dx**3
area = dx**2

# mesh grid
x_mid = np.linspace(0.5*dx, boxsize-0.5*dx, N)
X = x_mid

# Time
t_start = 0 * u.s
t_end = 10*60 * u.s

# time limit for stability in finite diffentiation to ensure numerical solution remains stable
t_lim = (0.5*dx**2/alph_therm).cgs

# pick a timestep that is a factor 2 times smaller then the limit
dt = t_lim / 2
print(dt)

# Initial Conditions temperature
T = np.zeros(N) * u.K
T[N//2-10:N//2+10] = 1e3 * u.K
print('mean temperature: %.3f %s' %(np.mean(T).value, T.unit))
#T[0] = 1e3 * u.K
#T[-1] = 1e3 * u.K

# loop time
t = t_start
 
fig, axs = plt.subplots()
line1, = axs.plot(x_mid, T)
#mesh = axs.pcolormesh([T], cmap=plt.cm.jet, vmin=0, vmax=100)
#mesh = axs.pcolormesh(x_mid.value, [0, 1], np.vstack([T.value, T.value]), cmap='jet', vmin=0, vmax=100)
#plt.colorbar(mesh, ax=axs)

# loop in time
while t < t_end:
    if t + dt > t_end:
        dt = t_end - t
    t += dt
    
    # time evolving temperature : temperature now
    W = T.copy()
    
    # loop in space
    for i in range(N):
        if(i == 0):
            # Left-side boundary condition
            T[i] = (W[i] + alph_therm * dt / dx**2 * (W[i+1] - 2.*W[i] + W[-1])).cgs
        elif(i == N-1):
            # Right-side boundary condition
            T[i] = (W[i] + alph_therm * dt / dx**2 * (W[0] - 2.*W[i] + W[i-1])).cgs
        else:
            T[i] = (W[i] + alph_therm * dt / dx**2 * (W[i+1] - 2.*W[i] + W[i-1])).cgs
    
    # System energy
    tot_E = np.sum((T*cst.k_B).to('eV'))

    # Plot results and update        
    axs.set_title('t = %.3f %s, E = %.3f %s' %(t.value, t.unit, tot_E.value, tot_E.unit))
    #mesh.set_array(np.vstack([T.value, T.value]))
    line1.set_xdata(x_mid)
    line1.set_ydata(T)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

    
plt.show(), plt.clf()
