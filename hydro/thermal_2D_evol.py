import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import astropy.units as u
import astropy.constants as cst

from astropy.cosmology import Planck18 as cosmo

# constants
gamma = 5./3     # adiabatic index for ideal gas
alph_therm = 3.35e-4 *u.m**2/u.s # 111. * u.mm**2/u.s # thermal diffusivity

# Simulation parameters
N = 64 # resolution
boxsize = 1. * u.m
dx = boxsize/N
vol = dx**3
area = dx**2

# mesh grid
x_mid = np.linspace(0.5*dx, boxsize-0.5*dx, N)
X, Y = np.meshgrid(x_mid, x_mid)  

# Time
t_start = 0 * u.s
t_end = 60 * u.s

# time limit for stability in finite diffentiation to ensure numerical solution remains stable
t_lim = (0.5*dx**2/alph_therm).cgs


# Initial Conditions temperature
#T = np.zeros((N,N)) * u.K
#T[N//2,N//2] = 1e2 * u.K
np.random.seed(2025)
T = (np.random.normal(loc=0., scale=50., size=(N,N))+1e2) * u.K

# pick a timestep that is a factor 2 times smaller then the limit
dt = t_lim / 4 #2**np.ndim(T)
print(dt)

# loop time
t = t_start
 
fig, axs = plt.subplots(figsize=(8, 7), constrained_layout=True)
mesh = axs.pcolormesh(x_mid.value, x_mid.value, T.value, cmap='jet', vmin=1e-3, vmax=T.value.mean())
plt.colorbar(mesh, ax=axs, label='T [%s]' %T.unit)
axs.set_xlabel('x [%s]' %x_mid.unit), axs.set_ylabel('y [%s]' %x_mid.unit)

# loop in time
while t < t_end:
    if t + dt > t_end:
        dt = t_end - t
    t += dt
    
    # time evolving temperature : temperature now
    W = T.copy()
    
    # loop in space
    for i, x_i in enumerate(x_mid):
        for j, y_j in enumerate(x_mid):          
            if(i == 0 and j == 0):          # bottom left corner
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[i+1,j] + W[-1,j] + W[i,j+1] + W[i,-1] - 4.*W[i,j])).cgs
            elif(i == N-1 and j == N-1):    # top right corner
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[0,j] + W[i-1,j] + W[i,0] + W[i,j-1] - 4.*W[i,j])).cgs    
            elif(i == N-1 and j == 0):      # top left corner
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[0,j] + W[i-1,j] + W[i,j+1] + W[i,-1] - 4.*W[i,j])).cgs
            elif(i == 0 and j == N-1):      # bottom right corner
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[i+1,j] + W[-1,j] + W[i,0] + W[i,j-1] - 4.*W[i,j])).cgs
            elif(i == 0 and j < N-1):       # bottom row
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[i+1,j] + W[-1,j] + W[i,j+1] + W[i,j-1] - 4.*W[i,j])).cgs
            elif(i < N-1 and j == 0):       # left column
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[i+1,j] + W[i-1,j] + W[i,j+1] + W[i,-1] - 4.*W[i,j])).cgs
            elif(i == N-1 and j > 0):       # top row
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[0,j] + W[i-1,j] + W[i,j+1] + W[i,-1] - 4.*W[i,j])).cgs
            elif(i > 0 and j == N-1):       # right column
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[i+1,j] + W[-1,j] + W[i,0] + W[i,j-1] - 4.*W[i,j])).cgs
            else:
                T[i,j] = (W[i,j] + alph_therm * dt / dx**2 * (W[i+1,j] + W[i-1,j] + W[i,j+1] + W[i,j-1] - 4.*W[i,j])).cgs
        
    # System energy
    tot_E = np.sum((T*cst.k_B).to('eV'))
    avrg_T = np.mean(T)

    # Plot results and update        
    axs.set_title('t = %.3f %s\nE = %.2f %s   T = %.2f %s' %(t.value, t.unit, tot_E.value, tot_E.unit, avrg_T.value, avrg_T.unit))
    mesh.set_array(T)
    #line1.set_xdata(x_mid)
    #line1.set_ydata(T)
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    plt.pause(0.01)

    
plt.show(), plt.clf()
