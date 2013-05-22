import numpy as np
import math 
import scipy.linalg as lin
import LDOS2 as ldos
from matplotlib import rc
import matplotlib.pyplot as mpl
from matplotlib import rc

Nx = 50
Ny = 50
pxmax = 3.14
pymax = 3.14
gamma = 0.05
theta = 0.0
h = 1.0
eta = 0.7
alpha = 2.0

rc('text', usetex = True)
rc('font', family = 'serif')

#read in data from diagonalization program
px, py, E, a = ldos.loadSolution(Nx, Ny, alpha, h, eta)

#transform the eigen-vectors from momentum space to real space
Rmin = -10.0
Rmax = 10.0
nR = 500
px, py = ldos.makeGrid(Nx, Ny, pxmax, pymax)
Rvals = np.linspace (Rmin, Rmax, nR)
psi3, psi4, probDensity = ldos.transformToRSpace(px, py, a, theta, Rvals)

#calculate the local density of states
Emin = -5.0 
Emax = +5.0 
nE = 500
Evals = np.linspace (Emin, Emax, nE)
LDOS = ldos.makeLDOS(psi3, psi4, probDensity, E, Evals, gamma)

#save calculated 
f = ldos.makeFilename2(alpha, Nx, Ny, eta, h)
np.savez(f, psi3=psi3, psi4=psi4, probDensity=probDensity)

#create colour plot of LDOS
print ("Creating color plot")
XX, YY = np.meshgrid(Rvals, Evals)
mpl.figure()
mpl.pcolor(XX, YY, LDOS)
mpl.colorbar()
mpl.xlim(Rmin, Rmax)
mpl.ylim(Emin, Emax)

theta_pi = theta / math.pi
if (abs(theta_pi) < 1e-4): 
	theta_s = r'$\theta = 0$'
elif (abs(theta_pi - 1.0) < 1e-4): 
	theta_s = r'$\theta = \pi$'
else:
	theta_s = r'$\theta = %g \pi$' % theta_pi

mpl.title(r'LDOS, $\alpha = %g$, $N = %d \times %d$, $\eta = %g$, $h = %g$,  %s' % (alpha, Nx, Ny, eta, h, theta_s))
mpl.xlabel(r'Distance $R$')
mpl.ylabel(r'Energy $\epsilon$')
#mpl.savefig("LDOS-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))
mpl.show()
