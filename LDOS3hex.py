import numpy as np
import math 
import scipy.linalg as lin
import LDOS2hex as ldoshex
from matplotlib import rc
import matplotlib.pyplot as mpl
from matplotlib import rc

Nx = 25
pxmax = 3.14
gamma = 0.02
theta = 0.1
h = 1.0
eta = 0.7
alpha = 2.0

rc('text', usetex = True)
rc('font', family = 'serif')

#read in data from diagonalization program
px, py, E, a = ldoshex.loadSolution(Nx, alpha, h, eta)

#transform the eigen-vectors from momentum space to real space
Rmin = -20.0
Rmax = 20.0
nR = 500

Rvals = np.linspace (Rmin, Rmax, nR)

def theta_r(r):
    if r > 0: 
    	return 0.0
    else:
    	return math.pi / 6.0
thetavals = np.vectorize(theta_r)(Rvals)
psi3, psi4, probDensity = ldoshex.transformToRSpace(px, py, a, thetavals, abs(Rvals))

#calculate the local density of states
Emin = -5.0 
Emax = +5.0 
nE = 500
Evals = np.linspace (Emin, Emax, nE)
LDOS = ldoshex.makeLDOS(psi3, psi4, probDensity, E, Evals, gamma)

#save calculated 
f = ldoshex.makeFilename2(alpha, Nx, eta, h)
np.savez(f, psi3=psi3, psi4=psi4, probDensity=probDensity)

#create colour plot of LDOS
print ("Creating color plot")
XX, YY = np.meshgrid(Rvals, Evals)
mpl.figure()
#fig, ax = plt.subplots(1)
mpl.pcolor(XX, YY, LDOS)
mpl.colorbar()
mpl.xlim(Rmin, Rmax)
mpl.ylim(Emin, Emax)

#theta_pi = theta / math.pi
#if (abs(theta_pi) < 1e-4): 
#	theta_s = r'$\theta = 0$'
#elif (abs(theta_pi - 1.0) < 1e-4): 
#	theta_s = r'$\theta = \pi$'
#else:
#	theta_s = r'$\theta = %g \pi$' % theta_pi
textstr1 = 'theta=pi/6'
textstr2 = 'theta=0'
#ax.text(0.05, 0.05, textstr1, transform=ax.transAxes, fontsize=14,verticalalignment='bottom', bbox=props)
#ax.text(0.95, 0.05, textstr2, transform=ax.transAxes, fontsize=14,verticalalignment='bottom', bbox=props)
mpl.title(r'LDOS, $\alpha = %g$, $N = %d, $\eta = %g$, $h = %g$' % (alpha, Nx, eta, h))
mpl.xlabel(r'Distance $R$')
mpl.ylabel(r'Energy $\epsilon$')
#mpl.savefig("LDOS-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))
mpl.show()
