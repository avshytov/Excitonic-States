import numpy as np
import math 
import scipy.linalg as lin
import ldosModule as ldos
from matplotlib import rc
import matplotlib.pyplot as mpl
from matplotlib import rc

Nx = 45
Ny = 45
pxmax = 3.14
pymax = 3.14
xmax=5
xmin=-xmax
ymax=5
ymin=-ymax
nx=90
ny=90
gamma = 0.05
Eval = -1.048
h = 1.0
eta = 0.7
alpha = 2.0

rc('text', usetex = True)
rc('font', family = 'serif')

#read in data from diagonalization program
px, py, E, a = ldos.loadSolution(Nx, Ny, alpha, h, eta)

Rx=np.linspace(xmin,xmax,nx)
Ry=np.linspace(ymin,ymax,ny)

psi3, psi4, probDensity = ldos.transformToRSpace(px, py, a, Rx, Ry)

#calculate the local density of states
LDOS = ldos.makeLDOS(psi3, psi4, probDensity, E, Eval, gamma, nx, ny)

#save calculated 
f = ldos.makeFilename2(alpha, Nx, Ny, eta, h, Eval)
np.savez(f, psi3=psi3, psi4=psi4, probDensity=probDensity)

#create colour plot of LDOS
print ("Creating color plot")
XX, YY = np.meshgrid(Rx, Ry)
mpl.figure()

mpl.pcolor(XX, YY, LDOS)
mpl.colorbar()
mpl.xlim(xmin, xmax)
mpl.ylim(ymin, ymax)

mpl.title(r'LDOS, $\alpha = %g$, $N = %d \times %d$, $\eta = %g$, $h = %g$, $\epsilon = %g$' % (alpha, Nx, Ny, eta, h, Eval))
mpl.xlabel(r'$x$')
mpl.ylabel(r'$y$')
#mpl.savefig("LDOS-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))
mpl.show()
