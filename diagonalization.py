import numpy as np
import math 
import scipy.linalg as lin
import LDOS2 as ldos
import matplotlib.pyplot as mpl
from matplotlib import rc

rc('text', usetex = True)
rc('font', family = 'serif')

Nx = 50
Ny = 50
pxmax = 3.14
pymax = 3.14
alpha = 2.0
h = 1.0
eta = 0.7

#calculate e-values and e-vectors
px, py, E, a = ldos.solveAndSave(Nx, Ny, pxmax, pymax, alpha, h, eta)

#save eigenvalues in seperate file for ease of use with wavefunction analysis
eigFile = open("ENERGIESalpha=%g-h=%g-eta=%g.txt" % (alpha,h,eta),'w')

N= 2 * Nx * Ny

for n in range (N):
	eigFile.write(str(n))
	eigFile.write('\t')
	eigFile.write(str(E[n]))
	eigFile.write('\n')
	
eigFile.close()

#create eigenvalue histogram
binsVal = 100
mpl.figure()
mpl.xlabel("Energy")
mpl.ylabel("Number of states")
mpl.hist(E, bins=binsVal)
mpl.title (r"\textbf{Energy Eigenvalues }\n $\alpha =$ %g, $h =$ %g, $\eta =$ %g" % (alpha,h,eta))
mpl.savefig("ENERGIES-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))
mpl.show()
