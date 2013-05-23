import numpy as np
import math 
import scipy.linalg as lin
import LDOS2hex as ldoshex
from matplotlib import rc
import matplotlib.pyplot as mpl

Nx = 25
pxmax = 3.14
gamma = 0.02
theta = 0.1
h = 1.0
eta = 0.7
alpha = 2.0
lowerBound = int(input("Enter index of peaks bottom "))
upperBound = int(input("Enter index of peaks top "))
eigNumber = int(input("How many states to be considered "))
indices = np.linspace(lowerBound,upperBound,eigNumber)
indices = indices.astype(int)			#indices should be integers

Rmin = -20.0
Rmax = 20.0
nR = 500
Rvals = np.linspace (Rmin, Rmax, nR)

#retrieve necessary entries from previously saved files
psi3, psi4, probDensity = ldoshex.loadSolution2(Nx, alpha, h, eta)
px, py, E, a = ldoshex.loadSolution(Nx, alpha, h, eta)
Pvals = np.sqrt(px**2 + py**2)

#plot the real part of the momentum space wavefunctions
# I do not understand this code. Does it plot both components of the 
# w.f. on the same graph? -- AVS
if False:
  mpl.figure()
  mpl.xlabel("P")
  mpl.ylabel(r"$Re{\psi_p}$")
  for k in range(eigNumber):
	print np.shape(Pvals), np.shape(a[indices[k], :])
	mpl.plot(Pvals,np.real(a[indices[k],:]),label = 'E = %g' % E[indices[k]])
  mpl.title ("RealWavefunctionsP")
  mpl.legend()
  mpl.savefig ("RealWavefunctionsP-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))

#plot the imaginary part of the momentum space	wavefunctions
if False: # See above -- AVS
  mpl.figure()
  mpl.xlabel("P")
  mpl.ylabel(r"$Im{\psi_p}$")
  for k in range(eigNumber):
	mpl.plot(Pvals,np.imag(a[indices[k],:]),label = 'E = %g' % E[indices[k]])
  mpl.title ("ImaginaryWavefunctionsP")
  mpl.legend()
  mpl.savefig ("ImaginaryWavefunctionsP-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))

#plot the probability density in momentum space
if False: # see above -- AVS
  mpl.figure()
  mpl.xlabel("P")
  mpl.ylabel(r"$|\psi_p|^2$")
  for k in range(eigNumber):
	mpl.plot(Pvals,np.abs(a[indices[k],:]),label = 'E = %g' % E[indices[k]])
  mpl.title ("ProbabilityDensityP")
  mpl.legend()
  mpl.savefig ("ProbabilityDensityP-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))

#plot the real part of the wavefunctions in position space
mpl.figure()
mpl.xlabel("R")
mpl.ylabel(r"$Re{\psi_R}$")
for k in range(eigNumber):
	mpl.plot(Rvals,np.real(psi3[indices[k], :]),label = 'E = %g (1r)' % E[indices[k]])
	mpl.plot(Rvals,np.real(psi4[indices[k], :]),label = 'E = %g (2r)' % E[indices[k]])
	mpl.plot(Rvals,np.imag(psi3[indices[k], :]),label = 'E = %g (1i)' % E[indices[k]])
	mpl.plot(Rvals,np.imag(psi4[indices[k], :]),label = 'E = %g (2i)' % E[indices[k]])
mpl.title ("RealWavefunctionsR")
mpl.legend()
mpl.savefig ("RealWavefunctionsR-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))

#plot the imaginary part of wavefunctions in position space
mpl.figure()
mpl.xlabel("R")
mpl.ylabel(r"$Im{\psi_R}$")
for k in range(eigNumber):
	mpl.plot(Rvals,np.imag(psi3[indices[k], :]),label = 'E = %g (1)' % E[indices[k]])
	mpl.plot(Rvals,np.imag(psi4[indices[k], :]),label = 'E = %g (2)' % E[indices[k]])
mpl.title ("ImaginaryWavefunctionsR")
mpl.legend()
mpl.savefig ("ImaginaryWavefunctionsR-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))

#plot probability density in position space
mpl.figure()
mpl.xlabel("R")
mpl.ylabel(r"$|\psi_R|^2$")
for k in range(eigNumber):
	mpl.plot(Rvals,probDensity[indices[k], :],label = 'E = %g' % E[indices[k]])
mpl.title ("ProbabilityDensityR")
mpl.legend()
mpl.savefig ("ProbabilityDensityR-alpha= %g-h= %g-eta= %g.pdf" % (alpha,h,eta))

mpl.show()
