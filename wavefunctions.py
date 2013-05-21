import numpy as np
import math 
import matplotlib.pyplot as mpl
import scipy.linalg as lin
from matplotlib import rc

rc('text', usetex = True)
rc('font', family = 'serif')

lowerBound = int(input("Enter index of peaks bottom "))
upperBound = int(input("Enter index of peaks top "))
eigNumber = int(input("How many states to be considered "))
indices = np.linspace(lowerBound,upperBound,eigNumber)
indices = indices.astype(int)			#indices should be integers

# Momentum space grid
Nx = 20
Ny = 20
pxmax = 3.14
pymax = 3.14  


gamma = 0.05  # LDOS broadening
eta   = 1.1   # eta = 6 * t' / t ?? Check
              # the kinetic energy is k^2 cos 3\phi + eta * k^2
alpha = 0.5   # Coulomb interaction strength
h     = 5.0   # layer spacing
	
N= 2 * Nx * Ny				#calculate size of matrix

H0 = np.zeros((N,N))		#allocate empty matrix
V = np.zeros((N,N))
H = np.zeros((N,N))

px = np.linspace(-pxmax, pxmax, Nx)
py = np.linspace(-pymax, pymax, Ny)
deltaPx = px[1] - px[0]
deltaPy = py[1] - py[0]
Cv = float(deltaPx * deltaPy) / 2.0 / math.pi

#potential energy terms
for iX1 in range (Nx):			
	for iY1 in range (Ny):
		for iX2 in range (iX1, Nx): # We do not need to scan the matrix
			                    	# twice
			for iY2 in range (Ny):

				i0 = (iX1*Ny + iY1)*2
				j0 = (iX2*Ny + iY2)*2

				px1 = px[iX1] 
				px2 = px[iX2] 
				
				py1 = py[iY1] 
				py2 = py[iY2] 

				phi1 = np.arctan2(py1, px1)
				phi2 = np.arctan2(py2, px2)

				# Never, ever use a == b with floating 
				# point values!
				if iX1 == iX2 and iY1 == iY2:			#avoid division by zero
					q = 0.5 * math.sqrt( deltaPx ** 2 + deltaPy ** 2 )
				else:
				        q = math.sqrt((px1-px2)**2+(py1-py2)**2)

				cosphi = math.cos( phi1 - phi2 ) # not np.cos!
				#cosphi = 1.0
				Uq = -1.0 / q * math.exp( - q * h )
				
				V[i0,j0] = Uq * cosphi
				V[j0,i0] = V[i0, j0] 

				V[j0 + 1, i0 + 1] = Uq
				V[i0 + 1, j0 + 1] = V[j0 + 1, i0 + 1]

V *= Cv

#kinetic energy terms
for iX1 in range (Nx):
	for iY1 in range (Ny):
		i0 = (iX1*Ny + iY1)*2
		px1 = px[iX1] 
		py1 = py[iY1] 
				
		p2   = px1**2 + py1**2
		phi1 = np.arctan2(py1, px1)
				
		if True:
			Ep = p2 * math.cos( 3.0 * phi1 )
		else:
			Ep = p2
					
		H0[ i0,     i0 + 1 ] = Ep
		H0[ i0 + 1, i0     ] = Ep
		
		H0[i0, i0] = H0[i0 + 1, i0 + 1] = p2 * eta 
		
				
H = H0 + alpha * V

print ("Diagonalising...")
(E,a) = lin.eigh(H)		#calculate eigenvalues, a[] are the row eigenvectors
print ("done")
	
Emin = -5.0 
Emax = +5.0 
nE = 500
Evals = np.linspace (Emin, Emax, nE)

Rmin = -10.0
Rmax = 10.0
nR = 500
Rvals = np.linspace (Rmin, Rmax, nR)

D    = np.zeros((len(Evals), len(E)))
for epsilon in range(len(Evals)):
	for n in range(len(E)):
		D[epsilon, n] = 1.0  / ((Evals[epsilon] - E[n])**2 + gamma**2) 
D *= gamma / math.pi 

for theta in [0.0]: 
	print ("Transforming eigenvectors to R-space")

	print ("Generating the Unitary transform")
	U = np.zeros((len(Rvals), N/2), dtype=complex)
	for iX in range(Nx):
		for iY in range(Ny):
			for R in range(len(Rvals)):
				i = (iX * Ny + iY)
				px_i = px[iX] 
				py_i = py[iY]
				pr = (px_i * math.cos(theta) + py_i * math.sin(theta)) * Rvals[R]
				U[R, i] = math.cos(pr) + 1j * math.sin(pr)

	print ("Doing unitary transformation to the R space")
	psi3 = np.zeros((N,len(Rvals)), dtype=complex)
	psi4 = np.zeros((N,len(Rvals)), dtype=complex)
	for n in range (len(E)):
	    psi3[n, :] = np.dot(U,  a[ ::2, n])
	    psi4[n, :] = np.dot(U,  a[1::2, n])

	print ("done")
	
	#calculate the probability densities
	probDensity = np.abs(psi3)**2 + np.abs(psi4)**2

Pvals = np.linspace(0,pxmax**2,N)
#plot the real part of the momentum space wavefunctions
mpl.figure()
mpl.xlabel("P")
mpl.ylabel(r"$Re{\psi_p}$")
for k in range(len(indices)):
	mpl.plot(Pvals,np.real(a[indices[k],:]),label = 'E = %g' % E[indices[k]])
mpl.title ("RealWavefunctionsP")
mpl.savefig ("RealWavefunctionsPNx%geta%galpha%gh%g.pdf" % (Nx,eta,alpha,h))

#plot the imaginary part of the momentum space	wavefunctions
mpl.figure()
mpl.xlabel("P")
mpl.ylabel(r"$Im{\psi_p}$")
for k in range(len(indices)):
	mpl.plot(Pvals,np.imag(a[indices[k],:]),label = 'E = %g' % E[indices[k]])
mpl.title ("ImaginaryWavefunctionsP")
mpl.savefig ("ImaginaryWavefunctionsPNx%geta%galpha%gh%g.pdf" % (Nx,eta,alpha,h))

#plot the probability density in momentum space
mpl.figure()
mpl.xlabel("P")
mpl.ylabel(r"$|\psi_p|^2$")
for k in range(len(indices)):
	mpl.plot(Pvals,np.abs(a[indices[k],:]),label = 'E = %g' % E[indices[k]])
mpl.title ("ProbabilityDensityP")
mpl.savefig ("ProbabilityDensityPNx%geta%galpha%gh%g.pdf" % (Nx,eta,alpha,h))

#plot the real part of the wavefunctions in position space
mpl.figure()
mpl.xlabel("R")
mpl.ylabel(r"$Re{\psi_R}$")
for k in range(len(indices)):
	mpl.plot(Rvals,np.real(psi3[indices[k], :]),label = 'E = %g (1)' % E[indices[k]])
	mpl.plot(Rvals,np.real(psi4[indices[k], :]),label = 'E = %g (2)' % E[indices[k]])
mpl.title ("RealWavefunctionsR")
mpl.savefig ("RealWavefunctionsRNx%geta%galpha%gh%g.pdf" % (Nx,eta,alpha,h))

#plot the imaginary part of wavefunctions in position space
mpl.figure()
mpl.xlabel("R")
mpl.ylabel(r"$Im{\psi_R}$")
for k in range(len(indices)):
	mpl.plot(Rvals,np.imag(psi3[indices[k], :]),label = 'E = %g (1)' % E[indices[k]])
	mpl.plot(Rvals,np.imag(psi4[indices[k], :]),label = 'E = %g (2)' % E[indices[k]])
mpl.title ("ImaginaryWavefunctionsR")
mpl.savefig ("ImaginaryWavefunctionsRNx%geta%galpha%gh%g.pdf" % (Nx,eta,alpha,h))

#plot probability density in position space
mpl.figure()
mpl.xlabel("R")
mpl.ylabel(r"$|\psi_R|^2$")
for k in range(len(indices)):
	mpl.plot(Rvals,probDensity[indices[k], :],label = 'E = %g' % E[indices[k]])
mpl.title ("ProbabilityDensityR")
mpl.savefig ("ProbabilityDensityRNx%geta%galpha%gh%g.pdf" % (Nx,eta,alpha,h))

mpl.legend()

mpl.show()
