import numpy as np
import math 
import matplotlib.pyplot as mpl
import scipy.linalg as lin
from matplotlib import rc

rc('text', usetex = True)
rc('font', family = 'serif')

# Momentum space grid
Nx = 25
Ny = 25
pxmax = 3.14 
pymax = 3.14  


gamma = 0.05  # LDOS broadening
eta   = 0.00  # t' / t ratio??
alpha = 2.0   # Coulomb interaction strength


if False:
	Nx = int(input("Nx = "))
	Ny = int(input("Ny = "))
	pxmax = float(input("PxMax = "))
	pymax = float(input("PyMax = "))
	gamma = float(input("gamma = "))
	
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
for iX1 in range (Nx):			#1,-0.8		0.005,
	for iY1 in range (Ny):
		for iX2 in range (Nx):
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
				Uq = -1.0 / q
				
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
				
		p    = math.sqrt(px1**2+py1**2)
		phi1 = np.arctan2(py1,px1)
				
		if True:
			Ep = p**2 * math.cos( 3.0 * phi1 )
		else:
			Ep = p**2
					
		H0[ i0,     i0 + 1 ] = Ep
		H0[ i0 + 1, i0     ] = Ep
		
		H0[i0, i0] = H0[i0 + 1, i0 + 1] = (px1 * px1 + py1 * py1) * 3.0 * eta 
		
				
H = H0 + alpha * V

print ("Diagonalising...")
(E,a) = lin.eigh(H)		#calculate eigenvalues, a are the row eigenvectors
print ("done")


if False:
	for icheck in [ 0, N-1, N/2]:
		Echeck = E[icheck]
		ev = a[:, icheck]
		print ("icheck = ", icheck, "norm:",  lin.norm(np.dot(H, ev) - Echeck * ev)) 


		
		
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
	print ("Calculating LDOS")
	#calculate the probability densities
	probDensity = np.abs(psi3)**2 + np.abs(psi4)**2

	#calculate LDOS
	LDOS = np.zeros((len(Evals), len(Rvals)))

	LDOS = np.dot(D, probDensity)

	print ("done")
	print(LDOS)

	XX, YY = np.meshgrid(Rvals, Evals)

	print ("Creating color plot")
	mpl.figure()
	mpl.pcolor(XX, YY, LDOS)
	mpl.colorbar()
	mpl.xlim(Rmin, Rmax)
	mpl.ylim(Emin, Emax)
	theta_pi = theta / math.pi
	
	if    (abs(theta_pi) < 1e-4): 
		theta_s = r'$\theta = 0$'
	elif (abs(theta_pi - 1.0) < 1e-4): 
		theta_s = r'$\theta = \pi$'
	else:
		theta_s = r'$\theta = %g \pi$' % theta_pi
		
	mpl.title(r'LDOS, $\alpha = %g$, $N_x = %d \times %d$, $\eta = %g$ %s' % (alpha, Nx, Ny, eta, theta_s))
	mpl.xlabel(r'Distance $R$')
	mpl.ylabel(r'Energy $\epsilon$')

mpl.figure()
mpl.hist(E, bins=100)

mpl.show()

#global DOS calculation
#def dos(epsilon):			#epsilon will be the elements of the array Evals
#    s = 0.0
#    for Ei in E: 			#loop through the eigenvalues
#        s += gamma / (math.pi  * ((epsilon - Ei)**2 + gamma**2))	#assign a value to the dos at epsilon by finding the 
#    return s														#proximity of each eigenvalue to epsilon and 
																	#calculating lorenzians
#Emin = min(E)
#Emax = max(E)
#dE = (Emax - Emin) / 1000.0
#Evals = np.arange (Emin, Emax, dE)	#defines an array called Evals, which contains
									#elements which are evenly spaced between Emin and Emax	
									#spacing is dE

#dosvals = np.vectorize(dos)(Evals)	#applies the function dos to each of the elements of Evals
									#and stores each result in an element of the array dosvals

#print eigenvalues and DOS values to seperate files
#energyFile = open('eigenvaluespxmax%gpymax%gNx%gNy%g.txt' % (pxmax,pymax,Nx,Ny),'w')
#dosFile = open('DOSgamma%gpxmax%gpymax%gNx%gNy%g.txt' % (gamma,pxmax,pymax,Nx,Ny),'w')

#for j in range (N):
#	print(E[j])
#	energyFile.write(str(E[j]))
#	energyFile.write('\n')

#energyArraySize = len(Evals)
#for m in range (energyArraySize):
#	dosFile.write(str(Evals[m]))
#	dosFile.write(',')
#	dosFile.write(str(dosvals[m]))
#	dosFile.write('\n')
	
#print("min E is = ", min(E))
#print("max E is = ", max(E))

#create DOS plot
#mpl.figure()
#mpl.xlabel("E (57.5eV)")
#mpl.ylabel("Global DOS")
#mpl.plot(Evals, dosvals)
#mpl.axis([-7358,-300,0,400])
#mpl.title ("\\textbf{Smoothened DOS}\n $\gamma =$ %g, $P_{x,max}$ = %g, $P_{y,max}$ = %g, $N_x$ = %g, $N_y$ = %g" % (gamma,pxmax,pymax,Nx,Ny))
#mpl.savefig("dosGamma%gPxmax%gPymax%gNx%gNy%g.pdf" % (gamma,pxmax,pymax,Nx,Ny))	#save DOS plot

#create histogram
#binsVal = 100
#mpl.figure()
#mpl.xlabel("E (57.5eV)")
#mpl.ylabel("Number of states")
#mpl.axis([-10,0,0,10])
#mpl.hist(E, bins=binsVal)
#mpl.title ("\\textbf{Energy Eigenvalues}\n Number of Bins = %g, $P_{x,max}$ = %g, $P_{y,max}$ = %g, $N_x$ = %g, $N_y$ = %g" % (binsVal,pxmax,pymax,Nx,Ny))
#mpl.savefig("energiesBins%gPxmax%gPymax%gNx%gNy%g.pdf" % (binsVal,pxmax,pymax,Nx,Ny))	#save histogram
#mpl.show()

#energyFile.close()
#dosFile.close()
