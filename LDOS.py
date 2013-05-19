import numpy as np
import math 
import matplotlib.pyplot as mpl
import scipy.linalg as lin
from matplotlib import rc

rc('text', usetex = True)
rc('font', family = 'serif')

Nx = int(input("Nx = "))
Ny = int(input("Ny = "))
pxmax = float(input("PxMax = "))
pymax = float(input("PyMax = "))
gamma = float(input("gamma = "))
N= 2 * Nx * Ny				#calculate size of matrix

H0 = np.zeros((N,N))		#allocate empty matrix
V = np.zeros((N,N))
H = np.zeros((N,N))

deltaPx = 2.0*float(pxmax)/float(Nx - 1.0)
deltaPy = 2.0*float(pymax)/float(Ny - 1.0)
alpha = 1.0
Calpha = alpha * float(deltaPx * deltaPy) / 2.0 / math.pi

ixvals = np.array(range(Nx))
iyvals = np.array(range(Ny))
px = pxmax * (ixvals * 2.0 / float(Nx - 1) - 1.0)
py = pymax * (iyvals * 2.0 / float(Ny - 1) - 1.0)

#potential energy terms
for iX1 in range (Nx):			#1,-0.8		0.005,
	for iY1 in range (Ny):
		for iX2 in range (Nx):
			for iY2 in range (Ny):

				i0 = (iX1*Ny + iY1)*2
				j0 = (iX2*Ny + iY2)*2

				px1 = px[iX1] #pxmax * ((iX1*2.0)/float(Nx - 1) - 1.0)
				px2 = px[iX2] #pxmax * ((iX2*2.0)/float(Nx - 1) - 1.0)
				
				py1 = py[iY1] #pymax * ((iY1*2.0)/float(Ny - 1) - 1.0)
				py2 = py[iY2] #pymax * ((iY2*2.0)/float(Ny - 1) - 1.0)

				phi1 = np.arctan2(py1,px1)
				phi2 = np.arctan2(py2,px2)

				if px1==px2 and py1==py2:			#avoid division by zero
					modulus = 0.5 * math.sqrt( deltaPx ** 2 + deltaPy ** 2 )

				else:
					modulus = math.sqrt((px1-px2)**2+(py1-py2)**2)

				if i0<=N:							#stops overfilling of matrix
					if j0<=N:
						V[i0,j0] = - (1.0 / modulus)* np.cos(phi1 - phi2)
						V[j0,i0] = - (1.0 / modulus)* np.cos(phi1 - phi2) 

				if i0+1<=N:
					if j0+1<=N:
						V[j0+1,i0+1] = - 1.0 / modulus

V *= Calpha

#kinetic energy terms
for iX1 in range (Nx):
	for iY1 in range (Ny):

				i0 = (iX1*Ny + iY1)*2

				px1 = px[iX1] #pxmax * ((iX1*2.0)/float(Nx - 1) - 1.0)
				py1 = py[iY1] #pxmax * ((iY1*2.0)/float(Ny - 1) - 1.0)
				
				p=math.sqrt(px1**2+py1**2)
				phi1 = np.arctan2(py1,px1)

				H0[i0,i0+1] = p**2 * np.cos(3 * phi1)
				H0[i0+1,i0] = p**2 * np.cos(3 * phi1)


H=V+H0

print ("Diagonalising...")
(E,a) = lin.eigh(H)		#calculate eigenvalues, a are the row eigenvectors
print ("done")


for icheck in [ 0, N-1, N/2]:
    Echeck = E[icheck]
    ev = a[:, icheck]
    print ("icheck = ", icheck, "norm:",  lin.norm(np.dot(H, ev) - Echeck * ev)) 

theta = 0.0

Emin = -3.0 #min(E)
Emax = +3.0 #max(E)
dE = (Emax - Emin) / 200.0
Evals = np.arange (Emin, Emax, dE)

Rmin = 0.0
Rmax = 5.0
dR = (Rmax - Rmin) / 200.0
Rvals = np.arange (Rmin, Rmax, dR)

print ("Transforming eigenvectors to R-space")
#find the eigenvectors in position space
if False:
	psi1 = np.zeros((N,len(Rvals)), dtype=complex)
	psi2 = np.zeros((N,len(Rvals)), dtype=complex)
	for n in range (len(E)):				#steps through eigenvalues
		for R in range (len(Rvals)):
			for iX in range (Nx):
				for iY in range (Ny):
					i = (iX*Ny + iY)*2				#lables momentum state
					px_i = px[iX] #pxmax * ((iX*2.0)/float(Nx - 1) - 1.0)
					py_i = py[iY] #pymax * ((iY*2.0)/float(Ny - 1) - 1.0)
					pr = (px_i * math.cos(theta) + py_i * math.sin(theta)) * Rvals[R]
					exp_pr = math.cos(pr) + 1j * math.sin(pr)
					psi1[n,R] += a[i,   n] * exp_pr
					psi2[n,R] += a[i+1 ,n] * exp_pr

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
	psi3[n, :] = np.dot(U,  a[::2, n])
	psi4[n, :] = np.dot(U,  a[1::2, n])
	#print ("n = ", n, "diff: ", lin.norm(psi3[n, :] - psi1[n, :]), lin.norm(psi4[n, :] - psi2[n, :]))
	#print ("psi1 = ", psi1, "psi2 = ", psi3)

print ("done")
print ("Calculating LDOS")
#calculate the probability densities
probDensity = np.abs(psi3)**2 + np.abs(psi4)**2
#probDensity = np.zeros((N,len(Rvals)))
#for n in range (len(E)):
#	for R in range (len(Rvals)):
#		probDensity[n,R] = abs( psi1[n,R] )**2  + abs( psi2[n, R] )**2

#calculate LDOS
LDOS = np.zeros((len(Evals), len(Rvals)))
D    = np.zeros((len(Evals), len(E)))

for epsilon in range(len(Evals)):
	for n in range(len(E)):
		D[epsilon, n] = 1.0  / ((Evals[epsilon] - E[n])**2 + gamma**2) 
D *= gamma / math.pi 

LDOS = np.dot(D, probDensity)

if False:
	LDOS1 = np.zeros((len(Evals),len(Rvals)))
	for epsilon in range(len(Evals)):
		for R in range (len(Rvals)):
			for n in range (len(E)):
				LDOS1[epsilon,R]+= gamma * probDensity[n,R] / (math.pi  * ((Evals[epsilon] - E[n])**2 + gamma**2))
	print("diff: ", lin.norm(LDOS1 - LDOS))


print ("done")
print(LDOS)

XX, YY = np.meshgrid(Rvals, Evals)

print ("Creating color plot")
mpl.figure()
mpl.pcolor(XX, YY, LDOS)
mpl.colorbar()
mpl.title('LDOS, alpha = %g' % alpha)
mpl.xlabel(r'Distance $R$')
mpl.ylabel(r'Energy $\epsilon$')

mpl.figure()
mpl.hist(E, bins=50)

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
