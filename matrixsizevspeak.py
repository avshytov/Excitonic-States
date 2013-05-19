import numpy as np
import math
import scipy.linalg as lin

Nx = [10,15,20,25,30,35,40,45,50]
pxmax = float(input("PxMax = "))
pymax = float(input("PyMax = "))
gamma = float(input("gamma = "))
alpha = 1.0
theta = 0.0

outputFile = open('matrixsizevspeak.txt','w')

for k in range (len(Nx)):
	Ny=Nx[k]
	N= 2 * Nx[k] * Ny						#calculate size of matrix
	print("k=",k)
	H0 = np.zeros((N,N))					#allocate empty matrix
	V = np.zeros((N,N))
	H = np.zeros((N,N))
	deltaPx = 2.0*float(pxmax)/float(Nx[k] - 1.0)
	deltaPy = 2.0*float(pymax)/float(Ny - 1.0)
	Calpha = alpha * float(deltaPx * deltaPy) / 2.0 / math.pi	#takes care of
																	#integral approximation
	ixvals = np.array(range(Nx[k]))				#[0,1,2,...,Nx-1]
	iyvals = np.array(range(Ny))				#[0,1,2,...,Ny-1]
	px = pxmax * (ixvals * 2.0 / float(Nx[k] - 1) - 1.0)
	py = pymax * (iyvals * 2.0 / float(Ny - 1) - 1.0)

	#potential energy terms
	for iX1 in range (Nx[k]):
		for iY1 in range (Ny):
			for iX2 in range (Nx[k]):
				for iY2 in range (Ny):

					i0 = (iX1*Ny + iY1)*2
					j0 = (iX2*Ny + iY2)*2

					px1 = px[iX1]	#calculates the momentum components of both states
					px2 = px[iX2]	#by calling the arrays px and py, which already
									#hold the components for any one given state, which
					py1 = py[iY1]	#are indexed by ixvals and iyvals arrays
					py2 = py[iY2]	#

					if px1==px2 and py1==py2:			#avoid division by zero
						modulus = 0.5 * math.sqrt( deltaPx ** 2 + deltaPy ** 2 )

					else:
						modulus = math.sqrt((px1-px2)**2+(py1-py2)**2)

					if i0<=N:							#stops overfilling of matrix
						if j0<=N:
							V[i0,j0] = - (1.0 / modulus)
							V[j0,i0] = - (1.0 / modulus)

					if i0+1<=N:
						if j0+1<=N:
							V[j0+1,i0+1] = - 1.0 / modulus

	V *= Calpha

	#kinetic energy terms
	for iX1 in range (Nx[k]):
		for iY1 in range (Ny):

					i0 = (iX1*Ny + iY1)*2

					px1 = px[iX1] 
					py1 = py[iY1] 
				
					p=math.sqrt(px1**2+py1**2)

					H0[i0,i0+1] = p**2
					H0[i0+1,i0] = p**2


	H=V+H0			#gives total matrix
	del V			#delete V and H0 to free up space
	del H0

	print ("Diagonalising...")
	(E,a) = lin.eigh(H)		#calculate eigenvalues, a are the row eigenvectors
	print ("done")

	del H					#delete H to free up space

	Emin = -3.0
	Emax = +3.0
	dE = (Emax - Emin) / 200.0
	Evals = np.arange (Emin, Emax, dE)	#array of values which will be the energy axis

	Rmin = 0.0
	Rmax = 5.0
	dR = (Rmax - Rmin) / 200.0
	Rvals = np.arange (Rmin, Rmax, dR)	#array of values will be radius axis

	print ("Transforming eigenvectors to R-space")
	#find the eigenvectors in position space

	print ("Generating the Unitary transform")
	U = np.zeros((len(Rvals), N/2), dtype=complex)
	for iX in range(Nx[k]):
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

	#Calculate the local density of states at each point in a radius,energy matrix
	print ("Calculating LDOS")
	probDensity = np.abs(psi3)**2 + np.abs(psi4)**2	#contains the sum of two spin terms

	#calculate LDOS
	LDOS = np.zeros((len(Evals), len(Rvals)))
	D    = np.zeros((len(Evals), len(E)))

	for epsilon in range(len(Evals)):
		for n in range(len(E)):
			D[epsilon, n] = 1.0  / ((Evals[epsilon] - E[n])**2 + gamma**2) 
	D *= gamma / math.pi 

	LDOS = np.dot(D, probDensity)

	#find the peak position
	def peak_position(dosMatrix,energyAxis):
		energySlice = dosMatrix[:,0]			#a slice of the DOS profile as energy varies, radius=0
		maxPos = np.argmax(energySlice)			#returns the index of energy slice which has the
												#highest value (top of the DOS peak)
		peak = energyAxis[maxPos]				#locates the peak using index of max DOS in array of 
												#energy values
		return peak								#the value with the highest count

	#print results to file
	outputFile.write(str(Nx[k]))
	outputFile.write(',')
	outputFile.write(str(peak_position(LDOS,Evals)))
	outputFile.write('\n')

	#delete matrices to free up space
	del LDOS
	del D
	del psi3
	del psi4
	del U

outputFile.close()
