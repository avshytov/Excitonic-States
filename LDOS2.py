#makeGrid(int Nx, int Ny, float pxmax, float pymax)
#		CONSTRUCTS THE AXES OF A 2D GRID IN MOMENTUM SPACE, WITH N(X/Y) POINTS BETWEEN p(x/y)max AND p(x/y)min
#		AND STORES THE AXIS POINTS IN TWO ARRAYS px AND py
#		RETURNS px, py
#makePotential(float px[], float py[], float alpha, float h)
#		CONSTRUCTS A 2D POTENTIAL ENERGY MATRIX V. LOOPS OVER THE MOMENTUM SPACE DEFINED BY px AND py, AND CALCULATES
#		FOR A POTENTIAL OF STRENGTH alpha BETWEEN STATES IN SEPERATE GRAPHENE PLANES OF SPATIAL SEPERATION h
#		RETURNS V
#makeKinetic(float px[], float py[], float eta)
#		CALCULATES THE 2D KINETIC ENERGY MATRIX H0 OF A STATE WITH MOMENTUM SPACE COORDINATES (px[i],py[j])
#		IN A SYSTEM CHARACTERIZED BY THE RATIO OF COMPETING HOPPING ENERGIES eta.
#		RETURNS H0
#transformToRSpace (float px[], float py[], float a[][], float theta, float Rvals[])
#		PERFORMS A UNITARY TRANSFORMATION OF THE MOMENTUM SPACE EIGENVECTORS STORED IN a, TO CALCULATE THE POSITION
#		SPACE EIGENVECTORS ALONG A DIRECTION theta IN R SPACE. RADIAL POINTS ARE DEFINED BY VALUES STORED IN ARRAY 
#		Rvals AND MOMENTUM GRID IS DEFINED BY px and py. DUE TO SPIN CONTRIBUTION THERE ARE TWO SPACE STATES FOR EACH
#		MOMENTUM STATE. HALF OF THESE ARE STORED IN MATRIX psi3 AND THE OTHER HALF IN psi4. ALL VECTORS ARE ROW VECTORS.
#		RETURNS psi3, psi4
#makeLDOS (float psi3[][], float psi4[][], float E[], float Evals[], float gamma[])
#		CALCULATES LOCAL DENSITY OF STATES ON AN ENERGY/RADIUS GRID WITH POINTS (Evals[i],R[[j]). FOR A GIVEN POINT, FUNCTION
#		SUMS THE CONTRIBUTION OF EACH EIGENVALUE IN E BY APPROXIMATING THE DELTA FUNCTION AS A LORENTZIAN OF HWHM gamma.
#		ALSO CALCULATES THE PROBABILITY DENSITY OF EACH STATE USING THE VECTORS FOUND IN psi3 AND psi4 FOR USE IN THE 
#		SUM
#		RETURNS 
#makeFilename(float alpha, int Nx, int Ny, float eta, float h)
#		USES THE PARAMETERS OF THE SYSTEM alpha (POTENTIAL STENGTH), eta (RATIO OF HOPPING CONTRIBUTIONS), h (LAYER SPACING)
#		Nx and Ny (MOMENTUM SPACE GRID SIZE) TO GENERATE A STRING TO BE USED AS A FILENAME FOR EXTERNAL USE. STRING HAS 
#		FORMAT fname = 'data-alpha=%g-N=%dx%d-eta=%g-h=%g.npz' % (alpha, Nx, Ny, eta, h)
#		RETURNS fname
#makeFilename2(float alpha, int Nx, int Ny, float eta, float h)
#		SAME AS makeFilename(), EXCEPT THE STRING HAS A DIFFERENT VALUE
#		fname = 'data2-alpha=%g-N=%dx%d-eta=%g-h=%g.npz' % (alpha, Nx, Ny, eta, h)
#		RETURNS fname
#solveAndSave(int Nx, int Ny, float pxmax, float pymax, float alpha, float h, float eta)
#		CALLS FUNCTIONS makePotential() AND makeKinetic() TO CONSTRUCT TOTAL SYSTEM HAMILTONIAN H. PERFORMS EIGENDECOMPOSITION
#		ON H, STORING EIGENVALUES IN ARRAY E AND MOMENTUM SPACE ROW EIGENVECTORS IN MATRIX a. ALSO CALLS makeGrid() TO COMPOSE
#		THE MOMENTUM SPACE, STORING AXES IN ARRAYS px AND py. STORES MOMENTUM GRID, E-VALUES AND E-VECTORS IN A FILE WITH NAME
#		GENERATED BY makeFilename() FOR EXTERNAL USE
#		RETURNS px, py, E, a
#loadSolution(int Nx, int Ny, float alpha, float h, float eta)
#		RETRIEVES DATA FROM A FILE WHICH WOULD HAVE BEEN CREATED PREVIOUSLY. IDENTIFIES WHICH FILE TO RETRIEVE FROM BY CALLING
#		makeFilename() AND PASSING PARAMETERS OF THE SYSTEM. RETRIEVES E-VALUES (E), E-VECTORS (a), AND MOMENTUM GRID (px AND py)
#		RETURNS data['px'], data['py'], data['E'], data['a']
#loadSolution2(int Nx, int Ny, float alpha, float h, float eta)
#		SAME AS loadSolution(), BUT READS DIFFERENT ENTRIES FROM A DIFFERENT FILE
#		RETURNS psi3, psi4, probDensity

import numpy as np
import math 
import scipy.linalg as lin


def makeGrid(Nx, Ny, pxmax, pymax):
	px = np.linspace(-pxmax, pxmax, Nx)
	py = np.linspace(-pymax, pymax, Ny)
	return px, py

def makePotential(px, py, alpha, h):

	Nx = len (px)
	Ny = len (py)
	N = 2 * Nx * Ny				#calculate size of matrix
	V = np.zeros((N,N))

	deltaPx = px[1] - px[0]
	deltaPy = py[1] - py[0]
	Cv = float(deltaPx * deltaPy) / 2.0 / math.pi

	for iX1 in range (Nx):			
		for iY1 in range (Ny):
			i0 = (iX1*Ny + iY1)*2
			px1 = px[iX1] 
			py1 = py[iY1] 
			phi1 = np.arctan2(py1, px1)
	
			for iX2 in range (iX1, Nx): # We do not need to scan the matrix twice
				for iY2 in range (Ny):
					j0 = (iX2*Ny + iY2)*2
					px2 = px[iX2] 
					py2 = py[iY2] 
					phi2 = np.arctan2(py2, px2)
					if iX1 == iX2 and iY1 == iY2:  #avoid division by zero
						q = 0.5 * math.sqrt( deltaPx ** 2 + deltaPy ** 2 )
						cosphi = 1.0
					else:
						q = math.sqrt((px1-px2)**2+(py1-py2)**2)
						cosphi = math.cos( phi1 - phi2 ) # not np.cos!
					Uq = -1.0 / q * math.exp( - q * h )

					V[i0,j0] = Uq * cosphi
					V[j0,i0] = V[i0, j0] 

					V[j0 + 1, i0 + 1] = Uq
					V[i0 + 1, j0 + 1] = V[j0 + 1, i0 + 1]

	V *= Cv
	return V

def makeKinetic(px, py, eta):
	Nx = len(px)
	Ny = len(py)
	N = 2 * Nx * Ny	
	H0 = np.zeros((N,N))

	for iX1 in range (Nx):
		for iY1 in range (Ny):
			i0 = (iX1*Ny + iY1)*2
			px1 = px[iX1] 
			py1 = py[iY1] 

			p2 = px1**2 + py1**2
			phi1 = np.arctan2(py1, px1)

			if True:
				Ep = p2 * math.cos( 3.0 * phi1 )
			else:
				Ep = p2
					
			H0[ i0,     i0 + 1 ] = Ep
			H0[ i0 + 1, i0     ] = Ep

			H0[i0, i0] = H0[i0 + 1, i0 + 1] = p2 * eta 
	return H0		

def transformToRSpace (px, py, a, theta, Rvals):
	nR = len(Rvals)
	Nx = len(px)
	Ny = len(py)
	N = 2 * Nx * Ny
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
	for n in range (N):
		psi3[n, :] = np.dot(U,  a[ ::2, n])
		psi4[n, :] = np.dot(U,  a[1::2, n])

	#calculate the probability densities
	probDensity = np.abs(psi3)**2 + np.abs(psi4)**2

	print ("done")
	return psi3, psi4, probDensity

def makeLDOS (psi3, psi4, probDensity, E, Evals, gamma):
	D	= np.zeros((len(Evals), len(E)))
	for epsilon in range(len(Evals)):
		for n in range(len(E)):
			D[epsilon, n] = 1.0  / ((Evals[epsilon] - E[n])**2 + gamma**2) 
	D *= gamma / math.pi 
	print ("Calculating LDOS")

	#calculate LDOS
	#LDOS = np.zeros((len(Evals), len(Rvals)))

	LDOS = np.dot(D, probDensity)
	print ("done")
	return LDOS

def makeFilename(alpha, Nx, Ny, eta, h):
	fname = 'data-alpha=%g-N=%dx%d-eta=%g-h=%g.npz' % (alpha, Nx, Ny, eta, h)
	return fname

def makeFilename2(alpha, Nx, Ny, eta, h):
	fname = 'data2-alpha=%g-N=%dx%d-eta=%g-h=%g.npz' % (alpha, Nx, Ny, eta, h)
	return fname

def solveAndSave(Nx, Ny, pxmax, pymax, alpha, h, eta):
	px, py = makeGrid (Nx, Ny, pxmax, pymax)
	H = makeKinetic(px, py, eta) + alpha * makePotential (px, py, alpha, h)
	print ("Diagonalising...")
	(E,a) = lin.eigh(H)		#calculate eigenvalues, a[] are the row eigenvectors
	print ("done")
	f = makeFilename(alpha, Nx, Ny, eta, h)#open(makeFilename(alpha, Nx, Ny, eta, h), 'w')
	np.savez(f, px=px, py=py, E=E, a=a)
	return px, py, E, a
  
def loadSolution(Nx, Ny, alpha, h, eta):
	f = makeFilename(alpha, Nx, Ny, eta, h)#open(makeFilename(alpha, Nx, Ny, eta, h))
	data = np.load(f)		# ValueError: string is smaller than requested size?????
	px = data['px']
	py = data['py']
	E = data['E']
	a = data['a']
	return px, py, E, a

def loadSolution2(Nx, Ny, alpha, h, eta):
	f = makeFilename2(alpha, Nx, Ny, eta, h)
	data = np.load(f)
	psi3 = data['psi3']
	psi4 = data['psi4']
	probDensity = data['probDensity']
	return psi3, psi4, probDensity

if __name__ == '__main__':
	import matplotlib.pyplot as mpl 
	from matplotlib import rc

	rc('text', usetex = True)
	rc('font', family = 'serif')
    
	Nx = 25
	Ny = 25
	pxmax = 3.14
	pymax = 3.14
	alpha = 2.0
	h = 1.0
	eta = 0.7
	#px, py = makeGrid (Nx, Ny, pxmax, pymax)
	#H = makeKinetic(px, py, eta) + alpha * makePotential (px, py, alpha, h)
	if True:
		px, py, E, a = solveAndSave(Nx, Ny, pxmax, pymax, alpha, h, eta)
	else:
		px, py, E, a = loadSolution(Nx, Ny, alpha, h, eta)   
	gamma = 0.05

	eigFile = open("eigenvalues.txt",'w')
	for n in range (len(E)):
		eigFile.write(str(n))
		eigFile.write('\t')
		eigFile.write(str(E[n]))
		eigFile.write('\n')
	eigFile.close()

	Rmin = -10.0
	Rmax = 10.0
	nR = 500
	Rvals = np.linspace (Rmin, Rmax, nR)

	theta = 0.0
	psi3, psi4, probDensity = transformToRSpace(px, py, a, theta, Rvals)

	Emin = -5.0 
	Emax = +5.0 
	nE = 500
	Evals = np.linspace (Emin, Emax, nE)

	LDOS = makeLDOS(psi3, psi4, probDensity, E, Evals, gamma)

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

	mpl.title(r'LDOS, $\alpha = %g$, $N = %d \times %d$, $\eta = %g$, $h = %g$,  %s' % (alpha, Nx, Ny, eta, h, theta_s))
	mpl.xlabel(r'Distance $R$')
	mpl.ylabel(r'Energy $\epsilon$')

	mpl.figure()
	mpl.hist(E, bins=100)

	mpl.show()
