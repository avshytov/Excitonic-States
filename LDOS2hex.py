# Hex grid in momentum space. 
# Note the change in semantics
#
# 0. No separate Nx and Ny values, only one parameter N defines 
#    the grid size. Respectively, one pmax value
#
# 1. The total number of grid points is now roughly 3*N^2, 
#     so that N=25 is equivalent to N=45 on the square grid
#
# 2. px and py arrays now list all grid points. Previously, 
#    they used to be Nx and Ny long, respectively. Now, 
#    both are 3*N*(N + 1) + 1 long
#
# 3. makegrid now returns px, py, and area element dS, which 
#    is to be used in momentum space integrations. (Finding it from px 
#    and py vectors can be algorithmically challenging.)
# 


#makeGrid(int Nx, float pmax)
#		CONSTRUCTS THE AXES OF A 2D GRID IN MOMENTUM SPACE, WITH N(X/Y) POINTS BETWEEN p(x/y)max AND p(x/y)min
#		AND STORES THE AXIS POINTS IN TWO ARRAYS px AND py
#		RETURNS px, py
#makePotential(float px[], float py[], float dS, float alpha, float h)
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

test_2dhydrogen = False

def makeGrid(nmax, pmax):
	ps = [(0.0, 0.0)]
	N = 3 * nmax * (nmax + 1) + 1
	for i in range (1, nmax + 1):
	    ki = pmax / float(nmax) * i # radius of the current shell
            #print "doing shell", i
	    for r in range (0, 6): # six directions
		#print "direction", r
		phi1 = 2.0 * math.pi / 6.0 * r
		phi2 = 2.0 * math.pi / 6.0 * (r + 1)
		x1, y1 = math.cos(phi1), math.sin(phi1) # left end
		x2, y2 = math.cos(phi2), math.sin(phi2) # right end
		for j in range (0, i):
		    w1 = (float(i) - float(j)) / float(i)
		    w2 = float(j) / float(i)
		    xj, yj = (x1 * w1 + x2 * w2), (y1 * w1 + y2 * w2)
	            #print xj, yj
		    ps.append((xj * ki, yj * ki))
		
	#px = np.linspace(-pxmax, pxmax, Nx)
	#py = np.linspace(-pymax, pymax, Ny)
	px = map(lambda p: p[0], ps)
	py = map(lambda p: p[1], ps)
	dp = pmax / nmax
        dS = dp**2 * math.sqrt(3) / 2.0
	print "dS = ", dS
	return np.array(px), np.array(py), dS

def makePotential(px, py, dS, alpha, h):

	Ns = len (px)
	N = 2 * Ns
	V = np.zeros((N,N))

	#deltaPx = px[1] - px[0]
	#deltaPy = py[1] - py[0]
	Cv = dS / 2.0 / math.pi

	for i in range(Ns):
	    i0 = 2 * i
	    px1 = px[i] 
	    py1 = py[i] 
	    phi1 = np.arctan2(py1, px1)
	    for j in range (i, Ns):
	        j0 = 2 * j
		px2 = px[j] 
		py2 = py[j] 
	        phi2 = np.arctan2(py2, px2)
	        if j == i:  #avoid division by zero
		   q = 0.5 * math.sqrt( dS )
		   cosphi = 1.0
		else:
		   q = math.sqrt((px1-px2)**2+(py1-py2)**2)
		   cosphi = math.cos( phi1 - phi2 ) # not np.cos!
		#print i, j, q, cosphi
		if test_2dhydrogen: cosphi = 1.0
		Uq = -1.0 / q * math.exp( - q * h )

		V[i0, j0] = Uq * cosphi
		V[j0, i0] = V[i0, j0] 

		V[j0 + 1, i0 + 1] = Uq
		V[i0 + 1, j0 + 1] = V[j0 + 1, i0 + 1]

	V *= Cv
	return V

def makeKinetic(px, py, eta):
	Ns = len(px)
	N = 2 * Ns	
	H0 = np.zeros((N,N))

	for i in range (Ns):
	    i0 = 2 * i
	    px1 = px[i] 
	    py1 = py[i] 

	    p2 = px1**2 + py1**2
	    phi1 = np.arctan2(py1, px1)

	    if not test_2dhydrogen:
	       Ep = p2 * math.cos( 3.0 * phi1 )
	    else:
	       Ep = p2
					
	    H0[ i0,     i0 + 1 ] = Ep
	    H0[ i0 + 1, i0     ] = Ep
            
	    if not test_2dhydrogen:
   	       H0[i0, i0] = H0[i0 + 1, i0 + 1] = p2 * eta
	
	
	return H0		

def transformToRSpace (px, py, a, thetavals, Rvals):
	nR = len(Rvals)
	Ns = len(px)
	N = 2 * Ns

	if type(thetavals) == type(0.0):
		theta = thetavals
		thetavals = np.zeros(np.shape(Rvals))
		thetavals[:] = theta

	if len(thetavals) != len(Rvals):
		import sys
		print ("Unequal sizes of thetavals and Rvals in transformToRSpace")
		sys.exit(1)
       	
	print ("Transforming eigenvectors to R-space")

	print ("Generating the Unitary transform")
	U = np.zeros((len(Rvals), N/2), dtype=complex)
	for i in range(Ns):
	    px_i = px[i] 
	    py_i = py[i]
	    for R in range(len(Rvals)):
		theta = thetavals[R]
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

def makeFilename(alpha, Nx, eta, h):
	fname = 'data-hex-alpha=%g-N=%d-eta=%g-h=%g.npz' % (alpha, Nx,eta, h)
	return fname

def makeFilename2(alpha, Nx, eta, h):
	fname = 'data2-hex-alpha=%g-N=%d-eta=%g-h=%g.npz' % (alpha, Nx, eta, h)
	return fname

def solveAndSave(Nx, pxmax, alpha, h, eta):
	px, py, dS = makeGrid (Nx, pxmax)
	H = makeKinetic(px, py, eta) + alpha * makePotential (px, py, dS, alpha, h)
	print ("Diagonalising...")
	(E,a) = lin.eigh(H)		#calculate eigenvalues, a[] are the row eigenvectors
	print ("done")
	f = makeFilename(alpha, Nx, eta, h)#open(makeFilename(alpha, Nx, Ny, eta, h), 'w')
	np.savez(f, px=px, py=py, E=E, a=a)
	return px, py, E, a
  
def loadSolution(Nx, alpha, h, eta):
	f = makeFilename(alpha, Nx, eta, h)#open(makeFilename(alpha, Nx, Ny, eta, h))
	data = np.load(f)		# ValueError: string is smaller than requested size?????
	px = data['px']
	py = data['py']
	E = data['E']
	a = data['a']
	return px, py, E, a

def loadSolution2(Nx, alpha, h, eta):
	f = makeFilename2(alpha, Nx, eta, h)
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
    
	q_e = 1.602e-19
	nm = 1e-9
	eps_0 = 8.85e-12
	a = 0.142
	eV = q_e
	meV = 0.001 * eV
	t_prime = 0.2 * 2.8 * eV
	vD = 1e+6
	hbar = 1.054e-34
	
	
	E_u = 9.0 / 4.0 * t_prime * a**2 / meV 
	C_0 = q_e**2 / 4.0 / math.pi / eps_0 / hbar / vD 
	print "C_0 = ", C_0
	eps_s = 3.9                 # environment (substrates)
	eps_g = math.pi / 2.0 * C_0 # intrinsic dielectric constant
	eps = eps_s + eps_g       
	print "eps_g = ", eps_g, "eps = ", eps
	
	Nx = 40
	pxmax = 3.14  / a * 0.3
	alpha = (q_e)** 2 / 4.0 / math.pi / eps_0 / eps / nm / E_u / meV 
	print "alpha = ", alpha
	
	print "alpha = ", alpha
	h = 1.3
	eta = 0.7
	
	#E_u = 9.0 * eta / 1.2
		
	#px, py = makeGrid (Nx, Ny, pxmax, pymax)
	#H = makeKinetic(px, py, eta) + alpha * makePotential (px, py, alpha, h)
	if True:
		px, py, E, a = solveAndSave(Nx, pxmax, alpha, h, eta)
	else:
		px, py, E, a = loadSolution(Nx, alpha, h, eta)   
	gamma = 1.0 / E_u

	eigFile = open("eigenvalues-hex.txt",'w')
	for n in range (len(E)):
		eigFile.write(str(n))
		eigFile.write('\t')
		eigFile.write(str(E[n]))
		eigFile.write('\n')
	eigFile.close()

	Rmin = 0.0
	Rmax = 10.0
	nR = 500
	Rvals = np.linspace (Rmin, Rmax, nR)

	theta = 0.0
	psi3, psi4, probDensity = transformToRSpace(px, py, a, theta, Rvals)

	Emin = -200.0 
	Emax =  100.0 
	nE = 500
	Evals = np.linspace (Emin/E_u, Emax/E_u, nE)

	LDOS = makeLDOS(psi3, psi4, probDensity, E, Evals, gamma)

	XX, YY = np.meshgrid(Rvals, Evals)

	print ("Creating color plot")
	mpl.figure()
	mpl.pcolor(XX, YY * E_u, LDOS)
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

	mpl.title(r'LDOS, $\alpha = %g$, $N = %d$, $\eta = %g$, $h = %g$,  %s' % (alpha, Nx, eta, h, theta_s))
	mpl.xlabel(r'Distance $R$, nm')
	mpl.ylabel(r'Energy $\epsilon$, meV')

	mpl.figure()
	mpl.hist(E * E_u, bins=100)

	mpl.show()
