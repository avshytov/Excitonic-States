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
	
	     for iX2 in range (iX1, Nx): # We do not need to scan the matrix
			                    	# twice
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
				
	    p2   = px1**2 + py1**2
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

	print ("done")
	return psi3, psi4

def makeLDOS (psi3, psi4, E, Evals, gamma):
    D    = np.zeros((len(Evals), len(E)))
    for epsilon in range(len(Evals)):
   	for n in range(len(E)):
	    D[epsilon, n] = 1.0  / ((Evals[epsilon] - E[n])**2 + gamma**2) 
    D *= gamma / math.pi 
    print ("Calculating LDOS")
    #calculate the probability densities
    probDensity = np.abs(psi3)**2 + np.abs(psi4)**2

    #calculate LDOS
    #LDOS = np.zeros((len(Evals), len(Rvals)))

    LDOS = np.dot(D, probDensity)
    print ("done")
    return LDOS

def makeFilename(alpha, Nx, Ny, eta, h):
    fname = 'data-alpha=%g-N=%dx%d-eta=%g-h=%g.npz' % (alpha, Nx, Ny, eta, h)
    return fname

def solveAndSave(Nx, Ny, pxmax, pymax, alpha, h, eta):
    px, py = makeGrid (Nx, Ny, pxmax, pymax)
    H = makeKinetic(px, py, eta) + alpha * makePotential (px, py, alpha, h)
    print ("Diagonalising...")
    (E,a) = lin.eigh(H)		#calculate eigenvalues, a[] are the row eigenvectors
    print ("done")
    f = open(makeFilename(alpha, Nx, Ny, eta, h), 'w')
    np.savez(f, px=px, py=py, E=E, a=a)
    return px, py, E, a
    
def loadSolution(Nx, Ny, alpha, h, eta):
    f = open(makeFilename(alpha, Nx, Ny, eta, h))
    data = np.load(f)
    return data['px'], data['py'], data['E'], data['a']

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
    if False:
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
    psi3, psi4 = transformToRSpace(px, py, a, theta, Rvals)

    Emin = -5.0 
    Emax = +5.0 
    nE = 500
    Evals = np.linspace (Emin, Emax, nE)

    LDOS = makeLDOS(psi3, psi4, E, Evals, gamma)

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

