import numpy as np
import math 
import scipy.linalg as lin
import LDOS2 as ldos


if __name__ == '__main__':
    import matplotlib.pyplot as mpl 
    from matplotlib import rc

    recalculate = False
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'recalculate': recalculate = True
        else:
            print ("Unknown argument:", sys.argv[1])
            sys.exit(1)
    
    
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
    if recalculate:
       px, py, E, a = ldos.solveAndSave(Nx, Ny, pxmax, pymax, alpha, h, eta)
    else:
       px, py, E, a = ldos.loadSolution(Nx, Ny, alpha, h, eta)   
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
    psi3, psi4 = ldos.transformToRSpace(px, py, a, theta, Rvals)

    Emin = -5.0 
    Emax = +5.0 
    nE = 500
    Evals = np.linspace (Emin, Emax, nE)

    LDOS = ldos.makeLDOS(psi3, psi4, E, Evals, gamma)

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

