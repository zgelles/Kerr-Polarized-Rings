import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import mpmath as mp

ek = np.frompyfunc(mp.ellipk, 1, 1)

def getparamsanalytic(spin, theta): #analytic solution for BL radius on beta axis (all other points need numerics)
    if spin == 0:
        return 3, np.sqrt(27)
    spin += 0j  #makes it complex type
    cuberootfac = (-9+9*spin**2+np.sqrt(3*spin**2*(-27+18*spin**2+spin**4)))**(1/3)
    r = 1-(-1/3)**(1/3)*(-3+spin**2)/cuberootfac-(-1)**(2/3)*cuberootfac/3**(2/3)
    r = np.real(r) #imaginary part should just be a numerical artefact
    
    delta = r**2-2*r+spin**2
    ell = ((r**2-spin**2)-r*delta)/(spin*(r-1))
    up, um = getangroots(spin, r)
    rho = np.sqrt(spin**2*((np.cos(theta))**2-up*um)+ell**2)
    
    return r, np.real(rho) #imaginary part should just be a numerical artefact


def rootfunc(params, comp=False): #set up system of equations to solve for numerical case
    global spin0
    global varphi0
    global theta0
    
    r, rho = params
    delta = r**2-2*r+spin0**2
    ell = ((r**2-spin0**2)-r*delta)/(spin0*(r-1))
    up, um = getangroots(spin0, r, comp=comp)

    return (rho-np.sqrt(spin0**2*((np.cos(theta0))**2-up*um)+ell**2), np.cos(varphi0)*rho*np.sin(theta0)+ell)


def getparamsnumerical(spin, theta, varphi): #numerical solution  - necessary if not on the beta axis
    if spin == 0:
        return 3, np.sqrt(27)
    global spin0
    global theta0
    global varphi0
    spin0 = spin
    theta0 = theta
    varphi0 = varphi

    
    r, rho = opt.fsolve(rootfunc, (3, np.sqrt(27)))
    return r, rho

def getangroots(spin, r, comp=False): #gets u_{\pm}
    delta = r**2-2*r+spin**2
    # if delta <= 0:
    #     print('hi', r/(spin*(r-1))**2*(-r**3+3*r-2*spin**2+2*np.sqrt(complex(delta)*(2*r**3-3*r**2+spin**2))), r/(spin*(r-1))**2*(-r**3+3*r-2*spin**2-2*np.sqrt(complex(delta)*(2*r**3-3*r**2+spin**2))))
    if comp:
        delta = complex(delta)
        uplus = r/(spin*(r-1))**2*(-r**3+3*r-2*spin**2+2*np.sqrt(delta*(2*r**3-3*r**2+spin**2)))
        uminus = r/(spin*(r-1))**2*(-r**3+3*r-2*spin**2-2*np.sqrt(delta*(2*r**3-3*r**2+spin**2)))
    else:
        uplus = r/(spin*(r-1))**2*(-r**3+3*r-2*spin**2+2*np.sqrt(delta*(2*r**3-3*r**2+spin**2)))
        uminus = r/(spin*(r-1))**2*(-r**3+3*r-2*spin**2-2*np.sqrt(delta*(2*r**3-3*r**2+spin**2)))
    return uplus, uminus 

def getcpm(r, b, spin, theta, varphi):
    alpha = b*np.cos(varphi)
    beta = b*np.sin(varphi)
    delta = r**2-2*r+spin**2
    psi = alpha-(r+1)/(r-1)*spin*np.sin(theta)
    chi = 1-delta/(r*(r-1)**2)
    cplus = ((1+np.sqrt(chi))/8/chi)**2*delta/2/r**4/chi*np.sqrt(beta**2+psi**2)
    drplus = (1+np.sqrt(1-spin**2))/r-1
    Q0 = 1+drplus/chi+drplus**2/4/chi
    Q2 = 2*np.sqrt(Q0)/(Q0+1-drplus**2/4/chi)
    cminus = -np.sqrt(1-chi)/(1+np.sqrt(chi))*np.sqrt((1+Q2)/(1-Q2))*cplus
    return cplus, cminus


def getparams(spin, theta, varphi):  #main function that prints impact parameter and lyapunov exponent
    if spin == 0:
        rho = np.sqrt(27)
        r = 3
        gamma = np.pi
        cplus = 1944 / (12 + 7 * np.sqrt(3))
        cminus = 648 * (26 * np.sqrt(3) - 45)
        return rho, gamma, cplus, cminus, r
        
    r, rho = np.abs(getparamsanalytic(spin, theta)) if (varphi-np.pi/2)%np.pi==0 else np.abs(getparamsnumerical(spin, theta, varphi))
#    print('BL radius: {}'.format(r))
    up, um = getangroots(spin, r)
    if (varphi-np.pi/2)%np.pi!=0:
        if np.abs(rootfunc((r, rho), comp=True)[0]) >= 1e-7 or np.abs(rootfunc((r, rho), comp=True)[1]) >= 1e-7:
            print('bad job getting bc')
    rdoubleprime = 3*spin**2-2*rho**2+12*r**2+spin**2*np.cos(2*theta)
    gamma = np.abs(np.sqrt(2*rdoubleprime/(-um*spin**2 +0j))*complex(ek(up/um)))
    cplus, cminus = np.abs(getcpm(r, rho, spin, theta, varphi))
    return rho, gamma, cplus, cminus, r


def plotfixedspin(spinvals, theta, dvarphi):
    colors = ['r', 'b', 'g', 'orange']
    phivals = np.arange(0,np.pi,dvarphi)
    
    for i in range(len(spinvals)):
        yvals = [getparams(spinvals[i], theta, varphi)[1] for varphi in phivals]
        plt.plot(phivals, yvals, color=colors[i])
        
    plt.ylim(0,np.pi+0.1)
    plt.show()
    return


# spin = 0.94
# theta = 17*np.pi/180
# varphi = 0

# #plotfixedspin([0.001, 0.5, 0.94, .985], 80*np.pi/180, 0.01)

# getparams(spin, theta, varphi)
# getparams(spin, theta, np.pi)
    
