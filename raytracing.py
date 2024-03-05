#Performs geodesic integration from equator to observer using elliptic integral formalism of arXiv:1910.12873 
#For inquiries, contact Zack Gelles at zgelles@princeton.edu

import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import scipy.optimize as op
import mpmath as mp
import matplotlib
from matplotlib import cm
import matplotlib.patches as pat
import getcritparams as getcrit #computes critical Lyapunov exponents for arbitrary a
import schwarzschildexact as sch #exact case of a=0 needs to be done separately
from scipy import integrate


#define elliptic functions (need the mpmath version to take complex args)
ek = np.frompyfunc(mp.ellipk, 1, 1)
ef = np.frompyfunc(mp.ellipf, 2, 1)
sn = np.frompyfunc(mp.ellipfun, 3, 1)
ep1 = np.frompyfunc(mp.ellippi, 3, 1)
epcomp = np.frompyfunc(mp.ellippi, 2, 1)

global listerr
listerr=[]


#get roots of radial potential
def getrootsrad(alpha, beta, spin, theta0, lam1=0, eta1=0, uselameta=False):
	lam = -alpha * np.sin(theta0) * (not uselameta) + lam1 * uselameta
	eta0 = (beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2) * (not uselameta) + eta1 * uselameta
	Aconst  = spin**2-eta0-lam**2 + 0j
	Bconst = 2*(eta0 + (lam - spin)**2)
	Cconst = -spin**2 * eta0
	Pconst = -1 * Aconst**2 / 12 - Cconst
	Qconst = -Aconst / 3 * (Aconst**2 / 36 - Cconst) - Bconst**2 / 8
	discr = -9 * Qconst + np.sqrt(12 * Pconst**3 + 81 * Qconst**2 + 0j)

	#root to resolvent cubic using version of Cardano's method consistent with Mathematica (and python)
	xi = (-2 * 3**(1/3) * Pconst + 2**(1/3) * discr**(2/3)) / (6**(2/3) * discr**(1/3)) - Aconst / 3

	zconst = np.sqrt(xi / 2+0j)
	rootfac1 = -1 * Aconst / 2 - zconst**2 + Bconst / 4 / zconst
	rootfac2 = -1 * Aconst / 2 - zconst**2 - Bconst / 4 / zconst
	finalconst1 = np.sqrt(np.real(rootfac1)+(1j)*np.imag(rootfac1))
	finalconst2 = np.sqrt(np.real(rootfac2)+(1j)*np.imag(rootfac2))
	
	r1 = -1 * zconst - finalconst1
	r2 = -1 * zconst + finalconst1
	r3 = zconst - finalconst2
	r4 = zconst + finalconst2

	return np.array([r1, r2, r3, r4])

#difference of two radial roots
def rij(listroots, i, j):
	return (listroots[i-1]-listroots[j-1])

#get angular roots u_+ and u_-
def getrootsang(alpha, beta, spin, theta0, lam1=0, eta1=0, uselameta=False):
	lam = -alpha * np.sin(theta0) * (not uselameta) + lam1 * uselameta
	eta0 = (beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2) * (not uselameta) + eta1 * uselameta
	deltatilde = 1 / 2 * (1 - (eta0 + lam**2) / spin**2)
	up = deltatilde + np.sqrt(deltatilde**2 + eta0 / spin**2)
	um = deltatilde - np.sqrt(deltatilde**2 + eta0 / spin**2)
	return up, um


#compute mino time gtheta
def getgtheta(alpha, beta, spin, theta0, mbar):
	if spin == 0:
		return getgthetasch(alpha, beta, spin, theta0, mbar)
		
	up, um = getrootsang(alpha, beta, spin, theta0)
	heavisidebeta = 1 if beta*np.cos(theta0) >= 0 else 0
	signbeta = 1 if beta*np.cos(theta0) >= 0 else -1
	argsin = np.abs(np.cos(theta0)) / np.sqrt(up) + 0j
	kcomplete = complex(ek(up / um))
	fobs = complex(ef(np.arcsin(argsin), up / um))

	if theta0 == 0: #special face-on case
		b = np.sqrt(alpha**2 + beta**2)
		gtp = (2 * mbar + 1) / np.sqrt(b**2 - spin**2) * complex(ek(spin**2 / (spin**2 - b**2)))
		return gtp
	
	return (1/np.sqrt(-spin**2 * um + 0j) * (2 * (mbar + heavisidebeta) * kcomplete - signbeta*fobs))

#Gtheta for off-equatorial emission
def getgthetanoneq(alpha, beta, spin, theta0, thetas, mbar):
	if spin == 0:
		return sch.gettau(np.sqrt(alpha**2+beta**2), np.arctan2(beta,alpha), mbar, theta0, thetas=thetas)
		
	up, um = getrootsang(alpha, beta, spin, theta0)
	heavisidebeta = 1 if beta*np.cos(theta0) >= 0 else 0
	signbeta = 1 if beta*np.cos(theta0) >= 0 else -1
	argsin = np.abs(np.cos(theta0)) / np.sqrt(up) + 0j
	argsins = np.cos(thetas) / np.sqrt(up) + 0j
	kcomplete = complex(ek(up / um))
	fobs = complex(ef(np.arcsin(argsin), up / um))
	fs = complex(ef(np.arcsin(argsins), up / um))

	pms = signbeta*(-1)**(mbar+heavisidebeta)

	if theta0 == 0: #special face-on case
		b = np.sqrt(alpha**2 + beta**2)
		gtp = (2 * mbar + 1) / np.sqrt(b**2 - spin**2) * complex(ek(spin**2 / (spin**2 - b**2)))
		return gtp
	
	return (1/np.sqrt(-spin**2 * um + 0j) * (2 * (mbar + heavisidebeta) * kcomplete - signbeta*fobs + pms * fs))

#special case for schwarzschild
def getgthetasch(alpha, beta, spin, theta0, mbar):
	b = np.sqrt(alpha**2 + beta**2)
	varphi = np.arctan2(beta, alpha)
	if theta0==0:
		return ((2 * mbar + 1) / b * np.pi / 2)
	m = mbar if beta*np.cos(theta0)<0 else mbar+1
	sinphi = 1 if beta >= 0 else -1
	return 1/b * (np.pi * m - sinphi*np.arcsin(np.cos(theta0) / np.sqrt(np.cos(theta0)**2 * np.cos(varphi)**2 + np.sin(varphi)**2)))


#inversion formula for r as a function of screen coords
def rinvert(alpha, beta, spin, theta0, mbar, thetas=np.pi/2):
	tau = getgtheta(alpha, beta, spin, theta0, mbar) if np.abs(thetas-np.pi/2)<1e-6 else getgthetanoneq(alpha, beta, spin, theta0, thetas, mbar)
	rvals = getrootsrad(alpha, beta, spin, theta0)
	k0 = (rij(rvals, 3, 2) * rij(rvals, 4, 1)) / (rij(rvals, 3, 1) * rij(rvals, 4, 2))
	arginvert = 1 / 2 * np.sqrt(rij(rvals, 3, 1) * rij(rvals, 4, 2)) * tau - complex(ef(np.arcsin(np.sqrt(rij(rvals, 3, 1)/rij(rvals, 4, 1))), k0))
	tempnum = rij(rvals, 4, 1) * (complex(sn('sn', arginvert, k0)))**2
	finalnum = (rvals[3] * rij(rvals, 3, 1) - rvals[2] * tempnum) / (rij(rvals, 3, 1) - tempnum)
	return finalnum

#equation whose root is the impact parameter
def geodesiceq(b):
	global spin1
	global varphi1
	global theta1
	global r1
	global mbar1
	global thetas1

	return np.abs(r1-rinvert(b*np.cos(varphi1),b*np.sin(varphi1),spin1,theta1,mbar1, thetas=thetas1))

#generate precise guess for higher order subimages (i.e. close to critical curve)
def getguess(alpha, beta, spin, theta0):
	up, um = getrootsang(alpha, beta, spin, theta0)
	iscloseto1 = np.abs(np.cos(theta0) / np.sqrt(up)) - 1 <= 1e-8 and np.real(np.cos(theta0) / np.sqrt(up)) > 1
	argsin = 1 if iscloseto1 else np.abs(np.cos(theta0) / np.sqrt(up))
	kcomplete = complex(ek(up / um))
	fobs = complex(ef(np.arcsin(argsin), up / um))
	return fobs/kcomplete, fobs / np.sqrt(-um * spin**2)


#numerically invert r(b) to find impact parameter
def findb(r, varphi, spin, theta0, mbar, thetas=np.pi/2):
	if spin==0:
		return sch.findb(r, varphi, theta0, mbar, thetas=thetas)
	global spin1
	global varphi1
	global theta1
	global r1
	global mbar1
	global listerr
	global thetas1
	
	spin1 = spin
	varphi1 = varphi
	theta1 = theta0
	spin1 = spin
	r1 = r
	mbar1 = mbar
	thetas1 = thetas

	#findroot params
	tol0 = 1e-10
	maxiter = 1000
	options0 = {'maxiter' : maxiter}

	guess = r + 1 + (spin**2 - 1) / 2 / r + (50 - 2 * spin**2 - 15 * np.pi) / 4 / r**2

	if mbar > 0:
		m = mbar if np.sin(varphi)*np.cos(theta0) < 0 else mbar + 1
		rho, gamma, cplus, cminus, rt = getcrit.getparams(spin, theta0, varphi)

		alphat = rho * np.cos(varphi)
		betat = rho * np.sin(varphi)
		deltat = rt**2-2*rt+spin**2
		psit = alphat-(rt+1)/(rt-1)*spin*np.sin(theta0)
		chit = 1-deltat/(rt*(r-1)**2)
		fac, f0 = getguess(rho * np.cos(varphi), rho * np.sin(varphi), spin, theta0)

		guess0 = r / rt - 1
		fac1 = 4 * chit / (1 + np.sqrt(chit)) * np.exp(np.sign(np.sin(varphi)*np.cos(theta0)) * 2 * rt * np.sqrt(chit) * f0 - m * gamma)
		guess0 -= fac1
		fac2 = (1 + np.sqrt(chit)) / (32 * rt**4 * chit**2) * np.sqrt(betat**2 + psit**2) * deltat * np.exp(-np.sign(np.sin(varphi)*np.cos(theta0)) * 2 * rt * np.sqrt(chit) * f0 + m * gamma)
		guess0 /= fac2

		guess = rho + 1 / cplus * np.exp(- (m + .5) * gamma)
		fac, f0 = getguess(rho * np.cos(varphi), rho * np.sin(varphi), spin, theta0)
		guess = np.abs(rho + 1 / cplus * np.exp(-gamma * (m - .5 * np.sign(np.sin(varphi)*np.cos(theta0)) * fac)))

		# if np.abs(spin) < .1:
		#     guess = sch.findb(r, varphi, theta, mbar, uselisterr=False)
	


	solb = op.root(geodesiceq, guess, method='lm', tol=tol0, options=options0)
	if solb.fun / r > 1e-5:
		print('err is {} at r={} and varphi = {} and theta = {}'.format(solb.fun/r, r, varphi, theta1))
	listerr.append(np.abs(solb.fun[0] / r))
	return (solb.x[0])




#we need to compute the mino time for a geodesic connecting the observer to a turning point to determine \pm_r
def gettaut(b, varphi, spin, theta0, lam1=0, eta1=0, uselameta=False):
	rootlist = getrootsrad(b*np.cos(varphi), b*np.sin(varphi), spin, theta0, lam1=lam1, eta1=eta1, uselameta=uselameta)
	r31 = rootlist[2]-rootlist[0]
	r32 = rootlist[2]-rootlist[1]
	r41 = rootlist[3]-rootlist[0]
	r42 = rootlist[3]-rootlist[1]
	prefac = 2 / np.sqrt(r31 * r42)
	k = r32 * r41 / r31 / r42
	x2 = np.arcsin(np.sqrt(r31 / r41))
	fobs = prefac * np.complex128(ef(x2, k))
	return np.abs(fobs)



#check sign of pr since it's otherwise ambiguous
def getsignpr(b, spin, theta0, varphi, mbar):
	if np.abs(spin)>0:
		bc = np.abs(getcrit.getparams(spin, theta0, varphi)[0])
	else:
		bc = np.sqrt(27)
	if b < bc:
		return 1
	gtheta = np.abs(getgtheta(b*np.cos(varphi), b*np.sin(varphi), spin, theta0, mbar))
	taut = gettaut(b, varphi, spin, theta0)
	return int(np.sign(taut - gtheta))

