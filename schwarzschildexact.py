#computes screen varphi and impact parameter as a function of phi and rs for a schwarzschild black hole-
#using formulas from 3dmc notes and arxiv:2010.07330, 2005.03856, 1910.12873

import numpy as np
import scipy.optimize as op
import mpmath as mp
import matplotlib.pyplot as plt
import scipy.integrate as sci

#define elliptic functions (need the mpmath version to take complex args)
sn = np.frompyfunc(mp.ellipfun, 3, 1)
ef = np.frompyfunc(mp.ellipf, 2, 1)


radperuas = 1/3600/1e6/180*np.pi

global afac
global cospsichange
global bchange 
global rchange
global rinvertbelo
afac = np.sqrt(6)*np.sqrt(-2+2**(2/3)*(25-3*np.sqrt(69))**(1/3)+2**(2/3)*(25+3*np.sqrt(69))**(1/3))
cospsichange = 1/6*(-6+afac-np.sqrt(-36-afac**2+432/afac))
bchange = np.sqrt(8)*((cospsichange-1)/cospsichange)**(3/2)/np.sqrt(-2+2*(cospsichange-1)/cospsichange)
rinvertbelo = lambda psi0,b0: np.sqrt((1-np.cos(psi0))**2/(1+np.cos(psi0))**2+b0**2/np.sin(psi0)**2)-(1-np.cos(psi0))/(1+np.cos(psi0))
rchange = rinvertbelo(np.arccos(cospsichange),bchange)

#get varphi from BL phi
def getvarphi(blphi, theta, n, thetas=np.pi/2):
	return (np.arctan2(np.sin(blphi)*np.sin(thetas-theta), np.cos(blphi)) + n * np.pi)

def getblphi(b, varphi, theta, n, thetas=np.pi/2):
	if np.abs(thetas-np.pi/2)<1e-6:
		return (np.arctan2(np.sin(varphi), np.cos(varphi)*np.sin(thetas-theta)) + n * np.pi)
	else:
		return getblphimino(b, varphi, theta, n, thetas=thetas)

def rbelocone(b, varphi, theta, n, thetas=np.pi/2):
	psi = np.abs(b*gettau(b, varphi, n, theta, thetas=thetas, rethit=False))
	psit = b*gettautot(b, theta, usebelo=True)/2 #np.arccos(-2/(np.abs(getradroots(b,theta)[-1])-2))
#	rinvert = lambda psi0: np.sqrt((1-np.cos(psi0))**2/(1+np.cos(psi0))**2+b**2/np.sin(psi0)**2)-(1-np.cos(psi0))/(1+np.cos(psi0))
	userb1 = np.logical_or(psit<=np.pi, psi<=np.pi)
	usecirc = np.logical_and(np.logical_not(userb1), psi<2*psit-np.pi)
	userb2 = np.logical_and(np.logical_not(userb1), np.logical_not(usecirc))
	return (userb1*rinvertbelo(psit-np.abs(psit-psi),b) + usecirc*b**2/8 + userb2*rinvertbelo(psi+2*np.pi-2*psit,b))



def getblphimino(b, varphi, theta0, n, thetas=np.pi/2):
	q = np.sqrt(b**2*np.cos(varphi)**2*np.cos(theta0)**2+b**2*np.sin(varphi)**2)
	lam = -b*np.cos(varphi)*np.sin(theta0)

	signdiff = np.sign((np.cos(theta)-np.cos(thetas)))
	m = n*(np.sin(varphi)*signdiff<0)+(n+1)*(np.sin(varphi)*signdiff>=0)
	sinphi = 1*np.logical_and(np.sin(varphi) > 0, varphi!=0) + -1*np.logical_and(np.sin(varphi)<0, varphi!=0) + np.sign(np.cos(theta))*(varphi==0)
	
	#sinphi = 1*(np.sin(varphi) >= 0) + -1*(np.sin(varphi)<0) #just need to be consistent about defining sinvarphi\geq 0 together
	pms = sinphi*(-1)**m  #sign(ptheta) at source

	arg1  = lambda theta: np.arctan(lam*np.cos(theta)/np.sin(theta)/np.sqrt(q**2-lam**2*np.cos(theta)**2/np.sin(theta)**2))
	lamgphi = m*np.pi-sinphi*arg1(theta0)+pms*arg1(thetas)
	return (-lamgphi-np.pi/2)%(2*np.pi) #assumes observer at phi=0

#compute mino time gtheta
def gettau(b, varphi, n, theta, thetas=np.pi/2, rethit=False):
	if theta==0 or theta == np.pi:
		return ((2 * n + 1) / b * np.pi / 2)

	if rethit:
		if not rinvertcone(b, varphi, n, theta, thetas, rethit = rethit): 
			return 0
		else:
			return 0


	q = np.sqrt(b**2*np.cos(varphi)**2*np.cos(theta)**2+b**2*np.sin(varphi)**2)
	lam = -b*np.cos(varphi)*np.sin(theta)

	signdiff = np.sign((np.cos(theta)-np.cos(thetas)))
	m = n*(np.sin(varphi)*signdiff<0)+(n+1)*(np.sin(varphi)*signdiff>=0)
	sinphi = 1*np.logical_and(np.sin(varphi) > 0, varphi!=0) + -1*np.logical_and(np.sin(varphi)<0, varphi!=0) + np.sign(np.cos(theta))*(varphi==0)

	#sinphi = 1*(np.sin(varphi) > 0 and varphi!=0) + -1*(np.sin(varphi)<0 and varphi!=0) + np.sign(np.cos(theta))*(varphi==0) #just need to be consistent about defining sinvarphi\geq 0 together
	pms = sinphi*(-1)**m #sign(ptheta) at source
	return 1/b * (np.pi * m - sinphi*np.arcsin(b*np.cos(theta)/ q+0j) 
				+pms*np.arctan(b/np.tan(thetas)/np.sqrt(q**2-lam**2/np.tan(thetas)**2+0j)))


#closed form expression for rs as a function of screen coords (to be inverted)
def getradroots(b, theta): #note!!! this needs to take the G+L roots for kerr and *then* set a=0 (else everything complex conjugated)
	fac1 = (-b**6 + 6*b**4 * (9 + np.sqrt(81-3*b**2 + 0j)))**(1/3)
	xi = (b**2 + fac1)**2 / 6 / fac1
	z0 = (xi / 2)**(1/2)
	a0 = -b**2
	b0 =  2*(b**2)
	rootfac1 = (-a0 / 2 - z0**2 + b0 / 4 / z0)**(1/2)
	rootfac2 = (-a0 / 2 - z0**2 - b0 / 4 / z0)**(1/2)
	r1 = -z0 - rootfac1
	r3 = z0 - rootfac2
	r4 = z0 + rootfac2
	return r1, r3, r4 #r2 = 0 by default


def getbbelo(rs, theta0, blphi):
	fac1 = rs**2*(1-np.sin(theta0)**2*np.sin(blphi)**2)
	fac2 = 2*rs*(1+np.sin(theta0)**2*np.sin(blphi)**2+2*np.sin(theta0)*np.sin(blphi))
	return np.sqrt(fac1+fac2)


def rinvertconeprelim(b, varphi, n, theta, thetac):
	tau1 = gettau(b, varphi, n, theta, thetas=thetac)
	bratio = np.complex128(np.sqrt(27) / b)
	rootfac = -bratio + np.sqrt(-1 + bratio**2)
	r1, r3, r4 = getradroots(b, theta)
	r31 = r3 - r1
	r41 = r4 - r1
	k = r3 * r41 / r31 / r4
	ffac = 1 / 2 * (r31 * r4)**(1/2)
	fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
	snnum1 = r41*(np.complex128(sn('sn', ffac * tau1 - fobs, k)))**2
	return np.real((r4 * r31 - r3 * snnum1) / (r31 - snnum1))


def rinvertcone(b, varphi, n, theta, thetac, usebelo=False, rethit=False):
	q = np.sqrt(b**2*np.cos(varphi)**2*np.cos(theta)**2+b**2*np.sin(varphi)**2)
	lam = -b*np.cos(varphi)*np.sin(theta)
	hitscone = np.logical_or(q**2-lam**2/np.tan(thetac)**2>0, np.abs(np.cos(theta))>np.abs(np.cos(thetac)))

	if rethit:
		return hitscone

	#if hitscone:
	x1=gettautot(b,theta, usebelo=usebelo)
	x2=gettau(b, varphi, n, theta, thetas=thetac)
	hitscone *= (x1>x2)

	#if hitscone:
	if not usebelo:
		return hitscone*rinvertconeprelim(b, varphi, n, theta, thetac)
	return hitscone*rbelocone(b, varphi, theta, n, thetas=thetac)


def rinvert(b, varphi, n, theta, thetas=np.pi/2, usetau=False, tau0=None):
	tau = gettau(b, varphi, n, theta, thetas=thetas) if usetau==False else tau0
	bratio = np.complex128(np.sqrt(27) / b)
	rootfac = -bratio + np.sqrt(-1 + bratio**2)
	r1, r3, r4 = getradroots(b, theta)
	r31 = r3 - r1
	r41 = r4 - r1
	k = r3 * r41 / r31 / r4
	ffac = 1 / 2 * (r31 * r4)**(1/2)
	fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
	snnum = r41*(np.complex128(sn('sn', ffac * tau - fobs, k)))**2
	rs = np.real((r4 * r31 - r3 * snnum) / (r31 - snnum))
	return rs

#screen impact param is the root of this equation
def geodesiceq(b):
	global varphi1
	global theta1
	global r1
	global n1
	global thetas1

	return np.abs(r1-rinvert(b, varphi1, n1, theta1, thetas=thetas1))

#find the root using scipy.optimize
def findb(r, varphi, theta, n, thetas=np.pi/2):
	global varphi1
	global theta1
	global r1
	global n1
	global thetas1
	# global listerr

	varphi1 = varphi
	theta1 = theta
	r1 = r
	n1 = n
	thetas1 = thetas

	#findroot params
	tol0 = 1e-10
	maxiter = 1000
	options0 = {'maxiter' : maxiter}

	#set seed values - add one for n=0 and subring approx for n>0
	if n==0:
		guess = r + 1
	else:
		bc = np.sqrt(27)
		cp = 1944 / (12 + 7 * np.sqrt(3))
		m = n if np.sin(varphi)*np.cos(theta) < 0 else n+1
		gamma = np.pi
		guess = bc + 1 / cp * np.exp(-gamma * (m + 0.5)) #face-on, bound orbit approximation, assumes outside crit curve (which I think is true for rs>4)
		
	solb = op.root(geodesiceq, guess, method='lm', tol=tol0, options=options0)

	#check to make sure numerics didn't mess anything up
	if solb.fun / r >= 1e-6:
		print('err is {} at r={} and varphi = {} and theta = {}'.format(solb.fun / r, r, varphi, theta))

	# listerr.append(solb.fun / r)
	return (solb.x[0])

#numerically compute screen coordinates
def getscreencoords(r, blphi, theta, n):
	varphi = getvarphi(blphi, theta, n)
	b = findb(r, varphi, theta, n)
	return b, varphi


def gettautot(b, theta0, usebelo=False):
	r1, r3, r4 = getradroots(b, theta0) #get radial roots - r4 will be complex if inside crit curve (which is fine)
	psitbelo = np.abs(np.arccos(-2/(np.abs(r4)-2)+0j))
	
	usepsibelo = (b>=bchange)
	psitexp = np.abs(np.arccos(cospsichange+0j))+np.log((rchange-3)/(r4-3))

	if usebelo:
		return 2/b*(usepsibelo*psitbelo+np.logical_not(usepsibelo)*psitexp)

	ifacout = lambda r: np.abs(1/np.sqrt(r*(r-r1)*(r-r3)))
	iintout = lambda u: ifacout(u**2+r4)
	irtotout = 4*sci.quad_vec(iintout, 0, np.inf)[0] #outside crit curve - needs change of variables to make sure integrand doesn't explode
#	print(irtotout)

	iintin = lambda r: 1/np.sqrt(r**4-r*(r-2)*b**2)
	irtotin = 2*sci.quad_vec(iintin, 2, np.inf)[0] #inside crit curve - no change of variables necessary since denom has no roots

	return np.nan_to_num((b<np.sqrt(27))*irtotin,nan=0) + np.nan_to_num((b>np.sqrt(27))*irtotout,nan=0)


def getpsimap(theta0, fovuasx, fovuasy, nx, ny):
	psizex = fovuasx/nx
	psizey = fovuasy/ny
	convertfac = 6.6742*1e-11*6.2*1e9*1.989*1e30/3.085678/1e16/16.9/1e6/(2.99792458*1e8)**2/radperuas
	xlist = np.flip(np.arange(0, -nx, -1) * psizex + (psizex * nx) / 2.0 - psizex / 2.0)
	ylist = np.flip(np.arange(0, -ny, -1) * psizey + (psizey * ny) / 2.0 - psizey / 2.0)
	x,y = np.meshgrid(xlist/convertfac,ylist/convertfac)
	bvals = np.sqrt(x**2+y**2)
	psivals = gettautot(bvals, theta0)*bvals
	return psivals


#now we need to find the winding number. there are a couple steps to this.
#first, compute psin using 3mdc formalism

def getpsin(theta, blphi, n):
	psib = np.arccos(-np.sin(theta) * np.sin(blphi)) % np.pi if n % 2 == 0 else -2 * np.pi + np.arccos(-np.sin(theta) * np.sin(blphi))
	psin = psib + int(n / 2) * (-1)**n * 2 * np.pi
	return psin


def getpsitr(r, varphi, theta, n, thetas=np.pi/2): #gets psit as a function of r
	b = findb(r, varphi, theta, n, thetas=thetas)
	return getpsit(b, theta)

#gets psit as a function of b
def getpsit(b, theta):
	r1, r3, r4 = getradroots(b, theta)
	r31 = r3 - r1
	r41 = r4 - r1
	k = r3 * r41 / r31 / r4
	argx2 = r31  / r41
	x2 = np.arcsin(np.sqrt(argx2))
	prefac = 2 * b / np.sqrt(r31 * r4)
	psit = np.abs(prefac * np.complex128(ef(x2, k)))
	return psit

#next, need sign(p^r) at the emission radius to determine if ray hits turning point
def getsignpr(b, r, theta, psin):
	# if b <= np.sqrt(27):
	#     return 1
	psit = getpsit(b, theta)
	out = (np.abs(psin) < psit).astype(int)
	out[np.abs(psin)<psit] = -1
	out[b <= np.sqrt(27)]=1
	return out
	# if np.abs(psin) < psit:
	#     return 1
	# elif np.abs(psin) > psit:
	#     return -1
	# return 0


def getw(bvals, varphivals, rs, theta0, mbar, thetas=np.pi/2): #assumes equatorial emission
	r1, r3, r4 = getradroots(bvals, theta0)

	r13 = r1-r3
	r14 = r1-r4
	r31 = r3-r1
	r34 = r3-r4
	r41 = r4-r1
	r43 = r4-r3

	tautot = gettau(bvals, varphivals, mbar, theta0, thetas=thetas)

	k = r3*r41/r31/r4
	tauturning = 2 / np.sqrt(r31 * r4) * np.complex128(ef(np.arcsin(np.sqrt(r31 / r41)), k)) #check if turning point is encountered
	w = np.heaviside(np.real((bvals>=np.sqrt(27)) * np.sign(tautot-tauturning)).astype(int), 0) #sign(pr) at source
	return w

def magnificationratio(bvals, varphivals, rs, theta0, mbar): #assumes equatorial emission
	r1, r3, r4 = getradroots(bvals, theta0)
	omegas = -rs**(-3/2)
	lam = -bvals*np.cos(varphivals)*np.sin(theta0)
	eta = bvals**2*np.cos(varphivals)**2*np.cos(theta0)**2+bvals**2*np.sin(varphivals)**2


	r13 = r1-r3
	r14 = r1-r4
	r31 = r3-r1
	r34 = r3-r4
	r41 = r4-r1
	r43 = r4-r3
	tautot = gettau(bvals, varphivals, mbar, theta0)

	rr = lambda r,b: r**4-r*(r-2)*b**2
	hlam = -lam/bvals**2*(tautot+np.sign(np.sin(varphivals))*np.cos(theta0)/np.sqrt(eta*np.sin(theta0)**2-lam**2*np.cos(theta0)**2))
	hq = 1/bvals**2*(-np.sqrt(eta)*tautot+np.sign(np.sin(varphivals))*np.cos(theta0)*lam**2/np.sqrt(eta)/np.sqrt(eta*np.sin(theta0)**2-lam**2*np.cos(theta0)**2))
	ell = -np.sign(np.sin(varphivals))/np.tan(theta0)/np.sqrt(eta-lam**2/np.tan(theta0)**2)

	k = r3*r41/r31/r4
	tauturning = 2 / np.sqrt(r31 * r4) * np.complex128(ef(np.arcsin(np.sqrt(r31 / r41)), k)) #check if turning point is encountered
	w = np.heaviside(np.real((bvals>=np.sqrt(27)) * np.sign(tautot-tauturning)).astype(int), 0) #sign(pr) at source

	print(w)

	d1 = 1/bvals**2*(r1+2*np.sqrt(bvals**2-3*r1**2/4+0j)/np.sqrt(bvals**2-27+0j))
	d3 = 1/bvals**2*(r3-2*np.sqrt(bvals**2-3*r3**2/4+0j)/np.sqrt(bvals**2-27+0j))
	d4 = 1/bvals**2*(r4+2*np.sqrt(bvals**2-3*r4**2/4+0j)/np.sqrt(bvals**2-27+0j))

	fvals = []
	gvals = []
	for i in range(len(w)):
		fint = lambda r: -omegas*r**4 / (rr(r, bvals[i]))**(3/2)
		gint = lambda r: r*(r-2) / (rr(r, bvals[i]))**(3/2)

		ftemp = sci.quad(fint, rs, np.inf)[0]
		gtemp = sci.quad(gint, rs, np.inf)[0]


		if w[i] == 1:
			fleib = -2*omegas*rs**3*d4[i]/(rs-2)/np.sqrt(rr(rs,bvals[i]))
			denom = lambda u: (u**2+r41[i])*(u**2+r4[i])*(u**2+r43[i])
			f1int = lambda u: -2*(u**2+r4[i])**3 * ((d4[i]-d1[i])*(u**2+r4[i])*(u**2+r43[i])+(d4[i])*(u**2+r41[i])*(u**2+r43[i])+(d4[i]-d3[i])*(u**2+r41[i])*(u**2+r4[i])) / (denom(u))**(3/2) / (u**2+r4[i]-2)
			f2int = lambda u: 12*d4[i]*(u**2+r4[i])**2/np.sqrt(denom(u))/(u**2+r4[i]-2)
			f3int = lambda u: -4*d4[i]*(u**2+r4[i])**3/(u**2+r4[i]-2)**2/np.sqrt(denom(u))
			ftot = lambda u: omegas*np.real(f1int(u)+f2int(u)+f3int(u))

			gleib = 2*d4[i]/np.sqrt(rr(rs, bvals[i]))
			gtot = lambda u: 2*np.real(((d4[i]-d1[i])*(u**2+r4[i])*(u**2+r43[i])+(d4[i])*(u**2+r41[i])*(u**2+r43[i])+(d4[i]-d3[i])*(u**2+r41[i])*(u**2+r4[i])) / (denom(u))**(3/2))

			upperbound = np.real(np.sqrt(rs-r4[i]))
			ftemp -= (fleib + sci.quad(ftot, 0, upperbound)[0])
			gtemp -= (gleib + sci.quad(gtot, 0, upperbound)[0])

		if np.abs(ftemp)>5:
			print(r4[i], lam[i], np.sqrt(eta[i]))


		fvals.append(ftemp)
		gvals.append(gtemp)

	# loc1=(np.where(np.abs(hlam*lam)==np.max(np.abs(lam*hlam)))[0][0])
	# print(varphivals[loc1]*180/np.pi)
	# hlam[loc1] = ell[loc1] = hq[loc1] = 0
	# loc2=(np.where(np.abs(hlam*lam)==np.max(np.abs(lam*hlam)))[0][0])
	# print(varphivals[loc2]*180/np.pi)
	# hlam[loc2] = ell[loc2] = hq[loc2] = 0

	fvals = np.array(fvals)
	gvals = np.array(gvals)

	dadlam = np.real(lam*gvals-hlam)
	dadq = np.real(np.sqrt(eta)*gvals-hq)
	dbdlam = np.real(lam*fvals+ell)
	dbdq = np.real(np.sqrt(eta)*fvals-lam/np.sqrt(eta)*ell)

	#plt.plot(np.real(hlam))
	# plt.plot(np.real(lam*fvals))
	# plt.plot(np.real(ell))
	# plt.plot(np.real(lam*hlam))
	#plt.show()
	boost = 1/np.sqrt(rs-2)
	gamma = 1/np.sqrt(1-boost**2)
	redshift = rs*np.sqrt(rs-2)/gamma/(rs**(3/2)+boost*lam*np.sqrt(rs-2)) #redshift in schwarzschild
	print(redshift)
	invjacobian = 1/np.abs(dbdlam*dadq-dbdq*dadlam)
	otherfac = 1/np.sqrt(rs**4-rs*(rs-2)*bvals**2)/np.sqrt(eta-lam**2/np.tan(theta0)**2)*np.sqrt(rs**3-3*rs**2)/rs**(3/2)
	#all other factors cancel in ratio between f0 and f1
	return (otherfac*invjacobian/redshift/np.sin(theta0)) #reduces ramesh's delta by one power due to pt/pr term in jacobian

#now compute alpha_n
def getalphan(b, r, theta, psin):
	signpr = getsignpr(b, r, theta, psin)
	arctannum = np.arctan(1 / np.sqrt(r**2/b**2/(1-2/r)-1))
	signpsin = np.sign(psin)
	out = signpsin * (np.pi-arctannum)
	mask = (signpr == 1)*(0<psin)*(psin<np.pi)
	out[mask] = (signpsin*arctannum)[mask]
	return out

	# if signpr == 1 or (0<psin<np.pi):
	#     return signpsin * arctannum
	# elif signpr == -1:
	#     return signpsin * (np.pi - arctannum)
	# return (signpsin * np.pi / 2)

#subtract from psi_n  to get total winding angle = xi_n^R
def getwindangle(b, r, blphi, theta, n):
	psin = getpsin(theta, blphi, n)
	alphan = getalphan(b, r, theta, psin)
	return (psin - alphan)



def gettime(bvals, varphivals, theta0, r, rmax, n):
	r1, r3, r4 = getradroots(bvals, theta0) #r2=0 by default
	print('hi2')
	r31 = r3-r1
	r41 = r4-r1
	tautot = gettau(bvals, varphivals, n, theta0)
	k = r3*r41/r31/r4
	tauturning = 2 / np.sqrt(r31 * r4) * np.complex128(ef(np.arcsin(np.sqrt(r31 / r41)), k)) #check if turning point is encountered
	nur = (bvals<np.sqrt(27)) + (bvals>=np.sqrt(27)) * np.sign(tauturning-tautot) #sign(pr) at source

	results = []
	for i in range(len(bvals)):
		gett0 = lambda r: r**3/(r-2)/np.sqrt(r**4-bvals[i]**2*r*(r-2))
		gett1 = lambda r: 2*r**3 / ((r-2)*np.sqrt(r*(r-r1[i])*(r-r3[i])))
		gett2 = lambda t: np.real(gett1(t**2+r4[i]))
		timevals, timeerr = sci.quad(gett0, r, rmax)
#		timevals, timeerr = sci.quad(gett2, np.real(np.sqrt(r-r4[i])), np.sqrt(rmax-r4[i]))
		if nur[i]<0:
			timevals2, timeerr2 = sci.quad(gett2, 0, np.real(np.sqrt(r-r4[i])))
			timevals += 2*timevals2
		results.append(timevals)

	return np.array(results)



def main():
	return 
	#print(rinvert(4, np.pi/4, 1, 17*np.pi/180, thetas=np.pi/2))
	#thetasvals = (10+np.arange(0,180,20))*np.pi/180
	# thetavals = np.array([1,20,60,70,80])*np.pi/180
	# npts = 100
	# varphi = np.pi/3
	# rvals = np.linspace(2.5,10,npts)
	# theta0 = 1*np.pi/180
	# rvals.astype('float32').tofile('datavecs/rvals.dat')
	# for thetas in thetasvals:
	# 	tdeg = round(thetas*180/np.pi)
	# 	bvals0 = np.array([findb(r, varphi, theta0, 0, thetas=thetas) for r in rvals])
	# 	bvals0.astype('float32').tofile('datavecs/bvals0inc1vp60thetas{}.dat'.format(tdeg))
	# 	bvals1 = np.array([findb(r, varphi, theta0, 1, thetas=thetas) for r in rvals])
	# 	bvals1.astype('float32').tofile('datavecs/bvals1inc1vp60thetas{}.dat'.format(tdeg))
	# 	wvals0 = getw(bvals0, varphi, rvals, theta0, 0, thetas=thetas)
	# 	wvals0.astype('float32').tofile('datavecs/wvals0inc1vp60thetas{}.dat'.format(tdeg))
	# 	wvals1 = getw(bvals1, varphi, rvals, theta0, 1, thetas=thetas)
	# 	print(wvals1)
	# 	wvals1.astype('float32').tofile('datavecs/wvals1inc1vp60thetas{}.dat'.format(tdeg))

	# return
	# print('hi')
	# global listerr;
	# listerr=[]

	# r=8
	# rmax = 8
	# n=0
	# theta0 = 80*np.pi/180
	# blphivals = np.arange(0, 2*np.pi, 2*np.pi/100)
	# varphivals = getvarphi(blphivals, theta0, n)

	# for i in range(len(varphivals)):
	# 	if varphivals[i] % np.pi <= 1e-8:
	# 		varphivals[i] += 1e-6

	# b = np.linspace(8,9,100)
	# print(rinvertcone(np.linspace(8,9,100), np.pi/3, 0, 80*np.pi/180, 1*np.pi/180))
	#print(rinvertcone(np.linspace(8,9,100), -np.pi/3, 0, np.pi-20*np.pi/180, np.pi-np.pi/3))
#	print([rinvert(rs, np.pi/3, 0, 20*np.pi/180) for rs in np.linspace(8,9,100)])


if __name__=="__main__":
	main()

