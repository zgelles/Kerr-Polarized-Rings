#Computes observed polarization using parallel transport of Penrose-Walker constant

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import mpmath as mp
from matplotlib import cm
import getcritparams as getcrit #computes critical Lyapunov exponents for arbitrary a
import schwarzschildexact as sch #exact case of a=0 needs to be done separately
from raytracing import *
from scipy import integrate

#first, make lorentz transformation matrix
def getlorentzboost(boost, chi):
	gamma = 1 / np.sqrt(1 - boost**2)
	coschi = np.cos(chi)
	sinchi = np.sin(chi)
	lorentzboost = np.array([[gamma, -gamma*boost*coschi, -gamma*boost*sinchi, 0],[-gamma*boost*coschi, (gamma-1)*coschi**2+1, (gamma-1)*sinchi*coschi, 0],[-gamma*boost*sinchi, (gamma-1)*sinchi*coschi, (gamma-1)*sinchi**2+1, 0],[0,0,0,1]])
	return lorentzboost



#gets kappa and EVPA - borrowing a good chunk from Daniel Palumbo's original script
def getevec(b, varphi, r, spin, theta0, boost, chi, bvec, normalret=True, mbar=0, use2=True, setman=False, retnew=False, retf=False):
	alpha = b * np.cos(varphi)
	beta = b * np.sin(varphi)
	eta0 = beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2


	if np.min(eta0) <= 0:
		print('vortical')
	
	d = r**2 - 2 * r + spin**2
	Xi = (r**2 + spin**2)**2 - d * spin**2
	omega = 2 * spin * r / Xi
	sigma = r**2
	lam = -alpha * np.sin(theta0)
	eta0 = beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2
	RR = (r**2 + spin**2 - spin * lam)**2 - d * (eta0 + (spin - lam)**2)
	bvec = np.asarray(bvec)
	bvec /= np.linalg.norm(bvec)

	#zamo frame tetrad
	emutetrad = np.array([[1/r*np.sqrt(Xi/d), 0, 0, omega/r*np.sqrt(Xi/d)], [0, np.sqrt(d)/r, 0, 0], [0, 0, 0, r/np.sqrt(Xi)], [0, 0, -1/r, 0]])

	#minkowski metric
	coschi = np.cos(chi)
	sinchi = np.sin(chi)
	minkmetric = np.diag([-1, 1, 1, 1])

	#fluid frame tetrad
	coordtransform = np.matmul(np.matmul(minkmetric, getlorentzboost(-boost, chi)), emutetrad)
	coordtransforminv = np.transpose(np.matmul(getlorentzboost(-boost, chi), emutetrad))

	#lowered momenta at source
	signpr = 1 if setman==True else getsignpr(b, spin, theta0, varphi, mbar)
	plowers = np.array([-1, signpr * np.sqrt(RR)/d, np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0), lam])

	rs = r

	#raised
	pt = 1 / (rs**2) * (-spin * (spin - lam) + (rs**2 + spin**2) * (rs**2 + spin**2 - spin * lam) / d)
	pr = signpr * np.sqrt(RR) / rs**2
	ptheta = np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0) / rs**2
	pphi = 1/(rs**2) * (-(spin -lam) + (spin * (rs**2 + spin**2 - spin * lam)) / d)

	#fluid frame momenta
	pupperfluid = np.matmul(coordtransform, plowers)
	redshift = 1 / (pupperfluid[0])

	lp = 1#pupperfluid[0]/pupperfluid[3]

	#fluid frame polarization
	fupperfluid = np.cross(pupperfluid[1:], bvec)
	fupperfluid = (np.insert(fupperfluid, 0, 0)) / (np.linalg.norm(pupperfluid[1:]))

	#print('angle', np.arcsin(np.sqrt(np.sum(np.square(fupperfluid))))*180/np.pi)
	#apply the tetrad to get kerr f
	kfuppers = np.matmul(coordtransforminv, fupperfluid)
	kft = kfuppers[0]
	kfr = kfuppers[1]
	kftheta = kfuppers[2]
	kfphi = kfuppers[3]

	if retf:
		return ptheta

	#kappa1 and kappa2
	AA = (pt * kfr - pr * kft) + spin * (pr * kfphi - pphi * kfr)
	BB = (rs**2 + spin**2) * (pphi * kftheta - ptheta * kfphi) - spin * (pt * kftheta - ptheta * kft)
	kappa1 = rs * AA
	kappa2 = -rs * BB
	if retnew:
		return [kappa1, kappa2]

	#screen appearance
	nu = -(alpha + spin * np.sin(theta0))
	ealpha = (beta * kappa2 - nu * kappa1) / (nu**2 + beta**2)
	ebeta = (beta * kappa1 + nu * kappa2) / (nu**2 + beta**2)
	intensity = redshift**4 * (ealpha**2 + ebeta**2) * lp

	if ebeta == 0 and not use2:
		angle = np.pi/2

	else:
		angle = np.arctan2(ebeta, ealpha) if use2==True else -np.arctan(ealpha / ebeta)

	#normbad = ealpha**2+ebeta**2
	ealpha *= redshift**2*np.sqrt(np.abs(lp))
	ebeta *= redshift**2*np.sqrt(np.abs(lp))

	# ealpha *= np.sqrt(intensity / normbad)
	# ebeta *= np.sqrt(intensity / normbad)

	return [intensity, angle] if normalret == True else [ealpha, ebeta]


#same but with beloborodov (from Ramesh Narayan's script)
def getevecbelo(varphi, r, theta0, boost, chi, bvec, normalret=True, use2=True):
	#lots of trig and classical scattering parameters
	phi = np.arctan2(np.sin(varphi), np.cos(varphi) * np.cos(theta0))
	sintheta = np.sin(theta0)
	costheta = np.cos(theta0)

	beta = boost
	gamma = 1. / np.sqrt(1. - beta**2)

    #redshifting
	gfac = np.sqrt(1. - 2./r)
	gfacinv = 1. / gfac
	
	sinphi = np.sin(phi)
	cosphi = np.cos(phi)
	cospsi = -sintheta * sinphi
	sinpsi = np.sqrt(1. - cospsi**2)
	
	 #invoke the approximation
	cosalpha = 1. - (1. - cospsi) * (1. - 2./r)
	sinalpha = np.sqrt(1. - cosalpha**2)
	sinxi = sintheta * cosphi / sinpsi
	cosxi = costheta / sinpsi
	
	coschi = np.cos(chi)
	sinchi = np.sin(chi)

	betax = boost * coschi
	betay = boost * sinchi


    #polarization in local frame
	bx = bvec[0]
	by = bvec[1]
	bz = bvec[2]

	bmag = np.sqrt(bx**2 + by**2 + bz**2)
	
	kPthat = gfacinv
	kPxhat = cosalpha * gfacinv
	kPyhat = -sinxi * sinalpha * gfacinv
	kPzhat = cosxi * sinalpha * gfacinv

	kFthat = gamma * (kPthat - betax * kPxhat - betay * kPyhat)
	kFxhat = -gamma * betax * kPthat + (1 + (gamma-1) * coschi**2) * kPxhat + (gamma-1) * coschi * sinchi * kPyhat
	kFyhat = -gamma * betay * kPthat + (gamma-1) * sinchi * coschi * kPxhat + (1 + (gamma-1) * sinchi**2) * kPyhat
	kFzhat = kPzhat

    #intensity weighting
	redshift = 1. / kFthat**4
	kcrossbx = kFyhat * bz - kFzhat * by
	kcrossby = kFzhat * bx - kFxhat * bz
	kcrossbz = kFxhat * by - kFyhat * bx
	sinzeta = np.sqrt(kcrossbx**2 + kcrossby**2 + kcrossbz**2) / (kFthat * bmag)
	

	fFthat = 0
	fFxhat = kcrossbx / (kFthat * bmag)
	fFyhat = kcrossby / (kFthat * bmag)
	fFzhat = kcrossbz / (kFthat * bmag)

    #boost out of local frame
	fPthat = gamma * (fFthat + betax * fFxhat + betay * fFyhat)
	fPxhat = gamma * betax * fFthat + (1 + (gamma-1) * coschi**2) * fFxhat + (gamma-1) * coschi * sinchi * fFyhat
	fPyhat = gamma * betay * fFthat + (gamma-1) * sinchi * coschi * fFxhat + (1 + (gamma-1) * sinchi**2) * fFyhat
	fPzhat = fFzhat

	kPrhat = kPxhat
	kPthhat = -kPzhat
	kPphat = kPyhat

	
	fPrhat = fPxhat
	fPthhat = -fPzhat
	fPphat = fPyhat

    #Penrose-Walker
	k1 = r * (kPthat * fPrhat - kPrhat * fPthat)
	k2 = -r * (kPphat * fPthhat - kPthhat * fPphat)

	if theta0 == 0:
		xalpha = r * cosphi * sinalpha / sinpsi * gfacinv
		if np.abs(r**2 * kPthhat**2 - xalpha**2) < 1e-10 and r**2 * kPthhat**2 - xalpha**2 < 0:
			ybeta = 0
		else:
			ybeta = np.sqrt(r**2 * kPthhat**2 - xalpha**2) * np.sign(sinphi)

	else:
		kOlp = r * kPphat
		kOlth = r * np.sqrt(kPthhat**2 - kPphat**2 * costheta**2 / sintheta**2) * np.sign(sinphi)

		xalpha = -kOlp / sintheta
		ybeta = kOlth
		
	nu = -xalpha
	
	den = np.sqrt((k1**2 + k2**2) * (ybeta**2 + nu**2))
	ealpha = (ybeta * k2 - nu * k1) / den
	ebeta = (ybeta * k1 + nu * k2) / den

	intensity = sinzeta**2 * redshift
#    print(intensity-(ealpha**2+ebeta**2))
	angle = np.arctan2(ebeta, ealpha) if use2 == True else -np.arctan(ealpha / ebeta)

	normbad = ealpha**2+ebeta**2
	
    #Screen values
	ealpha *= np.sqrt(intensity / normbad)
	ebeta *= np.sqrt(intensity / normbad)

	return [intensity, angle, xalpha, ybeta] if normalret == True else [ealpha, ebeta]
