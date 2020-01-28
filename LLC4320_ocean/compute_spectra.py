""" Script can be used to compute 2D spectrum and radially averaged 1D spectrum.
It also employs Helmholtz decomposition to compute spectra of rotational and divergent components.

Original by Cesar Rocha. Modifications by Hemant Khatri """

import numpy as np
from numpy import pi
from scipy.special import gammainc
from scipy import signal
import sys

try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

def TWODimensional_spec(phi,d1,d2,detrend=True,han_win=True):

    # phi = two dimensional real field
    # d1, d2 = grid interval in dimension 0 (along rows) and 1 (along columns)
    n2, n1 = phi.shape
    L1 = d1*n1
    L2 = d2*n2

    # wavenumber one (equals to dk1 and dk2)
    dk1 = 2.*np.pi/L1
    dk2 = 2.*np.pi/L2

    if(n1%2 == 1 or n2%2 == 1):
        sys.exit("2D field must have even dimentions")

    if detrend:
        phi = lin_detrend(phi)
    else:
        pass

    """ Applies hanning window to make the field smoothly go to zero at end points
    Or Else applies bump function, which quickly decays to zero at end points.
    This is better than hanning window, which removes signicant part of the signal """

    if han_win:
        phi = han_window(phi,n1,n2)
    else:
        phi = bump_window(phi,n1,n2)

    # calculate frequencies
    k1, k2, kk1, kk2 = calc_freq(L1, L2, n1, n2)

    # calculate power spectrum - 1/2|phi|^2
    spec = calc_spectrum(phi, dk1, dk2, n1, n2)

    # calculate isotropic spectrum
    ki, ispec = calc_ispec(k1,k2,spec)

    d = dict()
    d['k1'] = k1
    d['k2'] = k2
    d['kk1'] = kk1
    d['kk2'] = kk2
    d['spec'] = spec
    d['ki'] = ki
    d['ispec'] = ispec
    return d

def TWODimensional_helmholtz(zeta,div,d1,d2,detrend=True,han_win=True):
    """ Uses Helmholtz decomposition to compute spectra of
    roatatiaonl and divergent components.
    """
    # zeta, div = two dimensional vorticity and divergence fields
    # d1, d2 = grid interval in dimension 0 (along rows) and 1 (along columns)
    n2, n1 = zeta.shape
    L1 = d1*n1
    L2 = d2*n2

    # wavenumber one (equals to dk1 and dk2)
    dk1 = 2.*np.pi/L1
    dk2 = 2.*np.pi/L2

    if(n1%2 == 1 or n2%2 == 1):
        sys.exit("2D field must have even dimentions")

    if detrend:
        zeta = lin_detrend(zeta)
        div = lin_detrend(div)
    else:
        pass

    if han_win:
        zeta = han_window(zeta,n1,n2)
        div = han_window(div,n1,n2)
    else:
        zeta = bump_window(zeta,n1,n2)
        div = bump_window(div,n1,n2)

    # calculate frequencies
    k1, k2, kk1, kk2 = calc_freq(L1, L2, n1, n2)

    # calculate power spectrum - 1/2|phi|^2
    zeta_spec = calc_spectrum(zeta, dk1, dk2, n1, n2)
    div_spec = calc_spectrum(div, dk1, dk2, n1, n2)

    # Rotation and divergent sepctrum
    kappa2 = kk1**2 + kk2**2

    spec_psi = zeta_spec/kappa2
    #spec_psi[np.isnan(spec_psi)] = 0.
    spec_psi[0,0] = 0.
    spec_phi = div_spec/kappa2
    #spec_phi[np.isnan(spec_phi)] = 0.
    spec_phi[0,0] = 0.

    # calculate isotropic spectrum
    ki, ispec_psi = calc_ispec(k1,k2,spec_psi)
    ki, ispec_phi = calc_ispec(k1,k2,spec_phi)

    d = dict()
    d['k1'] = k1
    d['k2'] = k2
    d['kk1'] = kk1
    d['kk2'] = kk2
    d['spec_zeta'] = zeta_spec
    d['spec_div'] = div_spec
    d['ki'] = ki
    d['ispec_psi'] = ispec_psi
    d['ispec_phi'] = ispec_phi

    return d

def lin_detrend(phi):

    phi = signal.detrend(phi,axis=(-1),type='linear')
    phi = signal.detrend(phi,axis=(-2),type='linear')

    return phi

def han_window(phi,n1,n2):

    win1 =  np.hanning(n1)
    win1 =  np.sqrt(n1/(win1**2).sum())*win1
    win2 =  np.hanning(n2)
    win2 =  np.sqrt(n2/(win2**2).sum())*win2
    win = win1[np.newaxis,...]*win2[...,np.newaxis]
    phi = phi*win

    return phi

def bump_window(phi,n1,n2):

    a = 0.01 # controls how quickly function goes to zero
    x = np.linspace(-1,1,n1)
    y = np.linspace(-1,1,n2)
    win1 = np.exp(-a/(1-x**2)+a)
    win2 = np.exp(-a/(1-y**2)+a)
    win = win1[np.newaxis,...]*win2[...,np.newaxis]
    phi = phi*win

    return phi

def calc_freq(L1, L2, n1, n2):
    """
    calculate array of spectral variable (frequency or
    wavenumber) in cycles per unit of L
    """

    # wavenumber one (equals to dk1 and dk2)
    dk1 = 2.*np.pi/L1
    dk2 = 2.*np.pi/L2

    # wavenumber grids
    k2 = dk2*np.append( np.arange(0.,n2/2), np.arange(-n2/2,0.) )
    k1 = dk1*np.arange(0.,n1/2+1)

    kk1, kk2 = np.meshgrid(k1, k2)

    #kk1 = np.fft.fftshift(kk1, axes=0)
    #kk2 = np.fft.fftshift(kk2, axes=0)
    #kappa2 = kk1**2 + kk2**2
    #kappa = np.sqrt(kappa2)

    return k1, k2, kk1, kk2

def calc_spectrum(phi, dk1, dk2, n1, n2):

    phih = np.fft.rfft2(phi)
    spec = (phih*phih.conj()).real/(dk1*dk2*(n1*n2)**2)
    spec[0,:] = 0.5*spec[0,:] # multiply by 1/2 for kk1 == 0

    return spec

def calc_ispec(k,l,E,ndim=2):
    """ Calculates the azimuthally-averaged spectrum
        Parameters
        ===========
        - E is the two-dimensional spectral density
        - k is the wavenumber is the x-direction
        - l is the wavenumber in the y-direction
        Output
        ==========
        - kr: the radial wavenumber
        - Er: the azimuthally-averaged spectrum """

    dk = np.abs(k[2]-k[1])
    dl = np.abs(l[2]-l[1])

    k, l = np.meshgrid(k,l)

    wv = np.sqrt(k**2+l**2)

    if k.max()>l.max():
        kmax = l.max()
    else:
        kmax = k.max()

    #kmax = np.sqrt(l.max()**2 + k.max()**2)

    if ndim==3: # if the third axis is time or level
        nl, nk, nomg = E.shape
    elif ndim==2:
        nomg = 1

    dkr = np.sqrt(dk**2 + dl**2)
    kr =  np.arange(dkr/2.,kmax+dkr/2.,dkr)
    Er = np.zeros((kr.size,nomg))

    for i in range(kr.size):

        fkr = (wv>=kr[i]-dkr/2) & (wv<kr[i]+dkr/2)

        if (len(fkr) > 0):
            dth = np.pi / (fkr.sum())
            if ndim==2:
                Er[i] = (E[fkr]*(wv[fkr]*dth)).sum()
            elif ndim==3:
                Er[i] = (E[fkr]*(wv[fkr]*dth)).sum(axis=(0,1))

        else:
            Er[i] = 0.

    return kr, Er.squeeze()

def avg_per_decade(k,E,nbins = 10):
    """ Averages the spectra with nbins per decade
        Parameters
        ===========
        - E is the spectrum
        - k is the original wavenumber array
        - nbins is the number of bins per decade
        Output
        ==========
        - ki: the wavenumber for the averaged spectrum
        - Ei: the averaged spectrum """

    dk = 1./nbins
    logk = np.log10(k)

    logki = np.arange(np.floor(logk.min()),np.ceil(logk.max())+dk,dk)
    Ei = np.zeros_like(logki)

    for i in range(logki.size):

        f = (logk>logki[i]-dk/2) & (logk<logki[i]+dk/2)

        if f.sum():
            Ei[i] = E[f].mean()
        else:
            Ei[i] = 0.

    ki = 10**logki
    fnnan = np.nonzero(Ei)
    Ei = Ei[fnnan]
    ki = ki[fnnan]

    return ki,Ei

def spectral_slope(k,E,kmin,kmax,stdE):
    ''' compute spectral slope in log space in
        a wavenumber subrange [kmin,kmax],
        m: spectral slope; mm: uncertainty'''

    fr = np.where((k>=kmin)&(k<=kmax))

    ki = np.matrix((np.log10(k[fr]))).T
    Ei = np.matrix(np.log10(np.real(E[fr]))).T
    dd = np.matrix(np.eye(ki.size)*((np.abs(np.log10(stdE)))**2))

    G = np.matrix(np.append(np.ones((ki.size,1)),ki,axis=1))
    Gg = ((G.T*G).I)*G.T
    m = Gg*Ei
    mm = np.sqrt(np.array(Gg*dd*Gg.T)[1,1])
    yfit = np.array(G*m)
    m = np.array(m)[1]

    return m, mm

"""
tmp_u1 = u1
spec_u = sp.TWODimensional_spec(tmp_u1, 10000., 10000., detrend=False,han_win=False)
print('Spatial mean', np.mean(tmp_u1*tmp_u1))
print('Spectral mean', np.sum(spec_u.spec/(1.e8*u1.shape[0]*u1.shape[1])))
print('Iso spectral mean', np.sum(spec_u.ispec)*(spec_u.ki[1]-spec_u.ki[0]))

spec_u = sp.TWODimensional_spec(tmp_u1, 10000., 10000., detrend=False,han_win=False)
ki = spec_u['ki']
print('Spatial mean', np.mean(tmp_u1*tmp_u1*0.5))
print('Spectral mean', np.sum(spec_u['spec']/(1.e8*tmp_u1.shape[0]*tmp_u1.shape[1])))
print('Iso spectral mean', np.sum(spec_u['ispec'])*(ki[1]-ki[0]))
"""
