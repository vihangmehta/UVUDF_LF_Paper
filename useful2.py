import mpmath as mp
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from hmf import MassFunction

import conversions as conv

try:
    import mkl
    mkl.set_num_threads(15)
except ImportError:
    pass

plt.rcParams.update({'font.size':20,'font.weight':400,
                    #'mathtext.default':'regular',
                     'axes.linewidth': 2.0,
                     'xtick.major.width': 1.5,
                     'ytick.major.width': 1.5,
                     'xtick.minor.width': 1.2,
                     'ytick.minor.width': 1.2,
                     'xtick.major.size': 8,
                     'ytick.major.size': 8,
                     'xtick.minor.size': 5,
                     'ytick.minor.size': 5})
mp.mp.dps = 25

quad_args      = {'limit':250,'epsrel':1e-4,'epsabs':1e-10}
gammainc_vec   = np.vectorize(mp.gammainc)
derivative_vec = np.vectorize(scipy.misc.derivative,excluded=['func','dx'])
quad_vec       = np.vectorize(scipy.integrate.quad,excluded=['func','args','epsabs','epsrel','limit'])

def HMF(z=2.2,Mmin=9.,Mmax=15.,dlog10m=0.05,model='SMT'):
    """
    Returns the differential and cumulative
    Halo Mass Function for given redshift
    """
    hmf = MassFunction(Mmin=Mmin,Mmax=Mmax,dlog10m=dlog10m,z=z,hmf_model=model)
    return hmf.M, hmf.dndm, hmf.ngtm

def UV_LF(M,alpha,Mst,phi):
    """
    Returns the UV luminosity function for a given
    absolute UV magnitude
    """
    x  = 10**(0.4*(Mst - M))
    LF = 10**phi * 0.4 * np.log(10) * np.power(x,alpha+1) * np.exp(-x)
    return LF

def UV_LF_dustcor(M,alpha,Mst,phi,dust_model):
    """
    Returns the dust corrected UV luminosity function
    for a given absolute UV magnitude
    """
    frac = derivative_vec(dust_model.apply_dust,x0=M,dx=1e0)
    M    = dust_model.apply_dust(M)
    x    = 10**(0.4*(Mst - M))
    LF   = 10**phi * 0.4 * np.log(10) * np.power(x,alpha+1) * np.exp(-x)
    LF_  = LF * frac
    return LF_

def UV_CLF(M,alpha,Mst,phi):
    """
    Returns the cumulative UV luminosity function
    for a given absolute UV magnitude
    """
    x = 10**(0.4*(Mst - M))
    return 10**phi * gammainc_vec(alpha+1,x).astype(float)

@np.vectorize
def AGN_frac_S13(x):
    """
    Correct for the AGN fraction for Sobral13 Ha LF
    """
    res = 0.75*np.log10(x) + 0.06
    res = 0. if res<0. else res
    res = 1. if res>1. else res
    return res

def Ha_LF(L,alpha,Lst,phi,agn=False):
    """
    Returns the Ha luminosity function for a given
    Ha line luminosity
    """
    x  = 10**(L - Lst)
    LF = 10**phi * np.log(10) * np.power(x,alpha+1) * np.exp(-x)
    if agn: return LF * (1. - AGN_frac_S13(x))
    return LF

def Ha_LF_dustcor(L,alpha,Lst,phi,dust_model,agn=False):
    """
    Returns the dust corrected Ha luminosity function
    for a given Ha line luminosity
    """
    frac = derivative_vec(dust_model.apply_dust,x0=L,dx=1e0)
    L    = dust_model.apply_dust(L)
    x    = 10**(L - Lst)
    LF   = 10**phi * np.log(10) * np.power(x,alpha+1) * np.exp(-x)
    LF_  = LF * frac
    if agn: return LF_ * (1. - AGN_frac_S13(x))
    return LF_

def Ha_CLF(L,alpha,Lst,phi,agn=False):
    """
    Returns the cumulative Ha luminosity function
    for a given Ha line luminosity
    """
    if agn:
        return quad_vec(Ha_LF,L,99.9,args=(alpha,Lst,phi,agn),**quad_args)[0]
    else:
        x = 10**(L - Lst)
        return 10**phi * gammainc_vec(alpha+1,x).astype(float)

def SFR_K98(wave,x,imf='salp',inv=False):
    """
    Returns the SFR for given luminosity in
    UV (in units of Lnu) and Ha (total luminosity)
    using the Kennicutt 1998 relations.
    """
    ha_sfrd_factor = {'salp':1.}
    uv_sfrd_factor = {'salp':1., 'kroupa':1.7, 'chab':1.8}

    if (wave is 'ha' and imf not in ha_sfrd_factor.keys()) or \
       (wave is 'uv' and imf not in uv_sfrd_factor.keys()):
        raise Exception("Invalid IMF defined.")

    if   wave is 'ha':
        if inv:
            x   = x * ha_sfrd_factor[imf]
            res = x / 7.9e-42
        else:
            res = 7.9e-42 * x
            res = res / ha_sfrd_factor[imf]
    elif wave is 'uv':
        if inv:
            x   = x * uv_sfrd_factor[imf]
            res = x / 1.4e-28
        else:
            res = 1.4e-28 * x
            res = res / uv_sfrd_factor[imf]
    else:
        raise Exception('Invalid wavelength in SFR_K98(). Choose from: ha,uv')

    return res

def SFR_K12(wave,x,imf='salp',inv=False):
    """
    Returns the SFR for given luminosity in
    UV (in units of Lnu) and Ha (total luminosity)
    using the Kennicutt 2012 relations.
    """
    ha_sfrd_factor = {'salp':1.}
    uv_sfrd_factor = {'salp':1., 'kroupa':1.7, 'chab':1.8}

    if (wave is 'ha' and imf not in ha_sfrd_factor.keys()) or \
       (wave is 'uv' and imf not in uv_sfrd_factor.keys()):
        raise Exception("Invalid IMF defined.")

    if   wave is 'ha':
        if inv:
            x   = x * ha_sfrd_factor[imf]
            res = x / 10**-41.27
        else:
            res = x * 10**-41.27
            res = res / ha_sfrd_factor[imf]
    elif wave is 'uv':
        if inv:
            x   = x * uv_sfrd_factor[imf]
            res = x / 10**-43.35 / (conv.light/1500.)
        else:
            res = x * 10**-43.35 * (conv.light/1500.)
            res = res / uv_sfrd_factor[imf]
    else:
        raise Exception('Invalid wavelength in SFR_K12(). Choose from: ha,uv')

    return res

if __name__ == '__main__':
    print "No main() defined."