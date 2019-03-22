import numpy as np
import scipy.stats
import cosmolopy.distance as cd
import cosmolopy.constants as cc
import matplotlib.pyplot as plt
from functools32 import lru_cache

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

#From Planck (2015)
cosmo = {'omega_M_0' : 0.315,
         'omega_lambda_0' : 0.685,
         'omega_b_0' : 0.0490,
         'omega_n_0' : 0.0,
         'N_nu' : 0,
         'h' : 0.6731,
         'n' : 0.9655,
         'sigma_8' : 0.829,
         'baryonic_effects': False}
cosmo = cd.set_omega_k_0(cosmo)

class memoized(object):
   """
   Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}

   def __call__(self, *args):
    try:
        if not args in self.cache: self.cache[args] = self.func(*args)
        return self.cache[args]
    except:
        return self.func(*args)

def poisson_interval(k, sigma):
    """
    Returns the poisson error interval for N=k and
    2-sided CL=sigma using chisquared ppf expression
    """
    a = 1.-sigma
    lolim = scipy.stats.chi2.ppf(a/2, 2*k) / 2
    uplim = scipy.stats.chi2.ppf(1-a/2, 2*k + 2) / 2
    lolim[k==0] = 0.
    return np.array([lolim,uplim])

@lru_cache(maxsize=None)
def lum_dist(z):
    """
    Return the luminosity distance as redshift z
    in units of 'cm'
    """
    return cd.luminosity_distance(z,**cosmo) * cc.Mpc_cm

#@memoized
@lru_cache(maxsize=None)
def co_vol(z):
    """
    Returns the differential comoving volume
    at redshift z in units of 'Mpc^3'
    """
    return cd.diff_comoving_volume(z,**cosmo)

def age(z):
    """
    Returns the age of the Universe at redshift z
    in units of 'Gyr'.
    """
    return cd.age(z,**cosmo)/cc.Gyr_s

def lookback(z,z0=0.0):
    """
    Returns the lookback time for redshift z
    in units of 'Gyr'.
    """
    return cd.lookback_time(z,z0,**cosmo)/cc.Gyr_s

def gauss(x,x0,sig):
    """
    Returns the evaluation at x of a Gaussian centered
    at x0 with stdev sig
    """
    return np.exp( -0.5 * (x-x0) * (x-x0) / sig / sig )

def sch_shape(M,alpha,Mst):
    """
    Returns the UNnormalized Schechter Function shape
    given M, M*, and alpha (faint end slope)
    """
    x = 10**(0.4*(Mst - M))
    return 0.4 * np.log(10) * x**(alpha+1) * np.exp(-x)

def sch(M, P):
    """
    Returns the UNnormalized Schechter Function value
    given a magnitude and a set of parameters
    """
    (alpha, Mst) = P
    return sch_shape(M, alpha, Mst)

def norm_sch(M,alpha,Mst,phi):
    """
    Returns the Normalized Schechter Function value
    given a magnitude and a set of parameters
    """
    return 10**phi * sch_shape(M,alpha,Mst)

if __name__ == '__main__':
    print "No main() defined."