import numpy as np
import scipy.integrate
import scipy.interpolate

from useful import *
from uvudf_utils import filter_response, pivot_l

light = 2.998e18 # angs/s

def get_abs_from_app(app_mag,z):
    """
    Returns the absolute magnitude for a given apparent magnitude at redshift z.
    """
    dist = lum_dist(z) / cc.pc_cm
    if isinstance(app_mag,np.ndarray):
        cond = (np.abs(app_mag)!=99.)
        abs_mag = np.zeros(len(app_mag)) + 99.
        abs_mag[cond] = app_mag[cond] - 5*(np.log10(dist[cond]) - 1) + 2.5*np.log10(1+z[cond])
    else:
        abs_mag = app_mag - 5*(np.log10(dist) - 1) + 2.5*np.log10(1+z) if np.abs(app_mag)!=99. else app_mag
    return abs_mag

def get_app_from_abs(abs_mag,z):
    """
    Returns the apparent magnitude for a given absolute magnitude at redshift z.
    """
    dist = lum_dist(z) / cc.pc_cm
    app_mag = abs_mag + 5*(np.log10(dist) - 1) - 2.5*np.log10(1+z)
    return app_mag

def get_absM_from_Lnu(x,inv=False):
    """
    Arguments:
        x - Log UV Luminosity [log(ergs/s/Hz)]
        (if inv) UV absolute magnitude [AB]
    Returns:
        UV absolute magnitude [AB]
        (if inv) Log UV Luminosity [log(ergs/s/Hz)]
    Ref:
        CosmoloPy Magnitudes
    """
    MAB0 = -2.5 * np.log10(3631.e-23)
    const = 4. * np.pi * (10. * cc.pc_cm)**2.
    if not inv:
        return -2.5*(x - np.log10(const)) - MAB0
    else:
        return np.log10(const) + ((x+MAB0)/(-2.5))

def calc_bbmag(wave,flux,filt):
    """
    Calculate the broadband magnitude.
    """
    sel_filter = filter_response[filter_response['FILTER_NAME'] == filt]
    [filt_wave], [filt_sens], pivot = sel_filter['LAMBDA'], sel_filter['THROUGHPUT'], pivot_l[filt]
    filt_interp = scipy.interpolate.interp1d(filt_wave, filt_sens, bounds_error=False, fill_value=0, kind='linear')

    filt_sens = filt_interp(wave)
    flux = scipy.integrate.simps(flux*filt_sens*wave, wave) / scipy.integrate.simps(filt_sens*wave, wave)
    flux = (pivot**2/light) * flux
    mag  = -2.5*np.log10(flux) - 48.6
    return mag

def calc_sn(mag,dmag):
    """
    Mag  = -2.5*log(S)
    dMag = -2.5*log(S) + 2.5*log(S+N)
         =  2.5*log((S+N)/S)
         =  2.5*log(1+N/S)
    """
    if isinstance(dmag,np.ndarray):
        sn = dmag * 0
        sn[dmag==0] = 9999.
        sn[dmag!=0] = 1./(10**(dmag[dmag!=0]/2.5) - 1)
        sn[np.abs(mag)==99.] = 0
    else:
        sn = 1./(10**(dmag/2.5) - 1) if dmag!=0 else 9999
        sn = 0 if np.abs(mag)==99. else sn
    return sn

if __name__ == '__main__':
    print "No main() defined."