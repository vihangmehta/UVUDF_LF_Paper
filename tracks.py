import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt

import useful
from conversions import calc_bbmag
from uvudf_utils import pivot_l
from extinction import calzetti

from bc03.extract_bc03 import TemplateSED_BC03
from bc03.igm_attenuation import inoue_tau

def get_lib(fname,age):

    lib  = fitsio.getdata(fname,1)
    lib2 = fitsio.getdata(fname,2)

    idx = np.where(abs(lib2.age-age)==min(abs(lib2.age-age)))[0][0]
    wave,flux = lib['waves'],lib['age_%i'%idx]

    return wave,flux

def get_colcol(drop_filt,wave,flux,EBV,z0,zlim,igm=True):

    if   drop_filt=='f225w':
        filt1,filt2,filt3 = 'f225w','f275w','f336w'
    elif drop_filt=='f275w':
        filt1,filt2,filt3 = 'f275w','f336w','f435w'
    elif drop_filt=='f336w':
        filt1,filt2,filt3 = 'f336w','f435w','f606w'
    else:
        raise Exception("No config defined for drop_filt: %s" % drop_filt)

    flux[wave < 912] = 0.
    flux = flux * np.exp(-calzetti(wave,EBV))

    z_range = np.arange(z0,zlim+1e-6,0.02)
    colx,coly = np.zeros((2,len(z_range)))

    for i,z in enumerate(z_range):

        if igm:
            _flux = flux.copy() * np.exp(-inoue_tau(wave,z))
        else:
            _flux = flux.copy()
        _wave = wave.copy() * (1.+z)

        mag1 = calc_bbmag(_wave,_flux,filt=filt1)
        mag2 = calc_bbmag(_wave,_flux,filt=filt2)
        mag3 = calc_bbmag(_wave,_flux,filt=filt3)

        # mag1 = mag1 - 2.5*np.log10(np.exp(-calzetti(np.array([pivot_l[filt1],]),EBV)))
        # mag2 = mag2 - 2.5*np.log10(np.exp(-calzetti(np.array([pivot_l[filt2],]),EBV)))
        # mag3 = mag3 - 2.5*np.log10(np.exp(-calzetti(np.array([pivot_l[filt3],]),EBV)))

        coly[i] = mag1 - mag2
        colx[i] = mag2 - mag3

    return z_range,colx,coly

def get_tracks(drop_filt,bc03=True):

    if bc03:

        red_lims = {'f225w':3050.,'f275w':3650.,'f336w':4800.}
        zlim = (red_lims[drop_filt]/912. - 1.)

        wave,flux = np.genfromtxt('bc03/tracks/bc2003_hr_m62_chab_cSFR.spec',unpack=True)
        # a = TemplateSED_BC03(age=0.5,sfh='constant',metallicity=0.02,emlines=True,lya_esc=0.,
        #                      workdir='bc03/tracks/',rootdir='/data/highzgal/mehta/galaxev12/',library_version=2012,verbose=False)
        # a.generate_sed()
        # wave,flux = a.sed['waves'],a.sed['spec1']
        z1,colx1,coly1 = get_colcol(drop_filt,wave=wave,flux=flux,EBV=0.0 ,z0=1.,zlim=zlim)
        z2,colx2,coly2 = get_colcol(drop_filt,wave=wave,flux=flux,EBV=0.15,z0=1.,zlim=zlim)
        z3,colx3,coly3 = get_colcol(drop_filt,wave=wave,flux=flux,EBV=0.3 ,z0=1.,zlim=zlim)

        return z1, (colx1,coly1), (colx2,coly2), (colx3,coly3)

    else:

        zlims = {'f225w':1.1,'f275w':1.5,'f336w':2.0}

        wave,flux = np.genfromtxt('bc03/tracks/Geso.spec',unpack=True)
        z,colx,coly = get_colcol(drop_filt,wave=wave,flux=flux,EBV=0.0,z0=0.,zlim=zlims[drop_filt])

        return colx,coly

def plot():

    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,6),dpi=75,tight_layout=True)

    from plot_cuts import plot_colcut_patches

    plot_colcut_patches('f225w', ax1)
    (colx1,coly1), (colx2,coly2), (colx3,coly3) = get_tracks('f225w',bc03=True)
    ax1.plot(colx1,coly1,c='k',lw=1,ls='-')
    ax1.plot(colx2,coly2,c='k',lw=1,ls='--')
    ax1.plot(colx3,coly3,c='k',lw=1,ls=':')

    plot_colcut_patches('f275w', ax2)
    (colx1,coly1), (colx2,coly2), (colx3,coly3) = get_tracks('f275w',bc03=True)
    ax2.plot(colx1,coly1,c='k',lw=1,ls='-')
    ax2.plot(colx2,coly2,c='k',lw=1,ls='--')
    ax2.plot(colx3,coly3,c='k',lw=1,ls=':')

    plot_colcut_patches('f336w', ax3)
    (colx1,coly1), (colx2,coly2), (colx3,coly3) = get_tracks('f336w',bc03=True)
    ax3.plot(colx1,coly1,c='k',lw=1,ls='-')
    ax3.plot(colx2,coly2,c='k',lw=1,ls='--')
    ax3.plot(colx3,coly3,c='k',lw=1,ls=':')

    colx,coly = get_tracks('f225w',bc03=False)
    ax1.plot(colx,coly,c='g',lw=2,ls='-')
    colx,coly = get_tracks('f275w',bc03=False)
    ax2.plot(colx,coly,c='g',lw=2,ls='-')
    colx,coly = get_tracks('f336w',bc03=False)
    ax3.plot(colx,coly,c='g',lw=2,ls='-')

    ax1.set_xlabel('F275W - F336W')
    ax1.set_ylabel('F225W - F275W')
    ax2.set_xlabel('F336W - F435W')
    ax2.set_ylabel('F275W - F336W')
    ax3.set_xlabel('F435W - F606W')
    ax3.set_ylabel('F336W - F435W')

    for ax in [ax1,ax2,ax3]:
        ax.set_xlim(-0.5,2.5)
        ax.set_ylim(-0.5,2.5)

if __name__ == '__main__':

    plot()
    plt.show()