import os
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio

import useful
import uvudf_utils as utils
import plot_colcol
from conversions import calc_bbmag
from sample_selection import mk_dropout_cuts, mk_photoz_cuts

libdir = '/data/highzgal/mehta/pickles/'
liblist = [x for x in os.listdir(libdir) if 'uk' in x and '.dat' in x]

def read_lib(fname):

    lib = np.genfromtxt(libdir+fname,dtype=[('wave',float),('flux',float)],usecols=(0,1))
    return lib

def get_pickles_colcol(drop_filt):

    if drop_filt=='seq':
        filt1,filt2,filt3 = 'f606w','f775w','f850lp'
    else:
        filt1,filt2,filt3 = utils.filt_colcol(drop_filt)

    colx, coly = np.zeros((2,len(liblist)))

    for i,fname in enumerate(liblist):

        lib = read_lib(fname)

        mag1 = calc_bbmag(lib['wave'],lib['flux'],filt=filt1)
        mag2 = calc_bbmag(lib['wave'],lib['flux'],filt=filt2)
        mag3 = calc_bbmag(lib['wave'],lib['flux'],filt=filt3)

        coly[i] = mag1-mag2
        colx[i] = mag2-mag3

    return colx,coly

def get_pickles_sequence():

    colx,coly = get_pickles_colcol(drop_filt='seq')
    return scipy.interpolate.UnivariateSpline(colx,coly,k=3)

def plot_pickles_spectra():

    fig, ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    for fname in liblist:
        lib = read_lib(fname)
        ax.plot(lib['wave'],lib['flux'],lw=0.5)

    ax.set_xlim(1e3,2.6e4)
    ax.set_xscale('log')
    ax.set_xlabel('Wavelength [$\\AA$]')
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Flux')

def plot_pickles_sequence():

    fig, ax = plt.subplots(1,1,figsize=(8,6),dpi=75,tight_layout=True)

    colx,coly = get_pickles_colcol(drop_filt='seq')
    ax.scatter(colx,coly,facecolor='k',edgecolor='k',lw=0,s=10,alpha=1)

    func = get_pickles_sequence()
    xx = np.arange(-0.5,2,0.05)
    yy = func(xx)
    ax.fill_between(xx,yy-0.15,yy+0.15,color='k',alpha=0.1)

    catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
    catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]
    #plot_colcol.plot_colcol(catalog,'f606w','f775w','f850lp',ax,c='k',s=20,alpha=0.2,lw=0,use_MR_lims=True)

    for drop_filt,c in zip(['f225w','f275w','f336w'],['b','g','r']):

        drop = mk_dropout_cuts(catalog,drop_filt,calc_sn=True)
        phot = mk_photoz_cuts( catalog,drop_filt,calc_sn=True)

        cond0 = (drop['HLR_F435W'] <= 0.1/utils.pixscale) & (drop['MAG_F850LP'] >= 25.5)
        cond1 = (np.abs(drop['MAG_F606W']) != 99.) & (np.abs(drop['MAG_F775W']) != 99.) & (np.abs(drop['MAG_F850LP']) != 99.)
        plot_colcol.plot_colcol(drop[ cond0 & cond1],'f606w','f775w','f850lp',ax,marker='o',ec=c,fc=c,s=50,alpha=0.5,lw=0)
        plot_colcol.plot_colcol(drop[~cond0 & cond1],'f606w','f775w','f850lp',ax,marker='x',ec='none',fc=c,s=50,alpha=0.5,lw=1)

        cond3 = (np.abs((drop['MAG_F606W']-drop['MAG_F775W']) - func(drop['MAG_F775W']-drop['MAG_F850LP'])) <= 0.15)
        print drop_filt, len(drop[cond0&cond1&cond3]), len(drop)

    ax.set_xlabel('F775W - F850LP')
    ax.set_ylabel('F606W - F775W')

def plot_pickles_colcol():

    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),dpi=75,tight_layout=True)

    colx,coly = get_pickles_colcol(drop_filt='f225w')
    ax1.scatter(colx,coly,c='k',lw=0,s=25,alpha=0.5)
    colx,coly = get_pickles_colcol(drop_filt='f275w')
    ax2.scatter(colx,coly,c='k',lw=0,s=25,alpha=0.5)
    colx,coly = get_pickles_colcol(drop_filt='f336w')
    ax3.scatter(colx,coly,c='k',lw=0,s=25,alpha=0.5)

    ax1.set_xlabel('F275W - F336W')
    ax1.set_ylabel('F225W - F275W')
    ax2.set_xlabel('F336W - F435W')
    ax2.set_ylabel('F275W - F336W')
    ax3.set_xlabel('F435W - F606W')
    ax3.set_ylabel('F336W - F435W')

    for ax in [ax1,ax2,ax3]:
        ax.set_xlim(-1,5)
        ax.set_ylim(-1,6)

if __name__ == '__main__':

    #plot_pickles_spectra()
    #plot_pickles_colcol()
    plot_pickles_sequence()
    plt.show()
