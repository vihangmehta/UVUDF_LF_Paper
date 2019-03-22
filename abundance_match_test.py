import sys
import numpy as np
import scipy.interpolate
import scipy.optimize as so
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch, Rectangle
from matplotlib.font_manager import FontProperties

import useful2
import conversions as conv
import uvudf_utils as utils
import shmr
from dust import dust_uv, dust_ha, _Halpha_Dust
from abundance_match import abundance_matching

def fixed_factor(luv):

    luv = conv.get_absM_from_Lnu(luv,inv=True) + np.log10(conv.light / 1500.)
    
    ratios = np.zeros(0)
    for imf in ['_salp','_chab','_kroupa']:
        for metal in ['_m22','_m32','_m42','_m52','_m62']:
            x = fitsio.getdata('bc03/bc2003'+imf+metal+'_cSFR_luv.fits')
            ratio  = np.log10(x.luv_lambda*1500./x.lha_lambda)
            crit   = np.abs(x.age*1e3 - 100)
            idx    = np.where(crit == min(crit))[0][0]
            ratios = np.append(ratios,ratio[idx])

    range_ratios = np.arange(min(ratios),max(ratios),0.1)
    lha_nodust = np.hstack([luv - iratio for iratio in range_ratios])
    lha_nodust = np.percentile(lha_nodust,[50,0,100],axis=-1).T
    return lha_nodust

def test_Aha():

    M16 = fitsio.getdata('output/z2_LF_mehta16.fits')
    S13 = fitsio.getdata('output/z2_LF_sobral13.fits')

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6),dpi=75,sharey=True)
    fig.subplots_adjust(left=0.07,right=0.98,bottom=0.14,top=0.97,wspace=0)

    _massha, _lha = abundance_matching(S13,ref='S13',label='xlum')
    condha = (_lha[:,0] >= 42.-0.4)

    for dust,c in zip(['M99','R15'],['r','g']):
        
        _massuv, _luv = abundance_matching(M16,ref='M16_%s'%dust,label='xlum_%s'%dust)
        conduv = (_luv[:,0] <= dust_uv[dust].remove_dust(utils.lim_M['f275w_phot']))
        
        mass = _massuv[conduv]
        luv  = _luv[conduv]
        lha  = _lha[conduv]
    
        lha_nodust = fixed_factor(luv)
        x,y = np.meshgrid([0,1,2],[0,1,2])
        Aha = np.vstack([lha_nodust[:,_x]-lha[:,_y] for _x,_y in zip(x.ravel(),y.ravel())]).T
        # Aha = np.vstack([lha_nodust[:,i]-lha[:,0] for i in range(3)]).T
        Aha = np.percentile(Aha,[50,0,100],axis=-1).T

        label="using %s dust cor." % dust
        ax1.plot(lha[:,0],Aha[:,0],c=c,lw=2,alpha=0.8,label=label)
        ax1.fill_between(lha[:,0],Aha[:,1],Aha[:,2],color=c,lw=0,alpha=0.15)

        Mst = 10**shmr.shmr(np.log10(mass),z=2.2)
        ax2.plot(Mst,Aha[:,0],c=c,lw=2,alpha=0.8,label=label)
        ax2.fill_between(Mst,Aha[:,1],Aha[:,2],color=c,lw=0,alpha=0.15,zorder=1)

    dust_ha2 = _Halpha_Dust()
    tmp_lha  = np.arange(30,50,0.01)
    ax1.errorbar(dust_ha2.dom_Lha,dust_ha2.dom_Aha,xerr=[dust_ha2.dom_dLha1,dust_ha2.dom_dLha2],yerr=dust_ha2.dom_dAha,fmt='ko',markersize=0,c='k')
    ax1.scatter(dust_ha2.dom_Lha,dust_ha2.dom_Aha,c='k',s=50,lw=0, label='Dominguez+13')
    ax1.plot(tmp_lha, dust_ha['H01'].extinction(tmp_lha), c='k', lw=2.5, label='Hopkins01 (w/ offset, \n smoothed)')
    
    ax1.set_xlim(39.5,44.5)
    ax1.set_xlabel('log H$\\alpha$ Luminosity, L$_\mathrm{H\\alpha}$ [ergs/s]',fontsize=20)
    ax1.set_ylim(0,3.1)
    ax1.set_ylabel('H$\\alpha$ extinction, A$_\mathrm{H\\alpha}$',fontsize=20)
    ax1.legend(loc=2,fontsize=16)

    garn_best = lambda x: 0.91 + 0.77*(x-10) + 0.11*(x-10)**2 - 0.09*(x-10)**3
    Mst = 10**np.arange(8.5,11.5,0.1)
    ax2.plot(Mst,garn_best(np.log10(Mst)),c='k',lw=2,alpha=0.8,label='Garn & Best (2010), $z\sim0$',zorder=10)
    ax2.fill_between(Mst,garn_best(np.log10(Mst))-0.719**2/2,garn_best(np.log10(Mst))+0.719**2/2,color='k',lw=0,alpha=0.1,zorder=10)
    # ax2.errorbar(10**np.array([8.56,9.51,10.46]),np.array([0.57,0.57,1.26]),
    #                 xerr=[10**np.array([8.56,9.51,10.46])-10**np.array([8.56-1.38,9.51-0.35,10.46-0.61]),
    #                       10**np.array([8.56+0.58,9.51+0.34,10.46+1.40])-10**np.array([8.56,9.51,10.46])],
    #                 yerr=[0.71,0.65,1.22],color='k',lw=0,elinewidth=1,marker='o',markersize=8,mew=0,capthick=1,label='Dominguez+13, $z\sim1.5$')
    # ax2.errorbar(10**np.array([10.281,10.611,11.025]),np.array([0.93,1.11,1.66]),
    #                 xerr=[10**np.array([10.281,10.611,11.025])-10**np.array([10.039,10.521,10.761]),
    #                       10**np.array([10.477,10.739,11.635])-10**np.array([10.281,10.611,11.025])],
    #                 yerr=[0.71,0.65,1.22],color='b',lw=0,elinewidth=1,marker='o',markersize=8,mew=0,capthick=1,label='Kashino+13, $1.4<z<1.7$')

    ax2.set_xscale('log')
    ax2.set_xlim(5e7,3e11)
    ax2.set_xlabel('Stellar Mass, M$_\star$ [M$_\odot$]',fontsize=20)
    ax2.legend(loc=2,fontsize=16)

    fig.savefig('plots/abundance_match_test.png')
    fig.savefig('plots/abundance_match_test.pdf')

if __name__ == '__main__':
    
    test_Aha()
    plt.show()
