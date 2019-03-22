import sys
import numpy as np
import scipy.interpolate
import scipy.optimize as so
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Patch, Rectangle

import useful2
import conversions as conv
import uvudf_utils as utils
import shmr
import igimf
from dust import dust_uv, dust_ha

def to_solve(x,crit,func):
    #print x, func(x), crit
    return func(x) - crit

def abundance_matching(dataset,label,ref):

    mass,pdf,cdf = useful2.HMF(z=2.2)

    cond1 = (cdf <= 1e0) & (cdf >= 1e-9)
    mass,pdf,cdf = mass[cond1],pdf[cond1],cdf[cond1]

    cond2 = (dataset[ 'CLF']      >= 1e-50) & \
            (dataset['eCLF'][:,0] >= 1e-50) & \
            (dataset['eCLF'][:,1] >= 1e-50)
    dataset   = dataset[cond2]

    interpfn  = scipy.interpolate.interp1d(dataset[label],np.log10(dataset['CLF']))
    interpfn1 = scipy.interpolate.interp1d(dataset[label],np.log10(dataset['eCLF'][:,0]))
    interpfn2 = scipy.interpolate.interp1d(dataset[label],np.log10(dataset['eCLF'][:,1]))

    lims      = [min(dataset[label]),max(dataset[label])]
    xlum      = np.zeros((len(mass),3))
    xlum[:,0] = np.array([so.brentq(to_solve,lims[0],lims[1],args=(x,interpfn )) for x in np.log10(cdf)])
    xlum[:,1] = np.array([so.brentq(to_solve,lims[0],lims[1],args=(x,interpfn1)) for x in np.log10(cdf)])
    xlum[:,2] = np.array([so.brentq(to_solve,lims[0],lims[1],args=(x,interpfn2)) for x in np.log10(cdf)])

    return mass, xlum

def plot_uvtoha_ratio_vs_Muv():

    M16 = fitsio.getdata('output/z2_LF_mehta16.fits')
    H10 = fitsio.getdata('output/z2_LF_hayes10.fits')
    S13 = fitsio.getdata('output/z2_LF_sobral13.fits')

    fig, ax = plt.subplots(1,1,figsize=(10,9),dpi=75,tight_layout=False)
    fig.subplots_adjust(left=0.12,right=0.95,bottom=0.1,top=0.8)
    dax = ax.twiny()
    dax2= ax.twiny()

    _massha, _lha = abundance_matching(S13,ref='S13',label='xlum_H01')
    condha = (_lha[:,0] >= dust_ha['H01'].remove_dust(42.-0.4))

    trans = {}
    for dust,c in zip(['M99','R15'],['r','g']):
        
        _massuv, _luv = abundance_matching(M16,ref='M16_%s'%dust,label='xlum_%s'%dust)
        conduv = (_luv[:,0] <= dust_uv[dust].remove_dust(utils.lim_M['f275w_phot']))
        
        trans[dust] = scipy.interpolate.interp1d(dust_uv[dust].apply_dust(_luv[:,0]),np.log10(_massuv),bounds_error=False,fill_value='extrapolate')
        mass1, luv1 = _massuv[condha], _luv[condha]
        mass2, luv2 = _massuv[conduv], _luv[conduv]
        lha1 = _lha[_massha >= mass1[0]]
        lha2 = _lha[_massha >= mass2[0]]

        for mass,luv,lha,alpha in zip([mass1,mass2],[luv1,luv2],[lha1,lha2],[1.0,0.5]):

            xluv = dust_uv[dust].apply_dust(luv[:,0].copy())
            luv = conv.get_absM_from_Lnu(luv,inv=True) + np.log10(conv.light / 1500.)
            ratio = luv[:,0] - lha[:,0]

            x,y = np.meshgrid([0,1,2],[0,1,2])
            ratios = np.vstack([luv[:,_x]-lha[:,_y] for _x,_y in zip(x.ravel(),y.ravel())]).T
            ratio_err = np.percentile(ratios,[0,100],axis=-1).T

            label="using %s dust cor." % dust if alpha==1.0 else None
            ax.plot(xluv,ratio,c=c,lw=2,alpha=alpha,label=label,zorder=10)
            ax.fill_between(xluv,ratio_err[:,0],ratio_err[:,1],color=c,alpha=0.15*alpha,zorder=10)

    # for model,ls in zip(['standard','minimal1'],['-.','--']):

    #     muv, lha, sfr = igimf.get_igimf_uv_ha(model)
    #     luv = conv.get_absM_from_Lnu(muv,inv=True) + np.log10(conv.light / 1500.)
    #     ratio = luv - lha

    #     muv_ = dust_uv['M99'].apply_dust(muv)
    #     ax.plot(muv_,ratio,c='r',ls=ls,lw=3,zorder=5)
    #     muv_ = dust_uv['C14'].apply_dust(muv)
    #     ax.plot(muv_,ratio,c='g',ls=ls,lw=3,zorder=5)
    #     ax.plot(-99,-99,c='k',ls=ls,lw=3,label='%s IGIMF'%model.capitalize())

    ratios = np.zeros(0)
    for imf in ['_salp','_chab','_kroupa']:
        for metal in ['_m22','_m32','_m42','_m52','_m62']:
            x = fitsio.getdata('bc03/bc2003'+imf+metal+'_cSFR_luv.fits')
            ratio  = np.log10(x.luv_lambda*1500./x.lha_lambda)
            crit   = np.abs(x.age*1e3 - 100)
            idx    = np.where(crit == min(crit))[0][0]
            ratios = np.append(ratios,ratio[idx])

    ax.axhspan(min(ratios),max(ratios),edgecolor='k',facecolor='lightgray',lw=0,alpha=0.5,hatch='//',zorder=1)
    ax.add_patch(Rectangle((-99,-99),0,0,edgecolor='k',facecolor='lightgray',lw=0,alpha=0.5,hatch='//',label='constant SFR expectation \n from BC03 models'))
    ax.legend(fontsize=18,loc=3)

    ax.set_xlabel('Absolute UV Magnitude [AB]',fontsize=24)
    ax.set_xlim(-16,-23)
    ax.set_ylabel('log $\\nu$L$_{\\nu,1500}$/L$_{H\\alpha}$ (dust corrected)',fontsize=24)
    ax.set_ylim(1.45,2.55)
    ml = MultipleLocator(0.02)
    ax.yaxis.set_minor_locator(ml)

    dax.set_xlabel('Halo Mass, M$_h$ [M$_\odot$]',fontsize=24)
    xx = np.array([1e11,1e12,1e13,1e14])
    _xx = np.array([so.brentq(lambda x: trans['M99'](x) - i,*ax.get_xlim()) for i in np.log10(xx)])
    dax.set_xlim(ax.get_xlim())
    dax.set_xticks(_xx)
    dax.set_xticklabels(["10$^\mathrm{%i}$"%i for i in np.log10(xx)])
    
    #Minor ticks
    xx = np.zeros(0)
    for i in [10,11,12,13,14,15]: xx = np.append(xx,np.arange(1,10,1)*10**i)
    _xx = np.array([so.brentq(lambda x: trans['M99'](x) - i,-30,-8) for i in np.log10(xx)])
    _xx = _xx[(dax.get_xlim()[0]>=_xx) & (_xx>=dax.get_xlim()[1])]
    _xx = np.clip(_xx,dax.get_xlim()[1],dax.get_xlim()[0])
    dax.set_xticks(_xx,minor=True)

    dax2.spines['top'].set_position(('axes',1.15))
    dax2.set_xlabel('Stellar Mass, M$_\star$ [M$_\odot$]',fontsize=24)
    dax2.set_xlim(ax.get_xlim())

    xx = np.array([1e9,1e10,1e11,1.5e11])
    _xx = np.array([so.brentq(lambda x: shmr.shmr(trans['M99'](x),z=2.2) - i,*ax.get_xlim()) for i in np.log10(xx)])
    dax2.set_xticks(_xx)
    dax2.set_xticklabels(["10$^\mathrm{9}$","10$^\mathrm{10}$","10$^\mathrm{11}$","1.5$\\times$10$^\mathrm{11}$"])

    #Minor ticks
    xx = np.zeros(0)
    for i in [8,9,10]: xx = np.append(xx,np.arange(1,10,1)*10**i)
    xx = np.append(xx,[1e11,1.5e11])
    _xx = np.array([so.brentq(lambda x: shmr.shmr(trans['M99'](x),z=2.2) - i,-30,-8) for i in np.log10(xx)])
    _xx = _xx[(dax2.get_xlim()[0]>=_xx) & (_xx>=dax2.get_xlim()[1])]
    _xx = np.clip(_xx,dax2.get_xlim()[1],dax2.get_xlim()[0])
    dax2.set_xticks(_xx,minor=True)

    for axis in [ax,dax,dax2]:
        axis.tick_params(which='major', length=8, width=1.5)
        axis.tick_params(which='minor', length=4, width=1.2)

    _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()+dax.get_xticklabels()+dax2.get_xticklabels()]
    fig.savefig('plots/am_uv_vs_ha.png')
    fig.savefig('plots/am_uv_vs_ha.pdf')

def plot_uvtoha_ratio_vs_everything():

    M16 = fitsio.getdata('output/z2_LF_mehta16.fits')
    S13 = fitsio.getdata('output/z2_LF_sobral13.fits')

    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,6),dpi=75,tight_layout=False,sharey=True)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.15,top=0.98,wspace=0,hspace=0)

    _massha, _lha = abundance_matching(S13,ref='S13',label='xlum_H01')
    condha = (_lha[:,0] >= dust_ha['H01'].remove_dust(42.-0.4))

    trans = {}
    for dust,c in zip(['M99','R15'],['r','g']):
        
        _massuv, _luv = abundance_matching(M16,ref='M16_%s'%dust,label='xlum_%s'%dust)
        conduv = (_luv[:,0] <= dust_uv[dust].remove_dust(utils.lim_M['f275w_phot']))
        
        trans[dust] = scipy.interpolate.interp1d(dust_uv[dust].apply_dust(_luv[:,0]),np.log10(_massuv),bounds_error=False,fill_value='extrapolate')
        mass1, luv1 = _massuv[condha], _luv[condha]
        mass2, luv2 = _massuv[conduv], _luv[conduv]
        lha1 = _lha[_massha >= mass1[0]]
        lha2 = _lha[_massha >= mass2[0]]

        for mass,luv,lha,alpha in zip([mass1,mass2],[luv1,luv2],[lha1,lha2],[1.0,0.5]):

            xluv = dust_uv[dust].apply_dust(luv[:,0].copy())
            luv = conv.get_absM_from_Lnu(luv,inv=True) + np.log10(conv.light / 1500.)
            ratio = luv[:,0] - lha[:,0]

            x,y = np.meshgrid([0,1,2],[0,1,2])
            ratios = np.vstack([luv[:,_x]-lha[:,_y] for _x,_y in zip(x.ravel(),y.ravel())]).T
            ratio_err = np.percentile(ratios,[0,100],axis=-1).T

            label="using %s dust cor." % dust if alpha==1.0 else None
            ax1.plot(xluv,ratio,c=c,lw=2,alpha=alpha,label=label,zorder=10)
            ax1.fill_between(xluv,ratio_err[:,0],ratio_err[:,1],color=c,alpha=0.15*alpha,zorder=10)
            
            ax2.plot(mass,ratio,c=c,lw=2,alpha=alpha,label=label,zorder=10)
            ax2.fill_between(mass,ratio_err[:,0],ratio_err[:,1],color=c,alpha=0.15*alpha,zorder=10)
            
            mass_star = 10**shmr.shmr(np.log10(mass),z=2.2)
            ax3.plot(mass_star,ratio,c=c,lw=2,alpha=alpha,label=label,zorder=10)
            ax3.fill_between(mass_star,ratio_err[:,0],ratio_err[:,1],color=c,alpha=0.15*alpha,zorder=10)

            if alpha==0.5:

                xluv = xluv + 20
                fit1 = np.polyfit(xluv,ratio,deg=6)
                fn1  = np.poly1d(fit1)
                ax1.plot(xluv-20,fn1(xluv),c='k',lw=2,zorder=10)
                avg_err = np.mean((ratio_err[:,1]-ratio_err[:,0])/2.)
                print dust, 'UV Mag', fit1, avg_err, [min(xluv),max(xluv)]

                mass = np.log10(mass) - 12
                fit2 = np.polyfit(mass,ratio,deg=6)
                fn2  = np.poly1d(fit2)
                ax2.plot(10**(mass+12),fn2(mass),c='k',lw=2,zorder=10)
                avg_err = np.mean((ratio_err[:,1]-ratio_err[:,0])/2.)
                print dust, 'Halo Mass', fit2, avg_err, [min(mass),max(mass)]

                mass_star = np.log10(mass_star) - 10
                fit2 = np.polyfit(mass_star,ratio,deg=6)
                fn2  = np.poly1d(fit2)
                ax3.plot(10**(mass_star+10),fn2(mass_star),c='k',lw=2,zorder=10)
                avg_err = np.mean((ratio_err[:,1]-ratio_err[:,0])/2.)
                print dust, 'Stellar Mass', fit2, avg_err, [min(mass_star),max(mass_star)]

    ratios = np.zeros(0)
    for imf in ['_salp','_chab','_kroupa']:
        for metal in ['_m22','_m32','_m42','_m52','_m62']:
            x = fitsio.getdata('bc03/bc2003'+imf+metal+'_cSFR_luv.fits')
            ratio  = np.log10(x.luv_lambda*1500./x.lha_lambda)
            crit   = np.abs(x.age*1e3 - 100)
            idx    = np.where(crit == min(crit))[0][0]
            ratios = np.append(ratios,ratio[idx])

    for ax in [ax1,ax2,ax3]:
        ax.axhspan(min(ratios),max(ratios),edgecolor='k',facecolor='lightgray',lw=0,alpha=0.5,hatch='//',zorder=1)
        ax.add_patch(Rectangle((-99,-99),0,0,edgecolor='k',facecolor='lightgray',lw=0,alpha=0.5,hatch='//',label='constant SFR expectation \n from BC03 models'))
        ax.legend(fontsize=14,loc=3)
        ml = MultipleLocator(0.02)
        ax.yaxis.set_minor_locator(ml)

    ax1.set_ylabel('log $\\nu$L$_{\\nu,1500}$/L$_{H\\alpha}$ (dust corrected)',fontsize=20)
    ax1.set_ylim(1.45,2.55)
    ax1.set_xlabel('Absolute UV Magnitude [AB]',fontsize=20)
    ax1.set_xlim(-16,-23)

    ax2.set_xlabel('Halo Mass, M$_h$ [M$_\odot$]',fontsize=20)
    ax2.set_xlim(1e11,3e14)
    ax2.set_xscale('log')

    ax3.set_xlabel('Stellar Mass, M$_\star$ [M$_\odot$]',fontsize=20)
    ax3.set_xlim(1e8,1.6e11)
    ax3.set_xscale('log')

    for axis in [ax1,ax2,ax3]:
        axis.tick_params(which='major', length=8, width=1.5)
        axis.tick_params(which='minor', length=4, width=1.2)

    _ = [label.set_fontsize(20) for label in ax1.get_xticklabels()+ax1.get_yticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()]
    fig.savefig('plots/am_uv_vs_ha_2.png')
    fig.savefig('plots/am_uv_vs_ha_2.pdf')

if __name__ == '__main__':
    
    plot_uvtoha_ratio_vs_Muv()
    #plot_uvtoha_ratio_vs_everything()
    plt.show()