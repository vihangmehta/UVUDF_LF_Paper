import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio

import useful

def plot_spec_vs_age():

    fig,axes = plt.subplots(2,2,figsize=(12,6),dpi=75,sharex=True)
    fig.subplots_adjust(left=0.05,right=0.95,bottom=0.15,top=0.95,wspace=0,hspace=0)
    axes = axes.flatten()

    #for metal,ls in zip(['_m32','_m42','_m62'],['-','--','-.']):
    for ax,sfh in zip(axes,['_ssp','_burst10','_burst100','_cSFR']):

        x = fitsio.getdata('bc03/bc2003_chab_m32'+sfh+'.fits')
        ages = x.dtype.names[1::10]
        colors = plt.cm.gist_rainbow_r(np.linspace(0.15,1,len(ages)))

        for age,c in zip(ages,colors):
            cond = (5e2<=x['waves']) & (x['waves'] <= 1e4)
            ax.plot(x['waves'][cond],x[age][cond],color=c,lw=1.5,ls='-',alpha=0.8)

        ax.text(0.98,0.95,sfh[1:],va='top',ha='right',fontsize=14,fontweight=600,transform=ax.transAxes)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_yticks([])
        xticks = np.array([1,2,3,5,8,10])*1e3
        ax.set_xticks(xticks)
        ax.set_xticklabels(["%i"%(x/1e3) for x in xticks])

    axes[-2].set_ylabel('L$_{\lambda}$ [ergs/s/cm$^2$/$\AA$]')
    axes[-2].set_xlabel('Wavelength [$\\times 10^3 \ \AA$]',fontsize=20)
    ax.set_xlim(5e2,1e4)
    fig.savefig('plots/bc03_spectra.png')

def plot_lum_vs_age():

    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

    #for metal,ls in zip(['_m32','_m42','_m62'],['-','--','-.']):
    for sfh,c in zip(['_ssp','_burst10','_burst100','_cSFR'],['r','b','g','c']):

        x = fitsio.getdata('bc03/bc2003_chab_m32'+sfh+'_luv.fits')
        ax.plot(x.age*1e3,x.luv_lambda*1500.,color=c,lw=3,ls='-',alpha=1.0)
        ax.plot(x.age*1e3,x.lha_lambda,color=c,lw=1,ls='-',alpha=1.0)

    ax.axhline(-99,c='k',lw=3,ls='-',label='$\lambda$L$_{\lambda,1500}$')
    ax.axhline(-99,c='k',lw=1,ls='-',label='L$_{H\\alpha}$')
    ax.axhline(-99,c='r',lw=2,label='SSP')
    ax.axhline(-99,c='b',lw=2,label='10Myr Burst')
    ax.axhline(-99,c='g',lw=2,label='100Myr Burst')
    ax.axhline(-99,c='c',lw=2,label='constant SFR')

    ax.set_ylabel('$\lambda$L$_{\lambda,1500}$ or L$_{H\\alpha}$ [ergs/s/cm$^2$]',fontsize=20)
    ax.set_yscale('log')
    ax.set_ylim(1e25,1e45)
    ax.set_xlim(5e-1,14e3)
    ax.set_xscale('log')
    ax.set_xlabel('Age [Myr]',fontsize=20)
    ax.legend(fontsize=14,loc='best')
    fig.savefig('plots/bc03_lum.png')

def plot_lum_vs_age_2():

    fig,ax = plt.subplots(1,1,figsize=(12,8),dpi=75,tight_layout=True)

    #for metal,ls in zip(['_m32','_m42','_m62'],['-','--','-.']):
    for sfh,c in zip(['_ssp','_burst10','_burst100','_cSFR'],['r','b','g','c']):

        x = fitsio.getdata('bc03/bc2003_chab_m32'+sfh+'_luv.fits')

        i = np.argmin(np.abs(x.age-1e-3))
        ax.plot(x.age*1e3,x.luv_lambda/x.luv_lambda[i],color=c,lw=2,ls='-',alpha=1.0)
        ax.plot(x.age*1e3,x.lha_lambda/x.lha_lambda[i],color=c,lw=2,ls='--',alpha=1.0)

    ax.axhline(-99,c='k',lw=2,ls='-',label='UV')
    ax.axhline(-99,c='k',lw=2,ls='--',label='H$\\alpha$')
    ax.axhline(-99,c='r',lw=2,label='SSP')
    ax.axhline(-99,c='b',lw=2,label='10Myr Burst')
    ax.axhline(-99,c='g',lw=2,label='100Myr Burst')
    ax.axhline(-99,c='c',lw=2,label='constant SFR')

    ax.set_ylabel('norm. Luminosity [ergs/s/cm$^2$]',fontsize=20)
    ax.set_yscale('log')
    ax.set_xlim(1e0,14e3)
    ax.set_xscale('log')
    ax.set_xlabel('Age [Myr]',fontsize=20)
    ax.legend(fontsize=14,loc='best')
    # fig.savefig('plots/bc03_lum.png')

def plot_ratio_vs_age():

    fig,ax = plt.subplots(1,1,figsize=(10,7),dpi=75,tight_layout=True)

    ratios = np.zeros(0)
    for sfh,c in zip(['_ssp','_burst10','_burst100','_cSFR','_decl10','_decl100',    '_inc10',  '_inc100'],
                     [   'r',       'b',        'c',    'g',      'm',       'y','darkorange','limegreen'])[:4]:

        for metal,ls in zip(['_m32','_m42','_m62'],['-','--',':']):

            band = None
            for imf,lw in zip(['_salp','_chab','_kroupa'],[3,2,1]):

                x = fitsio.getdata('bc03/bc2003'+imf+metal+sfh+'_luv.fits')
                ratio = np.log10(x.luv_lambda*1500./x.lha_lambda)
                if band is not None:
                    band = np.vstack((band,ratio))
                else:
                    band = ratio

                if sfh=='_cSFR':
                    crit = np.abs(x.age*1e3 - 100)
                    ix = np.where(crit == min(crit))[0][0]
                    ratios = np.append(ratios,ratio[ix])

                #if imf=='_salp':
                ax.plot(x.age*1e3,ratio,color=c,lw=2,ls=ls,alpha=0.5,zorder=10)

            # band1, band2 = np.min(band,axis=0),np.max(band,axis=0)
            # ax.fill_between(x.age*1e3,band1,band2,color=c,lw=0.5,alpha=alpha)

    ax.axhspan(min(ratios),max(ratios),edgecolor='k',facecolor='lightgrey',lw=0,alpha=0.8,hatch='//',zorder=1)

    ax.axhline(-99,c='r',lw=2,label='SSP')
    ax.axhline(-99,c='b',lw=2,label='10Myr Burst')
    ax.axhline(-99,c='c',lw=2,label='100Myr Burst')
    # ax.axhline(-99,c='m',lw=2,label='10Myr exp. decl.')
    # ax.axhline(-99,c='y',lw=2,label='100Myr exp. decl.')
    # ax.axhline(-99,c='saddlebrown',lw=2,label='10Myr delayed exp.')
    # ax.axhline(-99,c='olive',      lw=2,label='100Myr delayed exp.')
    # ax.axhline(-99,c='darkorange', lw=2,label='10Myr exp. inc.')
    # ax.axhline(-99,c='limegreen',  lw=2,label='100Myr exp. inc.')
    ax.axhline(-99,c='g',lw=2,label='constant SFR')
    ax.axhline(-99,c='k',lw=2,alpha=1.0,ls='-' ,label='Z=0.02Z$_\odot$')
    ax.axhline(-99,c='k',lw=2,alpha=1.0,ls='--',label='Z=0.2Z$_\odot$')
    ax.axhline(-99,c='k',lw=2,alpha=1.0,ls=':' ,label='Z=Z$_\odot$')

    ax.set_ylabel('log ($\\nu$L$_{\\nu,UV}$/L$_{H\\alpha}$) (dust corrected)',fontsize=24)
    ax.set_ylim(0.9,7.1)
    ax.set_xlabel('Age [Myr]',fontsize=24)
    ax.set_xlim(8e-1,5e3)
    ax.set_xscale('log')
    ax.minorticks_on()
    ax.tick_params(which='minor',length=4,width=1)
    ax.tick_params(which='major',length=8,width=2)
    _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()]
    ax.legend(fontsize=16,loc=2,ncol=2,framealpha=0)

    fig.savefig('plots/bc03_uv_vs_ha.png')
    fig.savefig('plots/bc03_uv_vs_ha.pdf')

def plot_ewha_vs_age():

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)
    ax.minorticks_on()

    for sfh,c in zip(['_ssp','_burst10','_burst100','_cSFR'],['r','b','g','c']):

        line = fitsio.getdata('bc03/bc2003_chab_m32'+sfh+'_luv.fits')
        spec = fitsio.getdata('bc03/bc2003_chab_m32'+sfh+'.fits',1)

        assert len(line) == len(spec.dtype.names[1:])
        iwave = (np.abs(spec['waves']-6500.)).argmin()
        jwave = (np.abs(spec['waves']-6600.)).argmin()
        conti = np.array([np.mean([spec[x][iwave],spec[x][jwave]]) for x in spec.dtype.names[1:]])
        ha_ew = line.lha_lambda / conti

        ax.plot(line.age*1e3,ha_ew,color=c,lw=3,alpha=0.8)

    ax.axhline(-99,c='r',lw=2,label='SSP')
    ax.axhline(-99,c='b',lw=2,label='10Myr Burst')
    ax.axhline(-99,c='g',lw=2,label='100Myr Burst')
    ax.axhline(-99,c='c',lw=2,label='constant SFR')

    ax.set_ylabel('Rest EW$_{H\\alpha}$ [$\\AA$]',fontsize=20)
    ax.set_yscale('log')
    ax.set_ylim(9e1,3e4)
    ax.set_xlim(5e-1,5e2)
    ax.set_xscale('log')
    ax.set_xlabel('Age [Myr]',fontsize=20)
    ax.legend(fontsize=20,loc='best',frameon=False)

    fig.savefig('plots/bc03_ha_ew.png')

if __name__ == '__main__':

    #plot_spec_vs_age()
    # plot_lum_vs_age()
    plot_lum_vs_age_2()
    # plot_ratio_vs_age()
    #plot_ewha_vs_age()
    plt.show()
