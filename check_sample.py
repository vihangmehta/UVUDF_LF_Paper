import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.io.fits as fitsio

import useful
import veff
import mk_sample
import conversions as conv
import uvudf_utils as utils
from plot_colcol import plot_colcol, plot_colcut_patches
from sample_selection import mk_dropout_cuts, mk_photoz_cuts
from selection import SelectionFunction

def check_photoz():

    catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
    catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]    

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10),dpi=75,sharex=True)
    fig.subplots_adjust(wspace=0,hspace=0)

    ax1.scatter(catalog['MAG_B_F336W'],catalog['ODDS_B'],c='k',s=3,alpha=0.1)
    ax2.scatter(catalog['MAG_B_F336W'],catalog['CHISQ_B'],c='k',s=3,alpha=0.1)
    
    ax2.set_xlim(23,32)
    ax1.set_ylim(0,1.1)
    ax2.set_ylim(0,5)
    ax2.set_xlabel('F336W MAG')
    ax2.set_ylabel('CHI2')
    ax1.set_ylabel('ODDS')

def check_cuts():

    catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
    catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]

    fig, axes = plt.subplots(1,3,figsize=(20,7),dpi=75,tight_layout=True)

    for ax,drop_filt in zip(axes,['f225w','f275w','f336w']):

        drop = mk_dropout_cuts(catalog,drop_filt)
        ccol = mk_dropout_cuts(catalog,drop_filt,do_sn_cut=False)
        phot = mk_photoz_cuts( catalog,drop_filt)

        drop_not = drop[np.array([i not in phot['ID'] for i in drop['ID']])]
        ccol_not = ccol[np.array([i not in drop['ID'] for i in ccol['ID']])]
        phot_not = phot[np.array([i not in drop['ID'] for i in phot['ID']])]

        plot_colcut_patches(drop_filt, ax)

        filt1,filt2,filt3 = utils.filt_colcol(drop_filt)

        plot_colcol(drop_not, filt1, filt2, filt3, catalog=catalog, axis=ax,
                        marker='o', c='r', s=20, lw=2, alpha=1.0, label='%s Dropouts\n but wrong photo-z' % drop_filt.upper())
        plot_colcol(ccol_not, filt1, filt2, filt3, catalog=catalog, axis=ax,
                        marker='o', c='limegreen', s=20, lw=2, alpha=1.0, label='%s Dropouts colors\n but wrong S/N' % drop_filt.upper())
        plot_colcol(phot_not, filt1, filt2, filt3, catalog=catalog, axis=ax,
                        marker='o', c='b', s=20, lw=2, alpha=1.0, label='%.1f<photo-z<%.1f\n but wrong colors' % (utils.bpz_lims[drop_filt][0],utils.bpz_lims[drop_filt][1]))

        ax.set_xlabel('%s - %s' % (filt2.upper(),filt3.upper()))
        ax.set_ylabel('%s - %s' % (filt1.upper(),filt2.upper()))
        ax.set_xlim(-1,3)
        ax.set_ylim(-1,4)
        ax.legend(fontsize=12)

    fig.savefig('plots/check_cuts.png')

def check_sample_vols():

    fig = plt.figure(figsize=(12,10),dpi=75)
    
    gs1 = gridspec.GridSpec(2,3)
    gs1.update(left=0.1,right=0.98,top=0.98,bottom=0.58,wspace=0,hspace=0)
    axes1 = np.array([fig.add_subplot(ss) for ss in gs1]).reshape(2,3)
    
    gs2 = gridspec.GridSpec(2,3)
    gs2.update(left=0.1,right=0.98,top=0.50,bottom=0.10,wspace=0,hspace=0)
    axes2 = np.array([fig.add_subplot(ss) for ss in gs2]).reshape(2,3)

    for i,drop_filt in enumerate(['f225w','f275w','f336w']):

        for j,sample_type in enumerate(['dropout','photoz']):

            sample = mk_sample.mk_sample(drop_filt,sample_type=sample_type,return_all=True)
            cond  = (sample['SAMPLE_FLAG']==1)
            volfn = veff.VEff_Func(drop_filt=drop_filt,sample_type=sample_type)

            for axes,mag_label in zip([axes1,axes2],['M_1500','m_1500']):
            
                axes[j,i].scatter(sample[mag_label][cond],sample['Vfrac'][cond],c='k',s=15,lw=0,alpha=0.5)
                axes[j,i].scatter(sample[mag_label][~cond],sample['Vfrac'][~cond],c='r',s=15,lw=0,alpha=0.5)
                axes[j,i].axhline(0.50,lw=0.5,ls=':',c='k')
                axes[j,i].axhline(0.25,lw=0.5,ls='--',c='k')
                axes[j,i].axhline(0.10,lw=0.5,ls='-',c='k')
                axes[j,i].set_yscale('log')

            axes1[j,i].axvline(volfn.mag_limit(hlr=8),lw=0.5,c='k')
            
            axes1[j,i].set_xlim(-21.9,-14.5)
            axes1[j,i].set_ylim(3e-2,2e0)
            axes2[j,i].set_xlim(23.2,30)
            axes2[j,i].set_ylim(3e-2,2e0)

            if j!=1:
                axes1[j,i].set_xticklabels([])
                axes2[j,i].set_xticklabels([])
            if i!=0:
                axes1[j,i].set_yticklabels([])
                axes2[j,i].set_yticklabels([])

    axes1[1,1].set_xlabel("Rest 1500$\\AA$ Absolute Mag")
    axes2[1,1].set_xlabel("Rest 1500$\\AA$ Apparent Mag")
    axes1[1,0].set_ylabel("Vmax Correction")
    axes2[1,0].set_ylabel("Vmax Correction")

    fig.savefig('plots/check_sample_vols.png')

def check_sample_selfrac():

    fig,axes = plt.subplots(2,3,figsize=(12,9),dpi=75,sharex='col',sharey=True)
    fig.subplots_adjust(left=0.1,right=0.98,bottom=0.2,top=0.98,wspace=0,hspace=0)

    for i,drop_filt in enumerate(['f225w','f275w','f336w']):

        for j,sample_type in enumerate(['dropout','photoz']):

            selfn = SelectionFunction(drop_filt=drop_filt,sample_type=sample_type)
            xx,yy = np.arange(0.5,5,.01), np.arange(-22,-14,0.1)
            (gy,gx),gz = np.meshgrid(yy,xx), selfn.get_func(hlr=-99.)(xx,yy)
            im = axes[j,i].pcolormesh(gx,gy,gz,lw=0,cmap=plt.cm.inferno,vmin=0,vmax=1)

            sample = mk_sample.mk_sample(drop_filt=drop_filt,sample_type=sample_type,return_all=True)
            cond = (sample['SAMPLE_FLAG']==1)
            selfr = np.array([selfn(entry['M_1500'],entry['z'],hlr=entry['HLR_IN']) for entry in sample])
            axes[j,i].scatter(sample['BPZ'][cond],sample['M_1500'][cond],facecolor=selfr[cond],edgecolor='w',s=50,lw=0.2,cmap=plt.cm.inferno,vmin=0,vmax=1)
            axes[j,i].scatter(sample['BPZ'][~cond],sample['M_1500'][~cond],facecolor=selfr[~cond],edgecolor='r',s=50,lw=0.5,cmap=plt.cm.inferno,vmin=0,vmax=1)

    axes[1,0].set_ylabel("Absolute Magnitude")
    axes[0,0].set_ylim(-14.1,-21.9)
    axes[1,1].set_xlabel("Photo-z")
    axes[0,0].set_xlim(0.5,2.35)
    axes[0,1].set_xlim(0.9,2.9)
    axes[0,2].set_xlim(0.9,4.1)

    cbax = fig.add_axes([0.2,0.08,0.6,0.02])
    cbar = fig.colorbar(mappable=im,cax=cbax,orientation='horizontal')
    cbar.set_label('Relative Effiency')

    fig.savefig('plots/check_sample_selfrac.png')

def check_sample_volfrac():

    fig,axes = plt.subplots(2,3,figsize=(12,9),dpi=75,sharex='col',sharey=True)
    fig.subplots_adjust(left=0.1,right=0.98,bottom=0.2,top=0.98,wspace=0,hspace=0)

    for i,drop_filt in enumerate(['f225w','f275w','f336w']):

        for j,sample_type in enumerate(['dropout','photoz']):

            sample = mk_sample.mk_sample(drop_filt=drop_filt,sample_type=sample_type,return_all=True)
            cond = (sample['SAMPLE_FLAG']==1)
            im = axes[j,i].scatter(sample['BPZ'][cond],sample['M_1500'][cond],c=sample['Vfrac'][cond],s=50,lw=0.2,cmap=plt.cm.inferno,vmin=0,vmax=1)
            im = axes[j,i].scatter(sample['BPZ'][~cond],sample['M_1500'][~cond],c=sample['Vfrac'][~cond],s=50,marker='x',lw=1,cmap=plt.cm.inferno,vmin=0,vmax=1)

    axes[1,0].set_ylabel("Absolute Magnitude")
    axes[0,0].set_ylim(-14.1,-21.9)
    axes[1,1].set_xlabel("Photo-z")
    axes[0,0].set_xlim(0.5,2.35)
    axes[0,1].set_xlim(0.9,2.9)
    axes[0,2].set_xlim(0.9,4.1)

    cbax = fig.add_axes([0.2,0.08,0.6,0.02])
    cbar = fig.colorbar(mappable=im,cax=cbax,orientation='horizontal')
    cbar.set_label('Effective Volume Fraction')

    fig.savefig('plots/check_sample_volfrac.png')

def check_z():

    bins = np.arange(0,5,0.1)
    zz = np.arange(0.5,5,0.05)
    mag_range = [25,26,27,28,29]
    colors = ['b','c','limegreen','r','m']

    catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
    catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]

    for sample_type in ['dropout','photoz']:

        fig,axes = plt.subplots(1,3,figsize=(16,5),dpi=75)
        fig.subplots_adjust(left=0.055,right=0.94,bottom=0.14,top=0.95,hspace=0,wspace=0.3)
        daxes = []

        axes[0].text(0.95,0.95,"F225W Dropouts\n(z~1.65)",va='top',ha='right', fontsize=14,fontweight=600,transform=axes[0].transAxes)
        axes[1].text(0.50,0.95,"F275W Dropouts (z~2.2)",va='top',ha='center',fontsize=14,fontweight=600,transform=axes[1].transAxes)
        axes[2].text(0.05,0.95,"F336W Dropouts\n(z~3)",va='top',ha='left',  fontsize=14,fontweight=600,transform=axes[2].transAxes)

        axes[0].set_xlim(0.5,2.45)
        axes[1].set_xlim(0.9,2.9)
        axes[2].set_xlim(0.9,4.1)

        if sample_type=='dropout':
            axes[0].set_ylim(0,20)
            axes[1].set_ylim(0,25)
            axes[2].set_ylim(0,50)

        for ax,drop_filt,det_filt,filt_1500 in zip(axes,['f225w','f275w','f336w'],['f275w','f336w','f435w'],['f336w','f435w','f435w']):

            sample = mk_sample.mk_sample(drop_filt=drop_filt,sample_type=sample_type)
            selfunc = SelectionFunction(drop_filt=drop_filt,sample_type=sample_type)

            if sample_type=='dropout':
                cond = (conv.calc_sn(catalog[utils.filt_key[det_filt]],catalog[utils.dfilt_key[det_filt]]) > 5)
            elif sample_type=='photoz':
                cond = (conv.calc_sn(catalog[utils.filt_key[filt_1500]],catalog[utils.dfilt_key[filt_1500]]) > 5)
            #ax.hist(catalog['ZB_B'][cond], bins=bins,color='k',lw=1,histtype='step',alpha=0.8)
            ax.hist(sample['BPZ'],   bins=bins,color='k',lw=0,alpha=0.2,label='Photo-z')
            ax.hist(sample['GRISMZ'],bins=bins,color='b',lw=0,alpha=0.5,label='Grism-z')
            ax.hist(sample['SPECZ'], bins=bins,color='r',lw=0,alpha=0.5,label='Spec-z')
            
            dax = ax.twinx()
            daxes.append(dax)
            dax.set_ylim(0,1)
            dax.set_xlim(ax.get_xlim())

            if sample_type=='dropout':
                _mag_range = mag_range[:-1]
            else: _mag_range = mag_range
            
            for m,c in zip(_mag_range,colors):
                absM = np.array([conv.get_abs_from_app(m,_z) for _z in zz])
                selfr = selfunc.get_func(hlr=-99.)(zz,absM,grid=False)
                dax.plot(zz,selfr,c=c,lw=2,label='$m_{UV}$ = %i'%m)

            if sample_type=='photoz':
                norm = 1./np.max(selfunc.get_func(hlr=-99.)(zz,-22,grid=False))
                ax.set_ylim(0,0.9*norm*ax.get_ylim()[1])

            ax.set_xlabel('Redshift',fontsize=24)
            if ax==axes[0]: ax.set_ylabel('Number',fontsize=24)
            if ax==axes[2]: dax.set_ylabel('Completeness',fontsize=24)
            _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()]
            
        axes[0].legend(fontsize=14,loc=2,frameon=False)
        leg = daxes[2].legend(fontsize=14,loc=1,frameon=False,handlelength=0,handletextpad=0)
        for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
            txt.set_fontweight(600)
            txt.set_color(hndl.get_color())
            hndl.set_visible(False)

        fig.savefig('plots/check_z_%s.png'%sample_type)
        fig.savefig('plots/check_z_%s.pdf'%sample_type)

if __name__ == '__main__':

    # check_photoz()
    # check_cuts()
    # check_sample_vols()
    # check_sample_selfrac()
    # check_sample_volfrac()
    check_z()
    # check_size()
    # check_mag_size()

    plt.show()