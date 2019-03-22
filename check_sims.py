import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio

import useful
import conversions as conv
from uvudf_utils import filters, drop_filters, filt_det, filt_1500, filt_key, dfilt_key, bpz_lims, read_simulation_output

sim_input,sim_recov,sim_hlr = read_simulation_output(run0=True,run7=True,run9=True)

catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]
cond_bpz = (catalog['ODDS_B'] > 0.9) & (catalog['CHISQ2_B'] < 1.0)

def mag_size():

    fig, ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)
    #fig.subplots_adjust(left=0.06,right=0.98,bottom=0.14,top=0.92,wspace=0.25)

    sn = conv.calc_sn(catalog[filt_key['f435w']],catalog[dfilt_key['f435w']])
    cond_z = (1.0 < catalog['ZB_B']) & (catalog['ZB_B'] < 4.0)
    cond_m = (sn > 5.)

    ax.scatter(sim_input[filt_key['f435w']],sim_input['hlr'],c='r',s=10,lw=0,alpha=0.2,label='Simulation Input')
    ax.scatter(sim_recov[filt_key['f435w']],sim_hlr['f435w'],c='b',s=10,lw=0,alpha=0.2,label='Simulation Recovered')
    ax.scatter(catalog[filt_key['f435w']][cond_bpz&cond_z&cond_m],
               catalog['HLR_F435W'][cond_bpz&cond_z&cond_m], c='k',s=20,lw=0,alpha=1.0,label='Observed Catalog')

    ax.set_xlabel('F435W Magnitude [AB]')
    ax.set_xlim(21,32)
    ax.set_ylabel('F435W HLR [px]')
    ax.set_ylim(0,25)
    ax.legend(fontsize=12,loc=2)

    fig.savefig('plots/check_mag_size.png')

def z_size():

    fig,ax = plt.subplots(1,1,figsize=(10,7),dpi=75,tight_layout=True)

    dbin = 0.25
    bins = np.arange(0,20,dbin)
    binc = 0.5*(bins[1:]+bins[:-1])
    zz = np.arange(0,4.01,1)
    colors = plt.cm.brg(np.linspace(0,1,len(zz[:-1])))

    z0,z1 = 0,4
    cond = (z0 <= catalog['ZB_B']) & (catalog['ZB_B'] <= z1)
    hist = np.histogram(catalog['HLR_F435W'],bins=bins)[0]
    err = useful.poisson_interval(hist,0.6827)
    hist, err = hist/float(sum(hist))/dbin, err/float(sum(hist))/dbin
    ax.plot(binc,hist,c='k',lw=3,marker='o',markersize=10,mew=0,alpha=0.8,label='%.1f<z<%.1f'%(z0,z1))
    ax.fill_between(binc,err[0],err[1],color='k',lw=0,alpha=0.1)

    for z0,z1,c in zip(zz[:-1],zz[1:],colors):
        cond = (z0 <= catalog['ZB_B']) & (catalog['ZB_B'] <= z1)
        hist = np.histogram(catalog['HLR_F435W'][cond],bins=bins)[0]
        err = useful.poisson_interval(hist,0.6827)
        hist, err = hist/float(sum(hist))/dbin, err/float(sum(hist))/dbin
        ax.plot(binc,hist,c=c,lw=2,marker='o',markersize=5,mew=0,alpha=0.8,label='%.1f<z<%.1f'%(z0,z1))
        ax.fill_between(binc,err[0],err[1],color=c,lw=0,alpha=0.1)

    ax.set_yscale('log')
    ax.set_xlabel('F435W HLR [px]')
    ax.set_ylabel('Normalized Freq.')
    ax.set_ylim(2e-3,5e-1)
    ax.set_xlim(0,11)

    leg = ax.legend(fontsize=16,ncol=2,loc='best',frameon=False,markerscale=0,handlelength=0,handletextpad=0)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_fontweight(600)
        txt.set_color(hndl.get_color())
        hndl.set_visible(False)

    fig.savefig('plots/check_z_size.png')

def size():

    catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
    catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]

    z = np.arange(0,4.01,1)
    bins = np.arange(0,10,0.25)
    binc = 0.5*(bins[1:] + bins[:-1])
    colors = plt.cm.jet(np.linspace(0,1,len(z)-1))

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10),dpi=75,sharex=True,tight_layout=False)
    fig.subplots_adjust(left=0.12,right=0.95,bottom=0.09,top=0.96,hspace=0,wspace=0)

    z0,z1 = z[0],z[-1]
    cond = (z0<=catalog['ZB_B']) & (catalog['ZB_B']<=z1)
    hist = ax1.hist(catalog['HLR_F435W'][cond],bins=bins,histtype='stepfilled',color='k',alpha=0.1,label="%.1f<z<%.1f"%(z0,z1))[0]
    err = useful.poisson_interval(hist,sigma=0.68) - hist
    hist, err = hist / float(sum(hist)), err / float(sum(hist))
    cond_hist = (hist!=0)
    
    for z0,z1,c in zip(z[:-1],z[1:],colors):
        
        cond = (z0<=catalog['ZB_B']) & (catalog['ZB_B']<=z1)
        _hist = ax1.hist(catalog['HLR_F435W'][cond],bins=bins,histtype='step',color=c,lw=2,alpha=0.8,label="%.1f<z<%.1f"%(z0,z1))[0]
        _err = useful.poisson_interval(_hist,sigma=0.68) - _hist
        _hist, _err = _hist / float(sum(_hist)), _err / float(sum(_hist))

        ratio = _hist[cond_hist] / hist[cond_hist].astype(float)
        ratio_err = np.sqrt((err/hist)**2+(_err/_hist)**2) * ratio
        ax2.plot(binc[cond_hist],ratio,color=c,lw=2,marker='o',markersize=0,mew=0)
        ax2.fill_between(binc[cond_hist],ratio-ratio_err[0],ratio+ratio_err[1],color=c,lw=0,alpha=0.2)
        ax2.axhline(1,c='k',ls='--',lw=1)

    ax1.legend(fontsize=14)
    ax1.set_ylabel('N')
    ax2.set_ylabel('N/N$_{tot}$')
    ax2.set_xlabel('F435W HLR [pixel]')
    ax2.set_ylim(1/3e1,3e1)
    ax2.set_yscale('log')
    fig.savefig('plots/check_size.png') 

def obs_size():

    catlist = ['catalogs/udf_sample_f225w_dropout.fits','catalogs/udf_sample_f275w_dropout.fits','catalogs/udf_sample_f336w_dropout.fits',
               'catalogs/udf_sample_f225w_photoz.fits' ,'catalogs/udf_sample_f275w_photoz.fits' ,'catalogs/udf_sample_f336w_photoz.fits' ]
    
    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)
    
    bins = np.arange(0,20,0.25)
    binc = 0.5*(bins[1:] + bins[:-1])
    dbin = bins[1:] + bins[:-1]
    hist = np.histogram(sim_hlr['f435w'],bins=bins)[0]
    ax.plot(binc,hist/float(max(hist))/dbin,color='k',lw=2,ls='--',alpha=0.8)

    bins = np.arange(0,20,1)
    binc = 0.5*(bins[1:] + bins[:-1])
    dbin = bins[1:] + bins[:-1]
    colors = plt.cm.jet(np.linspace(0,1,len(catlist)-1))
    
    for catname,c in zip(catlist,colors):

        catalog = fitsio.getdata(catname)
        hist = np.histogram(catalog['HLR_F435W'],bins=bins)[0]
        ax.plot(binc,hist/float(max(hist))/dbin,color=c,lw=2,alpha=0.5,label=' '.join(catname.split('/')[1].split('.')[0].split('_')[2:]))

    ax.legend(fontsize=14)
    ax.set_ylabel('N')
    ax.set_xlabel('F435W HLR [pixel]')

def sersic():

    fig,axes = plt.subplots(3,4,figsize=(16,12),dpi=75,sharey=True)
    fig.subplots_adjust(left=0.04,right=0.98,bottom=0.06,top=0.94,wspace=0,hspace=0.55)
    axes = axes.flatten()
    axes[-1].set_visible(False)
    axes = axes[:-1]
    daxes = np.zeros(0)

    bins_mag = np.arange(20,35,0.2)
    binc_mag = 0.5*(bins_mag[1:]+bins_mag[:-1])
    bins_hlr = np.arange(0,10,0.1)
    binc_hlr = 0.5*(bins_hlr[1:]+bins_hlr[:-1])

    for filt,ax in zip(filters,axes):

        sn = conv.calc_sn(sim_recov[filt_key[filt]],sim_recov[dfilt_key[filt]])
        missed = sim_input[sn >= 5.0]

        _hist1 = np.histogram(sim_input[filt_key[filt]][sim_input['n']==1],bins=bins_mag)[0]
        _hist4 = np.histogram(sim_input[filt_key[filt]][sim_input['n']==4],bins=bins_mag)[0]
        hist1 = np.histogram(missed[filt_key[filt]][missed['n']==1],bins=bins_mag)[0]
        hist4 = np.histogram(missed[filt_key[filt]][missed['n']==4],bins=bins_mag)[0]
        cond1 = (_hist1 != 0)
        cond4 = (_hist4 != 0)
        ax.plot(binc_mag[cond1],1.*hist1[cond1]/_hist1[cond1],c='r',drawstyle='steps-mid',lw=1.5,alpha=0.75,label='n=1')
        ax.plot(binc_mag[cond4],1.*hist4[cond4]/_hist4[cond4],c='b',drawstyle='steps-mid',lw=1.5,alpha=0.75,label='n=4')
        ax.set_xlabel('Input ' + filt_key[filt])
        ax.set_ylim(0,2.4)
        ax.set_yticks([0,0.5,1,1.2,1.7,2.2])
        ax.set_yticklabels([0,0.5,1,0,0.5,1])
        ax.set_xlim(21,round(max(max(binc_mag[cond1]),max(binc_mag[cond4])),0)+1)

        ax.axhline(1.2,lw=1.5,c='k')
        ax.spines['top'].set_visible(False)
        dax = ax.twiny()
        daxes = np.append(daxes,dax)

        _hist1 = np.histogram(sim_input['hlr'][sim_input['n']==1],bins=bins_hlr)[0]
        _hist4 = np.histogram(sim_input['hlr'][sim_input['n']==4],bins=bins_hlr)[0]
        hist1 = np.histogram(missed['hlr'][missed['n']==1],bins=bins_hlr)[0]
        hist4 = np.histogram(missed['hlr'][missed['n']==4],bins=bins_hlr)[0]
        cond1 = (_hist1 != 0)
        cond4 = (_hist4 != 0)
        dax.plot(binc_hlr[cond1],1.*hist1[cond1]/_hist1[cond1] + 1.2,c='r',drawstyle='steps-mid',lw=1.5,alpha=0.75)
        dax.plot(binc_hlr[cond4],1.*hist4[cond4]/_hist4[cond4] + 1.2,c='b',drawstyle='steps-mid',lw=1.5,alpha=0.75)
        dax.set_xlim(-0.1,10.1)
        dax.set_xticks([2,4,6,8,10])
        dax.set_ylim(0,2.4)
        dax.set_xlabel('Input HLR')

    axes[-1].legend(fontsize=14,loc='center',bbox_to_anchor=[1.5,0.5])

def sersic2d():

    fig,axes = plt.subplots(3,4,figsize=(16,12),dpi=75,sharex=True,sharey=True)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.06,top=0.98,wspace=0,hspace=0.12)
    axes = axes.flatten()
    axes[-1].set_visible(False)
    axes = axes[:-1]

    hists,bins_mag,bins_hlr = {},{},{}
    std = np.zeros(0)

    for filt in filters:

        sn = conv.calc_sn(sim_recov[filt_key[filt]],sim_recov[dfilt_key[filt]])
        missed = sim_input[sn >= 5.0]

        bins_mag[filt] = scipy.stats.mstats.mquantiles(sim_input[filt_key[filt]], np.linspace(0,1,40))
        bins_hlr[filt] = scipy.stats.mstats.mquantiles(sim_input['hlr']         , np.linspace(0,1,20))
        hists1  = np.histogram2d(missed[filt_key[filt]][missed['n']==1],missed['hlr'][missed['n']==1],bins=[bins_mag[filt],bins_hlr[filt]])[0].T.astype(float)
        hists4  = np.histogram2d(missed[filt_key[filt]][missed['n']==4],missed['hlr'][missed['n']==4],bins=[bins_mag[filt],bins_hlr[filt]])[0].T.astype(float)
        _hists1 = np.histogram2d(sim_input[filt_key[filt]][sim_input['n']==1],sim_input['hlr'][sim_input['n']==1],bins=[bins_mag[filt],bins_hlr[filt]])[0].T.astype(float)
        _hists4 = np.histogram2d(sim_input[filt_key[filt]][sim_input['n']==4],sim_input['hlr'][sim_input['n']==4],bins=[bins_mag[filt],bins_hlr[filt]])[0].T.astype(float)
        hists1 = np.ma.masked_array(hists1,mask=hists1==0)
        hists4 = np.ma.masked_array(hists4,mask=hists4==0)
        _hists1 = np.ma.masked_array(_hists1,mask=_hists1==0)
        _hists4 = np.ma.masked_array(_hists4,mask=_hists4==0)
        hists[filt] = (hists1/_hists1.astype(float)) - (hists4/_hists4.astype(float))
        std = np.append(std,np.ma.std(hists[filt]))

    for filt,ax in zip(filters,axes):
        img = ax.pcolormesh(bins_mag[filt],bins_hlr[filt],hists[filt],cmap=plt.cm.RdYlBu,vmin=0-2*np.max(std),vmax=0+2*np.max(std))
        ax.set_xlabel('Input %s'%filt.upper())

    cbax = fig.add_axes([0.82,0.07,0.02,0.25])
    cbar = fig.colorbar(img,cax=cbax,orientation='vertical',label='(n=1 completeness) - \n (n=4 completeness)')

    axes[8].set_ylabel('Input HLR')
    axes[0].set_xlim(22,32)
    axes[0].set_ylim(0,10)

def mag_size_number_density():

    fig,axes = plt.subplots(3,4,figsize=(16,12),dpi=75,sharex=True,sharey=True)
    fig.subplots_adjust(left=0.08,right=0.98,bottom=0.08,top=0.98,wspace=0,hspace=0.12)
    axes = axes.flatten()
    axes[-1].set_visible(False)
    axes = axes[:-1]

    hists,bins_mag,binc_mag,bins_hlr,binc_hlr,vmin,vmax = {},{},{},{},{},1e4,0
    levels = [50,125]

    for filt in filters:

        bins_x = scipy.stats.mstats.mquantiles(sim_input[filt_key[filt]], np.linspace(0,1,40))
        bins_y = scipy.stats.mstats.mquantiles(sim_input['hlr']         , np.linspace(0,1,20))
        dx = np.median(bins_x[1:] - bins_x[:-1])
        dy = np.median(bins_y[1:] - bins_y[:-1])

        bins_mag[filt] = np.arange(15,50,dx)
        bins_hlr[filt] = np.arange(-1,12,dy)
        binc_mag[filt] = 0.5*(bins_mag[filt][1:]+bins_mag[filt][:-1])
        binc_hlr[filt] = 0.5*(bins_hlr[filt][1:]+bins_hlr[filt][:-1])
        hists[filt] = np.histogram2d(sim_input[filt_key[filt]],sim_input['hlr'],bins=[bins_mag[filt],bins_hlr[filt]])[0].T
        hists[filt] = np.ma.masked_array(hists[filt],mask=hists[filt]==0)
        vmax = max(vmax,np.max(hists[filt]))
        vmin = min(vmin,np.min(hists[filt]))

    for filt,ax in zip(filters,axes):
        img = ax.pcolormesh(bins_mag[filt],bins_hlr[filt],hists[filt],cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
        ax.contour(binc_mag[filt],binc_hlr[filt],hists[filt],levels=levels,linewidths=[1.0,2.0],colors='w')
        ax.set_xlabel('Input %s'%filt.upper())

    cbax = fig.add_axes([0.85,0.08,0.02,0.25])
    dbax = cbax.twinx()
    dbax.set_ylim(0,vmax)
    dbax.xaxis.set_visible(False)
    dbax.yaxis.set_visible(False)
    dbax.axhline(levels[0],c='w',lw=1.0,zorder=0)
    dbax.axhline(levels[1],c='w',lw=2.0,zorder=0)
    cbar = fig.colorbar(img,cax=cbax,orientation='vertical')

    axes[8].set_ylabel('Input HLR')
    axes[0].set_xlim(22,32)
    axes[0].set_ylim(0,10)

def check_comp(_filt):

    sim_input,sim_recov,sim_hlr = read_simulation_output(run0=False,run7=True,run9=True)

    filt, dfilt = filt_key[_filt],dfilt_key[_filt]
    sn = conv.calc_sn(sim_recov[filt], sim_recov[dfilt])
    cond = (sim_input[filt] < 25)

    input_mags = sim_input[cond]
    recov_mags = sim_input[cond & (sn>=5.)]
    missd_mags = sim_input[cond & (sn< 5.)]
    print _filt, len(recov_mags), len(input_mags), len(missd_mags)

    ibreak = np.where(np.diff(missd_mags['ID']) < 0)[0][0] + 1
    run = np.zeros(len(missd_mags))
    run[:ibreak], run[ibreak:] = 7, 9

    s = int(5./0.03)
    N = int(np.ceil(np.sqrt(len(missd_mags))))

    fig,axes = plt.subplots(N,N,figsize=(10,10),dpi=75)
    fig.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.98,hspace=0,wspace=0)
    axes = axes.flatten()
    _ = [ax.set_xticks([]) for ax in axes]
    _ = [ax.set_yticks([]) for ax in axes]

    fig2,axes2 = plt.subplots(N,N,figsize=(10,10),dpi=75)
    fig2.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.98,hspace=0,wspace=0)
    axes2 = axes2.flatten()
    _ = [ax.set_xticks([]) for ax in axes2]
    _ = [ax.set_yticks([]) for ax in axes2]

    for i,(entry,_run,_ax,_ax2) in enumerate(zip(missd_mags,run,axes,axes2)):

        sys.stdout.write("\rWorking on %i of %i ... " % (i+1,len(missd_mags)))
        sys.stdout.flush()

        _iter = entry['ID'] / 150
        _id   = entry['ID'] % 150
        _x,_y = int(entry['xpx']), int(entry['ypx'])

        img = fitsio.getdata("/data/highzgal/PUBLICACCESS/UVUDF/simulations/run_%i/iter_%i/cpro/udf_run_merge_template/v.fits" % (_run,_iter))
        stamp = img[_y-s/2:_y+s/2,_x-s/2:_x+s/2]

        img = fitsio.getdata("/data/highzgal/PUBLICACCESS/UVUDF/simulations/run_%i/iter_%i/cpro/udf_run_merge_template/det_segm.fits" % (_run,_iter))
        seg = img[_y-s/2:_y+s/2,_x-s/2:_x+s/2]

        med,std = np.median(stamp),np.std(stamp)
        _stamp = np.clip(stamp,med-5*std,med+5*std)
        med,std = np.median(_stamp),np.std(_stamp)
        vmin,vmax = med-3*std,med+3*std

        _ax.imshow(stamp,cmap=plt.cm.Greys_r,vmin=vmin,vmax=vmax,origin='lower',interpolation='none')
        _ax2.imshow(seg,cmap=plt.cm.Greys_r,vmin=0,vmax=np.max(seg),origin='lower',interpolation='none')

    for _i in range(i+1,N**2):
        axes[_i].set_visible(False)
        axes2[_i].set_visible(False)

    print "done."

if __name__ == '__main__':
    
    obs_size()
    #mag_size()
    #z_size()
    #sersic()
    #sersic2d()
    #mag_size_number_density()
    #check_comp(_filt='f435w')
    #check_comp(_filt='f606w')
    plt.show()