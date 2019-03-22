import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import mk_sample

filters = ['a','w','u','b','v','i','z','d','j','s','h']

def mk_stamps(drop_filt,sample_type,cut_type='no_cut',s=67):

    if   cut_type=='neg_nuv':
        sample = mk_sample.mk_sample(drop_filt,sample_type=sample_type,return_all=True,return_catalog=True)
        sample = sample[(sample['MAG_B_F225W']==-99.) & (sample['MAG_B_F275W']==-99.) & (sample['MAG_B_F336W']==-99.)]
        sample = np.sort(sample,order='MAG_B_F435W')[::-1]
        label  = 'MAG_F435W'
        fname  = 'stamps/stamps_%s%s_nnuv.jpg' % (sample_type[:4],drop_filt[1:-1])
        print "NUV weirdness:", len(sample)
        if len(sample)==0: return

    elif cut_type=='neg_hlr':
        sample = mk_sample.mk_sample(drop_filt,sample_type=sample_type,return_all=True,return_catalog=True)
        sample = sample[sample['HLR_F435W']<0]
        sample = np.sort(sample,order='HLR_F435W')
        label  = 'HLR_F435W'
        fname  = 'stamps/stamps_%s%s_nhlr.jpg' % (sample_type[:4],drop_filt[1:-1])
        print "Negative HLR:", len(sample)
        if len(sample)==0: return

    elif cut_type=='no_cut':
        sample = mk_sample.mk_sample(drop_filt,sample_type=sample_type,return_all=True)
        sample = np.sort(sample,order='M_1500')
        label  = 'M_1500'
        fname  = 'stamps/stamps_%s%s.jpg' % (sample_type[:4],drop_filt[1:-1])
        print "Full sample:", len(sample)

    ncols = (len(sample) / 50) + 1
    isplit = (np.arange(ncols)*50)[1:]

    fig,axes = plt.subplots(1,ncols,figsize=(ncols*(len(filters)+1),min(len(sample),50)),dpi=50)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)

    if ncols==1: axes = [axes,]

    for i,filt in enumerate(filters):

        img = fitsio.getdata('/data/highzgal/PUBLICACCESS/UVUDF/simulations/orig_run/data/%s.fits' % filt)

        for ax,_sample in zip(axes,np.split(sample,isplit)):

            for j,entry in enumerate(_sample):

                xc,yc = int(entry['X']),int(entry['Y'])
                stamp = img[yc-s/2:yc+s/2,xc-s/2:xc+s/2]

                med,std = np.median(stamp), np.std(stamp)
                _stamp = np.clip(stamp,med-5*std,med+5*std)
                med,std = np.median(_stamp[s/5:4*s/5,s/5:4*s/5]),np.std(_stamp[s/5:4*s/5,s/5:4*s/5])
                vmin, vmax = med-3*std, med+3*std

                extent = [s*i,s*(i+1),s*j,s*(j+1)]
                ax.imshow(stamp,cmap=plt.cm.Greys_r,vmin=vmin,vmax=vmax,interpolation='none',extent=extent)
                
                if i==0:
                    ax.text(s*i+2,s*j    +2,"%i"  %entry['ID'], color='w',va='top',   ha='left',
                                        fontsize=14,fontweight=600,path_effects=[PathEffects.withStroke(linewidth=2,foreground="k" if entry['SAMPLE_FLAG']==1 else "r")])
                    ax.text(s*i+2,s*(j+1)-2,"%.2f"%entry[label],color='w',va='bottom',ha='left',
                                        fontsize=14,fontweight=600,path_effects=[PathEffects.withStroke(linewidth=2,foreground="k" if entry['SAMPLE_FLAG']==1 else "r")])

    seg = fitsio.getdata('/data/highzgal/PUBLICACCESS/UVUDF/simulations/orig_run/cpro/udf_run_merge_template/det_segm.fits')

    for ax,_sample in zip(axes,np.split(sample,isplit)):

        for j,entry in enumerate(_sample):

            xc,yc = int(entry['X']),int(entry['Y'])
            id0 = seg[yc,xc]
            stamp = seg[yc-s/2:yc+s/2,xc-s/2:xc+s/2].astype(float)
            idx = np.unique(stamp)
            idx = idx[(idx!=0) & (idx!=id0)]

            for ix,ic in zip(idx,[0.9,0.2,0.7,0.35]): stamp[stamp==ix] = ic
            stamp[stamp==id0] = 0.5

            extent = [s*len(filters),s*(len(filters)+1),s*j,s*(j+1)]
            ax.imshow(stamp,cmap=plt.cm.hot,vmin=0,vmax=1,interpolation='none',extent=extent)

        ax.vlines(s*(np.arange(len(filters)+2)),0,s*len(_sample),    color='k',lw=2,alpha=0.8)
        ax.hlines(s*(np.arange(len(_sample)+1)),0,s*(len(filters)+1),color='k',lw=2,alpha=0.8)

    for ax in axes:
        
        ax.axis("off")
        ax.set_xlim(0,s*(len(filters)+1))
        if ncols>1: ax.set_ylim(s*50,0)
        else: ax.set_ylim(s*len(sample),0)

    fig.savefig(fname)

if __name__ == '__main__':

    for drop_filt in ['f225w','f275w','f336w']:
        for sample_type in ['dropout','photoz']:
            for cut_type in ['neg_nuv','neg_hlr','no_cut']:
                mk_stamps(drop_filt=drop_filt,sample_type=sample_type,cut_type=cut_type)