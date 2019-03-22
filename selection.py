import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import useful
from conversions import calc_sn, get_abs_from_app
from uvudf_utils import filt_det, filt_1500, filt_key, dfilt_key, read_simulation_output
from sample_selection import mk_dropout_cuts, mk_photoz_cuts

class SelectionFunction():

    def __init__(self,drop_filt,sample_type):

        self.drop_filt = drop_filt
        self.sample_type = sample_type
        self.func_list = {}

        self.input,self.recov,_ = read_simulation_output(run0=True,run7=True,run9=True)

        self.bins_h = scipy.stats.mstats.mquantiles(self.input['hlr'], np.linspace(0,1,4))
        self.binc_h = 0.5*(self.bins_h[1:]+self.bins_h[:-1])

        self.round_hlr = np.vectorize(self.round_hlr,excluded=['self',])
        self.get_2Dpivot_z = np.vectorize(self.get_2Dpivot_z,excluded=['self',])

        if   self.drop_filt == 'f225w':
            self.dz, self.nbinM = 0.12, 35
        elif self.drop_filt == 'f275w':
            self.dz, self.nbinM = 0.14, 35
        elif self.drop_filt == 'f336w':
            self.dz, self.nbinM = 0.16, 35
        else:
            raise Exception("No drop_filt:%s" % self.drop_filt)

        if   self.sample_type == 'dropout':
            self.smooth = 0.5
        elif self.sample_type == 'photoz':
            self.smooth = 0.8
        else:
            raise Exception("No sample_type:%s" % self.sample_type)

    def __call__(self,M,z,hlr):

        func = self.get_func(hlr)
        return func(z,M).flatten()

    def round_hlr(self,hlr):

        if hlr == -99.: return hlr
        idx = np.clip(np.digitize(hlr,self.bins_h)-1,0,len(self.binc_h)-1)
        return self.binc_h[idx]

    def get_uid(self,hlr):

        _hlr = self.round_hlr(hlr)
        return str(_hlr)

    def get_func(self,hlr):

        uid = self.get_uid(hlr)
        if uid not in self.func_list:
            self.func_list[uid] = self.mk_2Dfunc(hlr)
        return self.func_list[uid]

    def get_pivot_z(self):

        func = self.mk_1Dfunc(hlr=-99)
        pivot_z = scipy.integrate.quad(lambda z: z*func(z), 0.5, 5.0, limit=250, epsabs=1e-4, epsrel=1e-4)[0] / \
                  scipy.integrate.quad(lambda z:   func(z), 0.5, 5.0, limit=250, epsabs=1e-4, epsrel=1e-4)[0]
        return pivot_z

    def get_2Dpivot_z(self,M):

        _func = self.get_func(hlr=-99)
        func = lambda z: _func(z,M)
        pivot_z = scipy.integrate.quad(lambda z: z*func(z), 0.5, 5.0, limit=250, epsabs=1e-4, epsrel=1e-4)[0] / \
                  scipy.integrate.quad(lambda z:   func(z), 0.5, 5.0, limit=250, epsabs=1e-4, epsrel=1e-4)[0]
        return pivot_z

    def mk_2Dfunc(self,hlr,full=False):

        if hlr==-99:
            cond_hlr = np.ones(len(self.input),dtype=bool)
        else:
            cond_hlr = (self.round_hlr(self.input['hlr']) == self.round_hlr(hlr))
        
        _cat_input = self.input[cond_hlr]
        _cat_recov = self.recov[cond_hlr]

        if self.sample_type=='dropout':
            cond = mk_dropout_cuts(catalog=_cat_recov,drop_filt=self.drop_filt,
                                   verbose=False,return_cond=True)
        elif self.sample_type=='photoz':
            cond = mk_photoz_cuts(catalog=_cat_recov,drop_filt=self.drop_filt,
                                   verbose=False,return_cond=True)
        else: raise Exception("Invalid sample type in SelectionFunction().")

        cat_input = _cat_input
        cat_recov = _cat_input[cond]
        
        bins_z = np.arange(0,5,self.dz)
        binc_z = 0.5*(bins_z[1:]+bins_z[:-1])
        dbin_z = bins_z[1:]-bins_z[:-1]

        bins_M = scipy.stats.mstats.mquantiles(cat_input['abs_M'], np.linspace(0,1,self.nbinM))
        binc_M = 0.5*(bins_M[1:]+bins_M[:-1])
        dbin_M = bins_M[1:]-bins_M[:-1]

        hist_input = np.histogram2d(cat_input['z'],cat_input['abs_M'],bins=[bins_z,bins_M])[0]
        hist_recov = np.histogram2d(cat_recov['z'],cat_recov['abs_M'],bins=[bins_z,bins_M])[0]
        
        hist_comp = np.zeros(hist_input.shape)
        if   self.sample_type=='dropout': cond = (hist_input >= 5)
        elif self.sample_type=='photoz':  cond = (hist_input >= 10)
        hist_comp[cond] = hist_recov[cond] / hist_input[cond].astype(float)

        ypos, xpos = np.meshgrid(binc_M,binc_z)
        yedges, xedges = np.meshgrid(bins_M,bins_z)
        dx = xedges[1:,1:] - xedges[:-1,:-1]
        dy = yedges[1:,1:] - yedges[:-1,:-1]

        newx = np.arange(0.5,5.01,self.dz)
        newy = np.arange(-25,-12.01,0.2)
        ngry, ngrx = np.meshgrid(newy,newx)
        ix = np.clip(np.digitize(ngrx,bins=bins_z),1,len(bins_z)-1) - 1
        iy = np.clip(np.digitize(ngry,bins=bins_M),1,len(bins_M)-1) - 1
        ngrz = hist_comp[ix,iy]
        _func = scipy.interpolate.RectBivariateSpline(newx,newy,ngrz,kx=1,ky=2,s=self.smooth,bbox=[newx[0],newx[-1],newy[0],newy[-1]])
        func = lambda *args,**kwargs: np.clip(_func(*args,**kwargs),0.,1.)

        if full:
            return {'func':func,
                    'bins_x':bins_z,'bins_y':bins_M,
                    'binc_x':binc_z,'binc_y':binc_M,
                    'xedges':xedges,'yedges':yedges,
                    'input':hist_input,
                    'recov':hist_recov,
                    'comp':hist_comp}
        else:
            return func

    def mk_1Dfunc(self,hlr,full=False):

        if hlr==-99.:
            cond_hlr = np.ones(len(self.input),dtype=bool)
        else:
            cond_hlr = (self.round_hlr(self.input['hlr']) == self.round_hlr(hlr))

        _cat_input = self.input[cond_hlr]

        if self.sample_type=='dropout':
            cond = mk_dropout_cuts(catalog=_cat_input,drop_filt=self.drop_filt,
                                   calc_sn=False,do_sn_cut=False,
                                   verbose=False,return_cond=True)
        elif self.sample_type=='photoz':
            cond = mk_photoz_cuts(catalog=_cat_input,drop_filt=self.drop_filt,zlabel='z',
                                   calc_sn=False,do_sn_cut=False,
                                   verbose=False,return_cond=True)
        else: raise Exception("Invalid sample type in SelectionFunction().")

        cat_input = _cat_input
        cat_recov = _cat_input[cond]

        bins_z = np.arange(0,5,self.dz)
        binc_z = 0.5*(bins_z[1:]+bins_z[:-1])
        dbin_z = bins_z[1:]-bins_z[:-1]

        hist_input = np.histogram(cat_input['z'],bins=bins_z)[0]
        hist_recov = np.histogram(cat_recov['z'],bins=bins_z)[0]
        
        hist_comp = np.zeros(hist_input.shape)
        cond = (hist_input >= 10)
        hist_comp[cond] = hist_recov[cond] / hist_input[cond].astype(float)

        _func = scipy.interpolate.UnivariateSpline(binc_z,hist_comp,k=3,s=0)
        func = lambda *args: np.clip(_func(*args),0,1)

        if full:
            return {'func':func,
                    'bins':bins_z,
                    'binc':binc_z,
                    'input':hist_input,
                    'recov':hist_recov,
                    'comp':hist_comp}
        else:
            return func

    def plot(self,hlr):

        fig,axes = plt.subplots(2,3,figsize=(15,8),dpi=75,sharex=True)
        fig.subplots_adjust(left=0.08,right=0.98,bottom=0.08,top=0.92,hspace=0.0)
        hlr_title = " -- HLR=%.2f" % hlr if hlr else " -- all HLRs"
        fig.suptitle("%s %s sample%s" % (self.drop_filt.upper(),self.sample_type,hlr_title))
        
        axes = axes.flatten()
        axes[2].set_visible(False)
        axes = np.delete(axes,2)
        
        output = self.mk_2Dfunc(hlr=hlr,full=True)
        im1 = axes[0].pcolormesh(output['xedges'],output['yedges'],np.ma.masked_array(output['input'],mask=output['input']==0),cmap=plt.cm.viridis,vmin=1,vmax=np.max(output['input']))
        im1 = axes[1].pcolormesh(output['xedges'],output['yedges'],np.ma.masked_array(output['recov'],mask=output['recov']==0),cmap=plt.cm.viridis,vmin=1,vmax=np.max(output['input']))
        im2 = axes[2].pcolormesh(output['xedges'],output['yedges'],np.ma.masked_array(output['comp'],mask=output['input']==0),cmap=plt.cm.inferno,vmin=0,vmax=1)
    
        xx = np.linspace(np.min(output['xedges']),np.max(output['xedges']),250)
        yy = np.linspace(np.min(output['yedges']),np.max(output['yedges']),250)
        gy,gx = np.meshgrid(yy,xx)
        gz = output['func'](xx,yy)
        im2 = axes[3].pcolormesh(gx,gy,np.ma.masked_array(gz,mask=gz<0),cmap=plt.cm.inferno,vmin=0,vmax=1)

        cbar1 = fig.add_axes([0.72,0.55,0.03,0.35])
        cbax1 = fig.colorbar(mappable=im1,cax=cbar1,orientation='vertical')
        cbax1.ax.set_ylabel('Number')

        cbar2 = fig.add_axes([0.85,0.55,0.03,0.35])
        cbax2 = fig.colorbar(mappable=im2,cax=cbar2,ticks=[0,0.2,0.4,0.6,0.8,1],orientation='vertical')
        cbax2.ax.set_ylabel('Sel. Func.')
        
        zz = np.arange(0.5,5,0.01)
        Mrange = np.arange(-21,-14,1.0)
        crange = plt.cm.rainbow(np.linspace(0,0.9,len(Mrange)))
        for M,c in zip(Mrange,crange):
            axes[4].plot(zz,output['func'](zz,M),c=c,lw=1.5,label="M=%.1f"%M)
            axes[4].axvline(self.get_2Dpivot_z(M),c=c,lw=1.5,ls='--',alpha=0.8)
        axes[4].axvline(self.get_pivot_z(),c='k',lw=1.5,ls='--',alpha=0.8)
        
        #output = selfn.mk_1Dfunc(hlr=hlr,full=True)
        #axes[4].plot(output['binc'],output['comp'],c='k',lw=2)
        axes[4].hlines([0,1],0,10,color='k',linestyles=':')
        axes[4].set_ylim(-0.09,1.09)
        axes[4].set_xlim(0.55,4.25)
        for ax in axes[:-1]: ax.set_ylim(-14.1,-21.9)
        axes[3].set_xlabel("Redshift")
        axes[2].set_ylabel("Abs. Mag.")
        
        leg = axes[4].legend(fontsize=14,loc="best",frameon=False,handlelength=0,handletextpad=0)
        for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
            txt.set_color(hndl.get_color())
            hndl.set_visible(False)

def mk_pretty_plot(drop_filt,sample_type,title,plot_type='hist'):

    selfn = SelectionFunction(drop_filt=drop_filt,sample_type=sample_type)
    selfn_output = selfn.mk_2Dfunc(hlr=-99.,full=True)
    mag_range = [-20,-19,-18,-17,-16]
    colors = ['b','c','limegreen','r','m']

    if   drop_filt == 'f225w':
        mag_range = mag_range
        if sample_type=='dropout': mag_range = mag_range[:-2]
        xlim = [0.5,2.35]
    elif drop_filt == 'f275w':
        mag_range = mag_range
        if sample_type=='dropout': mag_range = mag_range[:-1]
        xlim = [0.9,3.0]
    elif drop_filt == 'f336w':
        mag_range = mag_range
        xlim = [0.9,4.1]

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,12),dpi=75,tight_layout=False,sharex=True)
    fig.subplots_adjust(bottom=0.15,left=0.13,top=0.95,right=0.95,hspace=0.)
    cbar = fig.add_axes([0.13,0.06,0.82,0.02])

    gx = np.arange(0,5,0.01)
    for M,c in zip(mag_range,colors):
        gz = selfn_output['func'](gx,M,grid=False)
        ax1.plot(gx,gz,c=c,lw=2,label='M$_{UV}$ = %i'%M)
        #ax1.axvline(selfn.get_2Dpivot_z(M=M),c=c,lw=1.5,ls='--',alpha=0.8)
    #ax1.axvline(selfn.get_pivot_z(),c='k',lw=1.5,ls='--',alpha=0.8)

    if   'hist' in plot_type:
        im = ax2.pcolormesh(selfn_output['xedges'],selfn_output['yedges'],selfn_output['comp'],lw=0,alpha=1.0,vmin=0,vmax=1,cmap=plt.cm.Greys,rasterized=True)
    elif 'func' in plot_type:
        xx = np.linspace(np.min(selfn_output['xedges']),np.max(selfn_output['xedges']),500)
        yy = np.linspace(np.min(selfn_output['yedges']),np.max(selfn_output['yedges']),500)
        gy,gx = np.meshgrid(yy,xx)
        gz = selfn_output['func'](xx,yy)
        im = ax2.pcolormesh(gx,gy,gz,lw=0,alpha=1.0,vmin=0,vmax=1,cmap=plt.cm.Greys,rasterized=True)
    else:
        raise Exception("Invalid plot type.")
    
    cbax = fig.colorbar(mappable=im, cax=cbar, ticks=[0,0.2,0.4,0.6,0.8,1.0], orientation='horizontal')
    cbax.ax.set_xlabel('Relative Efficiency',fontsize=24)
    cbax.ax.tick_params(labelsize=20)

    ax1.set_ylim(0,1)
    _ = [label.set_visible(False) for label in ax1.get_xticklabels()]
    _ = [label.set_fontsize(16) for label in ax1.get_yticklabels()]
    ax1.set_ylabel("Relative Efficiency",fontsize=24)
    ax1.legend(fontsize=16,loc=2)

    _ = [label.set_fontsize(16) for label in ax2.get_xticklabels()+ax2.get_yticklabels()]
    ax2.set_xlim(xlim)
    ax2.set_ylim(-14.8,-20.9)
    ax2.set_xlabel('Redshift',fontsize=24)
    ax2.set_ylabel('Absolute UV Magnitude [AB]',fontsize=24)
    _ = [label.set_fontsize(20) for label in ax1.get_xticklabels()+ax1.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()]

    ax1.text(0.93,0.95,title,va='top',ha='right',fontsize=18,fontweight=600,transform=ax1.transAxes)

    for ax in [ax1,ax2]:
        ax.minorticks_on()
        ax.tick_params(which='major', length=5, width='1')
        ax.tick_params(which='minor', length=3, width='1')

    fig.suptitle("%s %s sample" % (drop_filt.upper(),sample_type.capitalize()),fontsize=24,fontweight=800)
    fig.savefig('plots/selfn_%s_%s.png' % (drop_filt,sample_type))
    fig.savefig('plots/selfn_%s_%s.pdf' % (drop_filt,sample_type))

if __name__ == '__main__':
    
    # for f in ['f225w','f275w','f336w'][1:-1]:
    #     for s in ['dropout','photoz'][:1]:
    #         selfn = SelectionFunction(drop_filt=f,sample_type=s)
    #         for h in [-99.,2.,4.,8.][:1]:
    #             selfn.plot(hlr=h)
    # plt.show()

    for f,title_drop,title_phot in zip(['f225w','f275w','f336w'],['z~1.65','z~2.2','z~3'],['1.4<z<1.9','1.8<z<2.6','2.4<z<3.6']):
        for s in ['dropout','photoz']:
            title = title_drop if s=='dropout' else title_phot
            mk_pretty_plot(drop_filt=f,sample_type=s,title=title)
        #plt.show()