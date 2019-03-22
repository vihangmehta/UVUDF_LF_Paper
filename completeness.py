import numpy as np
import scipy.stats
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import useful
from uvudf_utils import pixscale, filt_key, dfilt_key, read_simulation_output
from conversions import get_app_from_abs, calc_sn
from hlr_transform import HLR_Transform

catalog_input, catalog_recov, catalog_recov_hlr = read_simulation_output(run0=False,run7=True,run9=True)

class Comp_Func_1D():

    def __init__(self):

        self.comp_func_list = {}
        self.limit_list = {}

    def get(self,filt,sn_cut):

        if filt=='f606w' and sn_cut<5.:
            print "Warning: F606W for SN < 5 (%.1f provided) is not well constrained." % sn_cut
        if filt not in ['f225w','f275w','f336w','f435w','f606w']:
            print "Warning: %s completeness is not constrained." % filt.upper()

        uid = filt+'-'+str(sn_cut)
        if uid not in self.comp_func_list:
            self.comp_func_list[uid] = self.mk_func(filt,sn_cut)
        return self.comp_func_list[uid]

    def eval(self,app_mag=None,abs_mag=None,z=None,filt=None,sn_cut=None,func=None):

        if not app_mag:
            app_mag = get_app_from_abs(abs_mag,z)

        if not func:
            func = self.get(filt,sn_cut)

        res = func(app_mag)
        res = 0.00 if res<0 else res
        res = 1.00 if res>1 else res
        res = 0.00 if app_mag>31 else res
        return res

    def get_limit(self,filt,sn_cut):

        uid = filt+'-'+str(sn_cut)
        if uid not in self.limit_list:
            self.limit_list[uid] = self.mk_limit(filt,sn_cut)
        return self.limit_list[uid]

    def mk_limit(self,filt,sn_cut):

        func = self.get(filt,sn_cut)
        limit = scipy.optimize.brentq(lambda x: func(x) - 0.5, 25, 31)
        return limit

    def mk_func(self,filt,sn_cut,full=False):

        if   filt in ['f225w','f275w','f336w']: _s=75#30
        elif filt in ['f435w',]: _s=25#20
        elif filt in ['f606w',]: _s=25#25

        sn = calc_sn(catalog_recov[filt_key[filt]], catalog_recov[dfilt_key[filt]])

        input_mags = catalog_input[filt_key[filt]]
        recov_mags = catalog_input[filt_key[filt]][sn >= sn_cut]

        # binsize = len(input_mags[input_mags<32]) / _s
        # bins = np.sort(input_mags)[::binsize]
        # if bins[-1] != np.max(input_mags): bins = np.append(bins,np.max(input_mags))
        bins = scipy.stats.mstats.mquantiles(input_mags, np.linspace(0,1,_s))
        binc = 0.5*(bins[1:]+bins[:-1])
        dbin = bins[1:]-bins[:-1]

        hist_input, _ = np.histogram(input_mags, bins=bins)
        hist_recov, _ = np.histogram(recov_mags, bins=bins)
        hist_comp = hist_recov / hist_input.astype(float)
        # Artificially putting the max recovered as 100%
        #hist_comp = hist_comp / max(hist_comp)

        ext = np.arange(15,binc[0],0.5)
        fit_x = np.sort(np.concatenate((ext,binc)))
        fit_y = np.zeros(len(fit_x))
        fit_y[len(ext):] = hist_comp
        fit_y[fit_x<=24] = 1
        if sn_cut>5: hist_comp[binc>=30.6] = 0
        func  = scipy.interpolate.UnivariateSpline(fit_x,fit_y,k=2,s=0.002)

        if full: return func, (binc, hist_input/dbin, hist_recov/dbin, hist_comp)
        else: return func

    def mk_plot(self,filts=['f225w','f275w','f336w','f435w','f606w'],sn_range=[1,3,5],savename=None):

        if type(filts) is not list: filts = [filts,]
        if type(sn_range) is not list: sn_range = [sn_range,]
        fig, axes = plt.subplots(2,len(filts),figsize=(5*len(filts),8),dpi=75,sharey='row',sharex=True,tight_layout=True)
        axes = axes.reshape(2,-1)
        x = np.arange(20,35,0.01)

        for i,f in enumerate(filts):
            for sn_cut,c in zip(sn_range,['r','g','b','m','y','c']):

                func, (binc, hist_input, hist_recov, hist_comp) = self.mk_func(f,sn_cut,full=True)
                axes[0,i].step(binc, hist_recov, c=c, lw=1.5, where='mid', label='SN>%.1f'%sn_cut)
                if sn_cut == sn_range[-1]:
                    axes[0,i].step(binc, hist_input, c='k', lw=1.5, where='mid', label='Input')
                axes[1,i].step(binc, hist_comp, c=c, lw=1.5, where='mid', label='SN>%.1f'%sn_cut)
                axes[1,i].plot(x, func(x), c=c, lw=1.5)
                axes[1,i].axvline(self.get_limit(f,sn_cut), c=c, lw=1.5, ls='--')
                axes[1,i].set_xlabel(f+' mag')
                axes[1,i].set_xlim(23.5,31.5)
                axes[0,i].legend(fontsize=10, loc=2, fancybox=True, framealpha=0)

        axes[0,0].set_ylabel('N dM')
        axes[0,0].set_ylim(0,1.1*np.max(hist_input))
        axes[1,0].set_ylabel('Completeness')
        axes[1,0].set_ylim(0,1)

        if savename: fig.savefig(savename)

class Comp_Func_2D():

    def __init__(self):

        self.comp_func_list = {}
        self.eval = np.vectorize(self.eval,excluded=['self','sn_cut','func'])

    def get(self,filt,sn_cut):

        uid = filt+'-'+str(sn_cut)
        if uid not in self.comp_func_list:
            self.comp_func_list[uid] = self.mk_func(filt,sn_cut)
        return self.comp_func_list[uid]

    def eval(self,app_mag=None,abs_mag=None,z=None,hlr=None,filt=None,sn_cut=None,func=None):

        if not app_mag:
            app_mag = get_app_from_abs(abs_mag,z)

        if not func:
            func = self.get(filt,sn_cut)

        res = func(app_mag,hlr)[0,0]
        res = 0.00 if res < 0 else res
        res = 1.00 if res > 1 else res
        res = 0.00 if app_mag>31 else res
        return res

    def get_2dlimit(self,hlr,filt,sn_cut):

        func = self.get(filt,sn_cut)
        limit = scipy.optimize.brentq(lambda x: func(x,hlr) - 0.5, 25, 30.5)
        return limit

    def mk_func(self,filt,sn_cut,full=False):

        if   filt in ['f225w','f275w']: _s=75
        elif filt in ['f336w',]: _s=50#30
        elif filt in ['f435w',]: _s=30#25
        elif filt in ['f606w',]: _s=30#25

        filt,dfilt = filt_key[filt], dfilt_key[filt]
        sn = calc_sn(catalog_recov[filt], catalog_recov[dfilt])

        # Input Catalog
        input_mags = catalog_input[filt]
        input_hlrs = catalog_input['hlr']
        # Entries from the input catalog that were recovered
        recov_mags = catalog_input[filt][sn >= sn_cut]
        recov_hlrs = catalog_input['hlr'][sn >= sn_cut]

        # Bins in the Magnitude dimension
        # binsize_x = len(input_mags[input_mags<32]) / _s
        # bins_x = np.sort(input_mags)[::binsize_x]
        # if bins_x[-1] != np.max(input_mags): bins_x = np.append(bins_x,np.max(input_mags))
        bins_x = scipy.stats.mstats.mquantiles(input_mags, np.linspace(0,1,_s))
        binc_x = 0.5*(bins_x[1:]+bins_x[:-1])
        dbin_x = bins_x[1:]-bins_x[:-1]

        # Bins in the HLR dimension
        # binsize_y = len(input_hlrs) / 10
        # bins_y = np.sort(input_hlrs)[::binsize_y]
        # if bins_y[-1] != np.max(input_hlrs): bins_y = np.append(bins_y,np.max(input_hlrs))
        bins_y = scipy.stats.mstats.mquantiles(input_hlrs, np.linspace(0,1,10))
        binc_y = 0.5*(bins_y[1:]+bins_y[:-1])
        dbin_y = bins_y[1:]-bins_y[:-1]

        hist_input = np.histogram2d(input_mags, input_hlrs, bins=[bins_x,bins_y])[0]
        hist_recov = np.histogram2d(recov_mags, recov_hlrs, bins=[bins_x,bins_y])[0]

        yedges, xedges = np.meshgrid(bins_y,bins_x)
        dx = xedges[1:,1:] - xedges[:-1,:-1]
        dy = yedges[1:,1:] - yedges[:-1,:-1]

        hist_comp = hist_recov / hist_input.astype(float)
        hist_comp = hist_comp - np.min(hist_comp)
        hist_comp = hist_comp / np.max(hist_comp)

        ypos, xpos = np.meshgrid(binc_y,binc_x)
        hist_comp[xpos>=32] = 0
        hist_comp[hist_input<10] = 0
        hist_comp[(hist_recov<25) & (hist_input<100) & (xpos>31)] = 0

        extend = np.arange(15,22,0.5)
        binc_x = np.insert(binc_x,0,extend)
        ypos, xpos = np.meshgrid(binc_y,binc_x)
        zpos = np.zeros(xpos.shape)
        zpos[len(extend):,:] = hist_comp
        zpos[:len(extend),:] = 1

        #func = scipy.interpolate.CloughTocher2DInterpolator((xpos.ravel(),ypos.ravel()),zpos.ravel())
        #func = scipy.interpolate.LinearNDInterpolator((xpos.ravel(),ypos.ravel()),zpos.ravel())
        func = scipy.interpolate.RectBivariateSpline(binc_x,binc_y,zpos,kx=1,ky=1,s=0.3,bbox=[binc_x[0],binc_x[-1],binc_y[0],binc_y[-1]])

        if full: return func, (xedges, yedges, hist_input/dx/dy, hist_recov/dx/dy, hist_comp)
        else: return func

    def mk_plot(self,filt,sn_cut,savename=None):

        func, (xedges, yedges, hist_input, hist_recov, hist_comp) = self.mk_func(filt,sn_cut=sn_cut,full=True)

        fig = plt.figure(figsize=(16,12),dpi=75,tight_layout=True)

        ax1 = fig.add_subplot(221)
        cbar1 = ax1.pcolormesh(xedges,yedges,hist_input,lw=0,alpha=0.8,vmin=0,vmax=np.max(hist_input))
        plt.colorbar(cbar1)

        ax2 = fig.add_subplot(222)
        cbar2 = ax2.pcolormesh(xedges,yedges,hist_recov,lw=0,alpha=0.8,vmin=0,vmax=np.max(hist_input))
        plt.colorbar(cbar2)

        ax3 = fig.add_subplot(223)
        cbar3 = ax3.pcolormesh(xedges,yedges,hist_comp ,lw=0,alpha=0.8,vmin=0,vmax=1)
        plt.colorbar(cbar3)

        ax4 = fig.add_subplot(224)
        gx,gy = np.mgrid[18:40:250j,np.min(yedges):np.max(yedges):250j]
        gz = np.array([func(x,y)[0,0] for x,y in zip(gx.ravel(),gy.ravel())])
        gz = gz.reshape(gx.shape)
        gz[(1<gz)|(gz<0)] = np.NaN
        cbar4 = ax4.imshow(gz.T,origin='lower',extent=[np.min(gx),np.max(gx),np.min(gy),np.max(gy)],interpolation='none',alpha=0.8,vmin=0,vmax=1,aspect='auto')
        plt.colorbar(cbar4)

        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_xlim(23,30)
            ax.set_ylim(0.25,9.8)
            ax.set_xlabel('%s MAG' % filt)
            ax.set_ylabel('Input HLR')

        if savename: fig.savefig(savename)

    def mk_plot3d(self,filt,sn_cut):

        func, (xedges, yedges, hist_input, hist_recov, hist_comp) = self.mk_func(filt,sn_cut=sn_cut,full=True)
        gx,gy = np.mgrid[18:40:250j,np.min(yedges):np.max(yedges):250j]
        gz = np.array([func(x,y)[0,0] for x,y in zip(gx.ravel(),gy.ravel())])
        gz = gz.reshape(gx.shape)
        gz[gz<0] = np.NaN

        fig = plt.figure(figsize=(12,10),dpi=75,tight_layout=True)
        ax3d = fig.add_subplot(111,projection='3d')
        ax3d.plot_surface(gx,gy,gz,color='r',edgecolor='r',antialiased=True,alpha=0.5,lw=1.,shade=False)
        ax3d.scatter(0.5*(xedges[1:,1:]+xedges[:-1,:-1]),0.5*(yedges[1:,1:]+yedges[:-1,:-1]),hist_comp,lw=0)
        ax3d.set_zlim(0,1)
        ax3d.set_title(filt)
        ax3d.set_zlim(0,1)

def mk_pretty_plot(filt, sn_cut):

    tf  = HLR_Transform()
    c1s  = Comp_Func_1D()
    c2d = Comp_Func_2D()

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,12),dpi=75,tight_layout=False,sharex=True)
    fig.subplots_adjust(bottom=0.14,left=0.1,top=0.95,right=0.88,hspace=0.)
    cbar = fig.add_axes([0.13,0.05,0.78,0.02])

    func, (binc, hist_input, hist_recov, hist_comp) = c1s.mk_func(filt,sn_cut,full=True)
    x = np.arange(15,35,0.05)
    ax1.plot(binc, hist_comp, c='k', lw=3, drawstyle='steps-mid')
    #ax1.plot(x, func(x), c='k', lw=3)
    ax1.axvline(c1s.get_limit(filt,sn_cut), c='k', lw=2, ls='--')
    ax1.text(0.95,0.95,'SN>%.1f'%sn_cut,fontsize=20,va='top',ha='right',transform=ax1.transAxes)

    ax1.set_xlim(23.5,31.2)
    ax1.set_ylim(0,1)
    _ = [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.set_xlabel(filt.upper()+' Magnitude [AB]')
    ax1.set_ylabel('Completeness',fontsize=20)

    func, (xedges, yedges, hist_input, hist_recov, hist_comp) = c2d.mk_func(filt,sn_cut=sn_cut,full=True)
    # gx,gy = np.mgrid[18:35:500j,np.min(yedges):np.max(yedges):500j]
    # gz = np.reshape(func(gx.ravel(),gy.ravel(),grid=False), gx.shape)
    # gy = tf.transform(gy)
    yedges = tf.transform(yedges)
    ax2.text(0.95,0.95,'SN>%.1f'%sn_cut,fontsize=20,va='top',ha='right',transform=ax2.transAxes)
    im = ax2.pcolormesh(xedges,yedges,hist_comp,lw=0,alpha=1,vmin=0,vmax=1,cmap=plt.cm.Greys)
    cbax = fig.colorbar(mappable=im, cax=cbar, ticks=[0,0.2,0.4,0.6,0.8,1.0], orientation='horizontal')
    cbax.ax.set_xlabel('Completeness',fontsize=20)
    cbax.ax.tick_params(labelsize=16)

    ax2.set_xlim(23.5,31.2)
    ax2.set_ylim(2.1,7.9)
    ax2.set_xlabel(filt.upper()+' Magnitude [AB]',fontsize=20)
    ax2.set_ylabel('F435W HLR [px]',fontsize=20)

    ax2x = ax2.twinx()
    ax2x.set_ylim(ax2.get_ylim()[0]*pixscale,ax2.get_ylim()[1]*pixscale)
    ax2x.set_ylabel('F435W HLR [arcsec]',fontsize=20)

    for ax in [ax1,ax2]:
        ax.minorticks_on()
        ax.tick_params(which='major', length=5, width='1')
        ax.tick_params(which='minor', length=3, width='1')

    _ = [label.set_fontsize(16) for label in ax2x.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()+ax1.get_yticklabels()]
    fig.suptitle(filt.upper()+' Completeness Function',fontsize=25)
    fig.savefig('plots/comp_%s_%.1fsig.png'%(filt,sn_cut))

def mk_ppretty_plot(sn_cut=5):

    tf  = HLR_Transform()
    c1s = Comp_Func_1D()
    c2d = Comp_Func_2D()

    fig = plt.figure(figsize=(8,14),dpi=75)
    ggs  = gridspec.GridSpec(2,2)
    ggs.update(left=0.1,right=0.88,top=0.98,bottom=0.14,wspace=0,hspace=0.18)

    for i,filt in enumerate(['f275w','f336w','f435w','f606w']):

        gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=ggs[i],wspace=0,hspace=0)
        ax1 = plt.Subplot(fig,gs[0])
        ax2 = plt.Subplot(fig,gs[1])

        fig.add_subplot(ax1)
        fig.add_subplot(ax2)

        func, (binc, hist_input, hist_recov, hist_comp) = c1s.mk_func(filt,sn_cut,full=True)
        ax1.plot(binc, hist_comp, c='k', lw=2.5, drawstyle='steps-mid')
        ax1.axvline(c1s.get_limit(filt,sn_cut), c='k', lw=2, ls='--')
        if i in [0,1]: _x,_y,va,ha = 0.92,0.92,'top','right'
        else: _x,_y,va,ha = 0.08,0.08,'bottom','left'
        ax1.text(_x,_y,'%s\nSN>%.1f'%(filt.upper(),sn_cut),fontsize=18,fontweight=600,va=va,ha=ha,transform=ax1.transAxes)

        ax1.set_ylim(0,1)
        ax1.set_xticklabels([])
        #ax1.set_xlabel(filt.upper()+' Magnitude [AB]')
        if i in [0,2]: ax1.set_ylabel('Comp.',fontsize=18)
        else: ax1.set_yticklabels([])

        func, (xedges, yedges, hist_input, hist_recov, hist_comp) = c2d.mk_func(filt,sn_cut=sn_cut,full=True)
        # gx,gy = np.mgrid[18:35:500j,np.min(yedges):np.max(yedges):500j]
        # gz = np.reshape(func(gx.ravel(),gy.ravel(),grid=False), gx.shape)
        # gy = tf.transform(gy)
        yedges = tf.transform(yedges)
        #ax2.text(0.95,0.95,'SN>%.1f'%sn_cut,fontsize=20,va='top',ha='right',transform=ax2.transAxes)
        im = ax2.pcolormesh(xedges,yedges,hist_comp,lw=0,alpha=1,vmin=0,vmax=1,cmap=plt.cm.Greys,rasterized=True)

        ax2.set_ylim(2.1,7.9)
        ax2.set_xlabel(filt.upper()+' Magnitude [AB]',fontsize=18)
        if i in [0,2]: ax2.set_ylabel('F435W HLR [px]',fontsize=18)
        else: ax2.set_yticklabels([])

        ax2x = ax2.twinx()
        ax2x.set_ylim(ax2.get_ylim()[0]*pixscale,ax2.get_ylim()[1]*pixscale)
        ax2x.set_yticks([0.09,0.12,0.15,0.18,0.21])
        if i in [1,3]: ax2x.set_ylabel('F435W HLR [arcsec]',fontsize=18)
        else: ax2x.set_yticklabels([])
        _ = [label.set_fontsize(16) for label in ax2x.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()+ax1.get_yticklabels()]

        for ax in [ax1,ax2]:
            ax.set_xlim(24.5,30.5)
            ax.minorticks_on()
            #ax.tick_params(which='major', length=5, width='1')
            #ax.tick_params(which='minor', length=3, width='1')

    cbar = fig.add_axes([0.2,0.05,0.6,0.02])
    cbax = fig.colorbar(mappable=im, cax=cbar, ticks=[0,0.2,0.4,0.6,0.8,1.0], orientation='horizontal')
    cbax.ax.set_xlabel('Completeness',fontsize=18)
    cbax.ax.tick_params(labelsize=16)

    fig.savefig('plots/comp_func.png')
    fig.savefig('plots/comp_func.pdf')

if __name__ == '__main__':

    # c1d = Comp_Func_1D()
    # c1d.mk_plot()

    c2d = Comp_Func_2D()
    # c2d.mk_plot(  filt='f275w',sn_cut=5.0)
    # c2d.mk_plot(  filt='f336w',sn_cut=5.0)
    c2d.mk_plot(  filt='f435w',sn_cut=5.0)
    # c2d.mk_plot(  filt='f606w',sn_cut=5.0)
    # c2d.mk_plot3d(filt='f275w',sn_cut=5.0)
    # c2d.mk_plot3d(filt='f336w',sn_cut=5.0)
    c2d.mk_plot3d(filt='f435w',sn_cut=5.0)
    # c2d.mk_plot3d(filt='f606w',sn_cut=5.0)

    # mk_pretty_plot(filt='f225w',sn_cut=1.)
    # mk_pretty_plot(filt='f275w',sn_cut=1.)
    # mk_pretty_plot(filt='f336w',sn_cut=1.)
    # mk_pretty_plot(filt='f435w',sn_cut=1.)
    # mk_pretty_plot(filt='f606w',sn_cut=1.)

    # mk_pretty_plot(filt='f225w',sn_cut=5.)
    # mk_pretty_plot(filt='f275w',sn_cut=5.)
    # mk_pretty_plot(filt='f336w',sn_cut=5.)
    # mk_pretty_plot(filt='f435w',sn_cut=5.)
    # mk_pretty_plot(filt='f606w',sn_cut=5.)

    # mk_ppretty_plot()
    plt.show()
