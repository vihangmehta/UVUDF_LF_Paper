import numpy as np
import scipy.stats
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import corner

from LF_refs import M16_LF_pars

def get_binsize(data):
    # Freedman-Diaconis Rule
    return 2*(np.percentile(data,75)-np.percentile(data,25)) / len(data)**(1./3.)

class MCMC_Output():

    def __init__(self,drop_filt,sample_type,fname=None,best_pars=None,
                 nwalkers=10,ndim=3,burnin=200,
                 labels=['$\\alpha$','M$^\\star$','$\\phi^\\star$'],
                 verbose=False):

        self.drop_filt   = drop_filt
        self.sample_type = sample_type
        self.fname       = fname if fname else "output/mcmc_%s_%s.dat" % (drop_filt,sample_type[:4])
        self.verbose     = verbose
        self.best_pars   = np.array(best_pars) if best_pars else np.array(M16_LF_pars["%s_%s"%(drop_filt,sample_type[:4])]['coeff'])

        self.nwalkers = 10
        self.ndim     = 3
        self.burnin   = burnin
        self.labels   = labels
        
        self.chain    = np.genfromtxt(self.fname)
        self.setup()

    def setup(self):

        self.chain2d_full = self.chain
        self.chain3d_full = self.chain.reshape((-1,self.nwalkers,3))
        self.chain2d = self.chain3d_full[self.burnin:,:,:].reshape((-1,3))
        self.chain3d = self.chain3d_full[self.burnin:,:,:]
        self.steps   = self.chain3d_full.shape[0]

        self.mode3D,self.conf_ints3D,self.errors3D = self.get_3D_confidence_intervals()
        self.mode1D,self.conf_ints1D,self.errors1D = self.get_1D_confidence_intervals()

    def get_3D_confidence_intervals(self):

        xdata, ydata, zdata = self.chain2d.T

        limx = [min(xdata), max(xdata)]
        limy = [min(ydata), max(ydata)]
        limz = [min(zdata), max(zdata)]

        dbinx = get_binsize(xdata)
        dbiny = get_binsize(ydata)
        dbinz = get_binsize(zdata)

        binsx = np.arange(limx[0],limx[1],dbinx)
        binsy = np.arange(limy[0],limy[1],dbiny)
        binsz = np.arange(limz[0],limz[1],dbinz)

        bincx = 0.5*(binsx[1:]+binsx[:-1])
        bincy = 0.5*(binsy[1:]+binsy[:-1])
        bincz = 0.5*(binsz[1:]+binsz[:-1])

        hist = np.histogramdd(self.chain2d,bins=[binsx,binsy,binsz])[0]
        hist = hist / dbinx / dbiny / dbinz / self.chain2d.shape[0]

        def find_crit(x,c):
            _hist = hist.copy()
            _hist[hist < x] = 0
            res = scipy.integrate.simps(
                    scipy.integrate.simps(
                      scipy.integrate.simps(_hist,dx=dbinz,axis=2),
                                                    dx=dbiny,axis=1),
                                                      dx=dbinx,axis=0)
            return res - c

        crit68 = scipy.optimize.brentq(find_crit, 0, np.max(hist), args=(0.68,))
        crit95 = scipy.optimize.brentq(find_crit, 0, np.max(hist), args=(0.95,))

        ypos,xpos,zpos = np.meshgrid(bincy,bincx,bincz)
        cix = np.array([np.min(xpos[hist>=crit68]), np.max(xpos[hist>=crit68])])
        ciy = np.array([np.min(ypos[hist>=crit68]), np.max(ypos[hist>=crit68])])
        ciz = np.array([np.min(zpos[hist>=crit68]), np.max(zpos[hist>=crit68])])

        imax = np.argmax(hist.ravel())
        mode = np.array([xpos.ravel()[imax],ypos.ravel()[imax],zpos.ravel()[imax]])
        conf_ints = np.array([cix,ciy,ciz])
        errors = conf_ints - self.best_pars[:,np.newaxis]

        if self.verbose:
            print
            print "==== %s %s LF ====" % (self.drop_filt.upper(),self.sample_type.capitalize())
            print "-- 3D Confidence Intervals --"
            print "alpha: Fit=%8.3f Mode=%8.3f Err=[%8.3f,%8.3f] CI=[%8.3f,%8.3f]" % (self.best_pars[0],mode[0],errors[0][0],errors[0][1],conf_ints[0][0],conf_ints[0][1])
            print "Mst:   Fit=%8.3f Mode=%8.3f Err=[%8.3f,%8.3f] CI=[%8.3f,%8.3f]" % (self.best_pars[1],mode[1],errors[1][0],errors[1][1],conf_ints[1][0],conf_ints[1][1])
            print "phi:   Fit=%8.3f Mode=%8.3f Err=[%8.3f,%8.3f] CI=[%8.3f,%8.3f]" % (self.best_pars[2],mode[2],errors[2][0],errors[2][1],conf_ints[2][0],conf_ints[2][1])

        return mode, conf_ints, errors

    def get_1D_confidence_intervals(self):

        xdata, ydata, zdata = self.chain2d.T

        limx = [min(xdata), max(xdata)]
        limy = [min(ydata), max(ydata)]
        limz = [min(zdata), max(zdata)]

        dbinx = get_binsize(xdata)
        dbiny = get_binsize(ydata)
        dbinz = get_binsize(zdata)

        binsx = np.arange(limx[0],limx[1],dbinx)
        binsy = np.arange(limy[0],limy[1],dbiny)
        binsz = np.arange(limz[0],limz[1],dbinz)

        bincx = 0.5*(binsx[1:]+binsx[:-1])
        bincy = 0.5*(binsy[1:]+binsy[:-1])
        bincz = 0.5*(binsz[1:]+binsz[:-1])

        histx = np.histogram(xdata,bins=binsx)[0] / dbinx / len(xdata)
        histy = np.histogram(ydata,bins=binsy)[0] / dbiny / len(ydata)
        histz = np.histogram(zdata,bins=binsz)[0] / dbinz / len(zdata)

        cix = np.percentile(xdata,[16,84])
        ciy = np.percentile(ydata,[16,84])
        ciz = np.percentile(zdata,[16,84])

        mode = np.array([bincx[np.argmax(histx)],bincy[np.argmax(histy)],bincz[np.argmax(histz)]])
        conf_ints = np.array([cix,ciy,ciz])
        errors = conf_ints - self.best_pars[:,np.newaxis]

        if self.verbose:
            print "-- 1D Confidence Intervals --"
            print "alpha: Fit=%8.3f Mode=%8.3f Err=[%8.3f,%8.3f] CI=[%8.3f,%8.3f]" % (self.best_pars[0],mode[0],errors[0][0],errors[0][1],conf_ints[0][0],conf_ints[0][1])
            print "Mst:   Fit=%8.3f Mode=%8.3f Err=[%8.3f,%8.3f] CI=[%8.3f,%8.3f]" % (self.best_pars[1],mode[1],errors[1][0],errors[1][1],conf_ints[1][0],conf_ints[1][1])
            print "phi:   Fit=%8.3f Mode=%8.3f Err=[%8.3f,%8.3f] CI=[%8.3f,%8.3f]" % (self.best_pars[2],mode[2],errors[2][0],errors[2][1],conf_ints[2][0],conf_ints[2][1])
            print

        return mode, conf_ints, errors

    def plot_walkers(self,savename=None):

        fig,axes = plt.subplots(3,1,figsize=(12,6),dpi=75,sharex=True)
        fig.subplots_adjust(left=0.08,right=0.96,bottom=0.12,top=0.92,hspace=0,wspace=0)

        for j in range(self.ndim):
            for i in range(self.nwalkers):
                axes[j].plot(np.arange(self.steps)+1,self.chain3d_full[:,i,j],lw=0.5,alpha=0.8)
            axes[j].axhline(self.best_pars[j],c='k',lw=1.5,ls='--')
            axes[j].set_ylabel(self.labels[j])

        axes[2].set_xlabel('Step #')
        fig.suptitle("%s %s Sample" % (self.drop_filt.upper(),self.sample_type.capitalize()),fontsize=20)

        if savename: fig.savefig(savename)
        plt.show(block=False)

    def plot_corner(self,quantiles=[0.16,0.84],savename=None):

        fig = corner.corner(self.chain2d,
                            truths=self.best_pars,
                            labels=self.labels,
                            quantiles=quantiles,
                            show_titles=True,
                            title_kwargs={"fontsize":16},
                            label_kwargs={"fontsize":16})
        
        for ax in fig.get_axes():
            _ = [i.set_fontsize(14) for i in ax.get_xticklabels()]
            _ = [i.set_fontsize(14) for i in ax.get_yticklabels()]
        
        fig.text(0.9,0.9,"%s %s Sample" % (self.drop_filt.upper(),self.sample_type.capitalize()),
                  fontsize=20,va='top',ha='right',transform=fig.transFigure)
        fig.set_size_inches(10,8)

        if savename: fig.savefig(savename)
        plt.show(block=False)

    def plot_2D_contours(self,xdata,ydata,axis,c='k'):

        limx = min(xdata)-np.std(xdata),max(xdata)+np.std(xdata)
        limy = min(ydata)-np.std(ydata),max(ydata)+np.std(ydata)

        dbinx= get_binsize(xdata)*1.5
        dbiny= get_binsize(ydata)*1.5

        binsx = np.arange(limx[0],limx[1],dbinx)
        binsy = np.arange(limy[0],limy[1],dbiny)

        bincx = 0.5*(binsx[1:]+binsx[:-1])
        bincy = 0.5*(binsy[1:]+binsy[:-1])

        hist2d = np.histogram2d(xdata,ydata,bins=[binsx,binsy])[0]
        hist2d = hist2d / dbinx / dbiny / len(xdata)

        def find_crit(x,c):
            _hist2d = hist2d.copy()
            _hist2d[hist2d < x] = 0
            res = scipy.integrate.simps(
                    scipy.integrate.simps(_hist2d,x=bincy,axis=1),
                                                    x=bincx,axis=0)
            return res - c

        crit68 = scipy.optimize.brentq(find_crit, 0, np.max(hist2d), args=(0.68,))
        crit95 = scipy.optimize.brentq(find_crit, 0, np.max(hist2d), args=(0.95,))

        ypos, xpos = np.meshgrid(bincy,bincx)
        cix = np.array([np.min(xpos[hist2d>=crit68]), np.max(xpos[hist2d>=crit68])])
        ciy = np.array([np.min(ypos[hist2d>=crit68]), np.max(ypos[hist2d>=crit68])])

        axis.contour(bincx, bincy, hist2d.T,
                     levels=[crit95,crit68],
                     colors=[c,c],
                     linewidths=[0.5,1.5])

    def plot_corner2(self):

        fig,axes = plt.subplots(self.ndim,self.ndim,figsize=(10,8),dpi=75)
        fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0,wspace=0)
        fig.text(0.9,0.9,"%s %s Sample" % (self.drop_filt.upper(),self.sample_type.capitalize()),
                                            fontsize=20,va='top',ha='right',transform=fig.transFigure)

        for j in range(self.ndim):
            for i in range(self.ndim):
                if j<i: axes[j,i].set_visible(False)
                if j>0: axes[j,0].set_ylabel(self.labels[j])
                if j!=self.ndim-1: axes[j,i].set_xticklabels([])
                if i!=0: axes[j,i].set_yticklabels([])
                if i==j: axes[j,i].set_yticklabels([])
                axes[-1,i].set_xlabel(self.labels[i])
                axes[j,i].set_xlim(min(self.chain2d[:,i]),max(self.chain2d[:,i]))
                if i!=j: axes[j,i].set_ylim(min(self.chain2d[:,j]),max(self.chain2d[:,j]))
                _ = [i.set_fontsize(12) for i in axes[j,i].get_xticklabels()+axes[j,i].get_yticklabels()]

        for i,c in zip(range(self.ndim),['r','g','b']):
            hist,bins = axes[i,i].hist(self.chain2d[:,i],
                                  bins=np.arange(min(self.chain2d[:,i]),max(self.chain2d[:,i]),get_binsize(self.chain2d[:,i])),
                                  color=c,lw=0,histtype='stepfilled',alpha=0.5)[:2]
            binc = 0.5*(bins[1:]+bins[:-1])
            axes[i,i].vlines(self.conf_ints1D[i],0,1e4,linestyle=':',color=c,alpha=0.8)
            axes[i,i].axvline(self.best_pars[i],c=c,lw=1.5,alpha=0.8)
            axes[i,i].set_title("%s: $%.2f_{%.2f}^{+%.2f}$" % (self.labels[i],self.best_pars[i],self.errors1D[i][0],self.errors1D[i][1]))
            axes[i,i].set_ylim(0,1.2*np.max(hist))

        axes[1,0].scatter(self.chain2d[:,0],self.chain2d[:,1],c='r',s=3,lw=0,alpha=0.05)
        axes[2,0].scatter(self.chain2d[:,0],self.chain2d[:,2],c='b',s=3,lw=0,alpha=0.05)
        axes[2,1].scatter(self.chain2d[:,1],self.chain2d[:,2],c='g',s=3,lw=0,alpha=0.05)

        axes[1,0].scatter(self.best_pars[0],self.best_pars[1],c='r',marker='x',s=50,lw=2,alpha=1)
        axes[2,0].scatter(self.best_pars[0],self.best_pars[2],c='b',marker='x',s=50,lw=2,alpha=1)
        axes[2,1].scatter(self.best_pars[1],self.best_pars[2],c='g',marker='x',s=50,lw=2,alpha=1)

        self.plot_2D_contours(xdata=self.chain2d[:,0],ydata=self.chain2d[:,1],axis=axes[1,0],c='r')
        self.plot_2D_contours(xdata=self.chain2d[:,0],ydata=self.chain2d[:,2],axis=axes[2,0],c='b')
        self.plot_2D_contours(xdata=self.chain2d[:,1],ydata=self.chain2d[:,2],axis=axes[2,1],c='g')

        plt.show(block=False)

if __name__ == '__main__':
    
    # m = MCMC_Output(drop_filt='f275w',sample_type='dropout',verbose=True)
    # m = MCMC_Output(drop_filt='f336w',sample_type='dropout',verbose=True)
    
    # m = MCMC_Output(drop_filt='f225w',sample_type='photoz',verbose=True)
    m = MCMC_Output(drop_filt='f275w',sample_type='photoz',verbose=True)
    # m = MCMC_Output(drop_filt='f336w',sample_type='photoz',verbose=True)

    # m.plot_walkers()
    # m.plot_corner()
    m.plot_corner2()

    plt.show()