import sys
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from multiprocessing import Queue, Process
from functools32 import lru_cache

import useful
import uvudf_utils as utils
from selection import SelectionFunction

quad_args = {'limit':250,'epsrel':1e-4,'epsabs':1e-4}

class VEff_Func():

    def __init__(self,drop_filt,sample_type,lim_z=[0.5,5]):

        self.drop_filt   = drop_filt
        self.sample_type = sample_type
        self.lim_z       = lim_z
        self.selfn       = SelectionFunction(drop_filt=drop_filt,sample_type=sample_type)
        self.veff_func_list = {}

        self.__call__  = np.vectorize(self.__call__, excluded=['self',])
        self.calc_veff = np.vectorize(self.calc_veff,excluded=['self','hlr','lim_z'])
        self.calc_vol  = np.vectorize(self.calc_vol ,excluded=['self','lim_z'])
        self.mag_limit = np.vectorize(self.mag_limit,excluded=['self','frac_cut'])

    def calc_veff(self,M,hlr,lim_z=None):

        lim_z = self.lim_z if not lim_z else lim_z
        integrand = lambda z: useful.co_vol(z)*utils.get_sangle()*self.selfn(M,z,hlr=hlr)
        return scipy.integrate.quad(integrand,lim_z[0],lim_z[1],**quad_args)[0]

    def calc_vol(self,lim_z=None):

        lim_z = utils.bpz_lims[self.drop_filt] if not lim_z else lim_z
        integrand = lambda z: useful.co_vol(z)*utils.get_sangle()
        return scipy.integrate.quad(integrand,lim_z[0],lim_z[1],**quad_args)[0]

    def mag_limit(self,frac_cut=0.25,hlr=-99.):

        vtot = self.calc_vol()
        mag_limit = scipy.optimize.brentq(lambda M: self.__call__(M=M,    hlr=hlr) / vtot - \
                                         frac_cut * self.__call__(M=-22.5,hlr=hlr) / vtot,-20,-14)
        return mag_limit

    @lru_cache(maxsize=None)
    def __call__(self,M,hlr):

        func = self.get_func(hlr)
        return func(M)

    def get_uid(self,hlr):

        _hlr = self.selfn.round_hlr(hlr)
        uid  = "%.2f_[%.1f,%.1f]" % (_hlr,self.lim_z[0],self.lim_z[1])
        return uid

    def get_func(self,hlr):

        uid = self.get_uid(hlr)
        if uid not in self.veff_func_list:
            print "Computing %s %s VEff_Func for non-setup parameters: " \
                  "(hlr=%.2f, lim_z=%s)" % (self.drop_filt.upper(), 
                    self.sample_type.capitalize(), hlr, str(self.lim_z))
            self.veff_func_list[uid] = self.mk_func(hlr)
        return self.veff_func_list[uid]

    def mk_func(self,hlr):

        Mrange = np.arange(-25,-12,0.1)
        veff   = self.calc_veff(M=Mrange,hlr=hlr)
        return scipy.interpolate.InterpolatedUnivariateSpline(Mrange,veff,k=1)

    def setup(self,hlr_range=[-99.,2.,4.,8.]):

        def slave(queue,chunk):
            for hlr in chunk:
                uid = self.get_uid(hlr)
                func = self.mk_func(hlr)
                items = (uid,func)
                queue.put(items)
            queue.put(None)

        num_procs = 5
        hlr_range = np.unique(self.selfn.round_hlr(hlr_range))
        split = np.array_split(hlr_range, num_procs)
        queue = Queue()
        procs = [Process(target=slave, args=(queue,chunk)) for chunk in split]
        for proc in procs: proc.start()

        finished = i = 0
        while finished < num_procs:
            items = queue.get()
            if items == None:
                finished += 1
            else:
                uid, func = items
                self.veff_func_list[uid] = func
                i += 1

        for proc in procs: proc.join()

def mk_pretty_plot():

    fig1,axes1 = plt.subplots(2,3,figsize=(15,7),dpi=75,sharex=True,sharey=True)
    fig1.subplots_adjust(left=0.07,right=0.98,bottom=0.10,top=0.98,hspace=0,wspace=0)
    axes1 = axes1.flatten()
    axes1[0].set_visible(False)
    axes1 = axes1[1:]

    fig2,axes2 = plt.subplots(2,3,figsize=(15,7),dpi=75,sharex=True,sharey=True)
    fig2.subplots_adjust(left=0.07,right=0.98,bottom=0.10,top=0.98,hspace=0,wspace=0)
    axes2 = axes2.flatten()
    axes2[0].set_visible(False)
    axes2 = axes2[1:]

    Mrange = np.arange(-25,-12,0.05)
    HLRrange = [-99.,2.,4.,8.]
    colors = ['k','b','g','r']

    for i,(ax1,ax2,drop_filt,sample_type,label) in enumerate(zip(axes1, axes2,
                                                                [  'f275w',  'f336w', 'f225w', 'f275w', 'f336w'],
                                                                ['dropout','dropout','photoz','photoz','photoz'],
                                                                ['F275W Dropout','F336W Dropout','F225W Photo-z','F275W Photo-z','F336W Photo-z'])):

        print 'Computing Effective Volume %s ...' % label

        veff = VEff_Func(sample_type=sample_type,drop_filt=drop_filt)
        v0 = np.zeros(len(Mrange)) + veff.calc_vol(lim_z=utils.bpz_lims[drop_filt])
        veff.setup()

        for hlr,c in zip(HLRrange,colors):

            v1 = veff(Mrange,hlr)
            ax1.plot(Mrange,v1/v0,c=c,lw=2,ls='-',alpha=0.5)
            ax2.plot(Mrange,v1,c=c,lw=1.5,ls='-' ,alpha=0.8)

        ax2.plot(Mrange,v0,c='k',lw=1,ls='--',alpha=0.8)
        
        ax1.axhline(0.50,lw=0.5,ls=':' ,c='k')
        ax1.axhline(0.25,lw=0.5,ls='--',c='k')
        ax1.axhline(0.10,lw=0.5,ls='-' ,c='k')
        ax1.axvline(veff.mag_limit(hlr=8),c='k')
        
        ax1.set_yscale('log')
        ax2.set_yscale('log')

        ax1.text(0.05,0.05,label,va='bottom',ha='left',transform=ax1.transAxes)
        ax2.text(0.05,0.05,label,va='bottom',ha='left',transform=ax2.transAxes)

    axes1[2].set_xlim(-23.9,-12.1)
    axes1[2].set_xlabel('UV Absolute Magnitude')
    axes1[2].set_ylim(2e-2,2e0)
    axes1[2].set_ylabel('Effective Volume Correction')

    axes2[2].set_ylim(2e2,8e4)
    axes2[2].set_ylabel('Effective Volume')
    axes2[2].set_xlim(-23.9,-12.1)
    axes2[2].set_xlabel('UV Absolute Magnitude')

    fig1.savefig('plots/veff_corr.png')
    fig2.savefig('plots/veff.png')

if __name__ == '__main__':

    mk_pretty_plot()
    plt.show()