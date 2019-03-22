import numpy as np
import scipy.ndimage
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import useful

class UV_Dust():

    def __init__(self,law='M99'):

        self.law = law
        self.M0 = -19.5

        # K14 only
        self.beta0     = -1.71
        self.dbeta_dM  = -0.09
        self.sig_beta  =  0.36
        # print self.beta0, self.dbeta_dM, self.M0

        if   self.law=='M99': self.a, self.b = 4.43, 1.99
        elif self.law=='H13': self.a, self.b = 3.40, 1.60
        elif self.law=='C14': self.a, self.b = 5.32, 1.99
        elif self.law=='R15': self.a, self.b = 4.48, 1.84
        else: raise Exception("No IRX-beta law for %s defined." % self.law)

        self.avg_ext_ = lambda mag: self.a + 0.2*self.b**2*np.log(10)*self.sig_beta**2 + self.b*self.avg_beta(mag)
        self.c = -(self.a+ 0.2*self.b**2*np.log(10)*self.sig_beta**2)/self.b

        self.extinction = np.vectorize(self.extinction)
        self.mk_funcs()

    def avg_beta(self,M):
        if M >= self.M0:
            beta = (self.beta0 - self.c) * np.exp(self.dbeta_dM*(M-self.M0)/(self.beta0-self.c)) + self.c
        else:
            beta = self.dbeta_dM*(M-self.M0) + self.beta0
        return beta

    def extinction(self,mags):
        res = self.avg_ext_(mags)
        res = res if res >= 0 else 0
        return res

    def mk_funcs(self):
        mags = np.arange(-65,1,0.01)
        mags_cor = mags - self.extinction(mags)
        self.apply_dust  = scipy.interpolate.interp1d(mags_cor,mags)
        self.remove_dust = scipy.interpolate.interp1d(mags,mags_cor)

    def plot(self):

        fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

        m = np.arange(-25,-10,0.1)
        ax.plot(m, self.extinction(m), c='k', lw=2, label='Kurczynski14 (used here)')

        ax.set_xlabel('UV Magnitude [AB]')
        ax.set_ylabel('UV Extinction, A$_\mathrm{UV}$')
        ax.set_xlim(-14,-25)
        ax.set_ylim(-0.1,4.1)
        ax.legend(loc=2,fontsize=18)
        plt.show()

class _Halpha_Dust():

    def __init__(self,new=False):

        self.dom_Lha   = np.array([41.57,41.91,42.33])
        self.dom_dLha1 = np.array([ 0.48, 0.11, 0.29])
        self.dom_dLha2 = np.array([ 0.23, 0.13, 1.05])
        self.dom_Aha   = np.array([ 0.40, 0.21, 1.59])
        self.dom_dAha  = np.array([ 0.71, 0.73, 0.98])
        self.dom_HaHb  = np.array([ 3.29, 3.08, 5.01])
        self.dom_dHaHb = np.array([ 0.82, 0.79, 1.60])

        self.remove_dust = np.vectorize(self.remove_dust,excluded=['self','offset'])
        self.apply_dust  = np.vectorize(self.apply_dust,excluded=['self','offset'])
        if new: self.offset = self.calc_offset()
        else:   self.offset = -1.3683348 #- 0.4

    def to_solve(self,lum_,lum):
        sfr  = lum  + np.log10(7.9e-42)
        sfr_ = lum_ + np.log10(7.9e-42)
        res = sfr_ - (2.88 * 10**((sfr_-sfr)/2.614) - 3.834) / 0.797
        return res

    def offset_lum(self,lum,offset):

        return lum+self.offset if offset else lum

    def unoffset_lum(self,lum,offset):

        return lum-self.offset if offset else lum

    def remove_dust(self,lum,offset=True):

        lum_off = self.offset_lum(lum,offset)
        if lum_off > 39:
            res = scipy.optimize.fsolve(self.to_solve,x0=lum_off+25,args=(lum_off,))
        else: res = lum_off
        lum_off = max(lum_off,res)
        return self.unoffset_lum(lum_off,offset)

    def apply_dust(self,lum,offset=True):

        to_solve = lambda lum,lum_: self.to_solve(lum_,lum)
        lum_off = self.offset_lum(lum,offset)
        if lum_off > 37.5:
            res = scipy.optimize.fsolve(to_solve,x0=lum_off-25,args=(lum_off,))
        else: res = lum_off
        lum_off = min(lum_off,res)
        return self.unoffset_lum(lum_off,offset)

    def calc_Aha(self,lum,offset):

        return (self.remove_dust(lum,offset=offset) - lum) * 2.5

    def calc_HaHb(self,lum,offset):

        return 2.86 * 10**(self.calc_Aha(lum,offset=offset) / 3.33 / 1.97)

    def calc_offset(self):

        to_solve = lambda x: np.sum((self.calc_Aha(self.dom_Lha+x,offset=False) - self.dom_Aha)**2 / self.dom_dAha**2)
        return scipy.optimize.minimize(to_solve,x0=[-1,],bounds=[(-5,0),])['x']

    def calc_smoothed(self,smooth=0.5):

        dL     = 0.005
        sigma  = smooth / dL
        L      = np.arange(0,100,dL)
        Aha    = self.calc_Aha(L,offset=True)
        Aha_s  = scipy.ndimage.filters.gaussian_filter(Aha,sigma=sigma,mode='nearest')
        self.extinction = scipy.interpolate.interp1d(L,Aha_s)

    def plot(self):

        lha  = np.arange(30,50,0.01)

        fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

        ax.axvline(0,lw=0,label='log L$_\mathrm{H\\alpha}$ offset: %.2f' % self.offset)
        ax.plot(lha, self.calc_Aha(lha,offset=True), c='k', lw=1.5, label='Hopkins01 (w/ offset)')
        ax.plot(lha, self.calc_Aha(lha,offset=False), c='k', ls='--', lw=1.5, label='Hopkins01 (w/o offset)')
        ax.scatter(self.dom_Lha,self.dom_Aha,c='k',s=50,lw=0, label='Dominguez13')
        ax.errorbar(self.dom_Lha,self.dom_Aha,xerr=[self.dom_dLha1,self.dom_dLha2],yerr=self.dom_dAha,fmt='ko',markersize=0,c='k')

        dax = ax.twinx()
        dax.plot(lha, self.calc_HaHb(lha,offset=True), c='r', lw=1.5)
        dax.plot(lha, self.calc_HaHb(lha,offset=False), c='r', ls='--', lw=1.5)
        dax.scatter(self.dom_Lha,self.dom_HaHb,c='r',s=50,lw=0)
        dax.errorbar(self.dom_Lha,self.dom_HaHb,xerr=[self.dom_dLha1,self.dom_dLha2],yerr=self.dom_dHaHb,fmt='ro',markersize=0,c='r')
        dax.axhline(2.86,ls=':',c='r')

        ax.set_xlim(39,45)
        ax.set_xlabel('log L$_\mathrm{H\\alpha}$')
        ax.set_ylim(-0.1,3.1)
        ax.set_ylabel('A$_\mathrm{H\\alpha}$')
        dax.set_ylim(1.5,8.1)
        dax.set_ylabel('H$\\alpha$/H$\\beta$',color='r')
        _ = [_.set_color('r') for _ in dax.get_yticklabels()]
        _ = [_.set_color('r') for _ in dax.get_yticklines()]
        dax.spines['right'].set_edgecolor('r')
        ax.spines['right'].set_edgecolor('r')

        ax.legend(fontsize=18,loc=2)
        plt.show()

class Halpha_Dust():

    def __init__(self,new=False,smooth=0.5):

        self.old_dust = _Halpha_Dust(new=new)

        self.smooth = smooth
        self.dL     = 0.005
        self.sigma  = self.smooth / self.dL
        self.L      = np.arange(0,100,self.dL)
        self.Aha    = self.old_dust.calc_Aha(self.L,offset=True)
        self.Aha_s  = scipy.ndimage.filters.gaussian_filter(self.Aha,sigma=self.sigma,mode='nearest')

        self.extinction_raw = scipy.interpolate.interp1d(self.L,self.Aha  )
        self.extinction     = scipy.interpolate.interp1d(self.L,self.Aha_s)

        self.mk_funcs()

    def mk_funcs(self):
        L     = np.arange(0,100,0.1)
        L_cor = L + self.extinction(L) / 2.5
        self.apply_dust  = scipy.interpolate.interp1d(L_cor,L)
        self.remove_dust = scipy.interpolate.interp1d(L,L_cor)

    def plot(self):

        fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

        lum = np.arange(30,50,0.01)
        ax.plot(lum, self.extinction(lum), c='k', lw=2, label='Smoothed')
        ax.plot(lum, self.extinction_raw(lum), c='r', lw=2, label='Original')

        ax.set_xlabel('H$\\alpha$ Luminosity [AB]')
        ax.set_ylabel('H$\\alpha$ Extinction, A$_\mathrm{H\\alpha}$')
        ax.set_xlim(40,45)
        ax.set_ylim(-0.1,3.1)
        ax.legend(loc=2,fontsize=18)
        plt.show()

def mk_pretty_plot():

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,12),dpi=75,tight_layout=True)

    m = np.arange(-25,-10,0.1)
    ax1.plot(m, UV_Dust(law='C14').extinction(m), c='k', lw=2, label='Castellano14 + Kurczynski14')
    ax1.plot(m, UV_Dust(law='R15').extinction(m), c='g', lw=1.5, label='Reddy15 + Kurczynski14')
    ax1.plot(m, UV_Dust(law='H13').extinction(m), c='b', lw=1.5, label='Heinis13 + Kurczynski14')
    ax1.plot(m, UV_Dust(law='M99').extinction(m), c='r', lw=2, label='Meurer99 + Kurczynski14')

    ax1.set_xlabel('UV Magnitude [AB]',fontsize=24)
    ax1.set_ylabel('UV Extinction, A$_\mathrm{UV}$',fontsize=24)
    ax1.set_xlim(-14,-25)
    ax1.set_ylim(0,4.1)
    leg = ax1.legend(loc=2,fontsize=14)
    _ = [entry.set_fontproperties(FontProperties(size=14,weight=fw)) for entry,fw in zip(leg.get_texts(),[800,800,400,400,400])]

    ha  = _Halpha_Dust()
    ha2 = Halpha_Dust()

    lha  = np.arange(30,50,0.01)
    ax2.axvline(0,lw=0,label='log L$_\mathrm{H\\alpha}$ offset: %.2f' % -ha.offset)
    ax2.scatter(ha.dom_Lha,ha.dom_Aha,c='k',s=50,lw=0, label='Dominguez13')
    ax2.errorbar(ha.dom_Lha,ha.dom_Aha,xerr=[ha.dom_dLha1,ha.dom_dLha2],yerr=ha.dom_dAha,fmt='ko',markersize=0,c='k')
    ax2.plot(lha, ha.calc_Aha(lha,offset=False), c='k', ls='--', lw=1.5, label='Hopkins01 (no offset)')
    ax2.plot(lha, ha.calc_Aha(lha,offset=True), c='k', lw=1.5, label='Hopkins01 (w/ offset)')
    ax2.plot(lha, ha2.extinction(lha), c='r', lw=2, label='Hopkins01 (w/ offset, \n smoothed)')

    ax2.set_xlim(39,45)
    ax2.set_xlabel('log H$\\alpha$ Luminosity, L$_\mathrm{H\\alpha}$ [ergs/s]',fontsize=24)
    ax2.set_ylim(0,3.1)
    ax2.set_ylabel('H$\\alpha$ extinction, A$_\mathrm{H\\alpha}$',fontsize=24)
    leg = ax2.legend(loc="best",fontsize=14)
    _ = [entry.set_fontproperties(FontProperties(size=14,weight=fw)) for entry,fw in zip(leg.get_texts(),[400,400,400,800,400])]
    ax2.spines['right'].set_visible(False)

    dax = ax2.twinx()
    # dax.plot(lha, ha.calc_HaHb(lha,offset=False), c='r', ls='--', lw=1.5)
    # dax.plot(lha, ha.calc_HaHb(lha,offset=True), c='r', lw=1.5)
    # dax.scatter(ha.dom_Lha,ha.dom_HaHb,c='r',s=50,lw=0)
    # dax.errorbar(ha.dom_Lha,ha.dom_HaHb,xerr=[ha.dom_dLha1,ha.dom_dLha2],yerr=ha.dom_dHaHb,fmt='ro',markersize=0,c='r')
    # dax.axhline(2.86,ls=':',c='r')

    Aha_ylim  = np.asarray(ax2.get_ylim())
    HaHb_ylim = np.log10(2.86) + (Aha_ylim / 3.33 / 1.97)
    HaHb_yticks = np.array([3,4,5,6,7,8])
    dax.set_ylim(10**HaHb_ylim)
    dax.set_yscale('log')
    dax.set_yticks(HaHb_yticks)
    dax.set_yticklabels(HaHb_yticks)
    dax.set_ylabel('Balmer decrement, H$\\alpha$/H$\\beta$',fontsize=24)
    _ = [label.set_fontsize(20) for label in ax1.get_xticklabels()+ax1.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()+dax.get_yticklabels()]

    fig.savefig('plots/dust.png')
    fig.savefig('plots/dust.pdf')

def neb_vs_stellar():

    import useful2
    import conversions as conv

    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(17,5),dpi=75,tight_layout=True)
    fig.subplots_adjust(left=0.08,right=0.95,bottom=0.1,top=0.95)

    dust_ha = Halpha_Dust(new=False)

    Lha = np.arange(40,44,0.1)
    Aha = dust_ha.extinction(Lha)
    Lha_cor = dust_ha.remove_dust(Lha)
    sfr_ha = useful2.SFR_K12('ha',10**Lha_cor)
    sfr_uv = sfr_ha / (0.29 + 0.21*np.log10(sfr_ha) + 0.29*np.log10(sfr_ha)**2 + 0.004*np.log10(sfr_ha)**3)
    Luv_cor = conv.get_absM_from_Lnu(np.log10(useful2.SFR_K12('uv',sfr_uv,inv=True)))

    for dust,c in zip(['M99','R15'],['r','g']):
        
        dust_uv = UV_Dust(law=dust)
        Luv = dust_uv.apply_dust(Luv_cor)
        Auv = dust_uv.extinction(Luv)

        ax1.plot(sfr_ha,Aha-Auv,c=c,lw=2,alpha=0.8,label='using %s UV dust corr.'%dust)
        ax3.plot(sfr_uv,Auv,c=c,lw=2,alpha=0.8,label='%s UV dust corr.'%dust)

    ax1.axhline(0,c='k',lw=2,alpha=0.8)
    ax1.set_xlabel('SFR(H$\\alpha$) [M$_\odot$ yr$^{-1}$]')
    ax1.set_ylabel('$A_{H\\alpha} - A_{UV}$')
    ax1.set_xscale('log')
    ax1.set_xlim(3e-1,3e2)
    ax1.set_ylim(-1.5,2.5)
    ax1.legend(loc='best',fontsize=12)

    ax2.plot(sfr_ha,Aha,c='k',lw=2,alpha=0.8,label='applied H$\\alpha$ extinction')
    ax2.set_xlabel('SFR(H$\\alpha$) [M$_\odot$ yr$^{-1}$]')
    ax2.set_xscale('log')
    ax2.set_xlim(3e-1,3e2)
    #ax2.set_xlabel('H$\\alpha$ Luminosity')
    #ax2.set_xlim(40.5,43.5)
    ax2.set_ylabel('$A_{H\\alpha}$')
    ax2.set_ylim(0,4)
    ax2.legend(loc='best',fontsize=12)

    ax3.set_xlabel('SFR(UV) [M$_\odot$ yr$^{-1}$]')
    ax3.set_xscale('log')
    ax3.set_xlim(3e-1,3e2)
    #ax3.set_xlabel('UV Magnitude')
    #ax3.set_xlim(-16,-23)
    ax3.set_ylabel('$A_{UV}$')
    ax3.set_ylim(0,4)
    ax3.legend(loc='best',fontsize=12)

if __name__ == '__main__':
    
    #mk_pretty_plot()
    neb_vs_stellar()
    plt.show()

else:
    
    dust_uv = {'M99': UV_Dust(law='M99'),
               'C14': UV_Dust(law='C14'),
               'R15': UV_Dust(law='R15')}
    dust_ha = {'H01': Halpha_Dust(new=False)}