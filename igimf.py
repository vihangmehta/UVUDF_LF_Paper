import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

import conversions as conv
from dust import dust_uv

def SFR_vs_Ha(x,model):
    """
    x = log Lha
    y = log SFR
    """
    coeff = {'standard': [+4.38e-1,+9.64e-1,+3.21e-2,-8.47e-3,-9.30e-4,-2.67e-5],
             'maximal' : [+1.11e-0,+9.63e-1,+2.67e-2,-9.13e-3,-9.97e-4,-2.94e-5],
             'minimal1': [-1.58e-1,+8.85e-1,+4.23e-2,-5.70e-3,-7.51e-4,-2.32e-5],
             'minimal2': [-1.91e-1,+8.94e-1,+3.50e-2,-6.58e-3,-2.43e-4,+3.84e-5]}
    a = coeff[model]
    x = x - 41.
    return a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4 + a[5]*x**5

def SFR_vs_FUV(x,model):
    """
    x = Muv
    y = log SFR
    """
    coeff = {'standard': [-5.11,-0.087,0.0214,0.000683,8.43e-06],
             'minimal1': [-5.47,-0.124,0.0173,0.000533,6.52e-06]}
    a = coeff[model]
    return a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4

def get_igimf_uv_ha(model):

    muv = np.arange(-50,-10,0.01)
    lha = np.arange(+38,+44,0.01)
    sfr_uv = SFR_vs_FUV(muv,model)
    sfr_ha = SFR_vs_Ha(lha,model)

    cond = (min(sfr_ha)<sfr_uv) & (sfr_uv<max(sfr_ha))
    muv,sfr_uv = muv[cond], sfr_uv[cond]

    lha = scipy.interpolate.interp1d(sfr_ha,lha)(sfr_uv)
    return muv, lha, sfr_uv

def salpeter(m):
    alpha = -2.35
    return (10**m)**alpha

def kroupa(m):

    glob_norm = (10**2)**-2.35 / (10**2)**-2.3

    if   m <= 0.08:
        norm  = ((10**0.5)**-2.3 / (10**0.5)**-1.3) * ((10**0.08)**-1.3 / (10**0.08)**-0.3)
        alpha = -0.3
    elif 0.08 < m <= 0.5:
        norm  = ((10**0.5)**-2.3 / (10**0.5)**-1.3)
        alpha = -1.3
    else:
        norm  = 1.
        alpha = -2.3

    return (10**m)**alpha * norm * glob_norm
kroupa = np.vectorize(kroupa)

def chabrier(m):
    glob_norm = (10**2)**-2.35 / (10**2)**-2.3
    coeff = lambda m: 0.086 * np.exp(-(np.log10(m)-np.log10(0.22))**2 / (2*0.57**2))
    if   m < 1:
        norm = ((10**1)**-2.3 / coeff(10**1))
        res  = coeff(10**m) * norm
    else:
        res  = (10**m)**-2.3
    return res * glob_norm
chabrier = np.vectorize(chabrier)

def igimf(m,sfr):

    mmax = 0.75 * sfr + 4.83

    if   m <= 0.08:
        alpha = -1.30
    elif 0.08 < m <= 0.5:
        alpha = -2.35
    elif 0.5 < m <= 1.0:
        alpha = -2.35
    elif 1.0 < m <= mmax:
        alpha = -2.35
    else:
        return np.NaN

    return (10**m)**alpha

igimf = np.vectorize(igimf,excluded=['sfr',])

def plot_imfs():

    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

    m = np.arange(-1,2.01,0.05)

    ax.plot(m,salpeter(m),c='k',lw=2,label='Salpeter+55')
    ax.plot(m,kroupa(m),c='k',ls='--',lw=2,label='Kroupa+01')
    ax.plot(m,chabrier(m),c='k',ls=':',lw=2,label='Chabrier+03')
    
    # sfr_range = np.arange(-5,-1.99,0.5)
    # colors = plt.cm.gist_rainbow(np.linspace(0,0.9,len(sfr_range)))

    # for sfr,c in zip(sfr_range,colors):
    #     print 0.75 * sfr + 4.83
    #     ax.plot(m,igimf(m,sfr),c=c,label='IGIMF (log SFR = %.2f)'%sfr)

    ax.legend(loc='best',fontsize=12)
    ax.set_xlabel('log (m/M$_\odot$)')
    ax.set_ylabel('IMF, $\\xi(m) \ dm$')
    ax.set_yscale('log')

def main():

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,12),dpi=75,tight_layout=True)

    for model,ls in zip(['standard','minimal1'],['-','--']):

        muv, lha, sfr = get_igimf_uv_ha(model)
        luv = conv.get_absM_from_Lnu(muv,inv=True) + np.log10(conv.light / 1500.)
        ratio = luv - lha
        
        ax1.plot(sfr,ratio,c='k',ls=ls,lw=2,label=model.capitalize())

        muv_ = dust_uv['M99'].apply_dust(muv)
        ax2.plot(muv_,ratio,c='r',ls=ls,lw=2)

        muv_ = dust_uv['C14'].apply_dust(muv)
        ax2.plot(muv_,ratio,c='b',ls=ls,lw=2)

    ax2.plot(-99,-99,c='k',ls='-',lw=2,label='Standard')
    ax2.plot(-99,-99,c='k',ls='--',lw=2,label='Minimal1')
    ax2.plot(-99,-99,c='r',lw=2,label='using M99 dust')
    ax2.plot(-99,-99,c='b',lw=2,label='using C14 dust')

    ax1.set_xlabel('log SFR [M$_\odot$ yr$^{-1}$]')
    ax2.set_xlabel('Observed UV Magnitude [AB]')
    ax2.set_xlim(-16,-23)

    for ax in [ax1,ax2]:
        ax.set_ylabel('log $\\nu$L$_{\\nu,1500}$/L$_{H\\alpha}$ (dust corrected)')
        ax.set_ylim(1.55,2.65)
        ax.legend(loc='best',fontsize=12)

def test():

    def integrand(Mecl,m):
        imf  = 1 * (10**m)**-2.35
        ecmf = 1 * Mecl**-1.95
        return imf * ecmf

    max_Mecl = lambda sfr: 10**(0.0144 + 0.75*np.log10(sfr) + 6.77)
    igimf = lambda m,sfr: scipy.integrate.quad(integrand,5,max_Mecl(sfr),args=(m,))[0]
    igimf = np.vectorize(igimf,excluded=['sfr',])
    mass = np.arange(-1,2,0.05)

    for sfr,c in zip([1e2,1e1,1e0,1e-1,1e-2],['k','r','g','b','c']):
        plt.plot(mass,igimf(mass,sfr),c=c,lw=2,alpha=0.5)
        plt.yscale('log')

if __name__ == '__main__':
    main()
    #plot_imfs()
    #test()
    plt.show()