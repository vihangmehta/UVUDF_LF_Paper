import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio

import useful2
import uvudf_utils as utils
import conversions as conv
from LF_refs import z2_LF_refs
from dust import dust_uv, dust_ha

def plot_SFRFs():

    M16 = fitsio.getdata('output/z2_LF_mehta16.fits')
    H10 = fitsio.getdata('output/z2_LF_hayes10.fits')
    S13 = fitsio.getdata('output/z2_LF_sobral13.fits')

    M16 = M16[M16['xlum'] < -16]
    H10 = H10[H10['xlum'] > np.log10(2e41)]
    S13 = S13[S13['xlum'] > 42.-0.4]

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    ax.plot(        M16['SFR_M99'],M16[ 'SFRF_M99'],lw=1.5,color='r',label='M16 SFRF(UV) (M99 dust cor.)')
    ax.fill_between(M16['SFR_M99'],M16['eSFRF_M99'][:,0],
                                   M16['eSFRF_M99'][:,1],color='r',alpha=0.15)

    ax.plot(        M16['SFR_C14'],M16[ 'SFRF_C14'],lw=1.5,color='b',label='M16 SFRF(UV) (C14 dust cor.)')
    ax.fill_between(M16['SFR_C14'],M16['eSFRF_C14'][:,0],
                                   M16['eSFRF_C14'][:,1],color='b',alpha=0.15)

    ax.plot(        M16['SFR_R15'],M16[ 'SFRF_R15'],lw=1.5,color='g',label='M16 SFRF(UV) (R15 dust cor.)')
    ax.fill_between(M16['SFR_R15'],M16['eSFRF_R15'][:,0],
                                   M16['eSFRF_R15'][:,1],color='g',alpha=0.15)

    # ax.plot(        H10['SFR_H01'],H10[ 'SFRF_H01'],lw=1.5,color='m',label='H10 SFRF(H$\\alpha$)')
    # ax.fill_between(H10['SFR_H01'],H10['eSFRF_H01'][:,0],
    #                            H10['eSFRF_H01'][:,1],color='m',alpha=0.1)

    cond = (S13["eSFRF_H01"][:,0]!=0) & (S13["eSFRF_H01"][:,1]!=0)
    ax.plot(        S13['SFR_H01'],S13[ 'SFRF_H01'],lw=1.5,color='k',label='S13 SFRF(H$\\alpha$)')
    ax.fill_between(S13['SFR_H01'][cond],S13['eSFRF_H01'][:,0][cond],
                                   S13['eSFRF_H01'][:,1][cond],color='k',alpha=0.1)

    ax.set_yscale('log')
    ax.set_ylim(8e-5,1e-1)
    ax.set_ylabel('$\Phi$(SFR) dlog(SFR)',fontsize=24)
    ax.set_xscale('log')
    ax.set_xlim(10**-0.9,10**2.7)
    ax.set_xlabel('Star Formation Rate [M$_\odot$ yr$^{-1}$]',fontsize=24)
    _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()]
    ax.legend(fontsize=16,loc=3,framealpha=0)

    fig.savefig('plots/SFRFs.png')
    fig.savefig('plots/SFRFs.pdf')

def plot_CSFRFs():

    M16 = fitsio.getdata('output/z2_LF_mehta16.fits')
    H10 = fitsio.getdata('output/z2_LF_hayes10.fits')
    S13 = fitsio.getdata('output/z2_LF_sobral13.fits')

    M16 = M16[M16['xlum'] <= utils.lim_M['f275w_phot']]
    H10 = H10[H10['xlum'] >= np.log10(2e41)]
    S13 = S13[S13['xlum'] >= 42.-0.4]

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    ax.plot(        M16['SFR_M99'],M16[ 'CSFRF'],lw=1.5,c='r',label='CSFRF(UV) (M99 dust cor.)')
    ax.fill_between(M16['SFR_M99'],M16['eCSFRF'][:,0],
                                   M16['eCSFRF'][:,1],color='r',alpha=0.15)

    ax.plot(        M16['SFR_C14'],M16[ 'CSFRF'],lw=1.5,c='b',label='CSFRF(UV) (C14 dust cor.)')
    ax.fill_between(M16['SFR_C14'],M16['eCSFRF'][:,0],
                                   M16['eCSFRF'][:,1],color='b',alpha=0.15)

    ax.plot(        M16['SFR_R15'],M16[ 'CSFRF'],lw=1.5,c='g',label='CSFRF(UV) (R15 dust cor.)')
    ax.fill_between(M16['SFR_R15'],M16['eCSFRF'][:,0],
                                   M16['eCSFRF'][:,1],color='g',alpha=0.15)

    # ax.plot(        H10['SFR_H01'],H10[ 'CSFRF'],lw=1.5,c='m',label='H10 CSFRF(H$\\alpha$)')
    # ax.fill_between(H10['SFR_H01'],H10['eCSFRF'][:,0],
    #                                H10['eCSFRF'][:,1],color='m',alpha=0.1)

    ax.plot(        S13['SFR_H01'],S13[ 'CSFRF'],lw=1.5,c='k',label='S13 CSFRF(H$\\alpha$)')
    ax.fill_between(S13['SFR_H01'],S13['eCSFRF'][:,0],
                                   S13['eCSFRF'][:,1],color='k',alpha=0.1)

    ax.set_yscale('log')
    ax.set_ylim(8e-5,1e-1)
    ax.set_ylabel('$\Phi$(>SFR)',fontsize=24)
    ax.set_xscale('log')
    ax.set_xlim(10**-0.9,10**2.7)
    ax.set_xlabel('Star Formation Rate [M$_\odot$ yr$^{-1}$]',fontsize=24)
    _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()]
    ax.legend(fontsize=16,loc=3,framealpha=0)

    fig.savefig('plots/CSFRFs.png')

def SFRF_Text_Numbers():

    print
    print "Numbers for difference in Ha and UV CLFs:"
    print "%10s%8s%8s%10s%10s%10s" % ('SFR','Muv','Lha','CLFuv','CLFha','C(ha/uv)')

    for sfr in [0.1,1,10,20,30,40,50,80,100]:

        luv = np.log10(useful2.SFR_K12('uv',sfr,inv=True))
        muv = conv.get_absM_from_Lnu(luv)
        muv = dust_uv['M99'].apply_dust(muv)

        lha = np.log10(useful2.SFR_K12('ha',sfr,inv=True))
        lha = dust_ha['H01'].apply_dust(lha)

        cuv = useful2.UV_CLF(muv,*z2_LF_refs['mehta16']['coeff'])
        cha = useful2.Ha_CLF(lha,*z2_LF_refs['sobral13']['coeff'],agn=True)

        print "%10.2e%8.2f%8.2f%10.2e%10.2e%10.4f" % (sfr, muv, lha, cuv, cha, cha/cuv)

if __name__ == '__main__':
    
    plot_SFRFs()
    # plot_CSFRFs()
    #SFRF_Text_Numbers()
    plt.show()