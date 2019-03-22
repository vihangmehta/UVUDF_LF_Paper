import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from joblib import Parallel, delayed

import useful2
import conversions as conv
from extinction import calzetti
from dust import dust_uv, dust_ha
from LF_refs import UV_LF_refs, z2_LF_refs, M16_LF_pars
from mcmc_sim import make_dummy_mcmc

def MD14_UVLD_relation(z,a=0.015,b=2.7,c=5.6,d=2.9):
    """
    Madau & Dickinson (2014) unobscured UVLD relation
    """
    return a * (1+z)**b / (1+((1+z)/d)**c)

def calc_luv_density(coeff,lolim,hilim,dust='none'):
    """
    Returns the UV luminosity density for given UV LF
    """
    if   dust=='none':
      luv = lambda x: 10**conv.get_absM_from_Lnu(x,inv=True)
    elif dust in dust_uv.keys():
      luv = lambda x: 10**conv.get_absM_from_Lnu(dust_uv[dust].remove_dust(x),inv=True)
    elif isinstance(dust,float):
      luv = lambda x: 10**conv.get_absM_from_Lnu(x,inv=True) * np.exp(calzetti(np.array([1500,]),dust))
    else:
      raise Exception('Incorrect UV Dust.')

    intg = lambda x: useful2.UV_LF(x,*coeff) * luv(x)
    return scipy.integrate.quad(intg, lolim, hilim, points=np.arange(lolim,hilim,0.5), **useful2.quad_args)[0]

def calc_lha_density(coeff,lolim,hilim,dust='none',agn=False):
    """
    Returns the Ha luminosity density for given Ha LF
    """
    if   dust=='none':
      lha = lambda x: 10**x
    elif dust in dust_ha.keys():
      lha = lambda x: 10**dust_ha[dust].remove_dust(x)
    else:
      raise Exception('Incorrect Ha Dust.')

    intg = lambda x: useful2.Ha_LF(x,*coeff,agn=agn) * lha(x)
    return scipy.integrate.quad(intg, lolim, hilim, points=np.arange(lolim,hilim,0.5), **useful2.quad_args)[0]

def get_uvld(coeff,Mlim,imf='salp',dust='none',get_plt_err=False,N=500):

    try:
        chain = np.genfromtxt(coeff['fname'])
    except KeyError:
        chain = make_dummy_mcmc(coeff['coeff'],coeff['err'])

    chain = chain.reshape((-1,10,3))
    chain = chain[200:,:,:].reshape((-1,3))
    
    luvd  = calc_luv_density(coeff=coeff['coeff'],lolim=-50,hilim=Mlim,dust=dust)
    sfrd  = useful2.SFR_K12('uv', luvd,imf='salp')
    
    _luvd = np.array(Parallel(n_jobs=15)(delayed(calc_luv_density)(coeff=_coeff,lolim=-50,hilim=Mlim,dust=dust) \
                                    for _coeff in chain[np.random.randint(len(chain),size=N)]))
    _sfrd = useful2.SFR_K12('uv',_luvd,imf='salp')

    luvd_err = np.percentile(_luvd,q=[50-68.27/2,50+68.27/2],axis=-1)
    sfrd_err = np.percentile(_sfrd,q=[50-68.27/2,50+68.27/2],axis=-1)
    luvd_err = np.abs(luvd_err - luvd)
    sfrd_err = np.abs(sfrd_err - sfrd)

    if get_plt_err:
        return luvd, luvd_err[:,np.newaxis]
    
    return luvd/1e26, luvd_err/1e26, sfrd, sfrd_err

def mk_pretty_plot(Mlim):

    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

    ref_keys = np.array(UV_LF_refs.keys())
    _ref_keys = ref_keys.copy()
    _ref_keys[_ref_keys=='mehta16'] = 'zzz16'
    ref_keys = ref_keys[np.argsort([i[-2:]+i[:-2] for i in _ref_keys])]

    for ref_key in ref_keys:

        ref = UV_LF_refs[ref_key]
        (capsize,capthick) = (8,2) if "mehta" in ref_key else (None,1.5)
        print "---",ref['label'],"---"
        ax.scatter(-99,-99,s=ref['s'],lw=0,color=ref['c'],marker=ref['m'],label=ref['label'])
        
        for coeff in ref['LFs']:
            luvd,err = get_uvld(coeff=coeff,Mlim=Mlim,get_plt_err=True)
            ax.scatter( 1+coeff['z'],luvd,color=ref['c'],s=ref['s'],lw=0,marker=ref['m'],zorder=1)
            ax.errorbar(1+coeff['z'],luvd,yerr=err,color=ref['c'],elinewidth=1.5,capthick=capthick,capsize=capsize,alpha=0.8,zorder=1)
            print 'z=%.2f'%coeff['z'],luvd

    ax.text(0.03,0.95,"M$_\mathrm{\mathsf{UV}}$ < %i"%Mlim,fontsize=24,fontweight=600,va='top',ha='left',transform=ax.transAxes)

    ax.set_ylim(10**24.4,10**27.3)
    ax.set_yscale('log')
    ax.set_ylabel("$\\rho_{UV}$ [ergs/s/Hz/Mpc$^3$]",fontsize=24)

    ax.set_xscale('log')
    ax.set_xlim(0.9,12.9)
    ax.set_xticks(np.append(np.arange(10)+1,12))
    ax.set_xticklabels(np.append(np.arange(10)+1,12))
    ax.set_xlabel('(1+z)',fontsize=24)
    _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()]
    leg = ax.legend(fontsize=15,ncol=2,loc=8,scatterpoints=1)

    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        wht = 800 if "This work" in txt.get_text() else 400
        txt.set_fontproperties(FontProperties(weight=wht))

    fig.savefig('plots/uvld.png')
    fig.savefig('plots/uvld.pdf')

def SFRDuv_Table_Numbers(imf='salp'):

    print 
    print "Mehta (2016) : UVLD & SFRD Numbers for Table"
    print "All numbers are for '%s' IMF" % imf
    print "All luminosity densities are in units of 1e26"
    
    print "".join(['-']*70)
    print "%5s|%8s|%10s%8s%8s|%10s%8s%8s" % ('Dust','Mag Lim','LuvD','-err','+err','SFRD','-err','+err')
    print "".join(['-']*70)
    
    for ref in ['f225w_phot','f275w_phot','f336w_phot']:
        print "%s"%ref.center(70,' ')
        for dust in ['none','M99','C14','R15']:
            for hilim in [M16_LF_pars[ref]['coeff'][1]-2.5*np.log10(0.03),-13,-10]:
                luvd, lerr, sfrd, serr = get_uvld(coeff=M16_LF_pars[ref],Mlim=hilim,imf=imf,dust=dust)
                print "%5s|%8.2f|%10.4f%8.4f%8.4f|%10.4f%8.4f%8.4f" % (dust,hilim,luvd,lerr[0],lerr[1],sfrd,serr[0],serr[1])
            print "".join(['-']*70)

def SFRD_other_Numbers(imf='salp'):

    print
    print "Hayes (2010) Ha LF:"
    print "".join(['-']*35)
    print "%5s%8s%12s%10s" % ('Dust','Lum Lim','LHaD','SFRD')
    print "".join(['-']*35)
    for dust in ['none','H01']:
        for lolim in [30.,z2_LF_refs['hayes10']['coeff'][1]+np.log10(0.03)]:
            lhad = calc_lha_density(coeff=z2_LF_refs['hayes10']['coeff'],lolim=lolim,hilim=50.0,dust=dust)
            sfrd = useful2.SFR_K12('ha',lhad)
            print "%5s%8.2f%12.2e%10.4f" % (dust,lolim,lhad,sfrd)

    print
    print "Sobral (2013) Ha LF:"
    print "".join(['-']*35)
    print "%5s%8s%12s%10s" % ('Dust','Lum Lim','LHaD','SFRD')
    print "".join(['-']*35)
    for dust in ['none','H01']:
        for lolim in [30.,z2_LF_refs['sobral13']['coeff'][1]+np.log10(0.03)]:
            lhad = calc_lha_density(coeff=z2_LF_refs['sobral13']['coeff'],lolim=lolim,hilim=50.0,dust=dust,agn=True)
            sfrd = useful2.SFR_K12('ha',lhad)
            print "%5s%8.2f%12.2e%10.4f" % (dust,lolim,lhad,sfrd)

def SFRD_Text_Numbers():

    print
    print "Madau14 SFRD Relation (dust corrected): %.4f" % MD14_UVLD_relation(z=2.2)

    lim_uv  = z2_LF_refs['mehta16']['coeff'][1]-2.5*np.log10(0.03)
    _lim_uv = dust_uv['M99'].remove_dust(lim_uv)
    lim_sfr = useful2.SFR_K12('uv',10**conv.get_absM_from_Lnu(_lim_uv,inv=True))
    _lim_ha = useful2.SFR_K12('ha',lim_sfr,inv=True)
    lim_ha  = dust_ha['H01'].apply_dust(np.log10(_lim_ha))
    print lim_uv, lim_sfr, lim_ha, z2_LF_refs['sobral13']['coeff'][1]+np.log10(0.03)

    luv = calc_luv_density(coeff=z2_LF_refs['mehta16' ]['coeff'],lolim=-50,hilim=lim_uv,dust='M99')
    lha = calc_lha_density(coeff=z2_LF_refs['sobral13']['coeff'],lolim=lim_ha,hilim= 50,dust='H01',agn=True)
    
    print useful2.SFR_K12('uv',luv), useful2.SFR_K98('uv',luv)
    print useful2.SFR_K12('ha',lha), useful2.SFR_K98('ha',lha)

def UVLD_Numbers(Mlim=-13):

    print
    print "UVLD @ z~1.7"
    M16 = get_uvld(UV_LF_refs['mehta16']['LFs'][0],Mlim,dust='none')[0]
    A16 = get_uvld(UV_LF_refs['alavi16']['LFs'][0],Mlim,dust='none')[0]
    P16 = get_uvld(UV_LF_refs['parsa16']['LFs'][0],Mlim,dust='none')[0]
    print 'z:', \
          UV_LF_refs['mehta16']['LFs'][0]['z'], \
          UV_LF_refs['alavi16']['LFs'][0]['z'], \
          UV_LF_refs['parsa16']['LFs'][0]['z']
    print 'UVLD:',  M16, A16, P16
    print 'ratio:', M16/A16, M16/P16

    print
    print "UVLD @ z~2.2"
    M16 = get_uvld(UV_LF_refs['mehta16']['LFs'][1],Mlim,dust='none')[0]
    A16 = get_uvld(UV_LF_refs['alavi16']['LFs'][1],Mlim,dust='none')[0]
    P16 = get_uvld(UV_LF_refs['parsa16']['LFs'][2],Mlim,dust='none')[0]
    print 'z:', \
          UV_LF_refs['mehta16']['LFs'][1]['z'], \
          UV_LF_refs['alavi16']['LFs'][1]['z'], \
          UV_LF_refs['parsa16']['LFs'][2]['z']
    print 'UVLD:',  M16, A16, P16
    print 'ratio:', M16/A16, M16/P16

    print
    print "UVLD @ z~2.8"
    M16 = get_uvld(UV_LF_refs['mehta16']['LFs'][2],Mlim,dust='none')[0]
    A16 = get_uvld(UV_LF_refs['alavi16']['LFs'][2],Mlim,dust='none')[0]
    P16 = get_uvld(UV_LF_refs['parsa16']['LFs'][3],Mlim,dust='none')[0]
    print 'z:', \
          UV_LF_refs['mehta16']['LFs'][2]['z'], \
          UV_LF_refs['alavi16']['LFs'][2]['z'], \
          UV_LF_refs['parsa16']['LFs'][3]['z']
    print 'UVLD:',  M16, A16, P16
    print 'ratio:', M16/A16, M16/P16

if __name__ == '__main__':
    
    #SFRDuv_Table_Numbers()
    #SFRD_other_Numbers()
    #SFRD_Text_Numbers()
    #UVLD_Numbers()
    mk_pretty_plot(Mlim=-13)
    
    plt.show()