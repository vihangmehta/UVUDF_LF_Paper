import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from multiprocessing import Queue,Process
from joblib import Parallel, delayed

import useful2
import conversions as conv
from LF_refs import z2_LF_refs
from dust import dust_uv, dust_ha
from mcmc_sim import make_dummy_mcmc

def multiprocess(func,M,coeffs,num_procs=15,dust_model=None,log=""):

    split = np.array_split(np.arange(len(coeffs)),num_procs)

    def worker(queue,chunk):
        for i in chunk:
            if dust_model:
                res = func(M,*coeffs[i],dust_model=dust_model)
            else:
                res = func(M,*coeffs[i])
            items = i,res
            queue.put(items)
        queue.put(None)

    queue = Queue()
    procs = [Process(target=worker, args=(queue,chunk)) for chunk in split]
    for proc in procs: proc.start()

    index,result = [],[]
    finished, ifinished = 0,0
    while finished < num_procs:
        sys.stdout.write("\rProcessing errors %s ... %i/%i" % (log,ifinished,len(coeffs)))
        sys.stdout.flush()
        items = queue.get()
        if items == None:
            finished += 1
        else:
            i, res = items
            index.append(i)
            result.append(res)
            ifinished += 1
    for proc in procs: proc.join()
    sys.stdout.write("\rProcessing errors %s ... done!\033[K\n" % log)
    sys.stdout.flush()

    index, result = np.array(index), np.array(result)
    return result[np.argsort(index)]

def calc_z2_LFs(ref,N=1000):

    coeff = z2_LF_refs[ref]
    
    try:
        chain = np.genfromtxt(coeff["fname"])
    except KeyError:
        chain = make_dummy_mcmc(coeff["coeff"],coeff["err"])

    chain = chain.reshape((-1,10,3))
    chain = chain[200:,:,:].reshape((-1,3))

    if   coeff['wave'] is 'uv':
    
        x  = np.arange(-50,0,0.05)
        dust_dict       = dust_uv
        LF_func         = useful2.UV_LF
        LF_func_dustcor = useful2.UV_LF_dustcor
        CLF_func        = useful2.UV_CLF
    
    elif coeff['wave'] is 'ha':

        x = np.arange(30,50,0.01)
        dust_dict       = dust_ha
        LF_func         = lambda *args,**kwargs: useful2.Ha_LF(*args,agn=z2_LF_refs[ref]['agn'],**kwargs)
        LF_func_dustcor = lambda *args,**kwargs: useful2.Ha_LF_dustcor(*args,agn=z2_LF_refs[ref]['agn'],**kwargs)
        CLF_func        = lambda *args,**kwargs: useful2.Ha_CLF(*args,agn=z2_LF_refs[ref]['agn'],**kwargs)

    dtype = [('xlum',float,1),('SFR',float,1),('LF',float,1),('eLF',float,2),('SFRF',float,1),('eSFRF',float,2)]
    for dust in dust_dict.keys(): dtype.extend([("%s_%s"%(i[0],dust),i[1],i[2]) for i in dtype])
    dtype.extend([('CLF',float,1),('eCLF',float,2),('CSFRF',float,1),('eCSFRF',float,2)])

    LF = np.recarray(len(x),dtype=dtype)
    LF['xlum']   = x
    
    if   coeff['wave'] is 'uv':
        LF['SFR'] = useful2.SFR_K12(coeff['wave'],10**conv.get_absM_from_Lnu(LF['xlum'],inv=True))
    elif coeff['wave'] is 'ha':
        LF['SFR'] = useful2.SFR_K12(coeff['wave'],10**LF['xlum'])
    
    _LF  = multiprocess( LF_func,M=LF['xlum'],coeffs=chain[np.random.randint(len(chain),size=N)],log="%s  LF ( no dust)"%ref)
    _CLF = multiprocess(CLF_func,M=LF['xlum'],coeffs=chain[np.random.randint(len(chain),size=N)],log="%s CLF ( no dust)"%ref)

    LF[  'LF'] = LF_func( LF['xlum'],*coeff['coeff'])
    LF[ 'eLF'] = np.percentile( _LF,q=[50-68.27/2,50+68.27/2],axis=0).T
    LF[ 'CLF'] = CLF_func(LF['xlum'],*coeff['coeff'])
    LF['eCLF'] = np.percentile(_CLF,q=[50-68.27/2,50+68.27/2],axis=0).T

    LF[  'SFRF'] = 2.5*LF[ 'LF'] if coeff['wave'] is 'uv' else LF[ 'LF']
    LF[ 'eSFRF'] = 2.5*LF['eLF'] if coeff['wave'] is 'uv' else LF['eLF']
    LF[ 'CSFRF'] = LF['CLF']
    LF['eCSFRF'] = LF['eCLF']

    for dust in dust_dict.keys():
        
        LF['xlum_%s'%dust]   = dust_dict[dust].remove_dust(LF['xlum'])
        
        if   coeff['wave'] is 'uv':
            LF['SFR_%s'%dust] = useful2.SFR_K12(coeff['wave'],10**conv.get_absM_from_Lnu(LF['xlum_%s'%dust],inv=True))
        elif coeff['wave'] is 'ha':
            LF['SFR_%s'%dust] = useful2.SFR_K12(coeff['wave'],10**LF['xlum_%s'%dust])

        _LF = multiprocess(LF_func_dustcor,M=LF['xlum_%s'%dust],
                        coeffs=chain[np.random.randint(len(chain),size=N)],
                        dust_model=dust_dict[dust],
                        log="%s  LF (%s dust)"%(ref,dust))
        LF[ 'LF_%s'%dust] = LF_func_dustcor(LF['xlum_%s'%dust],*coeff['coeff'],dust_model=dust_dict[dust])
        LF['eLF_%s'%dust] = np.percentile(_LF,q=[50-68.27/2,50+68.27/2],axis=0).T

        LF[  'SFRF_%s'%dust] = 2.5*LF[ 'LF_%s'%dust] if coeff['wave'] is 'uv' else LF[ 'LF_%s'%dust]
        LF[ 'eSFRF_%s'%dust] = 2.5*LF['eLF_%s'%dust] if coeff['wave'] is 'uv' else LF['eLF_%s'%dust]

    fitsio.writeto('output/z2_LF_%s.fits' % ref, LF, clobber=True)

def plot_LFs():

    M16 = fitsio.getdata('output/z2_LF_mehta16.fits')
    H10 = fitsio.getdata('output/z2_LF_hayes10.fits')
    S13 = fitsio.getdata('output/z2_LF_sobral13.fits')

    fig,axes = plt.subplots(2,1,figsize=(10,11),dpi=75,tight_layout=True)

    axes[0].plot(        M16['xlum'],M16['LF'],lw=1.5,c='k',label='Mehta16')
    axes[0].fill_between(M16['xlum'],M16['eLF'][:,0],M16['eLF'][:,1],color='k',alpha=0.2)

    axes[1].plot(        H10['xlum'],H10['LF'],lw=1.5,c='r',label='Hathi10')
    axes[1].fill_between(H10['xlum'],H10['eLF'][:,0],H10['eLF'][:,1],color='r',alpha=0.2)
    axes[1].plot(        S13['xlum'],S13['LF'],lw=1.5,c='b',label='Sobral13')
    axes[1].fill_between(S13['xlum'],S13['eLF'][:,0],S13['eLF'][:,1],color='b',alpha=0.2)

    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylim(10**-4.75,10**-1.1)
        ax.set_ylabel('$\Phi$(L) dlogL')
        ax.legend(fontsize=20)

    axes[0].set_xlim(-14,-22)
    axes[0].set_xlabel('rest-UV absolute magnitude')
    axes[1].set_xlim(40.8,43.6)
    axes[1].set_xlabel('log H$\\alpha$ Luminosity')

    fig.savefig('plots/z2_LFs.png')

def plot_Hbeta_LF():

    fig,ax = plt.subplots(1,1,figsize=(10,7),dpi=75,tight_layout=True)

    Lha = np.arange(40,45,0.1)
    Lhb = Lha - np.log10(2.86)

    pars_hb = (-1.6,41.5,-3.18)
    LF_hb = useful2.Ha_LF(Lhb,*pars_hb)
    ax.plot(Lha,LF_hb,c='k',lw=2,label='Ciardullo13 (H$\\beta$)')
    print pars_hb[1] - np.log10(2.86)

    for ref,c in zip(['hayes10','sobral13'],['r','b']):

        pars_ha = z2_LF_refs[ref]['coeff']
        LF_ha = useful2.Ha_LF(Lha,*pars_ha,agn=z2_LF_refs[ref]['agn'])
        ax.plot(Lha,LF_ha,c=c,lw=2,alpha=0.8,label=ref.capitalize())

    ax.set_ylim(1e-5,1e0)
    ax.set_yscale('log')
    ax.set_ylabel('$\\phi$ [Mpc$^{-3}$]')
    ax.set_xlabel('H$\\alpha$ Luminosity')

if __name__ == '__main__':
    
    # calc_z2_LFs('mehta16')
    # calc_z2_LFs('hayes10')
    # calc_z2_LFs('sobral13')

    plot_LFs()
    #plot_Hbeta_LF()
    plt.show()