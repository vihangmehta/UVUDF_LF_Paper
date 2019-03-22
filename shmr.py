import numpy as np
import matplotlib.pyplot as plt

def parameters(z):

    a = 1./(1+z)

    nu = np.exp(-4*a*a)
    log_M1 = 11.514 + (-1.793*(a-1) - 0.251*z) * nu
    log_e  = -1.777 + (-0.006*(a-1)          ) * nu - 0.119 * (a-1)
    alpha  = -1.412 + ( 0.731*(a-1)          ) * nu
    delta  =  3.508 + ( 2.608*(a-1) - 0.043*z) * nu
    gamma  =  0.316 + ( 1.319*(a-1) + 0.279*z) * nu

    return {'M1': log_M1,
            'e' : log_e,
            'alpha': alpha,
            'delta': delta,
            'gamma': gamma}

def shmr(Mh,z):

    pars = parameters(z=z)
    f = lambda x: -np.log10(10**(pars['alpha']*x)+1) + \
                  pars['delta'] * (np.log10(1+np.exp(x)))**pars['gamma'] / (1+np.exp(10**-x))
    Mst = (pars['e']+pars['M1']) + f(Mh - pars['M1']) - f(0)
    return Mst

def plot():

    Mh = np.arange(10,15,0.1)
    zz = np.append(np.array([0.1,]),np.arange(7)+1)
    colors = plt.cm.gist_rainbow(np.linspace(0,0.95,len(zz)))

    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

    for z,c in zip(zz,colors):

        Mst = shmr(Mh,z=z)
        ax.plot(10**Mh,10**Mst,c=c,lw=2,alpha=0.9,label='z=%.1f'%z)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Halo Mass')
    ax.set_ylabel('Stellar Mass')
    ax.set_xlim(1e10,2e15)
    ax.set_ylim(1e7,1e12)
    ax.legend(loc=4,ncol=2,fontsize=14,frameon=False)

if __name__ == '__main__':
    
    #plot()
    print shmr(12,z=2.2)
    plt.show()