import numpy as np
import matplotlib.pyplot as plt

import useful
import corner

def plot_walkers(drop_filt,sample_type,nwalkers=10,best_pars=None):

    fname = "output/mcmc_%s_%s.dat" % (drop_filt,sample_type[:4])
    chain = np.genfromtxt(fname)
    chain = chain.reshape((-1,nwalkers,3))

    fig1,axes = plt.subplots(3,1,figsize=(12,6),dpi=75,sharex=True)
    fig1.subplots_adjust(left=0.08,right=0.96,bottom=0.12,top=0.92,hspace=0,wspace=0)

    for i in range(nwalkers):

        axes[0].plot(np.arange(chain.shape[0])+1,chain[:,i,0],lw=0.5,alpha=0.8)
        axes[1].plot(np.arange(chain.shape[0])+1,chain[:,i,1],lw=0.5,alpha=0.8)
        axes[2].plot(np.arange(chain.shape[0])+1,chain[:,i,2],lw=0.5,alpha=0.8)

    if best_pars:
        axes[0].axhline(best_pars[0],c='k',lw=1.5,ls='--')
        axes[1].axhline(best_pars[1],c='k',lw=1.5,ls='--')
        axes[2].axhline(best_pars[2],c='k',lw=1.5,ls='--')

    axes[0].set_ylabel('$\\alpha$')
    axes[1].set_ylabel('M$^\\star$')
    axes[2].set_ylabel('$\\phi^\\star$')
    axes[2].set_xlabel('Step #')
    fig1.suptitle("%s %s Sample" % (drop_filt.upper(),sample_type.capitalize()),fontsize=20)

    chain = chain[200:,:,:].reshape((-1,3))
    fig2 = corner.corner(chain,labels=['$\\alpha$','M$^\\star$','$\\phi^\\star$'],
                        quantiles=[0.16,0.5,0.84],show_titles=True,truths=best_pars,
                        title_kwargs={"fontsize":16},label_kwargs={"fontsize": 16})
    
    for ax in fig1.get_axes()+fig2.get_axes():
        _ = [i.set_fontsize(14) for i in ax.get_xticklabels()]
        _ = [i.set_fontsize(14) for i in ax.get_yticklabels()]
    
    fig2.text(0.9,0.9,"%s %s Sample" % (drop_filt.upper(),sample_type.capitalize()),
              fontsize=20,va='top',ha='right',transform=fig2.transFigure)
    fig2.set_size_inches(10,8)

if __name__ == '__main__':
    
    plot_walkers(drop_filt='f275w',sample_type='dropout',best_pars=[-1.31,-19.66,-2.21])
    #plot_walkers(drop_filt='f336w',sample_type='dropout',best_pars=[-1.35,-20.71,-2.40])

    #plot_walkers(drop_filt='f225w',sample_type='photoz',best_pars=[-1.20,-19.93,-2.12])
    #plot_walkers(drop_filt='f275w',sample_type='photoz',best_pars=[-1.32,-19.92,-2.31])
    #plot_walkers(drop_filt='f336w',sample_type='photoz',best_pars=[-1.40,-20.41,-2.43])
    
    plt.show()