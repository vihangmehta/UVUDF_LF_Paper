import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from joblib import Parallel, delayed

import useful
import uvudf_utils as utils
import numbers_LF as nLF
import mk_sample
import veff
from plot_mcmc import MCMC_Output
from LF_refs import M16_LF_pars

def setup_figure():

    fig, ax = plt.subplots(1,1,figsize=(12,7),dpi=100,tight_layout=False)
    fig.subplots_adjust(left=0.11,bottom=0.13,right=0.96,top=0.98)
    dax = fig.add_axes([0.22, 0.72, 0.17, 0.24])
    return fig,ax,dax

def axes_decoration(ax,drop_filt,title,fontsize=18):

    _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()]
    ax.set_ylabel('$\phi$ [Mpc$^{-3}$ mag$^{-1}$]',fontsize=24)
    ax.set_xlabel('Rest 1500$\AA$ magnitude',fontsize=24)
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(which='major', length=5, width='1')
    ax.tick_params(which='minor', length=3, width='1')
    ax.tick_params(axis="x",which='both',direction='in',top="on")
    ax.tick_params(axis="y",which='both',direction='in',right="on")
    ax.text(0.5,0.95,title,va='top',ha='center',fontsize=24,fontweight=600,transform=ax.transAxes)
    leg = ax.legend(loc=4,fontsize=14,scatterpoints=1,fancybox=True,frameon=False,handlelength=0,handletextpad=0)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_color())
        # wht = 800 if "This work" in txt.get_text() else 400
        wht = 800 if "Mehta+17" in txt.get_text() else 400
        txt.set_fontproperties(FontProperties(size=fontsize,weight=wht))
        hndl.set_visible(False)

def inset_decoration(dax):

    dax.set_xlabel(r'M$^\star$')
    dax.set_ylabel(r'$\alpha$')
    dax.minorticks_on()
    dax.tick_params(which='major', length=5, width='1')
    dax.tick_params(which='minor', length=3, width='1')
    dax.tick_params(axis="x",which='both',direction='in',top="on")
    dax.tick_params(axis="y",which='both',direction='in',right="on")

def plot_hist(sample, drop_filt, sample_type, axis, ec, fc, label=None, comp=True):

    binc, hist, err, _ = nLF.get_LF_numbers(sample=sample, drop_filt=drop_filt, sample_type=sample_type, comp=comp)
    axis.scatter(binc, hist, color=ec, edgecolor=ec, facecolor=fc, s=50, lw=1.5, alpha=1.0, label=label)
    if comp: axis.errorbar(binc, hist, yerr=err, fmt='ko', color=ec, ecolor=ec, markersize=0, lw=1.5, capthick=1.5, capsize=4, alpha=1.0)

def LF_worker(x,coeff):
    return useful.sch(x,coeff[:-1]) * 10**coeff[-1]

def plot_LF(coeff, lim_M, axis, c, lw, alpha, ls='-', zorder=5, label=None):

    x = np.arange(lim_M[0],lim_M[1],0.01)
    y = LF_worker(x,coeff)
    line, = axis.plot(x,y,color=c,lw=lw,ls=ls,alpha=alpha,label=label,zorder=zorder)
    return line

def plot_mcmc(drop_filt,sample_type,lim_M,axis,daxis,c,best_pars=None):

    mcmc = MCMC_Output(drop_filt=drop_filt,sample_type=sample_type)
    mcmc.plot_2D_contours(xdata=mcmc.chain2d[:,1],ydata=mcmc.chain2d[:,0],axis=daxis,c=c)
    daxis.scatter(mcmc.best_pars[1],mcmc.best_pars[0],c=c,marker='x',s=50,lw=2)

    x = np.arange(lim_M[0],lim_M[1],0.01)
    y = np.array(Parallel(n_jobs=15)(delayed(LF_worker)(x,coeff) for coeff in mcmc.chain2d[np.random.randint(len(mcmc.chain2d),size=1000)]))
    axis.fill_between(x,*np.percentile(y,[16,84],axis=0),color=c,lw=0,alpha=0.25)

def plot_others(pars,data,lim_M,axis,marker,c,s=20,lw=2,alpha=0.8,zorder=10,label=None,zlabel=None):

    if zlabel: label += ' (%s)' % zlabel
    plot_LF(coeff=pars, lim_M=lim_M, axis=axis, c=c, lw=lw, alpha=alpha, zorder=zorder, label=label)
    
    if data is not None:
        M, phi, err = data['M'], data['phi'], data['err']
        axis.scatter(M,phi,marker=marker,color=c,s=s,lw=lw,alpha=alpha)
        axis.errorbar(M,phi,yerr=err,fmt='ko',color=c,ecolor=c,markersize=0,lw=lw,capthick=lw,capsize=3,alpha=alpha)

def mk_pretty_plot_z1():

    drop_filt = 'f225w'
    sample_phot = mk_sample.mk_sample(drop_filt,sample_type='photoz')

    fig,ax,dax = setup_figure()

    bpz_pars = M16_LF_pars["f225w_phot"]["coeff"]
    bpz_lims = M16_LF_pars["f225w_phot"]["lims"]

    plot_LF(coeff=bpz_pars, lim_M=bpz_lims,
            axis=ax, c='k', lw=2.5, alpha=1.0, zorder=10, label='Mehta+17 (photo-z, 1.4<z<1.9)')
    #plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='none', comp=False)
    plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='k',    comp=True)

    numbers, pars, lims, labels, zlabels, colors, markers = nLF.get_other_numbers()
    for i in ['hathi','oesch','alavi']:
        plot_others(pars=pars[i]['z1'], data=numbers[i]['z1'], lim_M=lims[i]['z1'], 
                        axis=ax, c=colors[i], marker=markers[i], label=labels[i], zlabel=zlabels[i]['z1'], zorder=1)

    plot_mcmc(drop_filt=drop_filt,sample_type='photoz',lim_M=bpz_lims,axis=ax,daxis=dax,c='k')

    # AXES DECORATIONS
    ax.set_xlim(-22,-12)
    ax.set_ylim(1e-4,1e0)
    axes_decoration(ax,drop_filt,title='z~1.65')

    # INSET DECORATIONS
    dax.set_xticks([-21,-20])
    dax.set_yticks([-1.5,-1.25,-1])
    dax.set_xlim(-21.2,-19.15)
    dax.set_ylim(-1.5,-0.85)
    inset_decoration(dax)

    fig.savefig('plots/LF_z1_temp.png')
    # fig.savefig('plots/LF_z1.pdf')

def mk_pretty_plot_z2():

    drop_filt = 'f275w'
    sample_drop = mk_sample.mk_sample(drop_filt,sample_type='dropout')
    sample_phot = mk_sample.mk_sample(drop_filt,sample_type='photoz')

    fig,ax,dax = setup_figure()

    drz_pars = M16_LF_pars["f275w_drop"]["coeff"]
    drz_lims = M16_LF_pars["f275w_drop"]["lims"]
    bpz_pars = M16_LF_pars["f275w_phot"]["coeff"]
    bpz_lims = M16_LF_pars["f275w_phot"]["lims"]

    plot_LF(coeff=drz_pars, lim_M=drz_lims,
            axis=ax, c='r', lw=2.5, alpha=1.0, zorder=10, label='Mehta+17 (%s dropouts, z~2.2)' % drop_filt.upper())
    plot_LF(coeff=bpz_pars, lim_M=bpz_lims,
            axis=ax, c='k', lw=2.5, alpha=1.0, zorder=10, label='Mehta+17 (photo-z, 1.8<z<2.6)')
    #plot_LF(coeff=[-1.627,-20.07,-2.63], lim_M=bpz_lims, axis=ax, c='k', ls='--', lw=2.5, alpha=1.0, zorder=10)

    numbers, pars, lims, labels, zlabels, colors, markers = nLF.get_other_numbers()
    for i in ['reddy','hathi','oesch','alavi']:
        plot_others(pars=pars[i]['z2'], data=numbers[i]['z2'], lim_M=lims[i]['z2'], 
                        axis=ax, c=colors[i], marker=markers[i], label=labels[i], zlabel=zlabels[i]['z2'], zorder=1)

    #plot_hist(sample_drop, drop_filt=drop_filt, sample_type='dropout', axis=ax, ec='r', fc='none', comp=False)
    plot_hist(sample_drop, drop_filt=drop_filt, sample_type='dropout', axis=ax, ec='r', fc='r', comp=True)

    #plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='none', comp=False)
    plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='k', comp=True)

    plot_mcmc(drop_filt=drop_filt,sample_type='dropout',lim_M=drz_lims,axis=ax,daxis=dax,c='r')
    plot_mcmc(drop_filt=drop_filt,sample_type='photoz' ,lim_M=bpz_lims,axis=ax,daxis=dax,c='k')

    # AXES DECORATIONS
    ax.set_xlim(-22.8,-12)
    ax.set_ylim(2e-6,2e0)
    axes_decoration(ax,drop_filt,title='z~2.2')

    # INSET DECORATIONS
    dax.set_xticks([-21,-20,-19])
    dax.set_yticks([-2,-1.5,-1,-0.5])
    dax.set_xlim(-21.9,-18.5)
    dax.set_ylim(-2.3,-0.25)
    inset_decoration(dax)

    fig.savefig('plots/LF_z2_temp.png')
    # fig.savefig('plots/LF_z2.pdf')

def mk_pretty_plot_z3():

    drop_filt = 'f336w'
    sample_drop = mk_sample.mk_sample(drop_filt,sample_type='dropout')
    sample_phot = mk_sample.mk_sample(drop_filt,sample_type='photoz')

    fig,ax,dax = setup_figure()

    drz_pars = M16_LF_pars["f336w_drop"]["coeff"]
    drz_lims = M16_LF_pars["f336w_drop"]["lims"]
    bpz_pars = M16_LF_pars["f336w_phot"]["coeff"]
    bpz_lims = M16_LF_pars["f336w_phot"]["lims"]

    plot_LF(coeff=drz_pars, lim_M=drz_lims,
            axis=ax, c='r', lw=2.5, alpha=1.0, zorder=10, label='Mehta+17 (%s dropouts, z~3)' % drop_filt.upper())
    plot_LF(coeff=bpz_pars, lim_M=bpz_lims,
            axis=ax, c='k', lw=2.5, alpha=1.0, zorder=10, label='Mehta+17 (photo-z, 2.4<z<3.6)')

    numbers, pars, lims, labels, zlabels, colors, markers = nLF.get_other_numbers()
    for i in ['reddy','hathi','oesch','alavi']:
        plot_others(pars=pars[i]['z3'], data=numbers[i]['z3'], lim_M=lims[i]['z3'], 
                        axis=ax, c=colors[i], marker=markers[i], label=labels[i], zlabel=zlabels[i]['z3'], zorder=1)
    
    #plot_hist(sample_drop, drop_filt=drop_filt, sample_type='dropout', axis=ax, ec='r', fc='none', comp=False)
    plot_hist(sample_drop, drop_filt=drop_filt, sample_type='dropout', axis=ax, ec='r', fc='r',    comp=True)

    #plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='none', comp=False)
    plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='k',    comp=True)

    plot_mcmc(drop_filt=drop_filt,sample_type='dropout',lim_M=drz_lims,axis=ax,daxis=dax,c='r')
    plot_mcmc(drop_filt=drop_filt,sample_type='photoz' ,lim_M=bpz_lims,axis=ax,daxis=dax,c='k')

    # AXES DECORATIONS
    ax.set_xlim(-22.8,-12)
    ax.set_ylim(2e-6,2e0)
    axes_decoration(ax,drop_filt,title='z~3')

    # INSET DECORATIONS
    dax.set_xticks([-22,-21,-20])
    dax.set_yticks([-1.75,-1.5,-1.25,-1])
    dax.set_xlim(-22.55,-19.6)
    dax.set_ylim(-1.8,-0.85)
    inset_decoration(dax)

    fig.savefig('plots/LF_z3_temp.png')
    # fig.savefig('plots/LF_z3.pdf')

if __name__ == '__main__':

    mk_pretty_plot_z1()
    mk_pretty_plot_z2()
    mk_pretty_plot_z3()
    # plt.show()