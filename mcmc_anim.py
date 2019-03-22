import sys, time
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from matplotlib import animation

import useful
import mk_sample
import plot_LFs
import numbers_LF as nLF
from LF_refs import M16_LF_pars
from plot_mcmc import MCMC_Output

def mk_plot():

    drop_filt = 'f275w'
    sample_drop = mk_sample.mk_sample(drop_filt,sample_type='dropout')
    sample_phot = mk_sample.mk_sample(drop_filt,sample_type='photoz')

    fig,ax,dax = plot_LFs.setup_figure()

    ax.plot(-99,-99,c='r',lw=2.5,label='This work (%s dropouts, z~2.2)' % drop_filt.upper())
    ax.plot(-99,-99,c='k',lw=2.5,label='This work (photo-z, 1.8<z<2.6)')
    lines = [ax.plot(-99,-99,c='k',lw=2.5)[0],]

    plot_LFs.plot_hist(sample_drop, drop_filt=drop_filt, sample_type='dropout', axis=ax, ec='r', fc='r', comp=True)
    plot_LFs.plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='k', comp=True)

    # AXES DECORATIONS
    ax.set_xlim(-22.8,-12)
    ax.set_ylim(2e-6,2e0)
    plot_LFs.axes_decoration(ax,drop_filt,title='z~2.2')

    # INSET DECORATIONS
    dax.set_xticks([-21,-20,-19])
    dax.set_yticks([-2,-1.5,-1,-0.5])
    dax.set_xlim(-21.9,-18.5)
    dax.set_ylim(-2.3,-0.25)
    plot_LFs.inset_decoration(dax)

    return fig,ax,dax,lines

def finish(ax,dax,drz_lims,bpz_lims):
    
    ax.cla()
    dax.cla()

    drop_filt = 'f275w'
    sample_drop = mk_sample.mk_sample(drop_filt,sample_type='dropout')
    sample_phot = mk_sample.mk_sample(drop_filt,sample_type='photoz')

    ax.plot(-99,-99,c='r',lw=2.5,label='This work (%s dropouts, z~2.2)' % drop_filt.upper())
    ax.plot(-99,-99,c='k',lw=2.5,label='This work (photo-z, 1.8<z<2.6)')
    lines = [ax.plot(-99,-99,c='k',lw=2.5)[0],]

    plot_LFs.plot_hist(sample_drop, drop_filt=drop_filt, sample_type='dropout', axis=ax, ec='r', fc='r', comp=True)
    plot_LFs.plot_hist(sample_phot, drop_filt=drop_filt, sample_type='photoz', axis=ax, ec='k', fc='k', comp=True)

    numbers, pars, lims, labels, zlabels, colors, markers = nLF.get_other_numbers()
    for i in ['reddy','hathi','oesch','parsa','alavi']:
        plot_LFs.plot_others(pars=pars[i]['z2'], data=numbers[i]['z2'], lim_M=lims[i]['z2'], 
                        axis=ax, c=colors[i], marker=markers[i], label=labels[i], zlabel=zlabels[i]['z2'], zorder=1)

    plot_LFs.plot_mcmc(drop_filt='f275w',sample_type='dropout',lim_M=drz_lims,axis=ax,daxis=dax,c='r')
    plot_LFs.plot_mcmc(drop_filt='f275w',sample_type='photoz', lim_M=bpz_lims,axis=ax,daxis=dax,c='k')

    # AXES DECORATIONS
    ax.set_xlim(-22.8,-12)
    ax.set_ylim(2e-6,2e0)
    plot_LFs.axes_decoration(ax,drop_filt,title='z~2.2')

    dax.set_xticks([-21,-20,-19])
    dax.set_yticks([-2,-1.5,-1,-0.5])
    dax.set_xlim(-21.9,-18.5)
    dax.set_ylim(-2.3,-0.25)
    plot_LFs.inset_decoration(dax)

def plot_mcmc_LFs(ax,pars,lim_M,c):

    lines = []
    for par in pars:
        lines.append(plot_LFs.plot_LF(coeff=par, lim_M=lim_M, axis=ax, c=c, lw=1, alpha=0.5, zorder=10))
    return lines

def plot_mcmc_err(dax,i,mcmc,c):

    if c=='r': dax.cla()

    if i < 300:
        chain = mcmc.chain2d_full[:i*mcmc.nwalkers,:]
        dax.scatter(chain[:,1],chain[:,0],c=c,s=2,lw=0,alpha=0.2)
    else:
        chain = mcmc.chain3d_full[200:i,:,:].reshape((-1,3))
        mcmc.plot_2D_contours(xdata=chain[:,1],ydata=chain[:,0],axis=dax,c=c)

    dax.set_xticks([-21,-20,-19])
    dax.set_yticks([-2,-1.5,-1,-0.5])
    dax.set_xlim(-21.9,-18.5)
    dax.set_ylim(-2.3,-0.25)
    plot_LFs.inset_decoration(dax)

def mk_animation():

    startTime = time.time()

    global lines,plot_finish
    fig,ax,dax,lines = mk_plot()
    
    drz_lims = M16_LF_pars["f275w_drop"]["lims"]
    bpz_lims = M16_LF_pars["f275w_phot"]["lims"]
    
    mcmc_drop = MCMC_Output(drop_filt='f275w',sample_type='dropout')
    mcmc_phot = MCMC_Output(drop_filt='f275w',sample_type='photoz')
    
    steps = 500#mcmc_drop.steps
    nslow = 2
    frames = nslow*(steps+25)
    plot_finish = False

    def animate(i):

        global lines,plot_finish

        stepTime = time.time()
        sys.stdout.write('\rProcessing Frame %i of %i ... (E:%.2fs,R:%.2fs) ... ' % ( \
                            i+1, frames, stepTime-startTime, (stepTime-startTime)/(i+1)*(frames-i-1)))
        sys.stdout.flush()
        
        if i%nslow!=0:
            return
        else:
            i /= nslow

        if i <= steps:
            
            for line in lines:
                line.remove()
                del line

            lines_drop = plot_mcmc_LFs(ax=ax,pars=mcmc_drop.chain3d_full[i,:,:],lim_M=drz_lims,c='r')
            lines_phot = plot_mcmc_LFs(ax=ax,pars=mcmc_phot.chain3d_full[i,:,:],lim_M=bpz_lims,c='k')
            plot_mcmc_err(dax=dax,i=i,mcmc=mcmc_drop,c='r')
            plot_mcmc_err(dax=dax,i=i,mcmc=mcmc_phot,c='k')

            lines = lines_drop
            lines.extend(lines_phot)

        else:

            if not plot_finish:
                plot_finish = True
                finish(ax,dax,drz_lims,bpz_lims)
            else:
                return

    Writer = animation.writers['ffmpeg']
    mywriter = Writer(fps=30, bitrate=5000)
    anim = animation.FuncAnimation(fig, animate, frames=frames)
    anim.save(filename='mcmc_LF_z2.mp4', fps=30, writer=mywriter, extra_args=['-vcodec','libx264'])
    print "done"

if __name__ == '__main__':
    
    mk_animation()