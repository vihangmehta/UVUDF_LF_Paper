import numpy as np
import scipy.optimize
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import useful
import pickles
import tracks
import uvudf_utils as utils
import mk_sample
from sample_selection import mk_dropout_cuts, mk_photoz_cuts, mk_photoz_cuts_with_dropout_sncut
from completeness import Comp_Func_1D

comp = Comp_Func_1D()

def setup_figure(sample, title, zlabel, z_label):

    print "Making Figure:", title
    fig, axes = plt.subplots(1,3,figsize=(15,7),dpi=75,tight_layout=False)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.1,top=0.92,wspace=0.25)

    daxes = []
    for axis in axes:

        divider = make_axes_locatable(axis)
        daxis = divider.append_axes("top", 1.5, pad=0.8)
        plt.setp(daxis.get_yticklabels(),visible=False)
        daxes.append(daxis)

        axis.set_xlim(-1,5)
        axis.set_ylim(-1,6)
        daxis.set_xlabel(z_label)
        daxis.set_xlim(0.5,4.2)
        #daxis.set_ylim(0,1.1)

    plot_colcut_patches('f225w', axes[0])
    plot_colcut_patches('f275w', axes[1])
    plot_colcut_patches('f336w', axes[2])

    plot_colcol(sample, *utils.filt_colcol('f225w'), catalog=None, zlabel=zlabel, axis=axes[0], daxis=None, marker='o', c='k', s=5, lw=0.5, alpha=0.05)
    plot_colcol(sample, *utils.filt_colcol('f275w'), catalog=None, zlabel=zlabel, axis=axes[1], daxis=None, marker='o', c='k', s=5, lw=0.5, alpha=0.05)
    plot_colcol(sample, *utils.filt_colcol('f336w'), catalog=None, zlabel=zlabel, axis=axes[2], daxis=None, marker='o', c='k', s=5, lw=0.5, alpha=0.05)

    axes[0].set_xlabel('F275W - F336W')
    axes[0].set_ylabel('F225W - F275W')
    axes[1].set_xlabel('F336W - F435W')
    axes[1].set_ylabel('F275W - F336W')
    axes[2].set_xlabel('F435W - F606W')
    axes[2].set_ylabel('F336W - F435W')

    fig.suptitle(title)

    return fig, axes, daxes

def plot_colcut_patches(drop_filt, axis):

    if   drop_filt == 'f225w':
        #verts = [[-0.2,1.3], [0.73077,1.3], [1.2,1.91], [1.2,50], [-0.2,50]]
        verts = [[-0.5,0.75], [0.7006,0.75], [1.4,1.918], [1.4,50], [-0.5,50]]
        poly = patches.Polygon(verts, color='k', lw=2.0, alpha=0.15, closed=True)
        axis.add_patch(poly)
        return axis

    elif drop_filt == 'f275w':
        verts = [[-0.2,1.0], [0.5,1.0], [1.2,1.91], [1.2,50], [-0.2,50]]
        poly = patches.Polygon(verts, color='k', lw=2.0, alpha=0.15, closed=True)
        axis.add_patch(poly)
        return axis

    elif drop_filt == 'f336w':
        verts = [[-0.2,0.8], [0.34615,0.8], [1.2,1.91], [1.2,50], [-0.2,50]]
        poly = patches.Polygon(verts, color='k', lw=2.0, alpha=0.15, closed=True)
        axis.add_patch(poly)
        return axis

    else:
        raise Exception("Invalid dropout filter.")

def plot_z_hist(daxis, z, full_z, c, lw, ls='-'):

    bins = np.arange(0,5,0.1)
    binc = 0.5*(bins[1:] + bins[:-1])
    _hist = np.histogram(full_z, bins=bins)[0]
    hist  = np.histogram(z, bins=bins)[0]
    cond = (_hist > 0)
    norm = 1/float(sum(hist)) if sum(hist)>100 else 1/1000.
    tmp = norm*hist[cond]/_hist[cond].astype(float)
    daxis.step(binc[cond], norm*hist[cond]/_hist[cond].astype(float), c=c, lw=lw, ls=ls, where='mid')

def plot_colcol(data, filt1, filt2, filt3, catalog, axis, zlabel=None, daxis=None,
                    marker='o', c='k', ec=None, fc=None, s=10, lw=1, alpha=1, label=None):

    if not ec and not fc: ec,fc = c,c

    cut_data = data[(np.abs(data[utils.filt_key[filt2]])!=99.) & (np.abs(data[utils.filt_key[filt3]])!=99.)]
    det_data = cut_data[np.abs(cut_data[utils.filt_key[filt1]]) != 99.]
    non_data = cut_data[np.abs(cut_data[utils.filt_key[filt1]]) == 99.]

    detx, dety = det_data[utils.filt_key[filt2]] - det_data[utils.filt_key[filt3]], det_data[utils.filt_key[filt1]] - det_data[utils.filt_key[filt2]]
    axis.scatter(detx, dety, marker=marker, edgecolor=ec, facecolor=fc, s=s, lw=lw, alpha=alpha, label=label)

    for entry in non_data:
        nonx, nony = entry[utils.filt_key[filt2]] - entry[utils.filt_key[filt3]], comp.get_limit(filt1,1.) - entry[utils.filt_key[filt2]]
        axis.arrow(nonx, nony, 0, 0.08,
                   head_length=0.025,head_width=0.03,lw=lw,fc=fc,ec=ec,alpha=alpha*0.75)

    if daxis:
        plot_z_hist(daxis, data[zlabel], catalog[zlabel], c, lw)

def plot_sample_list(sample_list, catalog, zlabel, axes, daxes, marker, c, s, lw, alpha):

    sample225, sample275, sample336 = sample_list
    plot_colcol(sample225, *utils.filt_colcol('f225w'), catalog=catalog, zlabel=zlabel, axis=axes[0], daxis=daxes[0], marker=marker, c=c, s=s, lw=lw, alpha=alpha)
    plot_colcol(sample275, *utils.filt_colcol('f275w'), catalog=catalog, zlabel=zlabel, axis=axes[1], daxis=daxes[1], marker=marker, c=c, s=s, lw=lw, alpha=alpha)
    plot_colcol(sample336, *utils.filt_colcol('f336w'), catalog=catalog, zlabel=zlabel, axis=axes[2], daxis=daxes[2], marker=marker, c=c, s=s, lw=lw, alpha=alpha)

def plot_obs_cat():

    catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
    catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]

    fig, axes, daxes = setup_figure(catalog, title='UVUDF Catalog v2.0', zlabel='ZB_B', z_label='BPZ')

    drop225 = mk_sample.mk_sample(drop_filt='f225w',sample_type='dropout',return_catalog=True)
    drop275 = mk_sample.mk_sample(drop_filt='f275w',sample_type='dropout',return_catalog=True)
    drop336 = mk_sample.mk_sample(drop_filt='f336w',sample_type='dropout',return_catalog=True)
    drop_list = [drop225, drop275, drop336]
    plot_sample_list(drop_list, catalog=catalog, zlabel='ZB_B', axes=axes, daxes=daxes, marker='o', c='r', s=10, lw=2, alpha=0.4)

    phot225 = mk_sample.mk_sample(drop_filt='f225w',sample_type='photoz',return_catalog=True)
    phot275 = mk_sample.mk_sample(drop_filt='f275w',sample_type='photoz',return_catalog=True)
    phot336 = mk_sample.mk_sample(drop_filt='f336w',sample_type='photoz',return_catalog=True)
    phot_list = [phot225, phot275, phot336]
    plot_sample_list(phot_list, catalog=catalog, zlabel='ZB_B', axes=axes, daxes=daxes, marker='o', c='b', s=10, lw=2, alpha=0.5)

    grism225 = catalog[(utils.bpz_lims['f225w'][0] < catalog['GRISM_Z']) & (catalog['GRISM_Z'] < utils.bpz_lims['f225w'][1])]
    grism275 = catalog[(utils.bpz_lims['f275w'][0] < catalog['GRISM_Z']) & (catalog['GRISM_Z'] < utils.bpz_lims['f275w'][1])]
    grism336 = catalog[(utils.bpz_lims['f336w'][0] < catalog['GRISM_Z']) & (catalog['GRISM_Z'] < utils.bpz_lims['f336w'][1])]
    grism_list = [grism225, grism275, grism336]
    plot_sample_list(grism_list, catalog=catalog, zlabel='GRISM_Z', axes=axes, daxes=daxes, marker='s', c='limegreen', s=25, lw=2, alpha=1)

    specz225 = catalog[(utils.bpz_lims['f225w'][0] < catalog['SPECZ_Z']) & (catalog['SPECZ_Z'] < utils.bpz_lims['f225w'][1])]
    specz275 = catalog[(utils.bpz_lims['f275w'][0] < catalog['SPECZ_Z']) & (catalog['SPECZ_Z'] < utils.bpz_lims['f275w'][1])]
    specz336 = catalog[(utils.bpz_lims['f336w'][0] < catalog['SPECZ_Z']) & (catalog['SPECZ_Z'] < utils.bpz_lims['f336w'][1])]
    specz_list = [specz225, specz275, specz336]
    plot_sample_list(specz_list, catalog=catalog, zlabel='SPECZ_Z', axes=axes, daxes=daxes, marker='s', c='darkorange', s=25, lw=2, alpha=1)

    print "GRISM_Z:", len(grism225), len(grism275), len(grism336)
    print "SPEC_Z :", len(specz225), len(specz275), len(specz336)

    fig.savefig('plots/colcuts_sample.png')

def plot_sim_cat_input():

    catalog_input = utils.read_simulation_output(run0=True,run7=True,run9=True)[0]

    fig, axes, daxes = setup_figure(catalog_input, title='Simulations -- Input', zlabel='z', z_label='Input z')

    true225 = catalog_input[(utils.bpz_lims['f225w'][0]<catalog_input['z']) & (catalog_input['z']<utils.bpz_lims['f225w'][1])]
    true275 = catalog_input[(utils.bpz_lims['f275w'][0]<catalog_input['z']) & (catalog_input['z']<utils.bpz_lims['f275w'][1])]
    true336 = catalog_input[(utils.bpz_lims['f336w'][0]<catalog_input['z']) & (catalog_input['z']<utils.bpz_lims['f336w'][1])]
    true_list = [true225, true275, true336]
    plot_sample_list(true_list, catalog_input, zlabel='z', axes=axes, daxes=daxes, marker='o', c='limegreen', s=5, lw=2, alpha=0.3)

    phot225 = mk_photoz_cuts(catalog_input, 'f225w', zlabel='z', calc_sn=False)
    phot275 = mk_photoz_cuts(catalog_input, 'f275w', zlabel='z', calc_sn=False)
    phot336 = mk_photoz_cuts(catalog_input, 'f336w', zlabel='z', calc_sn=False)
    phot_list = [phot225, phot275, phot336]
    plot_sample_list(phot_list, catalog_input, zlabel='z', axes=axes, daxes=daxes, marker='o', c='b', s=5, lw=2, alpha=0.1)

    drop225 = mk_dropout_cuts(catalog_input, 'f225w', calc_sn=False)
    drop275 = mk_dropout_cuts(catalog_input, 'f275w', calc_sn=False)
    drop336 = mk_dropout_cuts(catalog_input, 'f336w', calc_sn=False)
    drop_list = [drop225, drop275, drop336]
    plot_sample_list(drop_list, catalog_input, zlabel='z', axes=axes, daxes=daxes, marker='o', c='r', s=5, lw=2, alpha=0.1)

    fig.savefig('plots/colcuts_sim_input.png')

def plot_sim_cat_recov():

    catalog_input,catalog_recov = utils.read_simulation_output(run0=True,run7=True,run9=True)[:-1]
    catalog_recov['ZB'] = catalog_input['z']

    fig, axes, daxes = setup_figure(catalog_recov, title='Simulations -- Recovered', zlabel='ZB', z_label='Input z')

    true225 = catalog_recov[(utils.bpz_lims['f225w'][0]<catalog_input['z']) & (catalog_input['z']<utils.bpz_lims['f225w'][1])]
    true275 = catalog_recov[(utils.bpz_lims['f275w'][0]<catalog_input['z']) & (catalog_input['z']<utils.bpz_lims['f275w'][1])]
    true336 = catalog_recov[(utils.bpz_lims['f336w'][0]<catalog_input['z']) & (catalog_input['z']<utils.bpz_lims['f336w'][1])]
    true_list = [true225, true275, true336]
    plot_sample_list(true_list, catalog_recov, zlabel='ZB', axes=axes, daxes=daxes, marker='o', c='limegreen', s=5, lw=2, alpha=0.3)

    phot225 = mk_photoz_cuts(catalog_recov, 'f225w')
    phot275 = mk_photoz_cuts(catalog_recov, 'f275w')
    phot336 = mk_photoz_cuts(catalog_recov, 'f336w')
    phot_list = [phot225, phot275, phot336]
    plot_sample_list(phot_list, catalog_recov, zlabel='ZB', axes=axes, daxes=daxes, marker='o', c='b', s=5, lw=2, alpha=0.1)
    
    drop225 = mk_dropout_cuts(catalog_recov, 'f225w')
    drop275 = mk_dropout_cuts(catalog_recov, 'f275w')
    drop336 = mk_dropout_cuts(catalog_recov, 'f336w')
    drop_list = [drop225, drop275, drop336]
    plot_sample_list(drop_list, catalog_recov, zlabel='ZB', axes=axes, daxes=daxes, marker='o', c='r', s=5, lw=2, alpha=0.1)

    fig.savefig('plots/colcuts_sim_recov.png')

def mk_pretty_plot():

    catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
    catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]

    for drop_filt,title in zip(['f225w','f275w','f336w'],['z~1.65','z~2.2','z~3']):

        filt1,filt2,filt3 = utils.filt_colcol(drop_filt)
        zlim = utils.bpz_lims[drop_filt]

        drop = mk_sample.mk_sample(drop_filt,sample_type='dropout',return_catalog=True)

        # Setup Figure and plot the selection regions
        fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=75,tight_layout=True)
        plot_colcut_patches(drop_filt, ax)

        # Plot the stars from Pickles library
        starx,stary = pickles.get_pickles_colcol(drop_filt=drop_filt)
        ax.scatter(starx,stary,c='gold',s=80,marker='*',lw=0, alpha=1)

        # Plot the BC03 SF tracks
        z, (colx1,coly1), (colx2,coly2), (colx3,coly3) = tracks.get_tracks(drop_filt,bc03=True)

        ax.plot(colx1,coly1,c='b',lw=2,ls='-',alpha=1)
        ax.plot(colx2,coly2,c='b',lw=2,ls='--',alpha=1)
        ax.plot(colx3,coly3,c='b',lw=2,ls=':',alpha=1)

        # cond = (zlim[0]<=z) & (z<=zlim[1])
        # ax.plot(colx1[cond],coly1[cond],c='b',lw=5,ls='-',alpha=1)
        # ax.plot(colx2[cond],coly2[cond],c='b',lw=5,ls='--',alpha=1)
        # ax.plot(colx3[cond],coly3[cond],c='b',lw=5,ls=':',alpha=1)

        ax.scatter(colx1[::5],coly1[::5],c='b',marker='x',s=50,lw=1.5)
        ax.scatter(colx2[::5],coly2[::5],c='b',marker='x',s=50,lw=1.5)
        ax.scatter(colx3[::5],coly3[::5],c='b',marker='x',s=50,lw=1.5)

        # Plot the Coleman low-z tracks
        colx,coly = tracks.get_tracks(drop_filt,bc03=False)
        ax.plot(colx,coly,c='limegreen',lw=2,alpha=1)

        # Plot the color-color points
        plot_colcol(catalog, filt1, filt2, filt3, None, axis=ax,
                        marker='o', c='k', s=15, lw=0.25, alpha=0.25)
        plot_colcol(drop, filt1, filt2, filt3, None, axis=ax,
                        marker='o', c='r', s=35, lw=1.5, alpha=1.0)

        _ = [label.set_fontsize(20) for label in ax.get_xticklabels()+ax.get_yticklabels()]
        ax.text(0.05,0.97,'%s Dropouts\n(%s)'%(drop_filt.upper(),title),va='top',ha='left',fontsize=18,fontweight=600,transform=ax.transAxes)
        ax.set_xlabel("%s - %s" % (filt2.upper(),filt3.upper()),fontsize=24)
        ax.set_ylabel("%s - %s" % (filt1.upper(),filt2.upper()),fontsize=24)
        ax.set_xlim(-0.5,2.5)
        ax.set_ylim(-1,4)

        fig.savefig('plots/sample_%s.png' % drop_filt)
        fig.savefig('plots/sample_%s.pdf' % drop_filt)

if __name__ == '__main__':

    #plot_obs_cat()
    #plot_sim_cat_input()
    #plot_sim_cat_recov()
    mk_pretty_plot()
    plt.show()