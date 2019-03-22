import numpy as np
import astropy.io.fits as fitsio
from numpy.lib.recfunctions import stack_arrays

pixscale = 0.03

filter_response = fitsio.getdata('config/udf_uvis_filters.fits')
filters = ['f225w','f275w','f336w','f435w','f606w','f775w','f850lp','f105w','f125w','f140w','f160w']
drop_filters = filters[:3]

pivot_l = {'f225w':2358.9983, 'f275w':2703.6938, 'f336w':3354.9336,
           'f435w':4322.8394, 'f606w':5912.3276, 'f775w':7699.3149, 'f850lp':9054.335,
           'f105w':10552.033, 'f125w':12486.069, 'f140w':13922.979, 'f160w':15369.161}

filt_key = {'f225w':'MAG_B_F225W','f275w':'MAG_B_F275W','f336w':'MAG_B_F336W',
            'f435w':'MAG_F435W','f606w':'MAG_F606W','f775w':'MAG_F775W','f850lp':'MAG_F850LP',
            'f105w':'MAG_F105W','f125w':'MAG_F125W','f140w':'MAG_F140W','f160w':'MAG_F160W'}

dfilt_key = {'f225w':'MAGERR_B_F225W','f275w':'MAGERR_B_F275W','f336w':'MAGERR_B_F336W',
             'f435w':'MAGERR_F435W','f606w':'MAGERR_F606W','f775w':'MAGERR_F775W','f850lp':'MAGERR_F850LP',
             'f105w':'MAGERR_F105W','f125w':'MAGERR_F125W','f140w':'MAGERR_F140W','f160w':'MAGERR_F160W'}

bpz_lims = {'f225w':[1.4,1.9], 'f275w':[1.8,2.6], 'f336w':[2.4,3.6]} # UVUDF determined from simulation dropouts

filt_det = {'f225w':'f275w', 'f275w':'f336w', 'f336w':'f435w'}

lim_M = {'f225w_drop':-18.46,'f275w_drop':-17.97,'f336w_drop':-17.37,
         'f225w_phot':-15.94,'f275w_phot':-16.30,'f336w_phot':-16.87}

@np.vectorize
def filt_1500(drop_filt,z):
    if   drop_filt=='f225w': filt_1500 = 'f435w'
    elif drop_filt=='f275w': filt_1500 = 'f435w' if z<2.2 else 'f606w'
    elif drop_filt=='f336w': filt_1500 = 'f606w'
    else: raise Exception("Incorrect drop_filt in filt_1500().")
    return filt_1500

def filt_colcol(drop_filt):

    if   drop_filt=='f225w': filt1,filt2,filt3 = 'f225w', 'f275w', 'f336w'
    elif drop_filt=='f275w': filt1,filt2,filt3 = 'f275w', 'f336w', 'f435w'
    elif drop_filt=='f336w': filt1,filt2,filt3 = 'f336w', 'f435w', 'f606w'
    else: raise Exception("Invalid dropout filter.")
    return filt1,filt2,filt3

def get_sangle():

    sr_in_deg2 = (np.pi/180.)**2
    area_in_deg2 = (7.3/3600.)
    return area_in_deg2 * sr_in_deg2

def mag_limit(filt,sig):

    depth_5sig = {'f225w':27.8, 'f275w':27.8, 'f336w':28.3, 'f435w':29.2, 'f606w':29.6, 'f775w':29.5,
                  'f850lp':28.9,'f105w':30.1, 'f125w':29.7, 'f140w':29.8, 'f160w':29.9,
                  'MAG_B_F225W':27.8, 'MAG_B_F275W':27.8, 'MAG_B_F336W':28.3, 'MAG_B_F435W':29.2,
                  'MAG_F435W':29.2, 'MAG_F606W':29.6, 'MAG_F775W':29.5, 'MAG_F850LP':28.9,
                  'MAG_F105W':30.1, 'MAG_F125W':29.7, 'MAG_F140W':29.8, 'MAG_F160W':29.9}
    depth = depth_5sig[filt] - 2.5*np.log10(sig/5.0)
    return depth

def read_simulation_output(run0=False,run7=True,run9=True):

    catalog_input0 = np.genfromtxt('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_input.txt',
                              dtype=[('ID',int),('z',float),('abs_M',float),('MAG_B_F225W',float),('MAG_B_F275W',float),
                                     ('MAG_B_F336W',float),('MAG_F435W',float),('MAG_F606W',float),('MAG_F775W',float),
                                     ('MAG_F850LP',float),('MAG_F105W',float),('MAG_F125W',float),('MAG_F140W',float),
                                     ('MAG_F160W',float),('xpx',float),('ypx',float),('n',int),('hlr',float),
                                     ('axr',float),('pa',float),('metallicity',float),('age',float),('tau',float),
                                     ('Av',float),('beta',float)],unpack=True)
    catalog_recov0 = fitsio.getdata('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_recov.fits')
    catalog_recov_hlr0 = np.genfromtxt('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_recov_hlr.txt',
                                      dtype=[('ID',float),
                                             ('f225w',float),('f275w',float),('f336w',float),
                                             ('f435w',float),('f606w',float),('f775w',float),('f850lp',float),
                                             ('f105w',float),('f125w',float),('f140w',float),('f160w',float),
                                             ('xpx',float),('ypx',float)],unpack=True)

    catalog_input7 = np.genfromtxt('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_input_run7.txt',
                              dtype=[('ID',int),('z',float),('abs_M',float),('MAG_B_F225W',float),('MAG_B_F275W',float),
                                     ('MAG_B_F336W',float),('MAG_F435W',float),('MAG_F606W',float),('MAG_F775W',float),
                                     ('MAG_F850LP',float),('MAG_F105W',float),('MAG_F125W',float),('MAG_F140W',float),
                                     ('MAG_F160W',float),('xpx',float),('ypx',float),('n',int),('hlr',float),
                                     ('axr',float),('pa',float),('metallicity',float),('age',float),('tau',float),
                                     ('Av',float),('beta',float)],unpack=True)
    catalog_recov7 = fitsio.getdata('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_recov_run7.fits')
    catalog_recov_hlr7 = np.genfromtxt('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_recov_hlr_run7.txt',
                                      dtype=[('ID',float),
                                             ('f225w',float),('f275w',float),('f336w',float),
                                             ('f435w',float),('f606w',float),('f775w',float),('f850lp',float),
                                             ('f105w',float),('f125w',float),('f140w',float),('f160w',float),
                                             ('xpx',float),('ypx',float)],unpack=True)

    catalog_input9 = np.genfromtxt('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_input_run9.txt',
                                  dtype=[('ID',int),('z',float),('abs_M',float),('MAG_B_F225W',float),('MAG_B_F275W',float),
                                         ('MAG_B_F336W',float),('MAG_F435W',float),('MAG_F606W',float),('MAG_F775W',float),
                                         ('MAG_F850LP',float),('MAG_F105W',float),('MAG_F125W',float),('MAG_F140W',float),
                                         ('MAG_F160W',float),('xpx',float),('ypx',float),('n',int),('hlr',float),
                                         ('axr',float),('pa',float),('metallicity',float),('age',float),('tau',float),
                                         ('Av',float),('beta',float)],unpack=True)
    catalog_recov9 = fitsio.getdata('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_recov_run9.fits')
    catalog_recov_hlr9 = np.genfromtxt('/data/highzgal/mehta/UVUDF/lum_func/catalog/sample_recov_hlr_run9.txt',
                                      dtype=[('ID',float),
                                             ('f225w',float),('f275w',float),('f336w',float),
                                             ('f435w',float),('f606w',float),('f775w',float),('f850lp',float),
                                             ('f105w',float),('f125w',float),('f140w',float),('f160w',float),
                                             ('xpx',float),('ypx',float)],unpack=True)

    if   run0 and not run7 and not run9:
        catalog_input = catalog_input0
        catalog_recov = catalog_recov0
        catalog_recov_hlr = catalog_recov_hlr0

    elif not run0 and run7 and not run9:
        catalog_input = catalog_input7
        catalog_recov = catalog_recov7
        catalog_recov_hlr = catalog_recov_hlr7

    elif not run0 and not run7 and run9:
        catalog_input = catalog_input9
        catalog_recov = catalog_recov9
        catalog_recov_hlr = catalog_recov_hlr9

    elif run0 and run7 and not run9:
        catalog_input = stack_arrays((catalog_input0,catalog_input7),usemask=False,asrecarray=True)
        catalog_recov = stack_arrays((catalog_recov0,catalog_recov7),usemask=False,asrecarray=True)
        catalog_recov_hlr = stack_arrays((catalog_recov_hlr0,catalog_recov_hlr7),usemask=False,asrecarray=True)

    elif run0 and not run7 and run9:
        catalog_input = stack_arrays((catalog_input0,catalog_input9),usemask=False,asrecarray=True)
        catalog_recov = stack_arrays((catalog_recov0,catalog_recov9),usemask=False,asrecarray=True)
        catalog_recov_hlr = stack_arrays((catalog_recov_hlr0,catalog_recov_hlr9),usemask=False,asrecarray=True)

    elif not run0 and run7 and run9:
        catalog_input = stack_arrays((catalog_input7,catalog_input9),usemask=False,asrecarray=True)
        catalog_recov = stack_arrays((catalog_recov7,catalog_recov9),usemask=False,asrecarray=True)
        catalog_recov_hlr = stack_arrays((catalog_recov_hlr7,catalog_recov_hlr9),usemask=False,asrecarray=True)

    elif run0 and run7 and run9:
        catalog_input = stack_arrays((catalog_input0,catalog_input7,catalog_input9),usemask=False,asrecarray=True)
        catalog_recov = stack_arrays((catalog_recov0,catalog_recov7,catalog_recov9),usemask=False,asrecarray=True)
        catalog_recov_hlr = stack_arrays((catalog_recov_hlr0,catalog_recov_hlr7,catalog_recov_hlr9),usemask=False,asrecarray=True)

    else:
      raise Exception("No simulation run selected.")

    return catalog_input, catalog_recov, catalog_recov_hlr

if __name__ == '__main__':
    print "No main() defined."