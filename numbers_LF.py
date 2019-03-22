import numpy as np
import scipy.integrate

import useful
import uvudf_utils as utils
import mk_sample

quad_args = {'limit':100,'epsrel':1e-4,'epsabs':1e-4}

def get_LF_numbers(sample, drop_filt, sample_type, comp=True, dbin=0.5, vmax=True):

    if vmax:

        if comp: vmax = sample['Vmax']
        else:    vmax = sample['Vtot']

        lim_M = utils.lim_M[drop_filt+"_"+sample_type[:4]]
        #lim_M = max(sample['M_1500'])
        bins = np.arange(abs(lim_M),25,dbin)[::-1] * (-1)
        binc = 0.5*(bins[1:] + bins[:-1])
        nums = np.histogram(sample['M_1500'],bins=bins)[0]
        hist = np.histogram(sample['M_1500'],bins=bins,weights=1./vmax)[0]

        digi = np.digitize(sample['M_1500'],bins=bins) - 1
        _ll,_ul = useful.poisson_interval(np.array([1.,]),0.6827).flatten()
        ue = np.array([(_ul-1.)/vmax[digi==i] for i in range(len(binc))])
        le = np.array([(1.-_ll)/vmax[digi==i] for i in range(len(binc))])
        ue = np.array([np.sqrt(np.sum(_ue**2)) for _ue in ue])
        le = np.array([np.sqrt(np.sum(_le**2)) for _le in le])
        err = np.vstack([le,ue])

        hist,err,binc = hist[hist!=0],err.T[hist!=0].T,binc[hist!=0]
        hist, err = hist / dbin, err / dbin

    return binc, hist, err, nums

def get_other_numbers():

    n_z1_alavi = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=0 , max_rows=8)
    n_z2_alavi = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=11, max_rows=8)
    n_z3_alavi = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=22, max_rows=8)
    n_z2_oesch = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('loerr',float),('hierr',float)],
                                    skip_header=33, max_rows=5)
    n_z2_hathi = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=41, max_rows=7)
    n_z2_sawic = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('R',float),('phi',float),('err',float)],
                                    skip_header=51, max_rows=9)
    n_z2_reddy = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=63, max_rows=10)
    n_z3_reddy = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=76, max_rows=9)
    n_z2_parsa = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=88, max_rows=16)
    n_z3_parsa = np.genfromtxt('numbers_LF.dat',dtype=[('M',float),('phi',float),('err',float)],
                                    skip_header=107,max_rows=15)

    labels  = {'alavi':  'Alavi+16',
               'oesch':  'Oesch+10',
               'hathi':  'Hathi+10',
               'cucciati':'Cucciati+12',
               'parsa':  'Parsa+16',
               'sawicki':'Sawicki12',
               'reddy':  'Reddy & Steidel09'}

    numbers = {'alavi':  {'z1': n_z1_alavi,
                          'z2': n_z2_alavi,
                          'z3': n_z3_alavi},
               'oesch':  {'z1': None,
                          'z2': None, #n_z2_oesch,
                          'z3': None},
               'hathi':  {'z1': None,
                          'z2': None, #n_z2_hathi,
                          'z3': None},
               'cucciati':{'z1': None,
                           'z2': None,
                           'z3': None},
               'parsa':  {'z1': None,
                          'z2': n_z2_parsa,
                          'z3': n_z3_parsa},
               'sawicki':{'z2': n_z2_sawic},
               'reddy':  {'z2': n_z2_reddy,
                          'z3': n_z3_reddy}}

    zlabels = {'alavi':  {'z1': '1.0<z<1.6',
                          'z2': '1.6<z<2.2',
                          'z3': '2.2<z<3.0'},
               'oesch':  {'z1': 'z~1.5',
                          'z2': 'z~1.9',
                          'z3': 'z~2.5'},
               'hathi':  {'z1': 'z~1.7',
                          'z2': 'z~2.1',
                          'z3': 'z~2.7'},
               'cucciati':{'z1': 'z~1.5',
                           'z2': 'z~2.1',
                           'z3': 'z~3.0'},
               'parsa':  {'z1': 'z~1.7',
                          'z2': 'z~2.25',
                          'z3': 'z~2.8'},
               'sawicki':{'z2': 'z~2.2'},
               'reddy':  {'z2': '1.9<z<2.7',
                          'z3': '2.7<z<3.4'}}

    pars = {'alavi':  {'z1': (-1.56,-19.74,np.log10(2.32e-3)),
                       'z2': (-1.72,-20.41,np.log10(1.50e-3)),
                       'z3': (-1.94,-20.71,np.log10(0.55e-3))},
            'oesch':  {'z1': (-1.46,-19.82,-2.64),
                       'z2': (-1.60,-20.16,-2.66),
                       'z3': (-1.73,-20.69,-2.49)},
            'hathi':  {'z1': (-1.27,-19.43,np.log10(0.00217)),
                       'z2': (-1.17,-20.39,np.log10(0.00157)),
                       'z3': (-1.52,-20.94,np.log10(0.00154))},
            'cucciati':{'z1': (-1.09,-19.60,np.log10(4.10e-3)),
                       'z2': (-1.30,-20.40,np.log10(3.37e-3)),
                       'z3': (-1.50,-21.40,np.log10(0.86e-3))},
            'parsa':  {'z1': (-1.33,-19.61,np.log10(0.00681)),
                       'z2': (-1.26,-19.71,np.log10(0.00759)),
                       'z3': (-1.31,-20.20,np.log10(0.00532))},
            'sawicki':{'z2': (-1.47,-21.00,-2.74),},
            'reddy':  {'z2': (-1.73,-20.70,np.log10(2.75e-3)),
                       'z3': (-1.73,-20.97,np.log10(1.71e-3))}}

    lims = {'alavi':  {'z1': (-20.0,-13.0),
                       'z2': (-20.0,-13.0),
                       'z3': (-20.0,-13.0)},
            'oesch':  {'z1': (-23.0,-19.2),
                       'z2': (-23.0,-19.2),
                       'z3': (-23.0,-19.2)},
            'hathi':  {'z1': (-21.0,-18.0),
                       'z2': (-21.5,-18.0),
                       'z3': (-22.0,-18.5)},
            'cucciati':{'z1': (-21.5,-18.4),
                       'z2': (-22.5,-19.8),
                       'z3': (-23.5,-20.5)},
            'parsa':  {'z1': (-22.0,-14.5),
                       'z2': (-22.0,-14.5),
                       'z3': (-23.0,-15.5)},
            'sawicki':{'z2': (-23.0,-17.9)},
            'reddy':  {'z2': (-23.0,-17.9),
                       'z3': (-23.0,-19.0)}}

    colors = {'alavi':   'blue',
              'oesch':   'darkorange',
              'hathi':   'darkcyan',
              'cucciati':'sienna',
              'parsa':   'green',
              'sawicki': 'yellow',
              'reddy':   'darkmagenta'}

    markers = {'alavi':  's',
               'oesch':  's',
               'hathi':  's',
               'cucciati':'s',
               'parsa':  's',
               'sawicki':'s',
               'reddy':  's'}

    return numbers, pars, lims, labels, zlabels, colors, markers

if __name__ == '__main__':

    for drop_filt in ['f225w','f275w','f336w']:
        for sample_type in ['dropout','photoz']:
          
          sample = mk_sample.mk_sample(drop_filt,sample_type=sample_type)
          binc, hist, err, nums = get_LF_numbers(sample=sample, drop_filt=drop_filt, sample_type=sample_type, comp=True)
          
          print drop_filt, sample_type
          print ','.join(["%.6f"%_ for _ in binc])
          print ','.join(["%i"  %_ for _ in nums])
          print ','.join(["%.6f"%(_*1e3) for _ in hist])
          print ','.join(["%.6f"%(_*1e3) for _ in err[0]])
          print ','.join(["%.6f"%(_*1e3) for _ in err[1]])