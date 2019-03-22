import numpy as np

from conversions import calc_sn as _calc_sn
from uvudf_utils import bpz_lims
from completeness import Comp_Func_1D

comp = Comp_Func_1D()

def mk_dropout_cuts(catalog, drop_filt, calc_sn=True, do_sn_cut=True, verbose=True, return_cond=False):

    m225 = catalog['MAG_B_F225W'].copy()
    m275 = catalog['MAG_B_F275W'].copy()
    m336 = catalog['MAG_B_F336W'].copy()
    m435 = catalog['MAG_F435W'].copy()
    m606 = catalog['MAG_F606W'].copy()

    if calc_sn:

        dm225 = catalog['MAGERR_B_F225W'].copy()
        dm275 = catalog['MAGERR_B_F275W'].copy()
        dm336 = catalog['MAGERR_B_F336W'].copy()
        dm435 = catalog['MAGERR_F435W'].copy()
        dm606 = catalog['MAGERR_F606W'].copy()

        sn225 = _calc_sn(m225,dm225)
        sn275 = _calc_sn(m275,dm275)
        sn336 = _calc_sn(m336,dm336)
        sn435 = _calc_sn(m435,dm435)
        sn606 = _calc_sn(m606,dm606)

    if drop_filt == 'f225w':

        # Hathi 10/Teplitz 13 cuts
        # m225[np.abs(m225) == 99.] = comp.get_limit('f225w',1)
        # cond0 = (np.abs(m275)!=99.) & (np.abs(m336)!=99.) & (np.abs(m435)!=99.)
        # cond1 = (m225-m275 > 1.3) & (-0.2 < m275-m336) & (m275-m336 < 1.2) & (m225-m275 > (0.35 + 1.3*(m275-m336))) & (m336-m435 > -0.5)
        # if calc_sn: cond2 = (sn275 > 5)
        # else: cond2 = (m275 < comp.get_limit('f275w',5)) & (np.abs(m275) != 99.)

        # Oesch 10 cuts
        m225[np.abs(m225) == 99.] = comp.get_limit('f225w',1)
        cond0 = (np.abs(m275)!=99.) & (np.abs(m336)!=99.)
        cond1 = (m225-m275 > 0.75) & (-0.5 < m275-m336) & (m275-m336 < 1.4) & (m225-m275 > (1.67*(m275-m336) - 0.42))
        if calc_sn: cond2 = (sn275 > 5)
        else: cond2 = (m275 < comp.get_limit('f275w',5)) & (np.abs(m275) != 99.)

    elif drop_filt == 'f275w':

        # Hathi 10/Teplitz 13 cuts
        m275[np.abs(m275) == 99.] = comp.get_limit('f275w',1)
        cond0 = (np.abs(m336)!=99.) & (np.abs(m435)!=99.)
        cond1 = (m275-m336 > 1.0) & (-0.2 < m336-m435) & (m336-m435 < 1.2) & (m275-m336 > (0.35 + 1.3*(m336-m435)))
        if calc_sn: cond2 = (sn336 > 5) & (sn225 < 1)
        else: cond2 = ((m336 < comp.get_limit('f336w',5)) & (np.abs(m336) != 99.)) & ((m225 > comp.get_limit('f225w',1)) | (np.abs(m225) == 99.))

        # Oesch 10 cuts
        # m275[np.abs(m275) == 99.] = comp.get_limit('f275w',1)
        # cond0 = (np.abs(m336)!=99.) & (np.abs(m435)!=99.)
        # cond1 = (m275-m336 > 1.0) & (-0.5 < m336-m435) & (m336-m435 < 1.1) & (m275-m336 > (2.2*(m336-m435) - 0.42))
        # if calc_sn: cond2 = (sn336 > 5) & (sn225 < 2)
        # else: cond2 = ((m336 < comp.get_limit('f336w',5)) & (np.abs(m336) != 99.)) & ((m225 > comp.get_limit('f225w',2)) | (np.abs(m225) == 99.))

    elif drop_filt == 'f336w':
        
        # Hathi 10/Teplitz 13 cuts
        m336[np.abs(m336) == 99.] = comp.get_limit('f336w',1)
        cond0 = (np.abs(m435)!=99.) & (np.abs(m606)!=99.)
        cond1 = (m336-m435 > 0.8) & (-0.2 < m435-m606) & (m435-m606 < 1.2) & (m336-m435 > (0.35 + 1.3*(m435-m606)))
        if calc_sn: cond2 = (sn435 > 5) & (sn275 < 1)
        else: cond2 = ((m435 < comp.get_limit('f435w',5)) & (np.abs(m435) != 99.)) & ((m275 > comp.get_limit('f275w',1)) | (np.abs(m275) == 99.))

        # Oesch 10 cuts
        # m336[np.abs(m336) == 99.] = comp.get_limit('f336w',1)
        # cond0 = (np.abs(m435)!=99.) & (np.abs(m606)!=99.)
        # cond1 = (m336-m435 > 1.0) & (-0.5 < m435-m606) & (m435-m606 < 1.0) & (m336-m435 > (2.0*(m435-m606)))
        # if calc_sn: cond2 = (sn435 > 5) & (sn275 < 2)
        # else: cond2 = ((m435 < comp.get_limit('f435w',5)) & (np.abs(m435) != 99.)) & ((m275 > comp.get_limit('f275w',2)) | (np.abs(m275) == 99.))

    else:
        raise Exception("Invalid selection filter.")

    if do_sn_cut: cond = cond0 & cond1 & cond2
    else: cond = cond0 & cond1

    if verbose:
        print "%s Dropouts%s: %i" % (drop_filt.upper(),'' if do_sn_cut else ' (no SN cut)', len(catalog[cond]))

    if return_cond: return cond
    return catalog[cond]

def mk_photoz_cuts(catalog, drop_filt, zlabel='ZB_B', calc_sn=True, do_sn_cut=True, verbose=True, return_cond=False):

    if zlabel=='ZB_B':
        zcond = (catalog['ODDS_B'] > 0.9) & (catalog['CHISQ2_B'] < 1.0)
    else:
        zcond = np.ones(len(catalog),dtype=bool)

    m225 = catalog['MAG_B_F225W'].copy()
    m275 = catalog['MAG_B_F275W'].copy()
    m336 = catalog['MAG_B_F336W'].copy()
    m435 = catalog['MAG_F435W'].copy()
    m606 = catalog['MAG_F606W'].copy()

    if calc_sn:

        dm225 = catalog['MAGERR_B_F225W'].copy()
        dm275 = catalog['MAGERR_B_F275W'].copy()
        dm336 = catalog['MAGERR_B_F336W'].copy()
        dm435 = catalog['MAGERR_F435W'].copy()
        dm606 = catalog['MAGERR_F606W'].copy()

        sn225 = _calc_sn(m225,dm225)
        sn275 = _calc_sn(m275,dm275)
        sn336 = _calc_sn(m336,dm336)
        sn435 = _calc_sn(m435,dm435)
        sn606 = _calc_sn(m606,dm606)

    z0,z1 = bpz_lims[drop_filt]

    if drop_filt == 'f225w':
        cond1 = (z0<catalog[zlabel]) & (catalog[zlabel]<z1)
        if calc_sn: cond2 = (sn435 > 5)
        else: cond2 = (m435 < comp.get_limit('f435w',5)) & (np.abs(m435) != 99.)

    elif drop_filt == 'f275w':
        cond1 = (z0<catalog[zlabel]) & (catalog[zlabel]<z1)
        if calc_sn: cond2 = ((catalog[zlabel]<=2.2) & (sn435 > 5)) | ((catalog[zlabel]>2.2) & (sn606 > 5))
        else: cond2 = ((catalog[zlabel]<=2.2) & (m435 < comp.get_limit('f435w',5)) & (np.abs(m435) != 99.)) | ((catalog[zlabel]>2.2) & (m606 < comp.get_limit('f606w',5)) & (np.abs(m606) != 99.))

    elif drop_filt == 'f336w':
        cond1 = (z0<catalog[zlabel]) & (catalog[zlabel]<z1)
        if calc_sn: cond2 = (sn606 > 5)
        else: cond2 = (m606 < comp.get_limit('f606w',5)) & (np.abs(m606) != 99.)

    else:
        raise Exception("Invalid selection filter.")

    if do_sn_cut: cond = zcond & cond1 & cond2
    else: cond = zcond & cond1

    if verbose:
        print "%.1f < photo-z < %.1f%s: %i" % (z0,z1,'' if do_sn_cut else ' (no SN cut)', len(catalog[cond]))

    if return_cond: return cond
    return catalog[cond]

def mk_photoz_cuts_with_dropout_sncut(catalog, drop_filt, zlabel='ZB_B', calc_sn=True, verbose=True, return_cond=False):

    m225 = catalog['MAG_B_F225W'].copy()
    m275 = catalog['MAG_B_F275W'].copy()
    m336 = catalog['MAG_B_F336W'].copy()
    m435 = catalog['MAG_F435W'].copy()
    m606 = catalog['MAG_F606W'].copy()

    if calc_sn:

        dm225 = catalog['MAGERR_B_F225W'].copy()
        dm275 = catalog['MAGERR_B_F275W'].copy()
        dm336 = catalog['MAGERR_B_F336W'].copy()
        dm435 = catalog['MAGERR_F435W'].copy()
        dm606 = catalog['MAGERR_F606W'].copy()

        sn225 = _calc_sn(m225,dm225)
        sn275 = _calc_sn(m275,dm275)
        sn336 = _calc_sn(m336,dm336)
        sn435 = _calc_sn(m435,dm435)
        sn606 = _calc_sn(m606,dm606)

    z0,z1 = bpz_lims[drop_filt]

    if drop_filt == 'f225w':
        cond1 = (z0<catalog[zlabel]) & (catalog[zlabel]<z1)
        if calc_sn: cond2 = (sn275 > 5)
        else: cond2 = (m275 < comp.get_limit('f275w',5)) & (np.abs(m275) != 99.)
        cond = cond1 & cond2

    elif drop_filt == 'f275w':
        cond1 = (z0<catalog[zlabel]) & (catalog[zlabel]<z1)
        if calc_sn: cond2 = (sn336 > 5) & (sn225 < 1)
        else: cond2 = ((m336 < comp.get_limit('f336w',5)) & (np.abs(m336) != 99.)) & ((m225 > comp.get_limit('f225w',1)) | (np.abs(m225) == 99.))
        cond = cond1 & cond2

    elif drop_filt == 'f336w':
        cond1 = (z0<catalog[zlabel]) & (catalog[zlabel]<z1)
        if calc_sn: cond2 = (sn435 > 5) & (sn275 < 1)
        else: cond2 = ((m435 < comp.get_limit('f435w',5)) & (np.abs(m435) != 99.)) & ((m275 > comp.get_limit('f275w',1)) | (np.abs(m275) == 99.))
        cond = cond1 & cond2

    else:
        raise Exception("Invalid selection filter.")

    if verbose:
        print "%.1f < photo-z < %.1f (with dropout SN cuts): %i" % (z0,z1,len(catalog[cond]))

    if return_cond: return cond
    return catalog[cond]

if __name__ == '__main__':
    print "No main() defined."
