import numpy as np
import astropy.io.fits as fitsio

import uvudf_utils as utils
import conversions as conv
import pickles
import veff
from sample_selection import mk_dropout_cuts, mk_photoz_cuts
from selection import SelectionFunction
from hlr_transform import HLR_Transform

hlr_trans = HLR_Transform()

def remove_stars(catalog):

    cond_star = (catalog['STAR'] == 1)
    cond_stellarity = (catalog['MAG_F850LP']<=25.5) & (catalog['HLR_F435W']<=0.1/utils.pixscale) & (catalog['STELLARITY']>0.8)

    seq = pickles.get_pickles_sequence()
    cond0 = (catalog['HLR_F435W'] <= 0.1/utils.pixscale) & (catalog['MAG_F850LP'] >= 25.5)
    cond1 = (np.abs(catalog['MAG_F606W']) != 99.) & (np.abs(catalog['MAG_F775W']) != 99.) & (np.abs(catalog['MAG_F850LP']) != 99.)
    cond2 = (np.abs((catalog['MAG_F606W']-catalog['MAG_F775W']) - seq(catalog['MAG_F775W']-catalog['MAG_F850LP'])) < 0.15)
    cond_color = cond0 & cond1 & cond2

    cond = cond_star | cond_stellarity | cond_color
    clean_cat = catalog[~cond]
    return clean_cat

def mk_sample(drop_filt,sample_type,return_all=False,return_catalog=False,new=False):

    if new: 

        catalog = fitsio.getdata('catalogs/udf_cat_v2.0.hlr.fits')
        catalog = catalog[(catalog['UVUDF_COVERAGE']==1) & (catalog['UVUDF_EDGEFLG']==0)]
        catalog = remove_stars(catalog)

        dtyp=[('ID',int),('X',float),('Y',float),('RA',float),('DEC',float),('z',float),('SAMPLE_TYPE','S8'),('FILT_DROP','S8'),
              ('m_1500',float),('dm_1500',float),('M_1500',float),('dM_1500',float),('SN_1500',float),('FILT_1500','S8'),
              ('m_DET' ,float),('dm_DET' ,float),('M_DET' ,float),('dM_DET' ,float),('SN_DET' ,float),('FILT_DET' ,'S8'),
              ('HLR_F435W',float),('HLR_IN',float),('SPECZ',float),('GRISMZ',float),('BPZ',float),
              ('Vmax',float),('Vtot',float),('Vfrac',float),('SAMPLE_FLAG',int)]

        if sample_type=='dropout':
            
            selfn = SelectionFunction(drop_filt=drop_filt,sample_type=sample_type)
            data = mk_dropout_cuts(catalog,drop_filt,verbose=False)
            sample = np.recarray(len(data), dtype=dtyp)
            sample['z'] = selfn.get_pivot_z()
            sample['SAMPLE_TYPE'] = 'dropout'
            sample['FILT_DROP'] = drop_filt
            sample['FILT_DET' ] = utils.filt_det[drop_filt]
            sample['FILT_1500'] = utils.filt_1500(drop_filt,sample['z'])

        elif sample_type=='photoz':
            
            data = mk_photoz_cuts(catalog,drop_filt,verbose=False)
            sample = np.recarray(len(data), dtype=dtyp)
            sample['SAMPLE_TYPE'] = 'photo-z'
            sample['FILT_DROP'] = drop_filt
            sample['z'] = data['ZB_B']
            sample['FILT_DET' ] = utils.filt_1500(drop_filt,sample['z'])
            sample['FILT_1500'] = utils.filt_1500(drop_filt,sample['z'])

        else:
            raise Exception("Invalid sample type provided. Choose from: 'dropout' or 'photoz'.")

        sample['ID']     = data['ID']
        sample['X']      = data['X']
        sample['Y']      = data['Y']
        sample['RA']     = data['RA']
        sample['DEC']    = data['DEC']
        sample['SPECZ']  = data['SPECZ_Z']
        sample['GRISMZ'] = data['GRISM_Z']
        sample['BPZ']    = data['ZB_B']

        for entry,data_entry in zip(sample,data):

            entry['m_1500' ] = data_entry[utils.filt_key[ entry['FILT_1500']]]
            entry['dm_1500'] = data_entry[utils.dfilt_key[entry['FILT_1500']]]
            entry['M_1500']  = conv.get_abs_from_app(entry['m_1500'], entry['z'])
            entry['dM_1500'] = entry['dm_1500']

            entry['m_DET'  ] = data_entry[utils.filt_key[ entry['FILT_DET']]]
            entry['dm_DET' ] = data_entry[utils.dfilt_key[entry['FILT_DET']]]
            entry['M_DET']   = conv.get_abs_from_app(entry['m_DET'], entry['z'])
            entry['dM_DET']  = entry['dm_DET']

        ### Computing the SN
        sample['SN_1500'] = conv.calc_sn(sample['m_1500'],sample['dm_1500'])
        sample['SN_DET']  = conv.calc_sn(sample['m_DET'],sample['dm_DET'])

        ### Computing the original half-light radii
        sample['HLR_F435W'] = data['HLR_F435W']
        sample['HLR_IN'] = hlr_trans.inv_transform(sample['HLR_F435W'])

        ### Block for computing the volumes
        veff_func = veff.VEff_Func(drop_filt=drop_filt, sample_type=sample_type)
        veff_func.setup()
        sample['Vmax'] = veff_func(M=sample['M_1500'],hlr=sample['HLR_IN'])
        sample['Vtot'] = veff_func.calc_vol()
        sample['Vfrac'] = sample['Vmax'] / sample['Vtot']
        cond = (sample['M_1500'] < veff_func.mag_limit(hlr=8))
        sample['SAMPLE_FLAG'][cond]  = 1
        sample['SAMPLE_FLAG'][~cond] = 0
        print "%s %s sample size: %i [/%i]" % (drop_filt.upper(),sample_type.capitalize(),len(sample[cond]),len(sample))

        ### Get the catalog entries for the sample
        idx = [np.where(catalog['ID']==entry['ID'])[0][0] for entry in sample]
        _catalog = np.recarray(len(sample),dtype=catalog.dtype.descr + [('SAMPLE_FLAG',int),])
        for x in catalog.dtype.names: _catalog[x] = catalog[idx][x]
        _catalog['SAMPLE_FLAG'] = sample['SAMPLE_FLAG']
        
        ### Save the sample
        fitsio.writeto('catalogs/udf_sample_%s_%s.fits'  % (drop_filt,sample_type),   sample, clobber=True)
        fitsio.writeto('catalogs/udf_catalog_%s_%s.fits' % (drop_filt,sample_type), _catalog, clobber=True)

    ### RETURN THE SAMPLES REGARDLESS OF 'new'
    if return_catalog:
        sample = fitsio.getdata('catalogs/udf_catalog_%s_%s.fits' % (drop_filt,sample_type))
    else:
        sample = fitsio.getdata('catalogs/udf_sample_%s_%s.fits' % (drop_filt,sample_type))
    
    if return_all: return sample
    return sample[sample['SAMPLE_FLAG']==1]

if __name__ == '__main__':

    mk_sample('f225w','photoz',new=True)
    # for f in ['f225w','f275w','f336w']:
    #     for s in ['dropout','photoz']:
    #         mk_sample(drop_filt=f,sample_type=s,new=True)