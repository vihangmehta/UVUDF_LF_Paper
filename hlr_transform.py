import numpy as np
import matplotlib.pyplot as plt

import useful
from conversions import calc_sn
from uvudf_utils import filt_key, dfilt_key, read_simulation_output

class HLR_Transform():

    def __init__(self):

        cat_input, cat_recov, cat_recov_hlr = read_simulation_output(run0=True,run7=False,run9=False)

        filt,dfilt = filt_key['f435w'], dfilt_key['f435w']
        sn = calc_sn(cat_recov[filt], cat_recov[dfilt])
        cond = (sn >= 25.) & (cat_recov_hlr['f435w'] > 0)

        self.input_hlr = cat_input['hlr'][cond]
        self.recov_hlr = cat_recov_hlr['f435w'][cond]
        self.sersic = cat_input['n'][cond]
        self.sersic_cond_1 = (self.sersic == 1)
        self.sersic_cond_4 = (self.sersic == 4)

        self.transform, (self._binc, self._run_med) = self.mk_transform()

    def mk_transform(self):

        bins = np.sort(self.input_hlr)[::250]
        binc = 0.5*(bins[1:]+bins[:-1])
        index = np.digitize(self.input_hlr,bins) - 1
        run_med = np.array([np.median((self.recov_hlr)[index==i]) for i in range(len(bins[:-1]))])
        fitp = np.polyfit(binc,run_med,2)
        func = np.poly1d(fitp)
        return func, (binc, run_med)

    def inv_transform(self,y):

        if isinstance(y,np.ndarray):
            res = np.array([(self.transform - yi).roots[1].real for yi in y])
            res[res > 9.85] = 9.85
            res[res < 0.25] = 0.25
            if not all(np.isfinite(res)):
                raise Exception('Evaluated a NaN in HLR inv_transform for '+str(y[~np.isfinite(res)]))
        else:
            res = (self.transform - y).roots[1].real
            res = 9.85 if res > 9.85 else res
            res = 0.25 if res < 0.25 else res
            if not np.isfinite(res):
                raise Exception('Evaluated a NaN in HLR inv_transform for '+str(y))
        return res

    def plot_transform(self,savename=None):

        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,12),dpi=75,tight_layout=False,sharex=True)
        fig.subplots_adjust(left=0.12,right=0.96,bottom=0.06,top=0.98,wspace=0,hspace=0)
        tmpx = np.arange(0,10,0.01)[1:]

        ax1.scatter(self.input_hlr,self.recov_hlr/self.input_hlr,c='k',lw=0,s=5,alpha=0.2)
        ax1.plot(tmpx,self.transform(tmpx)/tmpx,c='b',lw=2.0)
        ax2.scatter(self.input_hlr,self.recov_hlr,c='k',lw=0,s=5,alpha=0.2)
        ax2.plot(tmpx,self.transform(tmpx),c='b',lw=2.0)
        ax2.scatter(self._binc,self._run_med,c='r',lw=0,s=20)

        ax1.set_ylim(-1,10)
        ax1.set_ylabel('Output F435W / Input HLR')
        ax2.set_ylim(0,12)
        ax2.set_xlim(0,10)
        ax2.set_xlabel('Input HLR')
        ax2.set_ylabel('Output F435W HLR')

        if savename: fig.savefig(savename)

    def plot_inv_transform(self,savename=None):

        fig,ax = plt.subplots(1,1,figsize=(8,6),dpi=75,tight_layout=True)
        tmpx = np.arange(0,10,0.01)

        ax.scatter(self.recov_hlr[self.sersic_cond_1],self.input_hlr[self.sersic_cond_1],c='r',s=8,lw=0,alpha=0.5,label='n=1')
        ax.scatter(self.recov_hlr[self.sersic_cond_4],self.input_hlr[self.sersic_cond_4],c='b',s=8,lw=0,alpha=0.5,label='n=4')
        ax.plot(tmpx,self.inv_transform(tmpx),c='k',lw=2,label='Fit')
        ax.plot([0,10],[0,10],c='k',ls='--',lw=2,label='1:1')

        ax.set_xlabel('Recovered F435W HLR')
        ax.set_ylabel('Input HLR')
        ax.set_ylim(0,10)
        ax.set_xlim(0,10)
        ax.legend(fontsize=14,loc=2)

        if savename: fig.savefig(savename)

if __name__ == '__main__':

    tf = HLR_Transform()
    tf.plot_transform(savename='plots/hlr_transform.png')
    tf.plot_inv_transform(savename='plots/hlr_inv_transform.png')
    plt.show()
