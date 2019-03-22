import time
import numpy as np
import scipy.integrate
from multiprocessing import Queue, Process

import useful
import conversions as conv
import veff
import mk_sample

class Probability():

    def __init__(self,drop_filt,sample_type,P,sample=None):

        self.sample_type = sample_type
        self.drop_filt   = drop_filt
        self.P           = P

        if sample is None:
            return_all = self.drop_filt=='f275w' and self.sample_type=='dropout'
            self.sample = mk_sample.mk_sample(drop_filt=drop_filt,sample_type=sample_type,return_all=return_all)
        else: self.sample = sample

        self.veff_func = veff.VEff_Func(drop_filt=self.drop_filt,sample_type=self.sample_type)
        self.veff_func.setup()

        self.lim_M = [-25, self.veff_func.mag_limit(hlr=8)]
        print "%s %s sample: %i (%i[%i] with M_UV < %.2f)" % (self.drop_filt.upper(),self.sample_type.capitalize(),
                                        len(self.sample),
                                        len(self.sample[self.sample['M_1500'] < self.lim_M[1]]),
                                        len(self.sample[self.sample['SAMPLE_FLAG']==1]),
                                        self.lim_M[1])

        self.quad_args_num = {'epsabs':1e-4,'epsrel':1e-4,'limit':250}
        self.quad_args_den = {'epsabs':1e-4,'epsrel':1e-4,'limit':250}
        self.quad_args_phi = {'epsabs':1e-4,'epsrel':1e-4,'limit':250}

        self.time_num = 0
        self.time_den = 0
        self.time_phi = 0

    def compute(self):

        t0 = time.time()
        self.numerator = np.zeros(len(self.sample))
        self.denominator = np.zeros(len(self.sample))
        self.multiprocess()
        self.numerator[self.numerator <= 0] = np.exp(-300)
        self.calc_prob()
        self.time = time.time() - t0

    def integrand_num(self,M,entry):

        a = useful.sch(M,self.P)
        b = useful.gauss(M,entry['M_1500'],entry['dM_1500'])
        c = self.veff_func(M=M,hlr=entry['HLR_IN'])
        return a*b*c

    def integrand_den(self,M,entry):

        a = useful.sch(M,self.P)
        b = self.veff_func(M=M,hlr=entry['HLR_IN'])
        return a*b

    def integrand_phi(self,M):

        a = useful.sch(M,self.P)
        b = self.veff_func(M=M,hlr=-99.)
        return a*b

    def calc_numerator(self,i):

        t0 = time.time()
        entry = self.sample[i]
        quad_args_pts = self.quad_args_num.copy()
        quad_args_pts["points"] = [entry['M_1500'],] + [entry['M_1500']-(j+1)*entry['dM_1500'] for j in range(10)] + [entry['M_1500']+(j+1)*entry['dM_1500'] for j in range(10)]
        quad_args_pts["points"] = np.unique(np.clip(quad_args_pts["points"], self.lim_M[0], self.lim_M[1]))
        num = scipy.integrate.quad(self.integrand_num, self.lim_M[0], self.lim_M[1], args=(entry,), **quad_args_pts)[0]
        return (i,num,time.time()-t0)

    def calc_denominator(self,i):

        t0 = time.time()
        entry = self.sample[i]
        den = scipy.integrate.quad(self.integrand_den, self.lim_M[0], self.lim_M[1], args=(entry,), **self.quad_args_den)[0]
        return (i,den,time.time()-t0)

    def calc_phi(self):

        t0 = time.time()
        integral = scipy.integrate.quad(self.integrand_phi, self.lim_M[0], self.lim_M[1], **self.quad_args_phi)[0]
        phi = len(self.sample) / integral
        self.phi = np.log10(phi)
        self.time_phi = time.time() - t0
        return self.phi

    def multiprocess(self):

        num_procs = 15
        split = np.array_split(range(len(self.sample)), num_procs)

        def slave(queue, chunk):
            for x in chunk:
                itemn = self.calc_numerator(x)
                itemd = self.calc_denominator(x)
                items = (itemn, itemd)
                queue.put(items)
            queue.put(None)

        queue = Queue()
        procs = [Process(target=slave, args=(queue,chunk)) for chunk in split]
        for proc in procs: proc.start()

        finished = 0
        while finished < num_procs:
            items = queue.get()
            if items == None:
                finished += 1
            else:
                itemn, itemd = items
                numi, num, numt = itemn
                deni, den, dent = itemd
                self.numerator[numi] = num
                self.time_num += numt / len(self.sample)
                self.denominator[deni] = den
                self.time_den += dent / len(self.sample)

        for proc in procs: proc.join()

    def calc_prob(self):

        self.prob = self.numerator / self.denominator
        self.lnprob = np.log(self.numerator) - np.log(self.denominator)
        self.lnlike = np.sum(self.lnprob)

if __name__ == '__main__':

    p = Probability(drop_filt='f275w',sample_type='dropout',P=(-1.5,-20))
    p.compute()
    print p.lnlike

    p = Probability(drop_filt='f275w',sample_type='photoz',P=(-1.5,-20))
    p.compute()
    print p.lnlike