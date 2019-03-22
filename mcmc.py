import time, emcee
import numpy as np
from multiprocessing import Queue, Process

import mk_sample
from probability import Probability

class MCMC_MLE():

    def __init__(self,drop_filt,sample_type,x0,nwalkers=10,steps=1000,savefile=None):

        self.drop_filt   = drop_filt
        self.sample_type = sample_type
        self.nwalkers    = nwalkers
        self.steps       = steps
        self.x0          = x0
        self.ndim        = len(self.x0)
        self.savefile    = savefile

        return_all = self.drop_filt=='f275w' and self.sample_type=='dropout'
        self.sample = mk_sample.mk_sample(drop_filt=drop_filt,sample_type=sample_type,return_all=return_all)
        self.prob = Probability(drop_filt=self.drop_filt,sample_type=self.sample_type,sample=self.sample,P=self.x0)

    def lnprior(self, pars):

        (alpha, Mst) = pars
        if -3<alpha<0. and -25.0<Mst<-16.0:
            return np.log(1.)
        return -np.inf

    def lnlike(self, pars):

        lp = self.lnprior(pars)
        if not np.isfinite(lp):
            return -np.inf

        self.prob.P = pars
        self.prob.compute()
        return lp + self.prob.lnlike

    def run_mcmc(self):

        x0 = [self.x0 + 1e-2*np.random.randn(self.ndim) for i in range(self.nwalkers)]

        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnlike)

        with open(self.savefile, "w") as f:
            f.write("# MCMC chain for %i walkers, %i steps\n" % (self.nwalkers, self.steps))
            f.write("# %13s %15s %15s\n" % ('alpha', 'Lst', 'phi'))

        start, step = time.time(), time.time()

        for k,result in enumerate(sampler.sample(x0, iterations=self.steps, storechain=False)):

            position = result[0]
            nphi = self.get_phi(position)

            with open(self.savefile, "a") as f:
                for i,phi in zip(range(self.nwalkers), nphi):
                    for j in range(self.ndim):
                        f.write(" %15.10f" % position[i,j])
                    f.write(" %15.10f \n" % phi)

            step_time = time.time() - step
            tot_time = time.time() - start
            est_time = tot_time / (k+1) * self.steps
            rem_time = est_time - tot_time
            print "Step #%i of %i -- Time Taken: %.2f s (total: %.0f s) -- Time Remaining: %.0f s (total: %.0f s)" % (k+1, self.steps, step_time, tot_time, rem_time, est_time)
            step = time.time()

    def phi_slave(self, queue, position, chunk_steps):

        for i in chunk_steps:

            pars = tuple(position[i,j] for j in range(self.ndim))
            self.prob.P = pars
            phi = self.prob.calc_phi()
            queue.put((i,phi))

        queue.put(None)

    def get_phi(self, position):

        nphi = np.zeros(self.nwalkers)
        num_procs = self.nwalkers if self.nwalkers<15 else 15
        split_steps = np.array_split(range(self.nwalkers), num_procs)

        queue = Queue()
        procs = [Process(target=self.phi_slave, args=(queue, position, chunk_steps)) for chunk_steps in split_steps]
        for proc in procs: proc.start()

        finished = 0
        while finished < num_procs:
            items = queue.get()
            if items is None: finished += 1
            else:
                i, phi = items
                nphi[i] = phi
        for proc in procs: proc.join()

        return nphi

def main():

    MCMC_MLE(drop_filt='f275w',sample_type='dropout',x0=[-1.31,-19.66],
             savefile="output/mcmc_f275w_drop.dat").run_mcmc()

    MCMC_MLE(drop_filt='f336w',sample_type='dropout',x0=[-1.35,-20.71],
             savefile="output/mcmc_f336w_drop.dat").run_mcmc()


    MCMC_MLE(drop_filt='f225w',sample_type='photoz', x0=[-1.20,-19.93],
             savefile="output/mcmc_f225w_phot.dat").run_mcmc()

    MCMC_MLE(drop_filt='f275w',sample_type='photoz', x0=[-1.32,-19.92],
             savefile="output/mcmc_f275w_phot.dat").run_mcmc()

    MCMC_MLE(drop_filt='f336w',sample_type='photoz', x0=[-1.40,-20.40],
             savefile="output/mcmc_f336w_phot.dat").run_mcmc()

if __name__ == '__main__':

    main()