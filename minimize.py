import numpy as np
import scipy.optimize

import veff
from mk_sample import mk_sample
from probability import Probability

class Minimize():

    def __init__(self,drop_filt,sample_type,x0=(-1.5,-20),bounds=[(-3,0),(-25,-15)],log=None):

        self.drop_filt   = drop_filt
        self.sample_type = sample_type
        self.x0          = x0
        self.bounds      = bounds
        self.log         = log

        return_all = self.drop_filt=='f275w' and self.sample_type=='dropout'
        self.sample = mk_sample(drop_filt=self.drop_filt,sample_type=self.sample_type,return_all=return_all)
        self.prob = Probability(drop_filt=self.drop_filt,sample_type=self.sample_type,sample=self.sample,P=self.x0)

    def func(self,x):

        self.prob.P = x
        self.prob.compute()

        log_string = '%15.10f %15.10f : %20.10f : %.2f s\n' % (x[0], x[1], self.prob.lnlike, self.prob.time)
        if self.log is not None:
            with open(self.log,'a') as f: f.write(log_string)
        print log_string,
        return 1e6 - self.prob.lnlike

    def minimize(self):

        return scipy.optimize.fmin_l_bfgs_b(self.func, x0=self.x0, bounds=self.bounds, factr=1e5, epsilon=1e-8, approx_grad=True)[0]

    def compute(self):

        if self.log is not None:
            with open(self.log,'w') as f:
                f.write('#Log file for LF MLE \n' \
                        '#%s Dropouts\n' \
                        '#%s Sample\n' % (
                            self.drop_filt,self.sample_type))

        self.best_x = self.minimize()

        self.prob.P = self.best_x
        self.phi = self.prob.calc_phi()

        log_string = '%15.10f %15.10f -- Norm.: %15.10f\n' % (self.best_x[0], self.best_x[1], self.phi)
        if self.log is not None:
            with open(self.log,'a') as f: f.write(log_string)
        print "Finished minimization for %s %s -- Best Fit Parameters: " % (self.drop_filt,self.sample_type), self.best_x, self.phi

if __name__ == '__main__':

    # Minimize(drop_filt   = 'f275w',
    #          sample_type = 'dropout',
    #         #x0          = (-1.45,-19.44),  # Informed guesses (from MCMC)
    #          x0          = (-1.5,-20),      # Un-informed guesses
    #          bounds      =[(-2.0,-0.5),(-22,-18)],
    #          log         = "output/fit_f275w_drop.dat").compute()

    # Minimize(drop_filt   = 'f336w',
    #          sample_type = 'dropout',
    #         #x0          = (-1.35,-20.79),  # Informed guesses (from MCMC)
    #          x0          = (-1.5,-20),      # Un-informed guesses
    #          bounds      =[(-2.0,-0.5),(-22,-18)],
    #          log         = "output/fit_f336w_drop.dat").compute()


    Minimize(drop_filt   = 'f225w',
             sample_type = 'photoz',
            #x0          = (-1.19,-19.77),  # Informed guesses (from MCMC)
             x0          = (-1.5,-20),      # Un-informed guesses
             bounds      =[(-2.0,-0.5),(-22,-18)],
             log         = "output/fit_f225w_phot.dat").compute()

    # Minimize(drop_filt   = 'f275w',
    #          sample_type = 'photoz',
    #         #x0          = (-1.37,-20.05),  # Informed guesses (from MCMC)
    #          x0          = (-1.5,-20),      # Un-informed guesses
    #          bounds      =[(-2.0,-0.5),(-22,-18)],
    #          log         = "output/fit_f275w_phot.dat").compute()

    # Minimize(drop_filt   = 'f336w',
    #          sample_type = 'photoz',
    #         #x0          = (-1.42,-20.41),  # Informed guesses (from MCMC)
    #          x0          = (-1.5,-20),      # Un-informed guesses
    #          bounds      =[(-2.0,-0.5),(-22,-18)],
    #          log         = "output/fit_f336w_phot.dat").compute()
