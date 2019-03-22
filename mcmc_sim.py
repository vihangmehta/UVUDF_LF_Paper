import numpy as np
import matplotlib.pyplot as plt

from plot_mcmc import MCMC_Output
from LF_refs import UV_LF_refs, z2_LF_refs, UV_LF_refs

def make_dummy_mcmc(coeff,error):

    chain = np.genfromtxt('output/mcmc_LFsim.dat')
    true_coeff  = np.array([-1.59,42.71,-2.95])
    true_error1 = np.array([ 0.05, 0.10, 0.13])
    true_error2 = np.array([ 0.05, 0.10, 0.12])

    if coeff[1]<0:
        chain[:,1] *= -1
        true_coeff[1] *= -1

    chain[:,0] += coeff[0] - true_coeff[0]
    chain[:,1] += coeff[1] - true_coeff[1]
    chain[:,2] += coeff[2] - true_coeff[2]

    cond0 = (chain[:,0] < coeff[0])
    chain[:,0][ cond0] = coeff[0] - np.sqrt((coeff[0] - chain[:,0][ cond0])**2 * error[0]**2 / true_error1[0]**2)
    chain[:,0][~cond0] = coeff[0] + np.sqrt((coeff[0] - chain[:,0][~cond0])**2 * error[0]**2 / true_error2[0]**2)

    cond1 = (chain[:,1] < coeff[1])
    chain[:,1][ cond1] = coeff[1] - np.sqrt((coeff[1] - chain[:,1][ cond1])**2 * error[1]**2 / true_error1[1]**2)
    chain[:,1][~cond1] = coeff[1] + np.sqrt((coeff[1] - chain[:,1][~cond1])**2 * error[1]**2 / true_error2[1]**2)

    cond2 = (chain[:,2] < coeff[2])
    chain[:,2][ cond2] = coeff[2] - np.sqrt((coeff[2] - chain[:,2][ cond2])**2 * error[2]**2 / true_error1[2]**2)
    chain[:,2][~cond2] = coeff[2] + np.sqrt((coeff[2] - chain[:,2][~cond2])**2 * error[2]**2 / true_error2[2]**2)

    return chain

def check_dummy_mcmc():

    # Simulated Ha LF MCMC
    m = MCMC_Output(drop_filt='Ha LF',sample_type='simulation',
                    fname='output/mcmc_LFsim.dat',
                    best_pars=[-1.59,42.71,-2.95],
                    verbose=True)
    # m.plot_walkers()
    # m.plot_corner()
    m.plot_corner2()

    for coeff,label in zip([z2_LF_refs['sobral13'],UV_LF_refs['parsa16']['LFs'][0]],
                           ["Ha LF","UV LF"]):

        m.chain = make_dummy_mcmc(coeff['coeff'],coeff['err'])
        m.best_pars = np.array(coeff['coeff'])
        m.verbose = True
        m.drop_filt = label
        m.sample_type = " - Dummy MCMC Test"
        m.setup()

        m.plot_corner2()

if __name__ == '__main__':
    
    check_dummy_mcmc()
    plt.show()