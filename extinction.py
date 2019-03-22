import numpy as np

def calzetti_k(l):

    Rv = 4.05
    x = 1e4 / l
    k = np.zeros(len(l))

    cond1 = (0.63 <= l/1e4) & (l/1e4 <= 2.20)
    k[cond1] = 2.659*(-1.857 + 1.040*x[cond1]) + Rv

    cond2 = (0.12 <= l/1e4) & (l/1e4 < 0.63)
    k[cond2] = 2.659*(-2.156 + 1.509*x[cond2] - 0.198*x[cond2]**2 + 0.011*x[cond2]**3) + Rv

    ### ADDED for completeness ###
    cond3 = (l/1e4 < 0.12)
    x[cond3] = 1./0.12
    k[cond3] = 2.659*(-2.156 + 1.509*x[cond3] - 0.198*x[cond3]**2 + 0.011*x[cond3]**3) + Rv

    return k

def calzetti(l,EBV,return_k=False):

    k = calzetti_k(l)
    tau = np.log(10) * (0.4 * EBV * k)
    return tau