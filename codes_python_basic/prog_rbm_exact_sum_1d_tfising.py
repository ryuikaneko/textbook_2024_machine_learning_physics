import numpy as np; from scipy.optimize import minimize

def get_spin(state,site):
    return (state>>site)&1

def flip_spin(state,site):
    return state^(1<<site)

def set_W(Lh,mu=0.0,sigma=0.01,seed=12345):
    np.random.seed(seed=seed)
    return np.random.normal(mu,sigma,Lh)

def calc_ampRBM(L,Lh,W,state):
    amp = 1.0
    for i in range(Lh):
        theta = 0.0
        for j in range(L):
            theta += W[(i-j+Lh)%Lh] * (1.0-2.0*get_spin(state,j))
        amp *= 2.0 * np.cosh(theta)
    return amp

def calc_eneRBM(L,Lh,g,W):
    Nstate = 2**L; psiIpsi = 0.0; psiHpsi = 0.0
    for a in range(Nstate):
        ampr = calc_ampRBM(L,Lh,W,a); ampl = ampr; ampl_ampr = ampl*ampr
        psiIpsi += ampl_ampr
        for i in range(L):
            if get_spin(a,i) == get_spin(a,(i+1)%L):
                psiHpsi -= ampl_ampr
            else:
                psiHpsi += ampl_ampr
        for i in range(L):
            b = flip_spin(a,i); ampl = calc_ampRBM(L,Lh,W,b)
            psiHpsi += ampl*(-g)*ampr
    return psiHpsi/psiIpsi

def main():
    L = 2; Lh = 1; g = 1.0; W = set_W(Lh)
    result = minimize(lambda Wdummy: calc_eneRBM(L,Lh,g,Wdummy),W)
    print("RBM energy:  ",result.fun)
    print("Exact energy:",-2.0*np.sqrt(2.0))

main()
