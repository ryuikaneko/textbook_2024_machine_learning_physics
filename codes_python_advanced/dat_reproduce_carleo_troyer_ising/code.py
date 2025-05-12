import numpy as np
from numba import njit
import argparse
import os

@njit
def logtwocosh(x):
    # s always has real part >= 0
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p)

@njit
def init_rbm(N, M):
    """
    Initialize RBM parameters.
    a: visible biases (length N)
    b: hidden biases (length M)
    w: weight parameters (length M) such that W[j, i] = w[((j-i) mod M)]
    """
    alpha = M//N
#    a = np.random.randn(1) * 0.01
    a = np.random.randn(1) * 0.00
#    b = np.random.randn(alpha) * 0.01
    b = np.random.randn(alpha) * 0.0001
    w = np.random.randn(M) * 0.01
    return a, b, w

@njit
def compute_theta(s, b, w):
    """
    Compute the hidden unit inputs (theta) given the spin configuration s,
    hidden biases b, and weight parameters w.
    
    theta[j] = b[j] + sum_{i=0}^{N-1} w[((j-i) mod M)] * s[i]
    """
    N = s.shape[0]
    M = w.shape[0]
    theta = np.empty(M)
    for j in range(M):
        f = j//N
        jj = j%N
        fN = f*N
        temp = b[f]
        for i in range(N):
            temp += w[(jj - i)%N + fN] * s[i]
        theta[j] = temp
    return theta

@njit
def logpsi(s, a, b, w):
    """
    Compute the log of the variational wavefunction:
    
      log(psi(s)) = sum_i a[i]*s[i] + sum_j log(2*cosh(theta[j]))
      
    where theta[j] = b[j] + sum_i w[((j-i) mod M)] * s[i]
    """
    vis = 0.0
    N = s.shape[0]
    for i in range(N):
        vis += s[i]
    vis = a[0] * vis
    M = w.shape[0]
    theta = np.empty(M)
    for j in range(M):
        f = j//N
        jj = j%N
        fN = f*N
        for i in range(N):
            temp += w[(jj - i)%N + fN] * s[i]
        theta[j] = temp
    hid = 0.0
    for j in range(M):
        hid += logtwocosh(theta[j])
#        hid += np.log(2.0 * np.cosh(theta[j]))
    return vis + hid

@njit
def gradients(s, theta):
    """
    Compute the gradients of log(psi) with respect to the parameters,
    using the current spin configuration s and the current (updated) theta.
    
    Visible gradient: O_a = s                                      (length N)
    Hidden gradient:  O_b = tanh(theta)                            (length M)
    Weight gradient:  O_w[k] = sum_i s[i]*tanh(theta[(i+k) mod M]) (length M)
    """
    N = s.shape[0]
    M = theta.shape[0]
    alpha = M//N
    O_a = s.copy()
    O_b = np.zeros(alpha)
    for j in range(M):
        O_b[j//N] += np.tanh(theta[j])
    O_w = np.empty(M)
    for k in range(M):
        f = k//N  
        kk = k%N
        fN = f*N
        temp = 0.0
        for i in range(N):
            temp += s[i] * np.tanh(theta[(i + kk)%N + fN])
        O_w[k] = temp
    return O_a, O_b, O_w

@njit
def psi_ratio(s, theta, a, w, i):
    """
    Compute the ratio psi(s')/psi(s) when flipping spin s[i] only.
    
    The change in theta is given by:
      new_theta[j] = theta[j] - 2 * w[((j-i) mod M)] * s[i]
      
    The ratio is then
      ratio = exp( -2*a[i]*s[i] + sum_j [ log(2*cosh(new_theta[j])) - log(2*cosh(theta[j])) ] )
    """
    N = s.shape[0]
    M = theta.shape[0]
    old_si = s[i]
    new_theta = np.empty(M)
    for j in range(M):
        f = j//N
        jj = j%N
        fN = f*N
        new_theta[j] = theta[j] - 2.0 * w[(jj - i)%N + fN] * old_si
    d_vis = -2.0 * a[0] * old_si
    d_hid = 0.0
    for j in range(M):
        d_hid += logtwocosh(new_theta[j]) - logtwocosh(theta[j])
#        d_hid += np.log(2.0 * np.cosh(new_theta[j])) - np.log(2.0 * np.cosh(theta[j]))
    diff = d_vis + d_hid
    return np.exp(diff)

@njit
def metropolis_step(s, theta, a, b, w, h):
    """
    Perform one Metropolis sweep.
    For each attempted update, choose a random spin index i.
    If the update is accepted (based on the ratio computed via psi_ratio),
    flip s[i] and update theta incrementally as:
      theta[j] = theta[j] - 2 * w[((j-i) mod M)] * s[i]
    """
    N = s.shape[0]
    M = theta.shape[0]
    for _ in range(N):
        i = np.random.randint(0, N)
        ratio = psi_ratio(s, theta, a, w, i)
        if np.random.rand() < min(1.0, ratio**2):
            old_si = s[i]
            s[i] = -old_si
            for j in range(M):
                f = j//N
                jj = j%N
                fN = f*N
                theta[j] -= 2.0 * w[(jj - i)%N + fN] * old_si
    return s, theta

@njit
def local_energy(s, theta, a, b, w, h):
    """
    Compute the local energy:
      E = E_diag + E_off,
    where
      E_diag = - sum_i s[i]*s[i+1]  (with periodic boundary conditions)
      E_off  = - h * sum_i (psi(s^(i))/psi(s))
    """
    N = s.shape[0]
    E_diag = 0.0
    for i in range(N - 1):
        E_diag += s[i] * s[i+1]
    E_diag += s[N-1] * s[0]
    E_diag = -E_diag
    E_off = 0.0
    M = theta.shape[0]
    for i in range(N):
        ratio = psi_ratio(s, theta, a, w, i)
        E_off += -h * ratio
    return E_diag + E_off

@njit
def run_vmc(a, b, w, s, theta, h, num_sweeps, lr, num_iter):
    """
    Run the VMC optimization using the Stochastic Reconfiguration (SR) method.
    
    For each iteration:
      - Equilibrate the system by performing num_sweeps//10 Metropolis sweeps.
      - Perform num_sweeps Monte Carlo sweeps and record the local energy and gradients.
      - Compute the average energy, the average gradient vector, and the covariance matrix.
      - Solve the linear system to obtain the parameter update and update the parameters.
      - Recompute theta for the current configuration.
      
    The parameter vector consists of:
      visible biases a (length N),
      hidden biases b (length M),
      and weight parameters w (length M).
    Total parameter dimension = N + 2*M.
    """
    N = s.shape[0]
    alpha = b.shape[0]
    M = w.shape[0]
    total_param = 1 + alpha + M  # visible (1) + hidden (alpha) + weight (M)
    
    E_mean_list = np.empty(num_iter)
    a_ave = np.zeros(1)
    b_ave = np.zeros(alpha)
    w_ave = np.zeros(M)
    num_ave = 10
    
    for it in range(num_iter):
        # Equilibration: num_sweeps//10 Metropolis sweeps
#        for _ in range(num_sweeps//10):
        for _ in range(num_sweeps):
            s, theta = metropolis_step(s, theta, a, b, w, h)
        
        E_list = np.empty(num_sweeps)
        O_flat = np.empty((num_sweeps, total_param))
        
        # Monte Carlo sampling
        for sweep in range(num_sweeps):
            s, theta = metropolis_step(s, theta, a, b, w, h)
            E_list[sweep] = local_energy(s, theta, a, b, w, h)
            # Compute gradients using the current s and the already updated theta.
            O_a, O_b, O_w = gradients(s, theta)
            idx = 0
            for i in range(1):
                O_flat[sweep, idx] = O_a[i]
                idx += 1
            for j in range(alpha):
                O_flat[sweep, idx] = O_b[j]
                idx += 1
            for k in range(M):
                O_flat[sweep, idx] = O_w[k]
                idx += 1
        # Mean energy
        E_mean = 0.0
        for i in range(num_sweeps):
            E_mean += E_list[i]
        E_mean /= num_sweeps
        # Calculate O_mean
        O_mean = np.zeros(total_param)
        for i in range(num_sweeps):
            for j in range(total_param):
                O_mean[j] += O_flat[i, j]
        O_mean /= num_sweeps
        # Calculate <O E>
        OE = np.zeros(total_param)
        for i in range(num_sweeps):
            E_list_i = E_list[i]
            for j in range(total_param):
                OE[j] += O_flat[i, j] * E_list_i
        OE /= num_sweeps
        # Force F = <O E> - <O>*<E>
        F = OE - O_mean * E_mean

        if lr > 1e-12:
            # Compute covariance matrix S: S[j,k] = <O_j * O_k> - <O_j>*<O_k>
            EPSILON = max([100.0*0.9**it,1e-4])
            S = np.zeros((total_param, total_param))
            for i in range(num_sweeps):
                for j in range(total_param):
                    O_i_j = O_flat[i, j] - O_mean[j]
                    for k in range(total_param):
                        S[j, k] += O_i_j * (O_flat[i, k] - O_mean[k])
            S /= num_sweeps
            for j in range(total_param):
#                S[j, j] += S[j, j] * EPSILON
                S[j, j] += S[j, j] * EPSILON + 1e-10
            
            # Solve the linear system S * delta = -F for delta
            delta = np.linalg.solve(S, -F)
            maxdelta = np.max(np.abs(delta))
            if maxdelta > 1.0:
                delta /= maxdelta
            
            # Update parameters
#            a += lr * delta[:1]
            a += 0.0 * delta[:1]
            b += lr * delta[1:1+alpha]
            w += lr * delta[1+alpha:]
            
            # Average w for last num_ave steps
            if it >= num_iter - num_ave:
                a_ave += a
                b_ave += b
                w_ave += w
        
        # Recompute theta for the current spin configuration with updated parameters.
        theta = compute_theta(s, b, w)
        E_mean_list[it] = E_mean
        if it%(num_iter//5) == 0:
            print("Iteration", it, ": Energy =", E_mean)
    
    # Average w for last num_ave steps
    a_ave /= num_ave
    b_ave /= num_ave
    w_ave /= num_ave        
    
    return a_ave, b_ave, w_ave, s, theta, E_mean_list

@njit
def set_seed(seed):
    np.random.seed(seed)
    return 0

@njit
def main_data(a, b, w, N, M, h, lr, num_sweeps, num_iter, num_sweeps_aft, num_iter_aft):
    # Initialize spin configuration: each spin is +-1
    s = np.random.choice(np.array([-1, 1]), size=N)
    # Compute initial theta = b + sum_i w[((j-i) mod M)] * s[i]
    theta = compute_theta(s, b, w)
    
    if num_iter>0:
        print("# start opt")
        a_ave, b_ave, w_ave, s, theta, E_opt = run_vmc(a, b, w, s, theta, h, num_sweeps, lr, num_iter)
        print("# end opt")
    else:
        a_ave, b_ave, w_ave = a, b, w
        E_opt = np.empty(0)
    print("# start aft")
    _, _, _, s, theta, E_aft = run_vmc(a_ave, b_ave, w_ave, s, theta, h, num_sweeps_aft, 0.0, num_iter_aft)
    print("# end aft")
    
    return a_ave, b_ave, w_ave, s, theta, E_opt, E_aft

def load_file(filename):
    if os.path.exists(filename):
        data = np.loadtxt(filename,ndmin=1)
        print(f"Loaded data from {filename}:")
        print(data)
        return 1, data
    else:
        print(f"File '{filename}' does not exist.")
        return 0, np.empty(0)

def main():
    parser = argparse.ArgumentParser(description="RBM VMC for 1D transverse-field Ising model")
    parser.add_argument("--seed", type=int, default=12345, help="seed: random seed")
    parser.add_argument("--N", type=int, default=10, help="N: Number of spins (length of the 1D chain)")
    parser.add_argument("--M", type=int, default=10, help="M: Number of hidden units (and weight parameters)")
    parser.add_argument("--h", type=float, default=1.0, help="h: Transverse field strength")
    parser.add_argument("--lr", type=float, default=0.01, help="lr: Learning rate")
    parser.add_argument("--num_sweeps", type=int, default=1000, help="num_sweeps: Number of MC sweeps per iteration")
    parser.add_argument("--num_iter", type=int, default=5000, help="num_iter: Number of iterations")
    parser.add_argument("--num_sweeps_aft", type=int, default=100000, help="num_sweeps_aft: Number of MC sweeps per iteration after optimization")
    parser.add_argument("--num_iter_aft", type=int, default=32, help="num_iter_aft: Number of iterations after optimization")
    parser.add_argument("--ia", type=str, default="input_a", help="ia: input_a")
    parser.add_argument("--ib", type=str, default="input_b", help="ib: input_b")
    parser.add_argument("--iw", type=str, default="input_w", help="iw: input_w")
    args = parser.parse_args()
    
    # Set parameters
    seed = args.seed
    N = args.N
    M = args.M
    h = args.h
    lr = args.lr
    num_sweeps = args.num_sweeps
    num_iter = args.num_iter
    num_sweeps_aft = args.num_sweeps_aft
    num_iter_aft = args.num_iter_aft
    ia = args.ia
    ib = args.ib
    iw = args.iw
    set_seed(seed)
    
    Ncnt = 5
    for cnt in range(Ncnt):
        print("Parameters: N=", N, ", M=", M, ", h=", h, ", seed=", seed)
        
        # Initialize RBM parameters
        if cnt == 0:
            flag_ia, a = load_file(ia)
            flag_ib, b = load_file(ib)
            flag_iw, w = load_file(iw)
            if flag_ia * flag_ib * flag_iw == 0: # at least one of input files is missing
                a, b, w = init_rbm(N, M)
        else:
#            b = np.hstack([b, np.zeros_like(b)])
#            w = np.hstack([w, np.zeros_like(w)])
            b = np.hstack([b, np.random.randn(len(b)) * 0.0001])
            w = np.hstack([w, np.random.randn(len(w)) * 0.0001])
        print("a",a)
        print("b",b)
        print("w",w)
        
        # Run VMC
        a, b, w, s, theta, E_opt, E_aft = main_data(a, b, w, N, M, h, lr, num_sweeps, num_iter, num_sweeps_aft, num_iter_aft)
        E_ave = np.mean(E_aft)
        E_err = np.sqrt(np.var(E_aft)/len(E_aft))
        print("VMC: Energy =", E_ave, "+-", E_err)
        
        # Exact solution
        ks = (np.arange(N) + 0.5) * (2*np.pi/N)
        cosks = np.cos(ks)
        Eks = 2 * np.sqrt(1 + h**2 - 2*h*cosks)
        E0 = -0.5 * np.sum(Eks)
        print("Exact: Energy =", E0)
        print("Error: 1 - E_VMC/E_Exact =", 1.0 - E_ave/E0)
        print("")
        
        # Output files
        fname = "N{}".format(N)+"M{}".format(M)+"h{:.10f}".format(h)+"seed{}".format(seed)
        np.savetxt("dat_"+fname+"_E_opt", E_opt)
        np.savetxt("dat_"+fname+"_E_aft", E_aft)
        np.savetxt("dat_"+fname+"_a", a)
        np.savetxt("dat_"+fname+"_b", b)
        np.savetxt("dat_"+fname+"_w", w)
        temp = np.array([N,M,h,E_ave,E_err,E0,1.0-E_ave/E0,E_err/E0]).reshape(1,-1)
        np.savetxt("dat_"+fname+"_E_aft_ave_err", temp, header="N M h ene err ene_exact 1-ene/ene_exact err/ene_exact")
        
        # Double a hidden layer size
        M = 2*M

if __name__ == "__main__":
    main()

