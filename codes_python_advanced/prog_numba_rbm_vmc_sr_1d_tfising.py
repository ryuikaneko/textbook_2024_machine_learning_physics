import numpy as np
from numba import njit

@njit
def init_rbm(N, M):
    """
    Initialize RBM parameters.
    a: visible biases (length N)
    b: hidden biases (length M)
    w: weight parameters (length M) such that W[j, i] = w[((j-i) mod M)]
    """
    a = np.random.randn(N) * 0.01
    b = np.random.randn(M) * 0.01
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
    M = b.shape[0]
    theta = np.empty(M)
    for j in range(M):
        temp = b[j]
        for i in range(N):
            temp += w[((j - i) % M)] * s[i]
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
        vis += a[i] * s[i]
    M = b.shape[0]
    theta = np.empty(M)
    for j in range(M):
        temp = b[j]
        for i in range(N):
            temp += w[((j - i) % M)] * s[i]
        theta[j] = temp
    hid = 0.0
    for j in range(M):
        hid += np.log(2.0 * np.cosh(theta[j]))
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
    O_a = s.copy()
    O_b = np.empty(M)
    for j in range(M):
        O_b[j] = np.tanh(theta[j])
    O_w = np.empty(M)
    for k in range(M):
        temp = 0.0
        for i in range(N):
            temp += s[i] * np.tanh(theta[((i + k) % M)])
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
        new_theta[j] = theta[j] - 2.0 * w[((j - i) % M)] * old_si
    d_vis = -2.0 * a[i] * old_si
    d_hid = 0.0
    for j in range(M):
        d_hid += np.log(2.0 * np.cosh(new_theta[j])) - np.log(2.0 * np.cosh(theta[j]))
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
                theta[j] -= 2.0 * w[((j - i) % M)] * old_si
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
    E_diag = -np.sum(s * np.roll(s, -1))
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
    M = b.shape[0]
    total_param = N + 2 * M  # visible (N) + hidden (M) + weight (M)
    
    for it in range(num_iter):
        # Equilibration: num_sweeps//10 Metropolis sweeps
        for _ in range(num_sweeps//10):
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
            for i in range(N):
                O_flat[sweep, idx] = O_a[i]
                idx += 1
            for j in range(M):
                O_flat[sweep, idx] = O_b[j]
                idx += 1
            for k in range(M):
                O_flat[sweep, idx] = O_w[k]
                idx += 1
        # Mean energy
        E_mean = np.mean(E_list)
        # Calculate O_mean
        O_mean = np.zeros(total_param)
        for i in range(num_sweeps):
            for j in range(total_param):
                O_mean[j] += O_flat[i, j]
        O_mean /= num_sweeps
        # Calculate <O E>
        OE = np.empty(total_param)
        for j in range(total_param):
            OE[j] = np.mean(O_flat[:, j] * E_list)
        # Force F = <O E> - <O>*<E>
        F = OE - O_mean * E_mean
        
        # Compute covariance matrix S: S[j,k] = <O_j * O_k> - <O_j>*<O_k>
        EPSILON = 1e-3
        S = np.empty((total_param, total_param))
        for j in range(total_param):
            for k in range(total_param):
                S[j, k] = np.mean(O_flat[:, j] * O_flat[:, k]) - O_mean[j] * O_mean[k]
        S += np.eye(total_param) * EPSILON
        
        # Solve the linear system S * delta = -F for delta
        delta = np.linalg.solve(S, -F)
        
        # Update parameters
        a += lr * delta[:N]
        b += lr * delta[N:N+M]
        w += lr * delta[N+M:]
        
        # Recompute theta for the current spin configuration with updated parameters.
        theta = compute_theta(s, b, w)
        print("Iteration", it, ": Energy =", E_mean)
    
    return a, b, w, s, theta

@njit
def main():
    np.random.seed(123)
    N = 10        # Number of spins (length of the 1D chain)
    M = 10        # Number of hidden units (and weight parameters)
    h = 1.0       # Transverse field strength
    num_sweeps = 1000  # Number of MC sweeps per iteration
    lr = 0.01
    num_iter = 500
    
    # Initialize RBM parameters
    a, b, w = init_rbm(N, M)
    # Initialize spin configuration: each spin is +-1
    s = np.random.choice(np.array([-1, 1]), size=N)
    # Compute initial theta = b + sum_i w[((j-i) mod M)] * s[i]
    theta = compute_theta(s, b, w)
    
    a, b, w, s, theta = run_vmc(a, b, w, s, theta, h, num_sweeps, lr, num_iter)

if __name__ == "__main__":
    main()

