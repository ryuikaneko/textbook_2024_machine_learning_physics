import numpy as np
from numba import njit

@njit
def init_rbm(N, M):
    # visible bias a (length N), hidden bias b (length M), weight parameter w (length M)
    a = np.random.randn(N) * 0.01
    b = np.random.randn(M) * 0.01
    w = np.random.randn(M) * 0.01
    return a, b, w

@njit
def compute_theta(N, M, s, b, w):
    # theta[j] = b[j] + sum_{i=0}^{N-1} w[(j-i)%M] * s[i]
    theta = np.empty(M)
    for j in range(M):
        temp = b[j]
        for i in range(N):
            temp += w[(j - i) % M] * s[i]
        theta[j] = temp
    return theta

@njit
def logpsi(N, M, s, a, b, w):
    # visible part: sum_i a[i]*s[i]
    vis = 0.0
    for i in range(N):
        vis += a[i] * s[i]
    # hidden part: sum_j ln(2 cosh(theta_j)),  theta_j = b[j] + sum_i w[(j-i)%M]*s[i]
    theta = np.empty(M)
    hid = 0.0
    for j in range(M):
        temp = b[j]
        for i in range(N):
            temp += w[(j - i) % M] * s[i]
        theta[j] = temp
        hid += np.log(2.0 * np.cosh(temp))
    return vis + hid

@njit
def gradients(N, M, s, a, b, w):
    # First, calculate theta and tanh(theta)
    theta = np.empty(M)
    tanh_theta = np.empty(M)
    for j in range(M):
        temp = b[j]
        for i in range(N):
            temp += w[(j - i) % M] * s[i]
        theta[j] = temp
        tanh_theta[j] = np.tanh(temp)
    # Gradient of the visible part: O_a = s (length N)
    O_a = np.empty(N)
    for i in range(N):
        O_a[i] = s[i]
    # Gradient of the hidden part: O_b = tanh(theta) (length M)
    O_b = tanh_theta.copy()
    # Gradient of the weight parameter: for k=0,...,M-1,
    #   O_w[k] = sum_{i=0}^{N-1} s[i] * tanh_theta[(i+k)%M]
    O_w = np.empty(M)
    for k in range(M):
        temp = 0.0
        for i in range(N):
            temp += s[i] * tanh_theta[(i + k) % M]
        O_w[k] = temp
    return O_a, O_b, O_w

@njit
def psi_ratio(N, M, s, theta, a, b, w, i):
    # When flipping the i-th spin s[i], calculate psi(s')/psi(s)
    # visible part: -2*a[i]*s[i]
    # hidden part: for each j,  theta_j becomes -2*w[(j-i)%M]*s[i], so
    #   diff = sum_j { ln[2 cosh(theta[j]-2*w[(j-i)%M]*s[i])] - ln[2 cosh(theta[j])] }
    old_si = s[i]
    d_vis = -2.0 * a[i] * old_si
    d_hid = 0.0
    for j in range(M):
        new_theta = theta[j] - 2.0 * w[(j - i) % M] * old_si
        d_hid += np.log(2.0 * np.cosh(new_theta)) - np.log(2.0 * np.cosh(theta[j]))
    diff = d_vis + d_hid
    return np.exp(diff)

@njit
def metropolis_step(N, M, s, theta, a, b, w, h):
    # One sweep: randomly select a spin, propose a flip, update s and theta after acceptance check
    for _ in range(N):
        i = np.random.randint(0, N)
        ratio = psi_ratio(N, M, s, theta, a, b, w, i)
        if np.random.rand() < min(1.0, ratio * ratio):
            old_si = s[i]
            s[i] = -old_si
            # Incrementally update theta:
            # theta[j] <- theta[j] - 2*w[(j-i)%M]*old_si, for j=0,...,M-1
            for j in range(M):
                theta[j] -= 2.0 * w[(j - i) % M] * old_si
    return s, theta

@njit
def local_energy(N, M, s, theta, a, b, w, h):
    # Hamiltonian: H = - sum_i s[i]*s[i+1] - h* sum_i psi_ratio(s^(i))/psi(s)
    # Diagonal term (periodic boundary condition):
    E_diag = 0.0
    for i in range(N - 1):
        E_diag += s[i] * s[i+1]
    E_diag += s[N-1] * s[0]
    E_diag = -E_diag
    # Off-diagonal term:
    E_off = 0.0
    for i in range(N):
        ratio = psi_ratio(N, M, s, theta, a, b, w, i)
        E_off += -h * ratio
    return E_diag + E_off

@njit
def run_vmc(a, b, w, s, theta, h, num_sweeps, lr, num_iter):
    N = s.shape[0]
    M = b.shape[0]
    # Number of parameters: visible: N, hidden: M, weight: M --> total_param = N + M + M
    total_param = N + M + M
    # Gradient for each MC sample is a vector of total_param dimensions
    # O_flat: shape (num_sweeps, total_param)
    O_flat = np.empty((num_sweeps, total_param))
    E_list = np.empty(num_sweeps)
    
    for it in range(num_iter):
        # Thermalization: num_sweeps//10 Metropolis updates
        for _ in range(num_sweeps//10):
            s, theta = metropolis_step(N, M, s, theta, a, b, w, h)
        # MC sampling
        for sweep in range(num_sweeps):
            s, theta = metropolis_step(N, M, s, theta, a, b, w, h)
            E_list[sweep] = local_energy(N, M, s, theta, a, b, w, h)
            # Gradient calculation
            O_a, O_b, O_w = gradients(N, M, s, a, b, w)
            # O_a: length N, O_b: length M, O_w: length M
            # Flatten into a 1D vector
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
        E_mean = 0.0
        for i in range(num_sweeps):
            E_mean += E_list[i]
        E_mean /= num_sweeps
        # Calculate O_mean
        O_mean = np.zeros(total_param)
        for j in range(total_param):
            for i in range(num_sweeps):
                O_mean[j] += O_flat[i, j]
            O_mean[j] /= num_sweeps
        # Calculate <O E>
        OE = np.zeros(total_param)
        for j in range(total_param):
            for i in range(num_sweeps):
                OE[j] += O_flat[i, j] * E_list[i]
            OE[j] /= num_sweeps
        # Force F = <O E> - <O>*<E>
        F = np.empty(total_param)
        for j in range(total_param):
            F[j] = OE[j] - O_mean[j] * E_mean
        # Calculate covariance matrix S: S[j,k] = <O_j O_k> - <O_j><O_k>
        S_mat = np.zeros((total_param, total_param))
        for i in range(num_sweeps):
            for j in range(total_param):
                for k in range(total_param):
                    S_mat[j, k] += (O_flat[i, j] - O_mean[j]) * (O_flat[i, k] - O_mean[k])
        for j in range(total_param):
            for k in range(total_param):
                S_mat[j, k] /= num_sweeps
        for j in range(total_param):
            S_mat[j, j] += 1e-3
        # Solve S_mat * delta = -F
        delta = np.linalg.solve(S_mat, -F)
        # Update parameters
        for i in range(N):
            a[i] += lr * delta[i]
        for j in range(M):
            b[j] += lr * delta[N + j]
        for k in range(M):
            w[k] += lr * delta[N + M + k]
        # Recalculate theta after update
        theta = compute_theta(N, M, s, b, w)
        print("Iteration", it, ": Energy =", E_mean)
    return a, b, w, s, theta

def main():
    np.random.seed(123)
    N = 10      # Number of spins (length of 1D chain)
    M = 10      # Number of hidden units (and weight parameters)
    h = 1.0     # Strength of the transverse field
    num_sweeps = 1000  # Number of MC sampling per update step
    lr = 0.01
    num_iter = 500

    a, b, w = init_rbm(N, M)
    s = np.random.choice(np.array([-1, 1]), size=N)
    theta = compute_theta(N, M, s, b, w)
    a, b, w, s, theta = run_vmc(a, b, w, s, theta, h, num_sweeps, lr, num_iter)

if __name__ == "__main__":
    main()

