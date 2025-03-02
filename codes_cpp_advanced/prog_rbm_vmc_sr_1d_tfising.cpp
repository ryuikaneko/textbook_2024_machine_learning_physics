#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double EPSILON = 1e-3;

// Global Mersenne Twister RNG with fixed seed
std::mt19937 rng(123);
std::normal_distribution<double> normal_dist(0.0, 1.0);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

// Returns a random normal number
double rand_normal() {
    return normal_dist(rng);
}

// Returns a random uniform number between 0 and 1
double rand_uniform() {
    return uniform_dist(rng);
}

// Compute modulo that handles negative values correctly.
int mod_index(int x, int m) {
    int r = x % m;
    return (r < 0) ? r + m : r;
}

// Initialize the RBM parameters:
// a: visible biases (size N), b: hidden biases (size M), w: weight parameters (size M)
void init_rbm(int N, int M, std::vector<double>& a, std::vector<double>& b, std::vector<double>& w) {
    for (int i = 0; i < N; i++) {
        a[i] = rand_normal() * 0.01;
    }
    for (int j = 0; j < M; j++) {
        b[j] = rand_normal() * 0.01;
        w[j] = rand_normal() * 0.01;
    }
}

// Compute theta vector (hidden unit inputs):
// theta[j] = b[j] + sum_{i=0}^{N-1} w[ (j-i+M) mod M ] * s[i]
void compute_theta(int N, int M, const std::vector<int>& s, const std::vector<double>& b, 
                   const std::vector<double>& w, std::vector<double>& theta) {
    for (int j = 0; j < M; j++) {
        double temp = b[j];
        for (int i = 0; i < N; i++) {
            temp += w[ mod_index(j - i, M) ] * s[i];
        }
        theta[j] = temp;
    }
}

// Compute log(psi(s)) = sum_i a[i]*s[i] + sum_j log(2 cosh(theta[j]))
double logpsi(int N, int M, const std::vector<int>& s, const std::vector<double>& a, 
              const std::vector<double>& b, const std::vector<double>& w) {
    double vis = 0.0;
    for (int i = 0; i < N; i++) {
        vis += a[i] * s[i];
    }
    std::vector<double> theta(M);
    for (int j = 0; j < M; j++) {
        double temp = b[j];
        for (int i = 0; i < N; i++) {
            temp += w[ mod_index(j - i, M) ] * s[i];
        }
        theta[j] = temp;
    }
    double hid = 0.0;
    for (int j = 0; j < M; j++) {
        hid += std::log(2.0 * std::cosh(theta[j]));
    }
    return vis + hid;
}

// Compute the gradients using the current (already updated) theta.
// Returns:
//   O_a (visible gradient) = s (size N)
//   O_b (hidden gradient) = tanh(theta) (size M)
//   O_w (weight gradient) = for each k: sum_{i=0}^{N-1} s[i]*tanh(theta[(i+k) mod M]) (size M)
void gradients(int N, int M, const std::vector<int>& s, const std::vector<double>& theta,
               std::vector<double>& O_a, std::vector<double>& O_b, std::vector<double>& O_w) {
    O_a.resize(N);
    O_b.resize(M);
    O_w.resize(M);
    for (int i = 0; i < N; i++) {
        O_a[i] = s[i];
    }
    for (int j = 0; j < M; j++) {
        O_b[j] = std::tanh(theta[j]);
    }
    for (int k = 0; k < M; k++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            int idx = mod_index(i + k, M);
            sum += s[i] * std::tanh(theta[idx]);
        }
        O_w[k] = sum;
    }
}

// Compute the ratio psi(s')/psi(s) when flipping spin s[i] only.
// The new theta is computed as: new_theta[j] = theta[j] - 2 * w[ (j-i+M) mod M ] * s[i]
// new_theta is a temporary vector of size M (passed by reference)
double psi_ratio(int N, int M, const std::vector<int>& s, const std::vector<double>& theta,
                 const std::vector<double>& a, const std::vector<double>& w, int i, 
                 std::vector<double>& new_theta) {
    int old_si = s[i];
    double d_vis = -2.0 * a[i] * old_si;
    double d_hid = 0.0;
    new_theta.resize(M);
    for (int j = 0; j < M; j++) {
        new_theta[j] = theta[j] - 2.0 * w[ mod_index(j - i, M) ] * old_si;
        d_hid += std::log(2.0 * std::cosh(new_theta[j])) - std::log(2.0 * std::cosh(theta[j]));
    }
    double diff = d_vis + d_hid;
    return std::exp(diff);
}

// Perform one Metropolis sweep.
// For each attempted update, a random spin index i is chosen.
// If the update is accepted, flip s[i] and update theta incrementally:
//   theta[j] = theta[j] - 2 * w[ (j-i+M) mod M ] * s[i]
void metropolis_step(int N, int M, std::vector<int>& s, std::vector<double>& theta,
                     const std::vector<double>& a, const std::vector<double>& b, 
                     const std::vector<double>& w, double h, std::vector<double>& new_theta) {
    for (int k = 0; k < N; k++) {
        int i = std::rand() % N; // You can also use rand_uniform() if desired.
        double ratio = psi_ratio(N, M, s, theta, a, w, i, new_theta);
        double r = rand_uniform();
        double prob = (ratio * ratio < 1.0) ? ratio * ratio : 1.0;
        if (r < prob) {
            int old_si = s[i];
            s[i] = -old_si;
            // Incrementally update theta
            for (int j = 0; j < M; j++) {
                theta[j] -= 2.0 * w[ mod_index(j - i, M) ] * old_si;
            }
        }
    }
}

// Compute the local energy:
// Diagonal term: E_diag = - sum_i s[i]*s[(i+1)%N]
// Off-diagonal term: E_off = - h * sum_i psi_ratio(s^(i))/psi(s)
double local_energy(int N, int M, const std::vector<int>& s, const std::vector<double>& theta,
                    const std::vector<double>& a, const std::vector<double>& b, 
                    const std::vector<double>& w, double h, std::vector<double>& new_theta) {
    double E_diag = 0.0;
    for (int i = 0; i < N - 1; i++) {
        E_diag += s[i] * s[i+1];
    }
    E_diag += s[N-1] * s[0];
    E_diag = -E_diag;
    double E_off = 0.0;
    for (int i = 0; i < N; i++) {
        double ratio = psi_ratio(N, M, s, theta, a, w, i, new_theta);
        E_off += -h * ratio;
    }
    return E_diag + E_off;
}

// Solve the linear system A * delta = -F using Gaussian elimination.
// S_mat: coefficient matrix (size n*n), F: right-hand side (size n)
// A, B, delta: work vectors (size n*n, n, n respectively)
// Returns 0 on success, nonzero if the system cannot be solved.
int solve_linear_system(int n, const std::vector<double>& S_mat, const std::vector<double>& F,
                          std::vector<double>& A, std::vector<double>& B, std::vector<double>& delta) {
    A = S_mat; // Copy S_mat into A
    B.resize(n);
    for (int i = 0; i < n; i++) {
        B[i] = -F[i];
    }
    // Forward elimination
    for (int i = 0; i < n; i++) {
        int pivot = i;
        double max_val = std::fabs(A[i * n + i]);
        for (int k = i + 1; k < n; k++) {
            double val = std::fabs(A[k * n + i]);
            if (val > max_val) {
                max_val = val;
                pivot = k;
            }
        }
        if (max_val < 1e-12) return 1;
        if (pivot != i) {
            for (int j = 0; j < n; j++) {
                std::swap(A[i * n + j], A[pivot * n + j]);
            }
            std::swap(B[i], B[pivot]);
        }
        for (int k = i + 1; k < n; k++) {
            double factor = A[k * n + i] / A[i * n + i];
            for (int j = i; j < n; j++) {
                A[k * n + j] -= factor * A[i * n + j];
            }
            B[k] -= factor * B[i];
        }
    }
    // Back substitution
    delta.resize(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i * n + j] * delta[j];
        }
        delta[i] = (B[i] - sum) / A[i * n + i];
    }
    return 0;
}

// Run the VMC optimization using the Stochastic Reconfiguration (SR) method.
// Parameters:
//   a: visible biases (size N)
//   b: hidden biases (size M)
//   w: weight parameters (size M)
//   s: spin configuration (size N)
//   theta: current hidden inputs (size M)
//   num_sweeps: number of MC sweeps per iteration
//   lr: learning rate
//   num_iter: number of iterations
// Work arrays (all pre-allocated in main) are passed by reference.
void run_vmc(int N, int M, double h, int num_sweeps, double lr, int num_iter,
             std::vector<double>& a, std::vector<double>& b, std::vector<double>& w,
             std::vector<int>& s, std::vector<double>& theta,
             std::vector<double>& E_list, std::vector<double>& O_flat,
             std::vector<double>& O_mean, std::vector<double>& OE, std::vector<double>& F_vec,
             std::vector<double>& S_mat, std::vector<double>& delta, std::vector<double>& A,
             std::vector<double>& B, std::vector<double>& theta_temp, std::vector<double>& tanh_theta,
             std::vector<double>& new_theta, std::vector<double>& O_a_temp,
             std::vector<double>& O_b_temp, std::vector<double>& O_w_temp) {
    // Total number of parameters: visible (N) + hidden (M) + weights (M) = N + 2*M
    int total_param = N + 2 * M;
    
    for (int it = 0; it < num_iter; it++) {
        // Equilibration: num_sweeps/10 Metropolis sweeps
        for (int eq = 0; eq < num_sweeps/10; eq++) {
            metropolis_step(N, M, s, theta, a, b, w, h, new_theta);
        }
        // MC sampling
        for (int sweep = 0; sweep < num_sweeps; sweep++) {
            metropolis_step(N, M, s, theta, a, b, w, h, new_theta);
            E_list[sweep] = local_energy(N, M, s, theta, a, b, w, h, new_theta);
            // Compute gradients using the current s and the already updated theta.
            gradients(N, M, s, theta, O_a_temp, O_b_temp, O_w_temp);
            int idx = 0;
            for (int i = 0; i < N; i++) {
                O_flat[sweep * total_param + idx] = O_a_temp[i];
                idx++;
            }
            for (int j = 0; j < M; j++) {
                O_flat[sweep * total_param + idx] = O_b_temp[j];
                idx++;
            }
            for (int k = 0; k < M; k++) {
                O_flat[sweep * total_param + idx] = O_w_temp[k];
                idx++;
            }
        }
        // Compute the average energy
        double E_mean = 0.0;
        for (int i = 0; i < num_sweeps; i++) {
            E_mean += E_list[i];
        }
        E_mean /= num_sweeps;
        
        // Compute average gradient vector O_mean (length total_param)
        O_mean.assign(total_param, 0.0);
        for (int i = 0; i < num_sweeps; i++) {
            for (int j = 0; j < total_param; j++) {
                O_mean[j] += O_flat[i * total_param + j];
            }
        }
        for (int j = 0; j < total_param; j++) {
            O_mean[j] /= num_sweeps;
        }
        // Compute <O E> vector
        OE.assign(total_param, 0.0);
        for (int i = 0; i < num_sweeps; i++) {
            for (int j = 0; j < total_param; j++) {
                OE[j] += O_flat[i * total_param + j] * E_list[i];
            }
        }
        for (int j = 0; j < total_param; j++) {
            OE[j] /= num_sweeps;
        }
        // Compute force vector F = <O E> - <O> * <E>
        F_vec.resize(total_param);
        for (int j = 0; j < total_param; j++) {
            F_vec[j] = OE[j] - O_mean[j] * E_mean;
        }
        // Compute covariance matrix S: S[j,k] = <(O_j - <O_j>)(O_k - <O_k>)>
        S_mat.assign(total_param * total_param, 0.0);
        for (int i = 0; i < num_sweeps; i++) {
            for (int j = 0; j < total_param; j++) {
                double diff_j = O_flat[i * total_param + j] - O_mean[j];
                for (int k = 0; k < total_param; k++) {
                    double diff_k = O_flat[i * total_param + k] - O_mean[k];
                    S_mat[j * total_param + k] += diff_j * diff_k;
                }
            }
        }
        for (int j = 0; j < total_param * total_param; j++) {
            S_mat[j] /= num_sweeps;
        }
        for (int j = 0; j < total_param; j++) {
            S_mat[j * total_param + j] += EPSILON;
        }
        // Solve linear system: S * delta = -F
        if (solve_linear_system(total_param, S_mat, F_vec, A, B, delta) != 0) {
            std::cout << "Linear system solve failed at iteration " << it << std::endl;
            continue;
        }
        // Update parameters:
        // Visible biases update
        for (int i = 0; i < N; i++) {
            a[i] += lr * delta[i];
        }
        // Hidden biases update
        for (int j = 0; j < M; j++) {
            b[j] += lr * delta[N + j];
        }
        // Weight parameters update
        for (int k = 0; k < M; k++) {
            w[k] += lr * delta[N + M + k];
        }
        // Recompute theta for the current s using the updated parameters.
        compute_theta(N, M, s, b, w, theta);
        std::cout << "Iteration " << it << ": Energy = " << E_mean << std::endl;
    }
}

int main(){
    int N = 10;      // Number of spins (length of the 1D chain)
    int M = 10;      // Number of hidden units (and weight parameters)
    double h = 1.0;  // Transverse field strength
    int num_sweeps = 1000; // Number of MC sweeps per iteration
    double lr = 0.01;
    int num_iter = 500;
    
    // Allocate and initialize RBM parameters using std::vector
    std::vector<double> a(N), b(M), w(M);
    init_rbm(N, M, a, b, w);
    
    // Initialize spin configuration s (each spin is +-1)
    std::vector<int> s(N);
    for (int i = 0; i < N; i++) {
        s[i] = (rand_uniform() < 0.5) ? 1 : -1;
    }
    
    // Compute initial theta = b + sum_i w[ (j-i+M) mod M ] * s[i]
    std::vector<double> theta(M);
    compute_theta(N, M, s, b, w, theta);
    
    // Determine total number of parameters: visible (N) + hidden (M) + weights (M) = N + 2*M
    int total_param = N + 2 * M;
    
    // Allocate work vectors (all are allocated in main)
    std::vector<double> E_list(num_sweeps);
    std::vector<double> O_flat(num_sweeps * total_param);
    std::vector<double> O_mean(total_param);
    std::vector<double> OE(total_param);
    std::vector<double> F_vec(total_param);
    std::vector<double> S_mat(total_param * total_param);
    std::vector<double> delta(total_param);
    std::vector<double> A(total_param * total_param);
    std::vector<double> B(total_param);
    
    // Temporary vectors for computations
    std::vector<double> theta_temp(M), tanh_theta(M), new_theta(M);
    std::vector<double> O_a_temp(N), O_b_temp(M), O_w_temp(M);
    
    // Run the VMC optimization
    run_vmc(N, M, h, num_sweeps, lr, num_iter,
            a, b, w, s, theta,
            E_list, O_flat, O_mean, OE, F_vec, S_mat, delta, A, B,
            theta_temp, tanh_theta, new_theta,
            O_a_temp, O_b_temp, O_w_temp);
    
    return 0;
}

