#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON 1e-3

// Compute modulus that handles negative numbers correctly.
int mod_index(int x, int m) {
    int r = x % m;
    return (r < 0) ? r + m : r;
}

// Generate a normally distributed random number using the Box-Muller method.
double rand_normal() {
    double u1 = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);
    double u2 = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Initialize RBM parameters:
// a: visible biases (length N)
// b: hidden biases (length M)
// w: weight parameters (length M), such that W[j,i] = w[((j-i) mod M)]
void init_rbm(int N, int M, double* a, double* b, double* w) {
    int i, j;
    for (i = 0; i < N; i++) {
        a[i] = rand_normal() * 0.01;
    }
    for (j = 0; j < M; j++) {
        b[j] = rand_normal() * 0.01;
        w[j] = rand_normal() * 0.01;
    }
}

// Compute theta vector given the current spin configuration s.
// theta[j] = b[j] + sum_{i=0}^{N-1} w[ mod_index(j-i, M) ] * s[i]
void compute_theta(int N, int M, int* s, double* b, double* w, double* theta) {
    int i, j;
    for (j = 0; j < M; j++) {
        double temp = b[j];
        for (i = 0; i < N; i++) {
            temp += w[ mod_index(j - i, M) ] * s[i];
        }
        theta[j] = temp;
    }
}

// Compute log(psi(s)) = sum_i a[i]*s[i] + sum_j log(2*cosh(theta[j]))
double logpsi(int N, int M, int* s, double* a, double* b, double* w) {
    int i, j;
    double vis = 0.0;
    for (i = 0; i < N; i++) {
        vis += a[i] * s[i];
    }
    double hid = 0.0;
    for (j = 0; j < M; j++) {
        double temp = b[j];
        for (i = 0; i < N; i++) {
            temp += w[ mod_index(j - i, M) ] * s[i];
        }
        hid += log(2.0 * cosh(temp));
    }
    return vis + hid;
}

// Compute gradients using the current spin configuration s and current theta.
// Visible gradient: O_a[i] = s[i] (length N)
// Hidden gradient:  O_b[j] = tanh(theta[j]) (length M)
// Weight gradient:  O_w[k] = sum_{i=0}^{N-1} s[i] * tanh(theta[(i+k) mod M]) (length M)
void gradients(int N, int M, int* s, double* theta,
               double* O_a, double* O_b, double* O_w) {
    int i, j, k;
    // Visible gradient
    for (i = 0; i < N; i++) {
        O_a[i] = s[i];
    }
    // Hidden gradient
    for (j = 0; j < M; j++) {
        O_b[j] = tanh(theta[j]);
    }
    // Weight gradient
    for (k = 0; k < M; k++) {
        double sum = 0.0;
        for (i = 0; i < N; i++) {
            int idx = mod_index(i + k, M);
            sum += s[i] * tanh(theta[idx]);
        }
        O_w[k] = sum;
    }
}

// Compute the ratio psi(s')/psi(s) when flipping spin s[i] only.
// The new theta is computed as: new_theta[j] = theta[j] - 2 * w[ mod_index(j-i, M) ] * s[i]
// The ratio is given by:
//   ratio = exp( -2*a[i]*s[i] + sum_j [ log(2*cosh(new_theta[j])) - log(2*cosh(theta[j])) ] )
double psi_ratio(int N, int M, int* s, double* theta, double* a, double* w, int i, double* new_theta) {
    int j;
    int old_si = s[i];
    double d_vis = -2.0 * a[i] * old_si;
    double d_hid = 0.0;
    for (j = 0; j < M; j++) {
        new_theta[j] = theta[j] - 2.0 * w[ mod_index(j - i, M) ] * old_si;
        d_hid += log(2.0 * cosh(new_theta[j])) - log(2.0 * cosh(theta[j]));
    }
    double diff = d_vis + d_hid;
    return exp(diff);
}

// Perform one Metropolis sweep.
// For each attempted update, choose a random spin index i.
// If the update is accepted (based on psi_ratio), flip s[i] and update theta incrementally:
//   theta[j] = theta[j] - 2 * w[ mod_index(j-i, M) ] * s[i]  for all j.
void metropolis_step(int N, int M, int* s, double* theta, double* a, double* b, double* w, double h, double* new_theta) {
    int k, i, j;
    for (k = 0; k < N; k++) {
        i = rand() % N;
        double ratio = psi_ratio(N, M, s, theta, a, w, i, new_theta);
        double r = ((double) rand()) / ((double) RAND_MAX);
        double prob = (ratio * ratio < 1.0) ? ratio * ratio : 1.0;
        if (r < prob) {
            int old_si = s[i];
            s[i] = -old_si;
            for (j = 0; j < M; j++) {
                theta[j] -= 2.0 * w[ mod_index(j - i, M) ] * old_si;
            }
        }
    }
}

// Compute the local energy.
// Diagonal term: E_diag = - sum_i s[i]*s[i+1] (with periodic boundary conditions)
// Off-diagonal term: E_off = - h * sum_i psi_ratio(s^(i))/psi(s)
double local_energy(int N, int M, int* s, double* theta, double* a, double* b, double* w, double h, double* new_theta) {
    int i;
    double E_diag = 0.0;
    for (i = 0; i < N - 1; i++) {
        E_diag += s[i] * s[i+1];
    }
    E_diag += s[N-1] * s[0];
    E_diag = -E_diag;
    double E_off = 0.0;
    for (i = 0; i < N; i++) {
        double ratio = psi_ratio(N, M, s, theta, a, w, i, new_theta);
        E_off += -h * ratio;
    }
    return E_diag + E_off;
}

// Solve the linear system S * delta = -F using Gaussian elimination.
// S_mat: coefficient matrix (size n*n), F: right-hand side vector (size n)
// A, B, delta: work arrays (sizes n*n, n, n respectively)
// Returns 0 on success; nonzero if the system is singular.
int solve_linear_system(int n, double* S_mat, double* F, double* A, double* B, double* delta) {
    int i, j, k;
    for (i = 0; i < n * n; i++) {
        A[i] = S_mat[i];
    }
    for (i = 0; i < n; i++) {
        B[i] = -F[i];
    }
    // Forward elimination
    for (i = 0; i < n; i++) {
        int pivot = i;
        double max_val = fabs(A[i * n + i]);
        for (k = i + 1; k < n; k++) {
            double val = fabs(A[k * n + i]);
            if (val > max_val) {
                max_val = val;
                pivot = k;
            }
        }
        if (max_val < 1e-12)
            return 1;
        if (pivot != i) {
            for (j = 0; j < n; j++) {
                double temp = A[i * n + j];
                A[i * n + j] = A[pivot * n + j];
                A[pivot * n + j] = temp;
            }
            double temp = B[i];
            B[i] = B[pivot];
            B[pivot] = temp;
        }
        for (k = i + 1; k < n; k++) {
            double factor = A[k * n + i] / A[i * n + i];
            for (j = i; j < n; j++) {
                A[k * n + j] -= factor * A[i * n + j];
            }
            B[k] -= factor * B[i];
        }
    }
    // Back substitution
    for (i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (j = i + 1; j < n; j++) {
            sum += A[i * n + j] * delta[j];
        }
        delta[i] = (B[i] - sum) / A[i * n + i];
    }
    return 0;
}

// Run the VMC optimization using the Stochastic Reconfiguration (SR) method.
// Parameter vector consists of visible biases a (length N), hidden biases b (length M),
// and weight parameters w (length M). Total parameter dimension = N + 2*M.
// All temporary/work arrays are allocated in main and passed to this function.
void run_vmc(int N, int M, double h, int num_sweeps, double lr, int num_iter,
             double* a, double* b, double* w, int* s, double* theta,
             double* E_list, double* O_flat, double* O_mean, double* OE, double* F_vec,
             double* S_mat, double* delta, double* A, double* B,
             double* theta_temp, double* tanh_theta, double* new_theta,
             double* O_a_temp, double* O_b_temp, double* O_w_temp) {
    int total_param = N + 2 * M; // visible (N) + hidden (M) + weight (M)
    int i, j, k;
    for (int it = 0; it < num_iter; it++) {
        // Equilibration: perform num_sweeps/10 Metropolis sweeps
        for (i = 0; i < num_sweeps/10; i++) {
            metropolis_step(N, M, s, theta, a, b, w, h, new_theta);
        }
        // Monte Carlo Sampling
        for (i = 0; i < num_sweeps; i++) {
            metropolis_step(N, M, s, theta, a, b, w, h, new_theta);
            E_list[i] = local_energy(N, M, s, theta, a, b, w, h, new_theta);
            // Compute gradients using current s and the already updated theta.
            gradients(N, M, s, theta, O_a_temp, O_b_temp, O_w_temp);
            int idx = 0;
            for (j = 0; j < N; j++) {
                O_flat[i * total_param + idx] = O_a_temp[j];
                idx++;
            }
            for (j = 0; j < M; j++) {
                O_flat[i * total_param + idx] = O_b_temp[j];
                idx++;
            }
            for (k = 0; k < M; k++) {
                O_flat[i * total_param + idx] = O_w_temp[k];
                idx++;
            }
        }
        // Compute average energy
        double E_mean = 0.0;
        for (i = 0; i < num_sweeps; i++) {
            E_mean += E_list[i];
        }
        E_mean /= num_sweeps;
        // Compute average gradient vector O_mean (length total_param)
        for (j = 0; j < total_param; j++) {
            O_mean[j] = 0.0;
        }
        for (i = 0; i < num_sweeps; i++) {
            for (j = 0; j < total_param; j++) {
                O_mean[j] += O_flat[i * total_param + j];
            }
        }
        for (j = 0; j < total_param; j++) {
            O_mean[j] /= num_sweeps;
        }
        // Compute <O E> vector
        for (j = 0; j < total_param; j++) {
            OE[j] = 0.0;
        }
        for (i = 0; i < num_sweeps; i++) {
            for (j = 0; j < total_param; j++) {
                OE[j] += O_flat[i * total_param + j] * E_list[i];
            }
        }
        for (j = 0; j < total_param; j++) {
            OE[j] /= num_sweeps;
        }
        // Compute force vector F = <O E> - <O>*<E>
        for (j = 0; j < total_param; j++) {
            F_vec[j] = OE[j] - O_mean[j] * E_mean;
        }
        // Compute covariance matrix S_mat: S[j,k] = <O_j * O_k> - <O_j>*<O_k>
        for (j = 0; j < total_param * total_param; j++) {
            S_mat[j] = 0.0;
        }
        for (i = 0; i < num_sweeps; i++) {
            for (j = 0; j < total_param; j++) {
                double diff_j = O_flat[i * total_param + j] - O_mean[j];
                for (k = 0; k < total_param; k++) {
                    double diff_k = O_flat[i * total_param + k] - O_mean[k];
                    S_mat[j * total_param + k] += diff_j * diff_k;
                }
            }
        }
        for (j = 0; j < total_param * total_param; j++) {
            S_mat[j] /= num_sweeps;
        }
        for (j = 0; j < total_param; j++) {
            S_mat[j * total_param + j] += EPSILON;
        }
        // Solve linear system S_mat * delta = -F
        if (solve_linear_system(total_param, S_mat, F_vec, A, B, delta) != 0) {
            printf("Linear system solve failed at iteration %d\n", it);
            continue;
        }
        // Update parameters:
        // Update visible biases a
        for (i = 0; i < N; i++) {
            a[i] += lr * delta[i];
        }
        // Update hidden biases b
        for (j = 0; j < M; j++) {
            b[j] += lr * delta[N + j];
        }
        // Update weight parameters w
        for (k = 0; k < M; k++) {
            w[k] += lr * delta[N + M + k];
        }
        // Recompute theta for the current spin configuration with updated parameters.
        compute_theta(N, M, s, b, w, theta);
        printf("Iteration %d: Energy = %f\n", it, E_mean);
    }
}

int main(){
    int N = 10;      // Number of spins (length of the 1D chain)
    int M = 10;      // Number of hidden units (and weight parameters)
    double h = 1.0;  // Transverse field strength
    int num_sweeps = 1000;
    double lr = 0.01;
    int num_iter = 500;
    
    srand(123); // Seed the random number generator
    
    // Allocate memory for RBM parameters
    double* a = (double*) malloc(N * sizeof(double));
    double* b = (double*) malloc(M * sizeof(double));
    double* w = (double*) malloc(M * sizeof(double));
    init_rbm(N, M, a, b, w);
    
    // Allocate and initialize the spin configuration s (each spin is +-1)
    int* s = (int*) malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        s[i] = (rand() % 2) ? 1 : -1;
    }
    
    // Allocate memory and compute initial theta = b + sum_i w[ mod_index(j-i, M) ] * s[i]
    double* theta = (double*) malloc(M * sizeof(double));
    compute_theta(N, M, s, b, w, theta);
    
    // Total parameter dimension = visible (N) + hidden (M) + weights (M) = N + 2*M
    int total_param = N + 2 * M;
    
    // Allocate work arrays
    double* E_list = (double*) malloc(num_sweeps * sizeof(double));
    double* O_flat = (double*) malloc(num_sweeps * total_param * sizeof(double));
    double* O_mean = (double*) malloc(total_param * sizeof(double));
    double* OE = (double*) malloc(total_param * sizeof(double));
    double* F_vec = (double*) malloc(total_param * sizeof(double));
    double* S_mat = (double*) malloc(total_param * total_param * sizeof(double));
    double* delta = (double*) malloc(total_param * sizeof(double));
    double* A = (double*) malloc(total_param * total_param * sizeof(double));
    double* B = (double*) malloc(total_param * sizeof(double));
    
    // Allocate temporary arrays
    double* theta_temp = (double*) malloc(M * sizeof(double));
    double* tanh_theta = (double*) malloc(M * sizeof(double));
    double* new_theta = (double*) malloc(M * sizeof(double));
    double* O_a_temp = (double*) malloc(N * sizeof(double));
    double* O_b_temp = (double*) malloc(M * sizeof(double));
    double* O_w_temp = (double*) malloc(M * sizeof(double));
    
    // Run VMC optimization
    run_vmc(N, M, h, num_sweeps, lr, num_iter,
            a, b, w, s, theta,
            E_list, O_flat, O_mean, OE, F_vec, S_mat, delta, A, B,
            theta_temp, tanh_theta, new_theta, O_a_temp, O_b_temp, O_w_temp);
    
    // Free all allocated memory
    free(a);
    free(b);
    free(w);
    free(s);
    free(theta);
    free(E_list);
    free(O_flat);
    free(O_mean);
    free(OE);
    free(F_vec);
    free(S_mat);
    free(delta);
    free(A);
    free(B);
    free(theta_temp);
    free(tanh_theta);
    free(new_theta);
    free(O_a_temp);
    free(O_b_temp);
    free(O_w_temp);
    
    return 0;
}

