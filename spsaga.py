import numpy as np
from scipy import sparse
from datetime import datetime
from numba import njit
from tqdm import tqdm


@njit
def logloss_func_grad(p, b):
    p *= b
    if p > 0:
        exp_t = np.exp(-p)
        phi = 1. / (1 + exp_t)
        return np.log(1 + exp_t), (phi - 1) * b
    else:
        exp_t = np.exp(p)
        phi = exp_t / (1. + exp_t)
        return -p + np.log(1 + exp_t), (phi - 1) * b

@njit
def prox_L1(x, gamma):
    """Proximal operator for the L1 norm"""
    return np.fmax(x - gamma, 0) - np.fmax(- x - gamma, 0)


@njit
def _SPSAGA_epoch(x, memory_gradient, gradient_average, A_data, A_indices, A_indptr, b,
                  A_norms, alpha, beta, d, n_samples, Lipschitz, sample_indices, line_search):
    """Run one epoch of Sparse Proximal SAGA"""

    for i in sample_indices:
        p = 0.  # .. sparse dot product ..
        for j in range(A_indptr[i], A_indptr[i+1]):
            j_idx = A_indices[j]
            p += x[j_idx] * A_data[j]

        fi, grad_i = logloss_func_grad(p, b[i])
        if line_search and (A_norms[i] * grad_i * grad_i) > 1e-8:
            f_next, _ = logloss_func_grad(p - grad_i * A_norms[i] / Lipschitz[0], b[i])
            # .. correct Lipschitz constant based on local information ..
            Lipschitz[0] *= .999
            while f_next > (fi - 0.5 * A_norms[i] * grad_i * grad_i / Lipschitz[0]):
                Lipschitz[0] *= 2

        step_size = 1. / (3 * Lipschitz[0])
        # .. update coefficients ..
        for j in range(A_indptr[i], A_indptr[i+1]):
            j_idx = A_indices[j]
            delta = (grad_i - memory_gradient[i]) * A_data[j]
            incr = delta + d[j_idx] * (
                gradient_average[j_idx] + alpha * x[j_idx])
            x[j_idx] = prox_L1(x[j_idx] - step_size * incr, beta * step_size * d[j_idx])
            gradient_average[j_idx] += delta / n_samples
        memory_gradient[i] = grad_i

    return x


def SPSAGA(A, b, x0, alpha, beta, max_iter=100, line_search=False):
    """
    Solves an L1-regularized logistic regression problem of the form

         argmin_x logistic_loss + alpha * ||x||^2_2 + beta * ||x||_1
    """
    A = sparse.csr_matrix(A, dtype=np.float)
    n_samples, n_features = A.shape

    # .. allocate SAGA historical gradients ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)

    # .. to tracking execution time later ..
    trace_x, trace_time = np.zeros((max_iter, n_features)), []
    start = datetime.now()

    # .. construct the diagonal D matrix ..
    B = A.copy()
    B.data[:] = 1
    d = np.array(B.mean(0)).ravel()
    idx = (d != 0)
    d[idx] = 1 / d[idx]

    # .. Lipschitz constant and for step-size selection ..
    A_norms = np.squeeze(np.asarray(A.multiply(A).sum(1)))
    Lipschitz = np.array([A_norms.max()])
    sample_indices = np.arange(n_samples)

    # .. main iteration ..
    for it in tqdm(range(max_iter), desc='Sparse SAGA iteration counter'):
        trace_x[it] = x0
        trace_time.append((datetime.now() - start).total_seconds())
        np.random.shuffle(sample_indices)
        _SPSAGA_epoch(
            x0, memory_gradient, gradient_average, A.data, A.indices, A.indptr, b, A_norms,
            alpha, beta, d, n_samples, Lipschitz, sample_indices, line_search)

    return x0, trace_x, trace_time
