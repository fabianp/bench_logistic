#
# run as
#
#     $ run_bench.py [dataset]
#
# where [dataset] is one of rcv1, url, KDD10 or KDD12

import numpy as np
import copt as cp
from spsaga import SPSAGA
import sys
import os

dataset = sys.argv[1]
# ! WARNING, relies on a patched version of scikit-learn to be able to track
# the execution path of the logistic regression algorithm
from sklearn.linear_model.sag import sag_solver
from sklearn.linear_model.logistic import _logistic_loss

loader = getattr(cp.datasets, 'load_' + dataset, None)

if loader is not None:
    A, b = loader()
else:
    raise NotImplementedError

n_samples, n_features = A.shape

if not os.path.exists('data'):
    os.mkdir('data')


alpha = 1.0 / n_samples
x0 = np.zeros(n_features)
for it in range(3):
    print('Iteration %s' % it)
    _, _, _, trace_saga_x, trace_saga_time = sag_solver(
        A, b, sample_weight=None, loss='log', alpha=1., beta=0., max_iter=50,
               is_saga=True)
    trace_saga_func = [_logistic_loss(xi, A, b, 1.) for xi in trace_saga_x]
    np.save('data/saga_trace_time_%s_%s.npy' % (dataset, it), trace_saga_time - trace_saga_time[0])
    np.save('data/saga_trace_func_%s_%s.npy' % (dataset, it), trace_saga_func)

    x, trace_sps_x, trace_sps_time = SPSAGA(A, b, np.zeros(n_features), alpha, 0, max_iter=40, line_search=False)
    trace_sps_func = [_logistic_loss(xi, A, b, 1.) for xi in trace_sps_x]
    np.save('data/sps_trace_time_%s_%s.npy' % (dataset, it), trace_sps_time)
    np.save('data/sps_trace_func_%s_%s.npy' % (dataset, it), trace_sps_func)

    x, trace_sps_x, trace_sps_time = SPSAGA(A, b, np.zeros(n_features), alpha, 0, max_iter=40, line_search=True)
    trace_sps_func = [_logistic_loss(xi, A, b, 1.) for xi in trace_sps_x]
    np.save('data/spsls_trace_time_%s_%s.npy' % (dataset, it), trace_sps_time)
    np.save('data/spsls_trace_func_%s_%s.npy' % (dataset, it), trace_sps_func)
