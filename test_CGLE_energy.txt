
import numpy as np
import spectral
#from scipy import sparse
#import pytest
from equations import CGLEquation

import matplotlib.pyplot as plt



def test_CGLE(N, dtype = np.complex128):
    x_basis = spectral.Chebyshev(N, interval=(0, 3))
    domain = spectral.Domain([x_basis])
    x = x_basis.grid()
    u = spectral.Field(domain, dtype=dtype)

    u.require_grid_space()
    u.data = np.exp(-(x-0.5)**2/0.01)

    cgle = CGLEquation(domain, u)

    # check sparsity of M and L matrices

    assert len(cgle.problem.pencils[0].M.data) < 9*N
    assert len(cgle.problem.pencils[0].L.data) < 9*N

    final_dat = np.zeros((500, 256), dtype = dtype)

    for i in range(500):
        cgle.evolve(spectral.SBDF2, 2e-3, 1)

        u.require_coeff_space()
        u.require_grid_space(scales=256//N)
    
        final_dat[i, :]= u.data

    plt.figure()
    plt.imshow(abs(final_dat))

N_list = [32, 64, 128]

for N in N_list:
    test_CGLE(N)
    
