import numpy as np
import spectral
#from scipy import sparse
#import pytest
from equations import CGLEquation

#import matplotlib.pyplot as plt



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

    cgle.evolve(spectral.SBDF2, 2e-3, 5000)

    u.require_coeff_space()
    u.require_grid_space(scales=256//N)
    
    print(u.data[150])

    #xplotbasis = spectral.Chebyshev(256, interval=(0, 3))
    #xplot = xplotbasis.grid()
    #plt.figure()
    #plt.plot(xplot, u.data)

N_list = [32, 64, 128]

for N in N_list:
    test_CGLE(N)