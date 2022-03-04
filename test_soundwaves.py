
import numpy as np
import spectral
from scipy import sparse
import pytest
from equations import SoundWaves

import matplotlib.pyplot as plt

waves_const_errors = {32: 0.2, 64: 5e-3, 128: 1e-8}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_SoundWaves_const(N, dtype):
    x_basis = spectral.Chebyshev(N, interval=(0, 3))
    domain = spectral.Domain([x_basis])
    x = x_basis.grid()
    u = spectral.Field(domain, dtype=dtype)
    p = spectral.Field(domain, dtype=dtype)
    p0 = spectral.Field(domain, dtype=dtype)

    u.require_grid_space()
    u.data = np.exp(-(x-0.5)**2/0.01)

    p0.require_grid_space()
    p0.data = 1 + 0*x

    waves = SoundWaves(domain, u, p, p0)

    # check sparsity of M and L matrices
    assert len(waves.problem.pencils[0].M.data) < 5*N
    assert len(waves.problem.pencils[0].L.data) < 5*N

    waves.evolve(spectral.SBDF2, 2e-3, 5000)

    p.require_coeff_space()
    p.require_grid_space(scales=256//N)

    sol = np.loadtxt('waves_const.dat')

    xplotbasis = spectral.Chebyshev(256, interval=(0, 3))
    xplot = xplotbasis.grid()
    plt.figure()
    plt.plot(xplot, p.data)
    plt.plot(xplot, sol)
    



    

    error = np.max(np.abs(sol - p.data))

    assert error < waves_const_errors[N]

waves_variable_errors = {32: 4e-2, 64: 3e-4, 128: 1e-12}
@pytest.mark.parametrize('N', [32, 64, 128])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_SoundWaves_variable(N, dtype):
    x_basis = spectral.Chebyshev(N, interval=(0, 3))
    domain = spectral.Domain([x_basis])
    x = x_basis.grid()
    u = spectral.Field(domain, dtype=dtype)
    p = spectral.Field(domain, dtype=dtype)
    p0 = spectral.Field(domain, dtype=dtype)
    
    u.require_grid_space()
    u.data = np.exp(-(x-0.5)**2/0.01)

    p0.require_grid_space()
    p0.data = 0.1 + x**2/9

    waves = SoundWaves(domain, u, p, p0)

    # check sparsity of M and L matrices
    assert len(waves.problem.pencils[0].M.data) < 5*N
    assert len(waves.problem.pencils[0].L.data) < 5*N

    waves.evolve(spectral.SBDF2, 2e-3, 5000)

    p.require_coeff_space()
    p.require_grid_space(scales=256//N)

    sol = np.loadtxt('waves_variable.dat')
    
    xplotbasis = spectral.Chebyshev(256, interval=(0, 3))
    xplot = xplotbasis.grid()
    plt.figure()
    plt.plot(xplot, p.data)
    plt.plot(xplot, sol)

    error = np.max(np.abs(sol - p.data))
    
    print(error)

    assert error < waves_variable_errors[N]

