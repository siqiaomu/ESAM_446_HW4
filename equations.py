
import spectral
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

class SoundWaves:
    
    def __init__(self, domain, u, ux, p0):
        self.u = u
        self.ux = ux
        self.domain = domain
        self.dtype = dtype = u.dtype
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.ux_RHS = spectral.Field(domain, dtype=dtype)
        
        self.problem = spectral.InitialValueProblem(domain, [u, ux], [self.u_RHS, self.ux_RHS],
                                                    num_BCs=2, dtype=dtype)
        
        self.interval = domain.bases[0].interval
        
        self.p0 = p0
        
        pr = self.problem.pencils[0]
        
        self.N = N = domain.bases[0].N
        Z = np.zeros((N, N))
        
        diag =  np.arange(N-1)+1
        interval_length = self.interval[1] - self.interval[0] 
        self.D = D = (2/interval_length) * sparse.diags(diag, offsets=1)

        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0,2))
        
        M = sparse.csr_matrix((2*N+2,2*N+2))
        M[0:N, 0:N] = C
        M[N:2*N, N:2*N] = C
        pr.M = M
        
        # L matrix
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        cols = np.zeros((2*N,2))
        cols[  N-1, 0] = 1
        cols[2*N-1, 1] = 1
        corner = np.zeros((2,2))

        Z = np.zeros((N, N))
        L = sparse.bmat([[Z, D],
                         [D, Z]])
        L = sparse.bmat([[      L,   cols],
                         [BC_rows, corner]])
        L = L.tocsr()
        pr.L = L
        self.t = 0
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        ux = self.ux
        ux_RHS = self.ux_RHS
        dudx = self.dudx
        
        p0 = self.p0
        
        
        for i in range(num_steps):
            u.require_coeff_space()
            ux.require_coeff_space()       
            
            dudx.require_coeff_space()
            dudx_temp = self.D @ u.data
            
            dudx.data = spla.spsolve(self.C, dudx_temp)
            
            ux_RHS.require_grid_space()
            p0.require_grid_space()

            dudx.require_grid_space()

            ux_RHS.data = (1 - p0.data) * dudx.data
            ux_RHS.require_coeff_space()
            ux_RHS.data = self.C @ ux_RHS.data
            
            
            ts.step(dt, [0, 0])
            self.t += dt


class CGLEquation:

    def __init__(self, domain, u):
        self.u = u
        self.domain = domain
        self.dtype = dtype = u.dtype
        
        
        self.interval = domain.bases[0].interval
        
        self.b = 0.5
        self.c = - 1.76

        self.N = N = domain.bases[0].N
        
        diag =  np.arange(N-1)+1
        interval_length = self.interval[1] - self.interval[0]        
        self.D = D = (2/interval_length) * sparse.diags(diag, offsets=1)


        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0,2))

        self.ux = ux = spectral.Field(domain, dtype=dtype)
        
        ux.require_coeff_space()
        dudx_temp = self.D @ u.data        
        ux.data = spla.spsolve(self.C, dudx_temp)

        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.ux_RHS = spectral.Field(domain, dtype=dtype)
        
        self.problem = spectral.InitialValueProblem(domain, [u, ux], [self.u_RHS, self.ux_RHS],
                                                    num_BCs=2, dtype=dtype)
        
        pr = self.problem.pencils[0]

        
        
        M = sparse.csr_matrix((2*N+2,2*N+2))
        M[N:2*N, :N] = C
        pr.M = M
        
        # L matrix
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        cols = np.zeros((2*N,2))
        cols[  N-1, 0] = 1
        cols[2*N-1, 1] = 1
        corner = np.zeros((2,2))

        Z = np.zeros((N, N))
        L = sparse.bmat([[D, -C],
                         [-C, -(1 + 1j * self.b)*D]])
        L = sparse.bmat([[      L,   cols],
                         [BC_rows, corner]])
        L = L.tocsr()
        pr.L = L
        self.t = 0


        

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        ux = self.ux
        ux_RHS = self.ux_RHS

        for i in range(num_steps):

            u.require_coeff_space()
            ux.require_coeff_space()
            ux_RHS.require_coeff_space()
            
            u.require_grid_space(scales=3/2)
            ux.require_grid_space(scales=3/2)
            ux_RHS.require_grid_space(scales=3/2)

            ux_RHS.data = - (1 + 1j * self.c) * (np.absolute(u.data)**2) * u.data
            ux_RHS.require_coeff_space()
            ux_RHS.data = self.C @ ux_RHS.data
            
            
            ts.step(dt, [0, 0])
            self.t += dt




class BurgersEquation:
    
    def __init__(self, domain, u, nu):
        dtype = u.dtype
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = -nu*D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space()
            dudx.require_grid_space()
            u_RHS.require_grid_space()
            u_RHS.data = -u.data*dudx.data
            ts.step(dt)


class KdVEquation:
    
    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 3/2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = D@D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space(scales=self.dealias)
            dudx.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 6*u.data*dudx.data
            ts.step(dt)


class SHEquation:

    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        op = I + D@D
        p.L = op @ op + 0.3*I

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        u_RHS = self.u_RHS
        for i in range(num_steps):
            u.require_coeff_space()
            u.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 1.8*u.data**2 - u.data**3
            ts.step(dt)



