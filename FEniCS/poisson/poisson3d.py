from dolfin import *

import scipy as sp
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from scipy.io import savemat
from matplotlib.pylab import spy
from matplotlib.pylab import show
import numpy as np

# nx = 4
# ny = 4
# nz = 4

import argparse
import sys

parser = argparse.ArgumentParser(description="Usage.py -p Nx Ny\n")

# Add a mutually exclusive group
group = parser.add_mutually_exclusive_group()

# Required  argument if not in test mode
group.add_argument('-p', '--param', nargs=3,
                   help='Nx, Ny Nz')

# Parse args
args = parser.parse_args()

if(args.param):
    if len(args.param) == 3:
        message = ''
        try:
            nx = int(args.param[0])
        except ValueError:
            message = ('1st argument, Nx, should be an int')
            raise parser.error(message)
        try:
            ny = int(args.param[1])
        except ValueError:
            message = ('2nd argument, Ny, should be an int')
            raise parser.error(message)
        try:
            nz = int(args.param[2])
        except ValueError:
            message = ('3rd argument, Nz, should be an int')
            raise parser.error(message)
            
else:
    print("Usage: poisson.py Nx Ny")
    sys.exit()

print("Setup mesh...")

# Create mesh and define function space
# mesh = UnitSquareMesh(nx, ny)
mesh = UnitCubeMesh(nx, ny, nz)
# V = FunctionSpace(mesh, "Lagrange", 1)
# V = FunctionSpace(mesh, 'P', 1)
# V = FunctionSpace(mesh, 'DG', 0)
V = FunctionSpace(mesh, "P", 1)
# V = FunctionSpace(mesh, "N1E", 1)

#Define Dirichlet boundary (x = 0 or x = 1)
# def boundary(x):
#     return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Define source terms
# Expression: C++ syntax where x[0], x[1] and x[2] correspond
# respectively to x, y, and z
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2) + pow(x[2] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")

# Define the variational problem
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

print("Assemble matrix...")

# Alternatively we can use assembly_system function which takes
# boundary conditions into account
# A = PETScMatrix()
A, rhs = assemble_system(a, L, bc)

# A = assemble(a)
# bc.apply(A)

# print(A.is_symmetric(1e-10))
# print(A.nnz())
# A.zero()

# print(M.array())
# savemat('A.mat', {'A': M.array()})

A_mat = as_backend_type(A).mat()

A_sparray = sp.sparse.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size, dtype=np.float)
# print(A_sparray.count_nonzero())
# Remove zero entries
A_sparray.eliminate_zeros()
# print(A_sparray.count_nonzero())
print("Dump matrix...")

mmwrite('poisson3d_%dx%dx%d' % (nx, ny, nz), A_sparray, symmetry='symmetric')

# # Show matix structure
# spy(A_sparray)
# show()

# Compute solution
# u = Function(V)
# solve(a == L, u, bc)

# Save solution in VTK format
# file = File("poisson.pvd")
# file << u

# Plot solution
# plot(u, interactive=True)
