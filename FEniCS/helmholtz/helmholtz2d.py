# coding: utf-8
#
# Solving Helmhotz equation with FEniCS
# Author: Juan Luis Cano Rodríguez <juanlu@pybonacci.org>
# Inspired by: http://jasmcole.com/2014/08/25/helmhurts/
#
import sys
from dolfin import *

import scipy as sp
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from matplotlib.pylab import spy
from matplotlib.pylab import show

# try:
#     k = float(sys.argv[1])
#     print "Setting k equal to %.1f" % k
# except IndexError:
#     k = 50.0

import argparse
import sys

parser = argparse.ArgumentParser(description="Usage.py -p Nx Ny\n")

# Add a mutually exclusive group
group = parser.add_mutually_exclusive_group()

# Required  argument if not in test mode
group.add_argument('-p', '--param', nargs=3,
                   help='Nx, Ny k')

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
        # try:
        #     nz = int(args.param[2])
        # except ValueError:
        #     message = ('3rd argument, Nz, should be an int')
        #     raise parser.error(message)

        try:
            w = int(args.param[2])
        except ValueError:
            message = ('3rd argument, k, should be an int')
            raise parser.error(message)
            
else:
    print("Usage: poisson.py Nx Ny")
    sys.exit()

print("Setup mesh...")


## Problem data
E0 = Constant(0.0)
n = Constant(1.0)
# Wavenumber
k = Constant(w)  # 2.4 GHz / c

## Formulation
mesh = UnitSquareMesh(nx, ny)
# mesh = UnitCubeMesh(16, 16, 16)
V = FunctionSpace(mesh, "P", 1)

E = TrialFunction(V)
v = TestFunction(V)

# # Boundary conditions
# point = Point(0.5, 0.5)
# f = PointSource(V, point)

def E0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, E0, E0_boundary)

# f = Function(V)
# f.interpolate(Expression("(1-2*pi*pi*l*l)*cos(x[0]*pi*l)*cos(x[1]*pi*l)",l=k))

# Equation
a = ((k**2 / n**2) * inner(E, v) + inner(nabla_grad(E), nabla_grad(v))) * dx
# L = f * v * dx

# Assemble system
# A, rhs = assemble_system(a, L, bc)
# f.apply(rhs)

A = assemble(a)
bc.apply(A)
M_mat = as_backend_type(A).mat()
M_sparray = sp.sparse.csr_matrix(M_mat.getValuesCSR()[::-1], shape = M_mat.size)

# # Show matix structure
# spy(M_sparray)
# show()

print("Dump matrix...")
mmwrite('helmholtz2d_%dx%d_%d' % (nx, ny, w), M_sparray, symmetry='symmetric')

# # Solve system
# E = Function(V)
# E_vec = E.vector()
# solve(A, E_vec, rhs)

# # Plot and export solution
# plot(E, interactive=True)

# file = File("helmhurts.pvd")
# file << E
