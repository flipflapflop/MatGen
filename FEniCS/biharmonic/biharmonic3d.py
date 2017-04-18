"""This demo program solves the Biharmonic equation,

    nabla^4 u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 4 pi^4 sin(pi*x)*sin(pi*y)

and boundary conditions given by

    u(x, y)         = 0
    nabla^2 u(x, y) = 0

using a discontinuous Galerkin formulation (interior penalty method).
"""

# Copyright (C) 2009 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2011
#
# First added:  2009-06-26
# Last changed: 2012-11-12

# Begin demo

from dolfin import *

import scipy as sp
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from matplotlib.pylab import spy
from matplotlib.pylab import show

import argparse
import sys

parser = argparse.ArgumentParser(description="Usage.py -p Nx Ny Nz\n")

# Add a mutually exclusive group
group = parser.add_mutually_exclusive_group()

# Required  argument if not in test mode
group.add_argument('-p', '--param', nargs=3,
                   help='Nx, Ny, Nz')

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
    print("Usage: biharmonic.py -p Nx Ny Nz")
    sys.exit()

print("Setup mesh...")

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Make mesh ghosted for evaluation of DG terms
parameters["ghost_mode"] = "shared_facet"

# nx = 32
# ny = 32

# Create mesh and define function space
# mesh = UnitSquareMesh(nx, ny)
mesh = UnitCubeMesh(nx, ny, nz)
# V = FunctionSpace(mesh, "P", 3)
V = FunctionSpace(mesh, "P", 1)
# V = FunctionSpace(mesh, "CG", 2)

# Define Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Source(Expression):
    def eval(self, values, x):
        values[0] = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal component, mesh size and right-hand side
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
n = FacetNormal(mesh)
f = Source()

# Penalty parameter
alpha = Constant(8.0)

# Define bilinear form
a = inner(div(grad(u)), div(grad(v)))*dx \
  - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
  - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
  + alpha/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS

# Define linear form
L = f*v*dx

print("Assemble matrix...")

# M = assemble(a)
# bc.apply(M)
A, rhs = assemble_system(a, L, bc)

A_mat = as_backend_type(A).mat()
A_sparray = sp.sparse.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)
# Remove zero entries
A_sparray.eliminate_zeros()

print("Dump matrix...")

mmwrite('biharmonic3d_%dx%dx%d' % (nx, ny, nz), A_sparray, symmetry='symmetric')

# # Show matix structure
# spy(A_sparray)
# show()

# # Define linear form
# L = f*v*dx

# # Solve variational problem
# u = Function(V)
# solve(a == L, u, bc)

# # Save solution to file
# file = File("biharmonic.pvd")
# file << u

# # Plot solution
# plot(u, interactive=True)
