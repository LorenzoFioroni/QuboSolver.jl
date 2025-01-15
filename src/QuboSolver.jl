module QuboSolver

import Reexport: @reexport
@reexport using LinearAlgebra
@reexport using SparseArrays
using Random

BLAS.set_num_threads(1) # Set number of threads to 1 for better performance in parallelization

# QuboProblem
include("solver.jl")
include("solution.jl")
include("problem.jl")

# Utilities
include("utilities.jl")
include("random.jl")

# Solvers
include("solvers/solvers.jl")

end
