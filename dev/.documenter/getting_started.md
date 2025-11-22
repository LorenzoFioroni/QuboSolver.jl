
# Getting Started {#Getting-Started}

This package provides a suite of tools for solving _Quadratic Unconstrained Binary Optimization_ (QUBO) problems.  Its main feature is the **GCS** algorithm, which implements an efficient variational simulation of Quantum Annealing [Fioroni and Savona [1]](/resources/bibliography#fioroni2025entanglement).  In addition, we also provide several other solvers for benchmarking purposes [[2](/resources/bibliography#bowlesQuadraticUnconstrainedBinary2022)–[5](/resources/bibliography#gurobioptimizationllcGurobiOptimizerReference2024)]. 

The complete list of implemented solvers can be found [at the end of this page](/getting_started#solvers).

We will now provide a brief overview of the package and its features through a simple example. More information on the implemented functions and their usage can be found in the [API documentation](resources/api.md).

## Basic usage {#Basic-usage}

A QUBO problem is specified by a real-valued symmetric matrix $W$ and a vector $c$ through the relation

$$    s^\star = \text{arg}\!\!\!\min_{s \in \{0,1\}^N} \left[s^T W s + c^T s\right].$$

which also defines its solution $s^\star$. We can define the problem in Julia via the [`QuboProblem`](/resources/api#QuboSolver.QuboProblem) object:

```julia
using QuboSolver

N = 10 # Number of variables

W = randn(N, N)
W = triu(W, 1) + triu(W, 1)' # Symmetric matrix with zero diagonal
c = randn(N) # Linear term

problem = QuboProblem(W, c)
problem.W
```


```ansi
10×10 Matrix{Float64}:
  0.0         0.705993   -1.0565    …  -0.783526   -0.962989  -1.3637
  0.705993    0.0         0.148361     -1.0749      0.437572  -0.0685601
 -1.0565      0.148361    0.0          -0.942668   -0.342168   2.5139
  0.0110129  -0.903541    0.400704     -0.0090979  -0.912997   0.124062
  0.128204   -1.25475    -0.308811      0.561116    0.167196   0.442011
 -0.217172    1.16814    -0.710293  …   0.166245   -1.24566    0.655911
 -0.200951    1.58843    -0.576838     -0.248753   -0.652145  -0.20563
 -0.783526   -1.0749     -0.942668      0.0        -0.292977   0.770703
 -0.962989    0.437572   -0.342168     -0.292977    0.0       -1.21394
 -1.3637     -0.0685601   2.5139        0.770703   -1.21394    0.0
```


::: tip

Often times QUBO problems are defined in terms of binary variables $z_i \in \{0,1\}$ as opposed to the spin variables $s_i \in \{-1,1\}$ we used here. `QuboSolver.jl` provides the functions [`binary_to_spin`](/resources/api#QuboSolver.binary_to_spin) and [`spin_to_binary`](/resources/api#QuboSolver.spin_to_binary) to convert between the two representations.

::: details Code example

```julia
using QuboSolver

N = 10

W_bin = randn(N, N)
W_bin = triu(W, 1) + triu(W, 1)' # Symmetric matrix with zero diagonal
c_bin = randn(N) # Linear term

W_spin, c_spin = binary_to_spin(W_bin, c_bin) # Convert the problem to spin variables
spin_problem = QuboProblem(W_spin, c_spin)
```


:::

The resolution of the problem with one of the available solvers is done by importing the desired solver and calling the [`solve!`](/resources/api#QuboSolver.solve!) function with the problem as an argument.

In the following example we will use a brute-force solver to find the solution of a simple all-to-all problem with antiferromagnetic interactions

```julia
using QuboSolver
using QuboSolver.Solvers.BruteForce

W = [ 0.0 -1.0  1.0 -1.0
     -1.0  0.0 -1.0  1.0
      1.0 -1.0  0.0 -1.0
     -1.0  1.0 -1.0  0.0]
c = [1.0, 0.0, 0.0, 0.0] # Break the spin-flip symmetry
problem = QuboProblem(W, c)

bf_solution = solve!(problem, BruteForce_solver())
```


The [`solve!`](/resources/api#QuboSolver.solve!) function runs the specified solver to find the solution of the problem and returns a [`Solution`](/resources/api#QuboSolver.Solution) object containing the solution and other useful information:

```julia
julia> bf_solution
🟦🟨🟦🟨 - Energy: -13.0 - Solver: BruteForce_solver - Metadata count: 2
```


The output shows a graphical representation of the solution, with blue and yellow squares representing the values of the variables, the energy of the solution and additional information about the execution such as the runtime.

```julia
julia> println("Runtime: $(round(bf_solution["runtime"]*1e6)) μs")
Runtime: 306.0 μs
```


::: tip 

Each solver has its own set of parameters that can be passed to the [`solve!`](/resources/api#QuboSolver.solve!) function, and its own set of metadata that can be accessed from the [`Solution`](/resources/api#QuboSolver.Solution) object.  For more information on the available solvers and their parameters, please refer to the [API documentation](resources/api.md).

:::

We conclude this section with one additional example.  We will use the [`GCS_solver`](/resources/api#QuboSolver.Solvers.GCS.GCS_solver) solver to study the same problem as before:

```julia
using QuboSolver.Solvers.GCS

gcs_solution = solve!(problem, GCS_solver(), progressbar=false)
```


The output will be similar to the one we obtained with the brute-force solver:

```julia
julia> gcs_solution
🟦🟨🟦🟨 - Energy: -13.0 - Solver: GCS_solver - Metadata count: 1
```


and the runtime:

```julia
julia> println("Runtime: $(round(gcs_solution["runtime"]*1e3)) ms")
Runtime: 61.0 ms
```


Of course for such a small problem the brute-force solver is much faster than the GCS solver. The GCS solver is designed to be used for larger problems, where exact algorithms become intractable.

## A more challenging example {#A-more-challenging-example}

We will now focus on a more challenging example, where we will not be able to use the brute-force solver to find the solution.  The problem is an instance of the $3D$ Edwards-Anderson model [[6](/resources/bibliography#barahonaComputationalComplexityIsing1982)] defined on a $8 \times 8 \times 8$ cubic lattice. The total number of variables is thus $N = 512$, far beyond the capabilities of the brute-force solver. Yet, we can still solve the problem for its exact solution using the Gurobi library [[5](/resources/bibliography#gurobioptimizationllcGurobiOptimizerReference2024)].

`QuboSolver.jl` provides convenient functions to generate instances of some QUBO problem families. See the [API documentation](resources/api.md) for more information on the available problem families and their parameters.

```julia
using QuboSolver

W = rand(EdwardsAnderson(), 8; sparse=true)
problem = QuboProblem(W)
problem.W
```


```ansi
512×512 SparseMatrixCSC{Float64, Int64} with 2688 stored entries:
⎡⢿⣷⣄⠀⠀⠱⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤
⎢⠀⠙⢿⣷⣄⠀⠈⠱⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⢄⡀⠀⠙⠿⣧⣄⠀⠈⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠑⢆⡀⠀⠙⢿⣷⣄⠀⠈⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠑⢦⡀⠀⠙⢿⣷⡄⠀⠀⠓⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠙⢦⡀⠀⠉⢿⣷⣄⠀⠀⠓⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠙⢤⠀⠀⠙⢿⣷⣄⠀⠀⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢤⠀⠀⠙⠿⣧⣄⠀⠈⠳⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢤⡀⠀⠙⢿⣷⣄⠀⠈⠱⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠙⢿⣷⠀⠀⠈⠱⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢆⡀⠀⠀⢿⣷⣄⠀⠈⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢆⡀⠀⠙⢿⣷⣄⠀⠈⠓⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢦⡀⠀⠙⢻⣶⣄⠀⠀⠓⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⠀⠀⠙⢿⣷⣄⠀⠀⠓⣄⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢤⠀⠀⠙⢿⣷⣀⠀⠈⠳⣄⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢤⠀⠀⠘⢿⣷⣄⠀⠈⠳⢄⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠙⢿⣷⣄⠀⠈⠱⢄⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠙⢻⣶⣄⠀⠈⠑⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢆⡀⠀⠙⢿⣷⣄⠀⎥
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢆⠀⠀⠙⢿⣷⎦
```


The problem is now defined and we can use the solvers to find its solution the same way as before.  This time we will compare the results obtained by [`LQA_solver`](/resources/api#QuboSolver.Solvers.LQA.LQA_solver) and [`GCS_solver`](/resources/api#QuboSolver.Solvers.GCS.GCS_solver) to the exact solutions found by [`Gurobi_solver`](/resources/api#QuboSolver.Solvers.GurobiLib.Gurobi_solver). Both the GCS and LQA solvers implement variational simulations of quantum annealing, with the main difference between the two being the Ansatz used to describe the quantum state. While LQA uses a product-state Ansatz, GCS employs a more expressive Ansatz which is able to describe entangled states [[1](/resources/bibliography#fioroni2025entanglement), [2](/resources/bibliography#bowlesQuadraticUnconstrainedBinary2022)].

&lt;!– Executed offline to avoid issues with Gurobi licences on the Github runners –&gt;

```@julia
using QuboSolver.Solvers.GurobiLib
using QuboSolver.Solvers.LQA
using QuboSolver.Solvers.GCS

grb_sol = solve!(problem, Gurobi_solver(), output=false) # Exact solution
lqa_sol = solve!(problem, LQA_solver(), progressbar=false) # LQA solution
gcs_sol = solve!(problem, GCS_solver(), progressbar=false) # GCS solution
```


Let&#39;s now compare the solutions obtained by the two approximate methods with the exact solution found by Gurobi.

&lt;!– Executed offline to avoid issues with Gurobi licences on the Github runners –&gt;

```julia
julia> println("Gurobi energy: $(grb_sol.energy)")
Gurobi energy: -798.4562953218554

julia> println("GCS energy: $(gcs_sol.energy)")
GCS energy: -793.647667961538

julia> println("LQA energy: $(lqa_sol.energy)")
LQA energy: -762.4597885183423
```


We can also display the relative error achieved by the two methods:

&lt;!– Executed offline to avoid issues with Gurobi licences on the Github runners –&gt;

```julia
julia> mean_err(a, b) = round(100*abs(a - b)/abs(b), digits=2)
mean_err (generic function with 1 method)

julia> println("GCS relative error: $(mean_err(gcs_sol.energy, grb_sol.energy))%")
GCS relative error: 0.6%

julia> println("LQA relative error: $(mean_err(lqa_sol.energy, grb_sol.energy))%")
LQA relative error: 4.51%
```


The GCS solver is able to find a solution that is within much closer to the optimal solution compared to LQA.

::: tip

For a systematic study of the performance of the GCS solver and its comparison with the other implemented solvers, please refer to the article &quot;[Entanglement-assisted variational algorithm for discrete optimization problems](https://www.nature.com/articles/s42005-025-02338-0)&quot;.  Both result quality and runtime are discussed in detail there.

:::

## Solvers {#Solvers}

Finally, we want to provide a brief overview of the available solvers implemented in the package. For more information on each one of them, please refer to the [API documentation](resources/api.md).

::: warning A note on execution times

When comparing heuristics, the two most important aspects to consider are the **solution quality** and the **runtime** of the algorithm.

However, as argued in our article [[1](/resources/bibliography#fioroni2025entanglement)], the number of iterations required to each a given error level together with the cost of each iteration can provide a more robust measure compared to the runtime alone.  The reason is that the runtime of the algorithm can be affected by several factors, such as the hardware, the language used and most importantly the way the algorithm is implemented.

An analysis that correlates solution quality with the number of iterations each method has been run for and the cost of each iteration is a more robust metric and allows for a fairer comparison between different algorithms.

:::

### Brute force {#Brute-force}

[`BruteForce_solver`](/resources/api#QuboSolver.Solvers.BruteForce.BruteForce_solver) finds the exact solution of the problem by evaluating the energy of all possible configurations.  This solver is only suitable for small problems, as its runtime scales exponentially with the number of variables.

### Gurobi {#Gurobi}

[`Gurobi_solver`](/resources/api#QuboSolver.Solvers.GurobiLib.Gurobi_solver) uses the Gurobi optimization library [[5](/resources/bibliography#gurobioptimizationllcGurobiOptimizerReference2024)] to find the solution of the problem.  When the [`solve!`](/resources/api#QuboSolver.solve!) function is called, an additional parameter `mipgap` can be passed to the solver to specify the confidence level of the solution.  The default value is `0.0`, which means that the solver will try to find the exact solution (with an exponentially long runtime). To use this solver, the user must have the Gurobi library installed and properly configured in their system. Note that Gurobi is a commercial solver and requires a license to be used. 

### GCS {#GCS}

[`GCS_solver`](/resources/api#QuboSolver.Solvers.GCS.GCS_solver) implements the GCS algorithm [[1](/resources/bibliography#fioroni2025entanglement)], which is a variational simulation of Quantum Annealing using a parameterized Ansatz of generalized coherent states.  The solver emulates the dynamics of quantum annealing by using an efficient variational approach.  The quantum state is represented throuhout the annealng via an Ansatz of generalized coherent states, which are optimized using a gradient descent algorithm. Importantly, the states that GCS describes are entangled, leveraging the potential advantages of quantum computing for solving combinatorial optimization problems.

The computational cost of each iteration of the GCS solver is $O(N^3)$ for a general problem, and $O(N^2)$ for problems with a sparse structure. Here $N$ is the number of variables in the problem.

### Local Quantum Annealing {#Local-Quantum-Annealing}

[`LQA_solver`](/resources/api#QuboSolver.Solvers.LQA.LQA_solver) implements the local quantum annealing algorithm [[2](/resources/bibliography#bowlesQuadraticUnconstrainedBinary2022)], which is a variational simulation of quantum annealing similar to the one used in the GCS solver. The main difference is that the LQA solver uses a product state Ansatz, which is not able to describe entangled states.

The computational cost of each iteration of the LQA solver is $O(N^2)$ for a general problem, and $O(N)$ for problems with a sparse structure. Here $N$ is the number of variables in the problem.

### Simulated Annealing {#Simulated-Annealing}

[`SA_solver`](/resources/api#QuboSolver.Solvers.SA.SA_solver) implements a _vanilla_ version of the simulated annealing algorithm [[4](/resources/bibliography#kirkpatrickOptimizationSimulatedAnnealing1983)]. It uses a Metropolis-Hastings algorithm with single-spin flips and a linear cooling schedule to find the solution of the problem.

The computational cost of each iteration of the Simulated Annealing solver is $O(N^2)$ for a general problem, and $O(N)$ for problems with a sparse structure. Here $N$ is the number of variables in the problem.

### PTICM {#PTICM}

[`PTICM_solver`](/resources/api#QuboSolver.Solvers.PTICM.PTICM_solver) implements the PTICM algorithm [[7](/resources/bibliography#zhuEfficientClusterAlgorithm2015)], which is a parallel tempering algorithm with isoenergetic cluster moves. It is an interface to the [TAMC](https://github.com/USCqserver/tamc) library in Rust [[3](/resources/bibliography#bauzaScalingAdvantageApproximate2024)] (which is assumed to be installed in the system).

The computational cost of each iteration of the PTICM solver is $O(N^2)$ for a general problem, and $O(N)$ for problems with a sparse structure. Here $N$ is the number of variables in the problem.
