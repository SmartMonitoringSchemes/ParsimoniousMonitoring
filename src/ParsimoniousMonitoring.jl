module ParsimoniousMonitoring

using ArgCheck
using Base: IdentityUnitRange
using DataFrames
using DiscreteValueIteration
using Distributions
using HMMBase
using InteractiveUtils: @which
using IterTools: @ifsomething
using LinearAlgebra
using POMDPs
using POMDPModelTools
using ProgressMeter
using Random
using SparseArrays

# Extended functions
import Base: Tuple, eltype, getindex, iterate, length, rand, show
import POMDPs:
    action,
    actionindex,
    actions,
    dimensions,
    discount,
    reward,
    stateindex,
    states,
    transition,
    update

import POMDPModelTools: SparseTabularMDP

export MonitoringMDP,
    ContinuousBelief,
    DiscreteBelief,
    ConstantPolicy,
    GreedyPolicy,
    AnalyticalGreedyPolicy,
    RecedingHorizonPolicy,
    HeuristicPolicy,
    always_measure_policy,
    never_measure_policy,
    solve_sparse,
    expectation,
    predict,
    update,
    action,
    action_predictor,
    benchmark,
    benchmark_mc,
    Constant,
    SparseTabularMDP

include("utilities.jl")
include("belief.jl")
include("spaces.jl")
include("problem.jl")
include("policies.jl")
include("heuristic.jl")
include("receding_horizon.jl")
include("benchmark.jl")
include("missings.jl")
include("sparse.jl")

end
