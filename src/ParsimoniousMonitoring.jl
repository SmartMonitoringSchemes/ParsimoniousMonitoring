module ParsimoniousMonitoring

using ArgCheck
using Base: IdentityUnitRange
using DataFrames
using DiscreteValueIteration
using Distributions
using HMMBase
using IterTools: @ifsomething
using LinearAlgebra
using POMDPs
using POMDPModelTools
using ProgressMeter
using Random

# Extended functions
import Base: Tuple, eltype, iterate, length, rand
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

export MonitoringMDP,
    ContinuousBelief,
    DiscreteBelief,
    ConstantPolicy,
    GreedyPolicy,
    RecedingHorizonPolicy,
    always_measure_policy,
    never_measure_policy,
    solve_sparse,
    expectation,
    predict,
    update,
    benchmark,
    benchmark_mc

include("utilities.jl")
include("belief.jl")
include("spaces.jl")
include("problem.jl")
include("policies.jl")
include("receding_horizon.jl")
include("benchmark.jl")

end
