module ParsimoniousMonitoring

using ArgCheck
using Base: IdentityUnitRange
# using DiscreteValueIteration
using HMMBase
using IterTools: @ifsomething
using LinearAlgebra
using POMDPs
using POMDPModelTools
using Random

# Extended functions
import Base: Tuple, eltype, iterate, length, rand
import POMDPs:
    action, actionindex, actions, dimensions, discount, reward, stateindex, states, transition

export MonitoringMDP, ConstantPolicy, always_measure_policy, never_measure_policy

include("spaces.jl")
include("problem.jl")
include("policies.jl")
include("receding_horizon.jl")

end
