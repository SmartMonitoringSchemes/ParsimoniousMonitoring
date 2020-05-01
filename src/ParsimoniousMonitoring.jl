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
    actionindex, actions, dimensions, discount, reward, stateindex, states, transition

export MonitoringMDP

include("spaces.jl")
include("problem.jl")

end
