module ParsimoniousMonitoring

using ArgCheck
using Base: IdentityUnitRange
using IterTools: @ifsomething
using HMMBase
using POMDPs
using POMDPModelTools

# Extended functions
import Base: Tuple, eltype, iterate, length, rand
import POMDPs: actionindex, actions, dimensions, discount, reward, stateindex, states, transition

export MonitoringMDP

include("spaces.jl")
include("problem.jl")

end
