using Test
using Distributions
using HMMBase
using ParsimoniousMonitoring
using POMDPModelTools

import ParsimoniousMonitoring: BooleanActionSpace, DiscreteBeliefSpace, index

function allocs(f)
    f(nothing)
    _, _, _, _, memallocs = @timed f(nothing)
    memallocs.malloc + memallocs.realloc + memallocs.poolalloc + memallocs.bigalloc
end

@testset "Allocations" for s in
                           [BooleanActionSpace(4), DiscreteBeliefSpace([10, 10], [5, 5])]
    allocs1 = allocs(_ -> collect(s))
    allocs2 = allocs(_ -> collect(s.indices))
    @show allocs1, allocs2
    @test allocs1 <= allocs2 <= 10
end

@testset "Spaces" for s in [BooleanActionSpace(4), DiscreteBeliefSpace([10, 10], [5, 5])]
    @test typeof(rand(s)) == eltype(s)
    @test length(collect(s)) == length(s)
    @test index(s, collect(s)[10]) == 10
end

hmm = HMM(
    [0.9 0.1; 0.1 0.9],
    [DiscreteNonParametric([0.0], [1.0]), DiscreteNonParametric([10.0], [1.0])],
)

@testset "DiscreteBelief" begin
    @test_throws ArgumentError DiscreteBelief(-1, 0)
    @test Tuple(DiscreteBelief(1, 1)) == (1, 1)

    @test expectation(DiscreteBelief(0, 1), hmm) == 0.0
    @test expectation(DiscreteBelief(0, 2), hmm) == 10.0
    @test expectation(DiscreteBelief(1000, 2), hmm) ≈ 5.0

    @test predict(DiscreteBelief(100, 1)) == DiscreteBelief(101, 1)
    @test predict(DiscreteBelief(100, 1), 100) == DiscreteBelief(100, 1)

    @test update(DiscreteBelief(100, 1), 2) == DiscreteBelief(0, 2)
end

@testset "ContinuousBelief" begin
    @test_throws ArgumentError ContinuousBelief([1.0, 1.0])

    @test expectation(ContinuousBelief([1.0, 0.0]), hmm) == 0.0
    @test expectation(ContinuousBelief([0.0, 1.0]), hmm) == 10.0
    @test expectation(ContinuousBelief([0.5, 0.5]), hmm) ≈ 5.0

    @test predict(ContinuousBelief([1.0, 0.0]), hmm).belief == [0.9, 0.1]
    @test update(ContinuousBelief([0.5, 0.5]), hmm, 10.0).belief == [0.0, 1.0]
end

@testset "SparseTabularMDP" begin
    mdp = MonitoringMDP([hmm, hmm], [20, 20], [4, 4], 0.5)
    # Original implementation (from POMDPModelTools.jl)
    R = POMDPModelTools.reward_s_a(mdp)
    T = POMDPModelTools.transition_matrix_a_s_sp(mdp)
    terminal_states = POMDPModelTools.terminal_states_set(mdp)
    # Custom implementation
    smdp = SparseTabularMDP(mdp, show_progress = true)
    @test smdp.R == R
    @test smdp.T == T
    @test smdp.discount == mdp.discount
    @test smdp.terminal_states == terminal_states
end
