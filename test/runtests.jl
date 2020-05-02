using Test
using ParsimoniousMonitoring

import ParsimoniousMonitoring: BooleanActionSpace, DiscreteBeliefSpace

function allocs(f)
    f(nothing)
    _, _, _, _, memallocs = @timed f(nothing)
    memallocs.malloc + memallocs.realloc + memallocs.poolalloc + memallocs.bigalloc
end

@testset "Allocations" for s in [BooleanActionSpace(4)]
    allocs1 = allocs(_ -> collect(s))
    allocs2 = allocs(_ -> collect(s.indices))
    @show allocs1, allocs2
    @test allocs1 <= allocs2 <= 10
end

@testset "DiscreteBelief" begin
    @test_throws ArgumentError DiscreteBelief(-1, 0)
end
