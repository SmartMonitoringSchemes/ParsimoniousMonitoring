# https://juliapomdp.github.io/POMDPs.jl/stable/interfaces/#space-interface-1

## Discrete Belief Space

struct DiscreteBeliefSpace{P,I}
    τmax::Vector{Int}
    nstates::Vector{Int}
    indices::I
    function DiscreteBeliefSpace(τmax, nstates)
        @argcheck length(τmax) == length(nstates)
        @argcheck all(τmax .>= 0) && all(nstates .>= 1)
        # (a, b, c, d) => DiscreteBelief(a, b), DiscreteBelief(c, d)
        range = tuplejoin([(0:τ, 1:n) for (τ, n) in zip(τmax, nstates)]...)
        # https://discourse.julialang.org/t/linearindices-for-non-1-based-indices/26906/2
        indices = CartesianIndices(IdentityUnitRange.(range))
        new{length(τmax),typeof(indices)}(τmax, nstates, indices)
    end
end

function beliefs(I::CartesianIndex, P)
    ntuple(P) do i
        DiscreteBelief(I[(i-1)*2+1], I[(i-1)*2+2])
    end
end

# TODO: Check type inference for index functions
function index(s::DiscreteBeliefSpace{P}, beliefs::NTuple{P,DiscreteBelief}) where {P}
    tpl = ntuple(2P) do i
        a, b = divrem(i + 1, 2)
        Tuple(beliefs[a])[b+1]
    end
    LinearIndices(s.indices)[CartesianIndex(tpl)]
end

function iterate(s::DiscreteBeliefSpace{P}, args...) where {P}
    element, state = @ifsomething iterate(s.indices, args...)
    beliefs(element, P), state
end

eltype(s::DiscreteBeliefSpace{P}) where {P} = NTuple{P,DiscreteBelief}
length(s::DiscreteBeliefSpace) = length(s.indices)
rand(rng::AbstractRNG, s::DiscreteBeliefSpace{P}) where {P} =
    beliefs(rand(rng, s.indices), P)

## Action Space

struct BooleanActionSpace{P,I}
    indices::I
    function BooleanActionSpace(P)
        @argcheck P >= 1
        # {0,1}^P => 1 if we measure, 0 else
        range = Tuple(0:1 for _ = 1:P)
        # https://discourse.julialang.org/t/linearindices-for-non-1-based-indices/26906/2
        indices = CartesianIndices(IdentityUnitRange.(range))
        new{P,typeof(indices)}(indices)
    end
end

# TODO: Check type inference for index functions
function index(s::BooleanActionSpace{P}, action::NTuple{P,Bool}) where {P}
    LinearIndices(s.indices)[CartesianIndex(action)]
end

function iterate(s::BooleanActionSpace{P}, args...) where {P}
    element, state = @ifsomething iterate(s.indices, args...)
    # NOTE: @code_warntype iterate(s)
    # => Type inference fails for large tuples
    Bool.(Tuple(element)), state
end

eltype(s::BooleanActionSpace{P}) where {P} = NTuple{P,Bool}
length(s::BooleanActionSpace) = length(s.indices)
rand(rng::AbstractRNG, s::BooleanActionSpace{P}) where {P} =
    Bool.(Tuple(rand(rng, s.indices)))

# TODO: Test for memory allocations
# @time s = BooleanActionSpace(4);
# @time collect(s);
# @time collect(s.indices);
# s = DiscreteBeliefSpace([300,300],[10,10]);
# @time collect(s);
# @time collect(s.indices);
