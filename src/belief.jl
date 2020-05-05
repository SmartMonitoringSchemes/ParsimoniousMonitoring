## Discrete Belief

struct DiscreteBelief
    timesteps::Int # >= 0
    laststate::Int # >= 1
    function DiscreteBelief(timesteps, laststate)
        @argcheck timesteps >= 0 && laststate >= 1
        new(timesteps, laststate)
    end
end

Tuple(b::DiscreteBelief) = (b.timesteps, b.laststate)

function expectation(b::DiscreteBelief, m::HMM)
    expectation(ContinuousBelief(b, m), m)
end

function predict(b::DiscreteBelief, τmax = Inf)
    timesteps = min(b.timesteps + 1, τmax)
    DiscreteBelief(timesteps, b.laststate)
end

function update(b::DiscreteBelief, x)
    DiscreteBelief(0, x)
end

## Continuous Belief
## (used for simulation)

struct ContinuousBelief
    belief::Vector{Float64}
    function ContinuousBelief(belief)
        @argcheck isprobvec(belief)
        new(belief)
    end
end

function ContinuousBelief(b::DiscreteBelief, m::HMM)
    belief = (m.A^b.timesteps)[b.laststate, :]
    ContinuousBelief(belief)
end

function expectation(b::ContinuousBelief, m::HMM)
    sum(i -> b.belief[i] * mean(m.B[i]), 1:length(b.belief))
end

# One-step ahead prediction
function predict(b::ContinuousBelief, m::HMM)
    ContinuousBelief(transpose(m.A) * b.belief)
end

function update(b::ContinuousBelief, m::HMM, x)
    # TODO: Verify numerical stability
    belief = pdf.(m.B, x) .* (transpose(m.A) * b.belief)
    ContinuousBelief(belief / sum(belief))
end

getindex(b::ContinuousBelief, args...) = getindex(b.belief, args...)
