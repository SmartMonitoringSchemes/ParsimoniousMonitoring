Action{P} = NTuple{P,Bool}
State{P} = NTuple{P,DiscreteBelief}

struct MonitoringMDP{P} <: MDP{State{P},Action{P}}
    models::Vector{HMM}
    τmax::Vector{Int}
    costs::Vector{Float64}
    discount::Float64
    # Internal fields
    # We cache the action space and the state space
    # to improve the performance of actionindex/stateindex.
    # Note that these don't use much memory since the elements
    # are produced lazily by `iterate`.
    actions::BooleanActionSpace{P}
    states::DiscreteBeliefSpace{P}
end

function MonitoringMDP(models, τmax, costs, discount = 0.99)
    @argcheck length(models) == length(τmax) == length(costs)
    @argcheck 0 <= discount < 1
    actions = BooleanActionSpace(length(models))
    nstates = map(m -> size(m, 1), models)
    states = DiscreteBeliefSpace(τmax, nstates)
    MonitoringMDP{length(models)}(models, τmax, costs, discount, actions, states)
end

actions(mdp::MonitoringMDP) = mdp.actions
states(mdp::MonitoringMDP) = mdp.states
discount(mdp::MonitoringMDP) = mdp.discount

# TODO: Type inference fails since index type not in MDP struct
actionindex(mdp::MonitoringMDP, a) = index(mdp.actions, a)
stateindex(mdp::MonitoringMDP, s) = index(mdp.states, s)

## Transition Model

function transition(τmax::Int, model::HMM, b::DiscreteBelief, a::Bool)
    @argcheck b.timesteps <= τmax
    if a # Measure
        probas = (model.A^(b.timesteps+1))[b.laststate, :]
        states = map(i -> DiscreteBelief(0, i), 1:length(probas))
        return probas, states
    else # Don't measure
        timesteps = min(b.timesteps + 1, τmax)
        return [1.0], [DiscreteBelief(timesteps, b.laststate)]
    end
end

# Possible transitions from state s and action a
# TODO: Optimize
function transition(mdp::MonitoringMDP{P}, s::State{P}, a::Action{P}) where {P}
    probas = Vector{Float64}[]
    states = Vector{DiscreteBelief}[]

    for (τmax, model, belief, action) in zip(mdp.τmax, mdp.models, s, a)
        probas_, states_ = transition(τmax, model, belief, action)
        push!(probas, probas_)
        push!(states, states_)
    end

    probas = splatmap(*, flatproduct(probas...))
    states = flatproduct(states...)

    SparseCat(states, probas)
end

## Reward Model
# TODO: Alternative reward for mdp with two paths (L - L(t))

# TODO: Optimize
function reward(mdp::MonitoringMDP, _, a::Action{P}, sp::State{P}) where {P}
    cost = dot(mdp.costs, a)

    delay = minimum(zip(mdp.models, sp)) do (model, belief)
        probas::Vector{Float64} = (model.A^belief.timesteps)[belief.laststate, :]
        sum(i -> mean(model.B[i])::Float64 * probas[i], 1:length(probas))
    end

    return -cost - delay
end
