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

function show(io::IO, mdp::MonitoringMDP{P}) where P
    print(io, "MonitoringMDP with $(P) paths")
    print(io, ", $(length(actions(mdp))) actions and $(length(states(mdp))) states (ρ = $(mdp.discount))\n")
    for (i, (model, τmax, cost)) in enumerate(zip(mdp.models, mdp.τmax, mdp.costs))
        println(io, "Path $(i): $(typeof(model)) with $(size(model, 1)) states (c = $(cost), τmax = $(τmax))")
    end
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
function transition(mdp::MonitoringMDP{P}, s::State{P}, a::Action{P}) where {P}
    all_probas = Vector{Float64}[]
    all_states = Vector{DiscreteBelief}[]

    for (τmax, model, belief, action) in zip(mdp.τmax, mdp.models, s, a)
        probas_::Vector{Float64}, states_::Vector{DiscreteBelief} = transition(τmax, model, belief, action)
        push!(all_probas, probas_)
        push!(all_states, states_)
    end

    probas::Vector{Float64} = splatmap(*, flatproduct(all_probas...))
    states::Vector{State{P}} = flatproduct(all_states...)

    SparseCat(states, probas)
end

## Reward Model
# TODO: Alternative reward for mdp with two paths (L - L(t))

# Delay in state s' after acting and applying the  "minimum expected delay" routing decision.
function delay(mdp::MonitoringMDP{P}, sp::State{P}) where {P}
    minimum(zip(mdp.models, sp)) do (model, belief)
        probas::Vector{Float64} = (model.A^belief.timesteps)[belief.laststate, :]
        sum(i -> mean(model.B[i])::Float64 * probas[i], 1:length(probas))
    end
end

function reward(mdp::MonitoringMDP{P}, _, a::Action{P}, sp::State{P}) where {P}
    - dot(mdp.costs, a) - delay(mdp, sp)
end
