struct RecedingHorizonPolicy{S,A} <: Policy
    mdp::MDP{S,A}
    horizon::Int
end

function action(policy::RecedingHorizonPolicy, state)
    value, action = receding_horizon(policy.mdp, state, policy.horizon)
    action
end

Cache{S,A} = Dict{Tuple{S, Int}, Tuple{Float64, A}}

# From https://hal.laas.fr/hal-02413636/document -- Algorithm 1
# TODO: Cleanup type annotations, find why type inference fails
# TODO: Non-recursive version (DFS)
function receding_horizon(mdp::MDP{S,A}, state::S, horizon::Int, cache = Cache{S,A}()) where {S,A}
    @argcheck horizon >= 0

    if horizon == 0
        return 0.0, first(actions(mdp))::A
    end

    key = (state, horizon)
    if haskey(cache, key)
        return cache[key]
    end

    # (max, argmax)
    best::Tuple{Float64,A} = (-Inf, first(actions(mdp)))
    discount_factor = discount(mdp)

    for action::A in actions(mdp)
        dist = transition(mdp, state, action)
        util::Float64 = 0.0

        for (statep::S, proba::Float64) in weighted_iterator(dist)
            r = reward(mdp, state, action, statep)
            v, _ = receding_horizon(mdp, statep, horizon - 1, cache)
            util += proba * (r + discount_factor * v)
        end

        best = max(best, (util, action))
    end

    cache[key] = best

    best
end
