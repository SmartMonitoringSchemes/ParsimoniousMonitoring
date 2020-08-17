# https://hal.laas.fr/hal-02413636/document -- Heuristic 2

const HeuristicCache = Dict{Tuple{Int,Float64}, ValueIterationPolicy}

struct HeuristicPolicy{P} <: Policy
    mdp::MonitoringMDP{P}
    solver::Solver
    cache::HeuristicCache
end

function HeuristicPolicy(mdp, solver)
    HeuristicPolicy(mdp, solver, HeuristicCache())
end

function expected_delay(mdp::MonitoringMDP, s::State{P}, a::Action{P}, i::Int) where {P}
    indices = vcat(1:i-1, i+1:P)

    # > First, path i is excluded in the computation of the mean delay which depends upon
    # > the current state and the current actions of the other paths only.
    # > Second, the cost of monitoring of the other paths is not taken into account.
    # HACK: build an MDP excluding path `i` and with no monitoring costs
    others::MonitoringMDP{P-1} = MonitoringMDP(mdp.models[indices], mdp.τmax[indices], zeros(length(indices)), mdp.discount)
    dist = transition(others, s[indices], a[indices])

    sum(weighted_iterator(dist)) do (sp, p)
        p * delay(others, sp)
    end
end

function heuristic_step(policy::HeuristicPolicy, s::State{P}, a::Action{P}, i::Int) where {P}
    delay = expected_delay(policy.mdp, s, a, i)
    key = (i, delay)

    policy = get!(policy.cache, key) do
        # Surrogate MDP
        # p1: stochastic path
        # p2: virtual path
        p1 = policy.mdp.models[i]
        p2 = HMM(ones(1,1), [Constant(delay)])
        surrogate = MonitoringMDP(
            [p1, p2],
            [policy.mdp.τmax[i], 0],
            [policy.mdp.costs[i], 0],
            policy.mdp.discount
        )
        solve(policy.solver, surrogate)
    end

    action(policy, (s[i], DiscreteBelief(0,1)))[1]
end


function action(policy::HeuristicPolicy{P}, s::State{P}) where{P}
    a = zeros(Bool, P)
    for i = 1:P
        a[i] = heuristic_step(policy, s, Tuple(a), i)
    end
    Tuple(a)
end
