# https://hal.laas.fr/hal-02413636/document -- Heuristic 2

function surrogate_mdp(mdp::MonitoringMDP, s::State{P}, a::Action{P}, i::Int)::MonitoringMDP{2} where {P}
    indices = vcat(1:i-1, i+1:P)

    # > First, path i is excluded in the computation of the mean delay which depends upon
    # > the current state and the current actions of the other paths only.
    # > Second, the cost of monitoring of the other paths is not taken into account.
    # HACK: build an MDP excluding path `i` and with no monitoring costs
    others::MonitoringMDP{P-1} = MonitoringMDP(mdp.models[indices], mdp.τmax[indices], zeros(length(indices)), mdp.discount)
    dist = transition(others, s[indices], a[indices])
    expected_delay = sum(weighted_iterator(dist)) do (sp, p)
        p * delay(others, sp)
    end

    p1 = mdp.models[i] # Stochastic path
    p2 = HMM(ones(1,1), [Constant(expected_delay)]) # Deterministic path

    MonitoringMDP([p1, p2], [mdp.τmax[i], 0], [mdp.costs[i], 0], mdp.discount)
end

function heuristic_step(solver::Solver, mdp::MonitoringMDP, s::State{P}, a::Action{P}, i::Int) where {P}
    surrogate = surrogate_mdp(mdp, s, a, i)
    policy = solve(solver, surrogate)
    action(policy, (s[i], DiscreteBelief(0,1)))[1]
end

struct HeuristicPolicy{P} <: Policy
    mdp::MonitoringMDP{P}
    solver::Solver
end

# Algorithm 2
function action(policy::HeuristicPolicy{P}, s::State{P}) where{P}
    a = zeros(Bool, P)
    for i = 1:P
        a[i] = heuristic_step(policy.solver, policy.mdp, s, Tuple(a), i)
    end
    Tuple(a)
end
