# Basic policies
# See receding_horizon.jl for the receding horizon policy.

## Constant Policy

struct ConstantPolicy{P} <: Policy
    action::NTuple{P,Bool}
end

action(policy::ConstantPolicy, _) = policy.action

always_measure_policy(P) = ConstantPolicy(Tuple(ones(Bool, P)))
never_measure_policy(P) = ConstantPolicy(Tuple(zeros(Bool, P)))

## Generic Greedy Policy

struct GreedyPolicy{S,A} <: Policy
    mdp::MDP{S,A}
end

function action(policy::GreedyPolicy, s)
    best = (-Inf, first(actions(policy.mdp)))
    for a in actions(policy.mdp)
        dist = transition(policy.mdp, s, a)
        util = 0.0
        for (sp, proba) in weighted_iterator(dist)
            util += proba * reward(policy.mdp, s, a, sp)
        end
        best = max(best, (util, a))
    end
    best[2]
end

## Analytical greedy policy for two Markovian paths with an arbitrary number of states.
# - Works with continuous beliefs
# - Useful for plotting the policy on a grid
# - Treat HMMs as MCs
# JONS paper -- Table 3

struct AnalyticalGreedyPolicy <: Policy
    mdp::MonitoringMDP{2}
end

function action(policy::AnalyticalGreedyPolicy, s::NTuple{2,DiscreteBelief})
    beliefs = ntuple(2) do i
        ContinuousBelief(s[i], policy.mdp.models[i])
    end
    action(policy, beliefs)
end

function action(policy::AnalyticalGreedyPolicy, s::NTuple{2,ContinuousBelief})
    predictor = ntuple(2) do i
        predict(s[i], policy.mdp.models[i])
    end
    action_predictor(policy, predictor)
end

function action_predictor(policy::AnalyticalGreedyPolicy, p::NTuple{2,ContinuousBelief})
    c1, c2 = policy.mdp.costs
    m1, m2 = policy.mdp.models
    l1, l2 = map(mean, m1.B), map(mean, m2.B)

    exp1 = expectation(p[1], m1)
    exp2 = expectation(p[2], m2)

    # (false, false)
    R00 = -(exp2 <= exp1) * exp2 - (exp2 > exp1) * exp1

    # (false, true)
    R01 = -sum(j -> (l2[j] <= exp1) * l2[j] * p[2][j], 1:length(l2))
    R01 -= exp1 * sum(j -> (l2[j] > exp1) * p[2][j], 1:length(l2))
    R01 -= c2

    # (true, false)
    R10 = -sum(i -> (l1[i] < exp2) * l1[i] * p[1][i], 1:length(l1))
    R10 -= exp2 * sum(i -> (l1[i] >= exp2) * p[1][i], 1:length(l1))
    R10 -= c1

    # (true, true)
    R11 = -sum(1:length(l1)) do i
        sum(1:length(l2)) do j
            p[1][i] * p[2][j] * min(l1[i], l2[j])
        end
    end
    R11 -= c1 + c2

    actions = [(false, false), (false, true), (true, false), (true, true)]
    rewards = [R00, R01, R10, R11]

    actions[argmax(rewards)]
end

# A policy wrapper that caches actions.
# Not thread-safe!
struct CachedPolicy{T,S,A} <: Policy
    policy::T
    cache::Dict{S,A}
end

CachedPolicy(mdp::MDP{S,A}, policy) where {S,A} = CachedPolicy(policy, Dict{S,A}())

function action(policy::CachedPolicy, x)
    get!(policy.cache, x) do
        action(policy.policy, x)
    end
end
