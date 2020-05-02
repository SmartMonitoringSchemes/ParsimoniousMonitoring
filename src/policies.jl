# Basic policies
# See receding_horizon.jl for the receding horizon policy.

## Constant Policy

struct ConstantPolicy{P} <: Policy
    action::NTuple{P,Bool}
end

action(policy::ConstantPolicy, _) = policy.action

always_measure_policy(P) = ConstantPolicy(Tuple(ones(Bool, P)))
never_measure_policy(P) = ConstantPolicy(Tuple(zeros(Bool, P)))

## Greedy Policy

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
