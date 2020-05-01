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
