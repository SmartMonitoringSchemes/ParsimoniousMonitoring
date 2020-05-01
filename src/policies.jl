struct ConstantPolicy{P} <: Policy
    action::NTuple{P,Bool}
end

action(policy::ConstantPolicy, _) = policy.action

always_measure_policy(P) = ConstantPolicy(Tuple(ones(Bool, P)))
never_measure_policy(P) = ConstantPolicy(Tuple(zeros(Bool, P)))
