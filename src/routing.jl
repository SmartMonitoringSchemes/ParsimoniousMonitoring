abstract type RoutingPolicy end

struct ShortestPathPolicy <: RoutingPolicy end

action(policy::ShortestPathPolicy, delays) = argmin(delays)

struct ConstantPathPolicy <: RoutingPolicy
    path::Int
end

action(policy::ConstantPathPolicy, _) = policy.path