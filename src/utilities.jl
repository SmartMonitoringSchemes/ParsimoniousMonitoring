flatproduct(args...) = reshape(collect(Iterators.product(args...)), :)
splatmap(f, args...) = map(x -> f(x...), args...);

# https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/9
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...);

# A discrete probability distribution with a single value.
Constant(x) = DiscreteNonParametric([x], [1.0])
