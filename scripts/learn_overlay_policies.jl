import Pkg
Pkg.activate(@__DIR__)

using Distributions
using Glob
using JSON
using ParsimoniousMonitoring
using ProgressMeter

using POMDPs
using POMDPModelTools
using DiscreteValueIteration

### Move this in a dedicated module?

import Distributions: Normal, MixtureModel
import HMMBase: HMM

function HMM{F,T}(::Type{MixtureModel}, D::Type{<:Distribution}, d::Dict) where {F,T}
    a = Vector{T}(d["a"])
    A = Matrix{T}(hcat(d["A"]...))
    B = map(x -> MixtureModel(D, x), d["B"])
    HMM(a, A, B)
end

function MixtureModel(T::Type{<:Distribution}, d::Dict)
    components = map(T, d["components"])
    prior = Vector{Float64}(d["prior"]["p"])
    MixtureModel(components, prior)
end

Normal(d::Dict) = Normal(d["μ"], d["σ"])

###

function process(file, τmax, c, ρ)
    obj = JSON.parsefile(file)

    timeseries = map(x -> x["raw_rtt"], obj["series"])
    models = map(obj["series"]) do x
        HMM{Univariate,Float64}(MixtureModel, Normal, x["model"])
    end

    mdp = MonitoringMDP(models, [τmax, τmax], [c, c], ρ)
    smdp = SparseTabularMDP(mdp, show_progress = false)

    solver =
        SparseValueIterationSolver(max_iterations = 5000, belres = 1e-6, verbose = false)
    policy = solve_sparse(solver, mdp, smdp)

    res = Dict("action_map" => policy.action_map, "policy" => policy.policy)
    output_file = "$(file).policy"
    write(output_file, json(res))
end

function main(args)
    @show dataset = args[1]
    @show τmax = parse(Int, args[2])
    @show c = parse(Float64, args[3])
    @show ρ = parse(Float64, args[4])

    files = glob("*.model", dataset)
    @show length(files)

    p = Progress(length(files))

    Threads.@threads for file in files
        process(file, τmax, c, ρ)
        next!(p)
    end
end

main(ARGS)
