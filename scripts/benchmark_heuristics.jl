import Pkg
Pkg.activate(@__DIR__)

using CSV
using Distributions
using Glob
using HMMBase
using JSON
using DiscreteValueIteration
using ParsimoniousMonitoring

read_ts(filename) = CSV.read(filename, header = ["timestep", "rtt"])

function read_model(filename)
    obj = JSON.parsefile(filename)
    A = permutedims(hcat(obj["transmat"]...))
    B = map(obj["states"]) do (_, d)
        components = map(zip(d["means"], d["variances"])) do (μ, σ2)
            Normal(μ, sqrt(σ2))
        end
        MixtureModel(components, [d["weights"]...])
    end
    HMM(A,B)
end

function load_scenario(path)
    files = map(x -> splitext(x)[1], glob("*.csv", path))
    models = []
    series = []
    for file in files
        push!(models, read_model("$file.json"))
        push!(series, read_ts("$file.csv"))
    end
    data = hcat(map(x -> x.rtt, series)...)
    name = splitpath(path)[end]
    name, models, data
end


function main(args)
    scenario, models, data = load_scenario(args[1])
    @show scenario

    τmaxs = fill(100, length(models))
    costs = fill(0.5, length(models))
    discount = 0.99

    mdp = MonitoringMDP(models, τmaxs, costs, discount)
    @show mdp

    policies = Dict(
        "Greedy" => GreedyPolicy(mdp),
        "Heuristic" => HeuristicPolicy(mdp, SparseValueIterationSolver()),
        "RH-3" => RecedingHorizonPolicy(mdp, 3, shared_cache = true)
    )

    logbooks = Dict()
    for (name, policy) in policies
        @show name
        logbooks[name] = benchmark(mdp, policy, data, show_progress = true)
    end

    output_file = "$scenario.json"
    println("Writing results to $(output_file)")
    write(output_file, json(logbooks))
end

main(ARGS)