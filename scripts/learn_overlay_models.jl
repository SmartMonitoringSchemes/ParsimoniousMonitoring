import Pkg
Pkg.activate(@__DIR__)

using ConjugatePriors: NormalInverseChisq
using Distributions
using HMMBase
using HDPHMM
using Glob
using JSON
using ProgressMeter

@show Threads.nthreads()

function prior(data)
    obs_med, obs_var = robuststats(Normal, data)
    tp = TransitionDistributionPrior(
        Gamma(1, 1/0.01),
        Gamma(1, 1/0.01),
        Beta(500, 1)
    )
    op = DPMMObservationModelPrior{Normal}(
        NormalInverseChisq(obs_med, obs_var, 1, 10),
        Gamma(1, 0.5)
    )
    BlockedSamplerPrior(1.0, tp, op)
end

function infer(data; L = 10, LP = 5)
    config = MCConfig(
        init = KMeansInit(L),
        iter = 200,
        verb = false
    )
    chains = HDPHMM.sample(BlockedSampler(L, LP), prior(data), data, config = config)
    result = select_hamming(chains[1])
    HMM(result[4], result[2])
end

function process(file)
    obj = JSON.parsefile(file)
    res = Dict(
        "pair" => obj["pair"],
        "series" => map(s -> Dict(s..., "model" => infer([s["raw_rtt"]...])), obj["series"])
    )
    output_file = "$(splitext(file)[1]).model"
    write(output_file, json(res))
end

function retry(f; retries=1)
    n = 0
    while n < retries
        try
            return f(nothing)  
        catch
            n += 1
            stacktrace(catch_backtrace())
        end
    end
    nothing
end

function main(args)
    files = glob("*.series", args[1])
    @show length(files)

    p = Progress(length(files))

    Threads.@threads for file in files
        retry(_ -> process(file))
        next!(p)
    end
end

main(ARGS)
