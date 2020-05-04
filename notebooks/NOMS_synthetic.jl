import Pkg; Pkg.activate(@__DIR__)

using Distributions
using HMMBase
using JSON
using ParsimoniousMonitoring
using ProgressMeter

using POMDPs
using POMDPModelTools
using DiscreteValueIteration

function evaluate_policies(βs, τmax)
    results = Dict(β => Dict() for β in βs)
    solver = SparseValueIterationSolver(max_iterations=5000, belres=1e-6)
    
    @showprogress for β in βs
        @show β

        p1 = HMM([β 1-β; 1-β β], [Constant(0.01), Constant(100)])
        mdp = MonitoringMDP([p1, p1], [τmax, τmax], [25, 25], 0.99)
    
        smdp = SparseTabularMDP(mdp)
        optimal_policy = solve_sparse(solver, mdp, smdp)
    
        results[β]["value_iteration"] = evaluate(mdp, optimal_policy).(states(mdp))
        results[β]["greedy"] = evaluate(mdp, GreedyPolicy(mdp)).(states(mdp))
        results[β]["receding_h2"] = evaluate(mdp, RecedingHorizonPolicy(mdp, 2)).(states(mdp))
        results[β]["receding_h3"] = evaluate(mdp, RecedingHorizonPolicy(mdp, 3)).(states(mdp))
        results[β]["receding_h4"] = evaluate(mdp, RecedingHorizonPolicy(mdp, 4)).(states(mdp))
    end

    results
end

function main(βs, τmax)
    @show βs, τmax
    results = evaluate_policies(βs, τmax)
    filename = joinpath(@__DIR__, "NOMS_synthetic.json")
    write(filename, JSON.json(results))
end

main([0.1], 10)
