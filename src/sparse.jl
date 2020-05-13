# Faster implementation of SparseTabularMDP
# https://github.com/JuliaPOMDP/POMDPModelTools.jl/blob/master/src/sparse_tabular.jl
# NOTE: This does not handle terminal states, but there aren't any in `MonitoringMDP`.
function SparseTabularMDP(mdp::MonitoringMDP; show_progress = false)
    action_space = ordered_actions(mdp)
    state_space = ordered_states(mdp)

    na = length(action_space)
    ns = length(state_space)

    # - `T::Vector{SparseMatrixCSC{Float64, Int64}}`
    #    The transition model is represented as a vector of sparse matrices (one for each action).
    #   `T[a][s, sp]` the probability of transition from `s` to `sp` taking action `a`.
    T = Vector{SparseMatrixCSC{Float64, Int64}}(undef, na)

    # - `R::Array{Float64, 2}`
    #    The reward is represented as a matrix where the rows are states and the columns actions:
    #   `R[s, a]` is the reward of taking action `a` in sate `s`.
    R = fill(-Inf, (ns, na))

    for (aidx, a) in enumerate(action_space)
        I, J, V = Int[], Int[], Float64[]
        show_progress && (p = Progress(ns))
        for (sidx, s) in enumerate(state_space)
            dist = transition(mdp, s, a)
            r = 0.0
            for (sp, p) in weighted_iterator(dist)
                spidx = stateindex(mdp, sp)
                push!(I, sidx); push!(J, spidx); push!(V, p)
                r += p * reward(mdp, s, a, sp)
            end
            R[sidx, aidx] = r
            show_progress && next!(p)
        end
        T[aidx] = sparse(I, J, V, ns, ns)
    end

    SparseTabularMDP(T, R, Set(), mdp.discount)
end
