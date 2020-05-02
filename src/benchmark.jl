LogEntry = NamedTuple{(:s, :a, :sp, :r, :t, :delay, :delay_opt, :path, :time)}

function benchmark(
    mdp::MonitoringMDP,
    policy::Policy,
    data::Matrix{Float64},
    state = first(states(mdp));
    show_progress = false
)
    @argcheck length(mdp.models) == size(data, 2)
    T, P = size(data)
    logbook = LogEntry[]

    mdp_state = collect(state)
    hmm_state = map(x -> ContinuousBelief(x...), zip(mdp_state, mdp.models))

    show_progress && (prog = Progress(T))

    for t = 1:T
        s = Tuple(mdp_state)
        policy_time = @elapsed a = action(policy, s)

        for i = 1:P
            if a[i] # Measure
                hmm_state[i] = update(hmm_state[i], mdp.models[i], data[t, i])
                mdp_state[i] = update(mdp_state[i], argmax(hmm_state[i].belief))
            else # Don't measure
                hmm_state[i] = predict(hmm_state[i], mdp.models[i])
                mdp_state[i] = predict(mdp_state[i], mdp.τmax[i])
            end
        end

        sp = Tuple(mdp_state)
        r = reward(mdp, s, a, sp)

        # NOTE: The decision is taken on the discrete belief, not on the continuous belief.
        delays = map(x -> expectation(x...), zip(mdp_state, mdp.models))
        path = argmin(delays)

        show_progress && next!(prog)
        push!(logbook, LogEntry(s, a, sp, r, t, data[t,path], minimum(data[t,:]), path, policy_time))
    end

    logbook
end
