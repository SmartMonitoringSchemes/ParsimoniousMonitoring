# Parsimonious Monitoring in Routing Overlays

[![CI](https://github.com/SmartMonitoringSchemes/ParsimoniousMonitoring/workflows/CI/badge.svg)](https://github.com/SmartMonitoringSchemes/ParsimoniousMonitoring/actions?query=workflow%3ACI)
[![CompatHelper](https://github.com/SmartMonitoringSchemes/ParsimoniousMonitoring/workflows/CompatHelper/badge.svg)](https://github.com/SmartMonitoringSchemes/ParsimoniousMonitoring/actions?query=workflow%3ACompatHelper)
[![codecov](https://codecov.io/gh/SmartMonitoringSchemes/ParsimoniousMonitoring/branch/master/graph/badge.svg?token=A383yme5Ey)](https://codecov.io/gh/SmartMonitoringSchemes/ParsimoniousMonitoring)

This is an implementation of the following papers:

- Sandrine Vaton, Olivier Brun, Maxime Mouchet, Pablo Belzarena, Isabel Amigo, et al.. Joint Minimization of Monitoring Cost and Delay in Overlay Networks: Optimal Policies with a Markovian Approach. _Journal of Network and Systems Management_, Springer Verlag, 2019, 27 (1), pp.188-232. [⟨hal-01857738⟩](https://hal.archives-ouvertes.fr/hal-01857738)
- Maxime Mouchet, Martin Randall, Marine Ségneré, Isabel Amigo, Pablo Belzarena, et al.. Scalable Monitoring Heuristics for Improving Network Latency. _IEEE/IFIP Network Operations and Management Symposium (IEEE/IFIP NOMS 2020)_, Apr 2020, Budapest, Hungary. [⟨hal-02413636⟩](https://hal.archives-ouvertes.fr/hal-02413636)

## TODO

- Document caches / caching levels
- Document scripts

## Notebooks

Name | Description
:----|:-----------
[JONS Basic](/notebooks/JONS_Basic.ipynb) | Reproduction of sections 8.1 and 8.2 of JONS paper.
[JONS Pair](/notebooks/JONS_Pair.ipynb)   | Reproduction of section 8.3 of JONS paper.
[NOMS Paper Synthetic](/notebooks/NOMS_Paper_Synthetic.ipynb) | Reproduction of section 5.1 of NOMS paper.
[NOMS Paper NANOG](/notebooks/NOMS_Paper_NANOG.ipynb) | Reproduction of section 5.2 of NOMS paper.

## Policies

Name | Implementation | Description
:----|:---------------|:-----------
ConstantPolicy         | [policies.jl](/src/policies.jl) | `always_measure_policy(P)`, `never_measure_policy(P)`.
GreedyPolicy           | [policies.jl](/src/policies.jl) | Generic greedy policy.
AnalyticalGreedyPolicy | [policies.jl](/src/policies.jl) | Analytical greedy policy for 2 stochastic paths (JONS paper).
ValueIterationPolicy   | [DiscreteValueIteration.jl](https://github.com/JuliaPOMDP/DiscreteValueIteration.jl) | Standard VI. Implementation from POMDPs.jl.
RecedingHorizonPolicy  | [receding_horizon.jl](/src/receding_horizon.jl) | Heuristic 1 from NOMS paper.
HeuristicPolicy        | [heuristic.jl](/src/heuristic.jl) | Heuristic 2 from NOMS paper.

## Routing

Name | Implementation | Description
:----|:---------------|:-----------
ShortestPathPolicy | [routing.jl](/src/routing.jl) | Choose the path of minimal delay.
ConstantPathPolicy | [routing.jl](/src/routing.jl) | Always choose the same path.
