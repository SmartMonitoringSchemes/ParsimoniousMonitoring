using PyCall
using PyPlot

function save_thesis(filename, figure = gcf(); clean = true, hwr = nothing)
    tikzplotlib = pyimport("tikzplotlib")
    clean && tikzplotlib.clean_figure(fig)
    path = joinpath(@__DIR__, "..", "plots", "$(filename).tex")
    kwargs = Dict(
        :figure => figure,
        :textsize => 11,
        :show_info => true
    )
    if !isnothing(hwr)
        # TODO: Use \axis_width instead?
        kwargs[:axis_height] = "$(hwr)\\linewidth"
        kwargs[:axis_width] = "\\linewidth"
    end
    tikzplotlib.save(path; kwargs...)
end