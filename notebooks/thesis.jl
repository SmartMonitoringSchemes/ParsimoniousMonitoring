using PyCall
using PyPlot

# Colorblind friendly colors
# https://medialab.github.io/iwanthue/
colorblindmap = ["#001975", "#86cc4d", "#ea3b73", "#9dab11", "#a86f00"]

rc("axes", prop_cycle = plt.cycler(color = colorblindmap))

function save_thesis(filename, figure = gcf(); clean = true, hwr = nothing)
    tikzplotlib = pyimport("tikzplotlib")
    clean && tikzplotlib.clean_figure(fig)
    path = joinpath(@__DIR__, "..", "plots", "$(filename).tikz")
    kwargs = Dict(:figure => figure, :textsize => 11)
    kwargs[:extra_axis_parameters] = [
        raw"legend style={nodes={scale=0.8}}",
        raw"label style={font=\small}",
        raw"tick label style={font=\small}",
    ]
    if !isnothing(hwr)
        # TODO: Use \axis_width instead?
        kwargs[:axis_height] = "$(hwr)\\linewidth"
        kwargs[:axis_width] = "\\linewidth"
    end
    tikzplotlib.save(path; kwargs...)
end
