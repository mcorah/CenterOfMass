using PyPlot
using Convex
using PyCall
using HDF5, JLD
@pyimport matplotlib2tikz

#using CenterOfMass
include("CenterOfMass.jl")
include("setup_parameters.jl")

interior_q(x) = norm(x) <= 1

masses = [1.0]
resolution = 0.02
prior = initialize_prior(circle_ps, resolution, interior_q, masses)

critical_forces = get_critical_values(circle_ws, attachment_ws[5], prior, interior_q)

circle_polygon = hcat(circle_ps..., circle_ps[1])
#plot(circle_polygon[1,:][:], circle_polygon[2,:][:], color = "k", alpha = 1, linestyle = "solid")

xs = get_range(prior, 1)
ys = get_range(prior, 2)

xg = repmat(xs', length(ys), 1)
yg = repmat(ys, 1, length(xs))

critical_forces = max(critical_forces, zeros(size(critical_forces)))
plot_surface(xg, yg, critical_forces[:,:], cmap="BuPu", rstride=2, cstride=2, alpha=0.8, linewidth=0.25)

#extent= [minimum(xs), maximum(xs), minimum(ys), maximum(ys)]

#imshow(critical_forces[:,:]', cmap= "BuPu", vmin=minimum(critical_forces), vmax=maximum(critical_forces),
  #extent = extent, interpolation="nearest", origin="lower")
