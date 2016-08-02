using PyPlot
using Convex
using ProfileView
using Mapping
include("CenterOfMass.jl")

r_attach = 0.8

get_com() = r_attach * rand_in_circle()

circle_ps = map(theta->[cos(theta);sin(theta);0], thetas)
attachment_ps = map(theta->r_attach*[cos(theta);sin(theta);0], thetas)

# warm up
applied_f, boundary_fs = critical_force_from_points(circle_ps, get_com(), attachment_ps[1], 1)

times = Float64[]

Profile.clear()
@profile for ii = 1:1000
  time = @elapsed applied_f, boundary_fs = critical_force_from_points(circle_ps, get_com(), attachment_ps[1], 1)
  push!(times, time)
end

PyPlot.plt[:hist](times, 20)
ProfileView.view()


Void
