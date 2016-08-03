using PyPlot
using Convex
using ProfileView
using Mapping
include("CenterOfMass.jl")

np = 5
thetas = 2*pi * (1/np:1/np:1)
circle_ps = map(theta->[cos(theta);sin(theta);0], thetas)

na = 5
thetas = 2*pi * (1/na:1/na:1)
r_attach = 0.8
attachment_ps = map(theta->r_attach*[cos(theta);sin(theta);0], thetas)

get_com() = r_attach * rand_in_circle()

# warm up
applied_f, boundary_fs = critical_force_from_points(circle_ps, get_com(), attachment_ps[1], 1)

times = Float64[]

Profile.clear()
@profile for ii = 1:10000
  time = @elapsed applied_f, boundary_fs = critical_force_from_points(circle_ps,
  get_com(), attachment_ps[1], 1, critical_force_iterative)
  push!(times, time)
end

println("Mean time: $(mean(times))")

PyPlot.plt[:hist](times, 20)
ProfileView.view()


Void
