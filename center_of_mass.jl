using PyPlot
using Convex
using Mapping
#using CenterOfMass
include("CenterOfMass.jl")
plt[:close]("all")

m = 1
g = -9.8
z = [0;0;1]

np = 100
thetas = 2*pi * (1/np:1/np:1)
circle_ps = map(theta->[sin(theta);cos(theta);0], thetas)

# set up the occupancy prior
resolution = 0.02
prior = initialize_prior(circle_ps, resolution, pi, x->norm(x) <= 1)
fig, ax = plt[:subplots](1)
plot_grid(grid, :hot)
#ax[:set_ylim]([-1,1])
#ax[:set_xlim]([-1,1])

com_p = rand_in_circle()

applied_p = [1;0;0]

figure()
solve_and_plot(circle_ps, com_p, applied_p)
