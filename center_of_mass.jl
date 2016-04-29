using PyPlot
using Convex
using Mapping
#using CenterOfMass
include("CenterOfMass.jl")
plt[:close]("all")

m = 1
g = -9.8
z = [0;0;1]
sigma = 1

#np = 20
np = 10
thetas = 2*pi * (1/np:1/np:1)
circle_ps = map(theta->[cos(theta);sin(theta);0], thetas)

# set up the occupancy prior
resolution = 0.1

interior_q(x) = norm(x) <= 1

prior = initialize_prior(circle_ps, resolution, interior_q)

#fig, ax = plt[:subplots](1)
#plot_grid(prior, :hot)
#ax[:set_ylim]([-1,1])
#ax[:set_xlim]([-1,1])

com_p = rand_in_circle()
#com_p = [-0.6;-0.13;0.0]


# first point
#applied_p = [1;0;0]
applied_p = circle_ps[1]

figure()
applied_f, boundary_fs = solve_and_plot(circle_ps, com_p, applied_p)
f_hat = applied_f + sigma * randn()
update_prior!(prior, interior_q, circle_ps, applied_p, f_hat, sigma)

#fig, ax = plt[:subplots](1)
plot_grid(prior, :hot)
title("probability distribution of center-of-mass")

# selection of second point
csqmis = [(println("computed csqmi");compute_mutual_information(circle_ps, x, prior, sigma, interior_q)) for x = circle_ps]
figure()
plot_attachment_csqmis(circle_ps, circle_ps, csqmis)

# application at second point
applied_p = circle_ps[indmax(csqmis)]

figure()
applied_f, boundary_fs = solve_and_plot(circle_ps, com_p, applied_p)
f_hat = applied_f + sigma * randn()
update_prior!(prior, interior_q, circle_ps, applied_p, f_hat, sigma)

#fig, ax = plt[:subplots](1)
plot_grid(prior, :hot)
title("probability distribution of center-of-mass")

# selection of third point
csqmis = [(println("computed csqmi");compute_mutual_information(circle_ps, x, prior, sigma, interior_q)) for x = circle_ps]
figure()
plot_attachment_csqmis(circle_ps, circle_ps, csqmis)

# application at third point
applied_p = circle_ps[indmax(csqmis)]

figure()
applied_f, boundary_fs = solve_and_plot(circle_ps, com_p, applied_p)
f_hat = applied_f + sigma * randn()
update_prior!(prior, interior_q, circle_ps, applied_p, f_hat, sigma)

#fig, ax = plt[:subplots](1)
plot_grid(prior, :hot)
title("probability distribution of center-of-mass")
