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
#sigma = 0.1

np = 20
#np = 8
thetas = 2*pi * (1/np:1/np:1)
circle_ps = map(theta->[cos(theta);sin(theta);0], thetas)

# move to separate attachment points
na = 10
thetas = 2*pi * (1/na:1/na:1)
r_attach = 0.8
#r_attach = 0.6
attachment_ps = map(theta->r_attach*[cos(theta);sin(theta);0], thetas)

# set up the occupancy prior
resolution = 0.05
#resolution = 0.1

interior_q(x) = norm(x) <= 1

prior = initialize_prior(circle_ps, resolution, interior_q)

#fig, ax = plt[:subplots](1)
#plot_grid(prior, :hot)
#ax[:set_ylim]([-1,1])
#ax[:set_xlim]([-1,1])

# really I lose uniformity here
com_p = r_attach * rand_in_circle()
#com_p = [-0.6;-0.13;0.0]

pygui(false)
for ii = 1:3
  # selection of second point
  csqmis = [(println("computed csqmi");compute_mutual_information(circle_ps, x, prior, sigma, interior_q)) for x = attachment_ps]
  fig = figure(figsize=(6,6), dpi=600)
  plot_attachment_csqmis(circle_ps, attachment_ps, csqmis)

  # application at second point
  applied_p = attachment_ps[indmax(csqmis)]

  applied_f, boundary_fs = solve_and_plot(circle_ps, com_p, applied_p, attachment_ps)
  f_hat = applied_f + sigma * randn()
  update_prior!(prior, interior_q, circle_ps, applied_p, f_hat, sigma)

  #fig, ax = plt[:subplots](1)
  plot_grid(prior, "BuPu")
  fig[:axes][1][:get_yaxis]()[:set_visible](false)
  fig[:axes][1][:get_xaxis]()[:set_visible](false)
  ylim([-1.25,1.05])
  xlim([-1.05,1.05])
  axis("off")
  #title("probability distribution of center-of-mass")
  savefig("fig/belief_$(ii).png", pad_inches=0.01, bbox_inches="tight")
end
pygui(true)
