using PyPlot
using Convex
using Mapping
#using CenterOfMass
include("CenterOfMass.jl")
plt[:close]("all")

min_mass = 0.6
max_mass = 1.4
mass = min_mass + rand() * (max_mass - min_mass)
g = -9.8
z = [0;0;1]
sigma = 1
#sigma = 0.1

mass_resolution = 0.2
masses = collect(min_mass:mass_resolution:max_mass)

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
#resolution = 0.05
#resolution = 0.2
resolution = 0.1

@everywhere interior_q(x) = norm(x) <= 1 - resolution * 3/2

prior = initialize_prior(circle_ps, resolution, interior_q, masses, mass_resolution)

#fig, ax = plt[:subplots](1)
#plot_grid(prior, :hot)
#ax[:set_ylim]([-1,1])
#ax[:set_xlim]([-1,1])

# really I lose uniformity here
com_p = r_attach * rand_in_circle()
#com_p = [-0.6;-0.13;0.0]

critical_forces_by_point = map(attachment_ps) do point
  get_critical_values(attachment_ps, point, prior, interior_q, masses)
end

pygui(false)
for ii = 1:8
  # selection of second point
  csqmis = map(critical_forces_by_point) do critical_forces
    println("computing csqmi")
    compute_mutual_information(critical_forces, prior, sigma, resolution^2*mass_resolution)
  end

  #fig = figure(figsize=(6,6), dpi=600)

  # application at second point
  applied_p = attachment_ps[indmax(csqmis)]
  critical_forces = critical_forces_by_point[indmax(csqmis)]

  #@show critical_forces
  #fig, ax = plt[:subplots](1)
  #plot_field(reshape(critical_forces[1,:,:], (size(critical_forces, 2),
  #size(critical_forces, 3))))


  applied_f, boundary_fs = critical_force_from_points(circle_ps, com_p, applied_p, mass)

  f_hat = applied_f + sigma * randn()
  update_prior!(prior, critical_forces, f_hat, sigma, mass_resolution)

  cloud = to_cloud(prior, masses, interior_q)

  fig, ax = plt[:subplots](1)

  plot_attachment_csqmis(circle_ps, attachment_ps, csqmis)
  plot_solution(applied_p)

  scaled_ps = (15*cloud[4,:]'/maximum(cloud[4,:]))
  scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', "z", scaled_ps.^2, "purple", alpha = 0.5)

  # plot point
  scatter3D([com_p[1]], [com_p[2]], [mass], "z", 200, "red", marker="*", alpha=1)
  #scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', cloud[4,:]')

  #title("probability distribution of center-of-mass")
  xlabel("X")
  ylabel("Y")
  zlabel("M")
  fig[:axes][1][:get_yaxis]()[:set_visible](false)
  fig[:axes][1][:get_xaxis]()[:set_visible](false)
  #axis("off")
  savefig("fig/belief_3d$(ii).png", pad_inches=0.01, bbox_inches="tight")
end
pygui(true)
