using PyPlot
using Convex
#using CenterOfMass
include("CenterOfMass.jl")
include("setup_parameters.jl")

plt[:close]("all")

mass = min_mass + rand() * (max_mass - min_mass)
com_p = r_attach * rand_in_circle()

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

critical_forces_by_point = map(attachment_ws) do wrench
  get_critical_values(circle_ws, wrench, prior, interior_q)
end

normals_matrices = map(critical_forces_by_point) do forces
  normal_matrix(forces[:], sigma^2)
end

pygui(false)
for ii = 1:16
  # selection of second point
  n_attach = length(critical_forces_by_point)
  csqmis = map(1:n_attach) do ii
    println("computing csqmi")
    compute_mutual_information(get_data(prior)[:], normals_matrices[ii])
  end

  #fig = figure(figsize=(6,6), dpi=600)

  # application at second point
  applied_p = attachment_ps[indmax(csqmis)]
  critical_forces = critical_forces_by_point[indmax(csqmis)]

  #@show critical_forces
  #fig, ax = plt[:subplots](1)
  #plot_field(reshape(critical_forces[1,:,:], (size(critical_forces, 2),
  #size(critical_forces, 3))))

  applied_f = critical_force_from_points(circle_ps, com_p, applied_p, mass)

  f_hat = applied_f + sigma * randn()
  update_prior!(prior, critical_forces, f_hat, sigma)

  cloud = to_cloud(prior, interior_q)

  fig, ax = plt[:subplots](1)

  plot_attachment_csqmis(circle_ps, attachment_ps, csqmis)
  plot_solution(applied_p)

  scaled_ps = (15*cloud[4,:]'/maximum(cloud[4,:]))
  scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', "z", scaled_ps.^2, "purple", alpha = 0.5)

  # plot point
  scatter3D([com_p[1]], [com_p[2]], [mass], "z", 200, "red", marker="*", alpha=1)
  #scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', cloud[4,:]')

  circle_polygon = hcat(circle_ps..., circle_ps[1])
  plot3D(circle_polygon[1,:][:], circle_polygon[2,:][:], circle_polygon[3,:][:],
    color = "k", alpha = 1, linestyle = "solid")

  #title("probability distribution of center-of-mass")
  xlabel("X")
  ylabel("Y")
  zlabel("M")
  fig[:axes][1][:get_yaxis]()[:set_visible](false)
  fig[:axes][1][:get_xaxis]()[:set_visible](false)
  xlim(-1.0,1.0)
  ylim(-1.0,1.0)
  #axis("off")
  savefig("fig/belief_3d$(ii).png", pad_inches=0.01, bbox_inches="tight")
end
pygui(true)
