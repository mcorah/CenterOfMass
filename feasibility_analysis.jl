# accuracies: (0.7895149180648964,0.7598746490827197,0.37677090814128095)
include("CenterOfMass.jl")
include("setup_parameters.jl")
include("plotting_tools.jl")
using PyCall
@pyimport skimage.measure as measure

plt[:close]("all")
pygui(false)

feasibility_constraint = 0.9

folder = "feasibility"

actuator_limit = 6.0
max_robots = 4
n_trial = 20
n_measurement = 20

mass_resolution = 0.05
masses = collect(min_mass:mass_resolution:max_mass)
resolution = 0.05

function analyze_feasibility(robots)
  # initialize feasibility histogram
  prior = initialize_prior(circle_ps, resolution, interior_q, masses)
  feasibility = Histogram(get_range(prior))

  all_robots = collect(1:length(attachment_ws))
  chosen_ws = attachment_ws[robots]
  remaining_ws = attachment_ws[setdiff(all_robots, robots)]

  ranges = get_range(feasibility)
  for (ii, x) = enumerate(ranges[1])
    for (jj, y) = enumerate(ranges[2])
      for (kk, m) = enumerate(ranges[3])
        Convex.clearmemory()
        gravity_w = g * m * point_to_wrench([x;y])

        feasible = check_feasible_configuration(chosen_ws, remaining_ws,
          gravity_w, actuator_limit, max_robots)
        get_data(feasibility)[ii,jj,kk] = feasible
      end
    end
  end

  f = figure()
  f[:add_subplot](111,projection="3d")

  fill3d(hcat(circle_ps...), alpha=0.2)
  plot_attachment_points(attachment_ps)
  plot_occupied_points(attachment_ps[robots])
  plot_quadrotor([0;0;0.025], color="k", scale=0.15, alpha=0.0)

  cloud = to_cloud(feasibility, interior_q)
  scaled_ps = (15*cloud[4,:]'/maximum(cloud[4,:]))
  #scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', "z", scaled_ps.^2, "gray", alpha = 0.5)
  verts, faces = measure.marching_cubes(get_data(feasibility), level=0.5,
  spacing=(resolution,resolution,mass_resolution))
  verts[:,[1,2]] -= 1.0
  verts[:,3] += min_mass
  plot_trisurf(verts[:,1],verts[:,2],faces,verts[:,3], cmap="BuPu", lw=0.5)

  xlabel("X")
  ylabel("Y")
  zlabel("M")

  xlim(-1.0,1.0)
  ylim(-1.0,1.0)
  zlim(0.0,1.405)

  if ~isdir(folder)
    mkdir(folder)
  end
  savefig("$(folder)/feasibility_$(robots).png", pad_inches=0.01, bbox_inches="tight")
  feasibility_probability = sum(get_data(prior).*get_data(feasibility))
  feasibility_probability
end

f1 = analyze_feasibility(Int64[])
f2 = analyze_feasibility([5,10])
f3 = analyze_feasibility([1,3,5,8])
pygui(true)
@show (f1,f2,f3)
Void
