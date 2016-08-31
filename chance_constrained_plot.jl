using HDF5, JLD
using PyCall
@pyimport matplotlib2tikz

include("CenterOfMass.jl")
include("setup_parameters.jl")

save_plots = true
do_belief = true

plt[:close]("all")
pygui(!save_plots)

folder = "chance_constrained"

type TrialData
  belief::Histogram
  estimate
  error
  applied_f
  f_hat
  csqmi
  robots_measurement
  robots
  current_feasibility
  true_feasibility
  feasibility_no_limit
end

@load "$(folder)/data" data_array csqmi_ratio thetas actuator_limit feasibility_constraint

forces = map(x->x.applied_f, data_array)
num_measurement = map(x->length(x.robots_measurement), data_array)
num_robot = map(x->length(x.robots), data_array)
actuator_limits = actuator_limit * num_measurement
limits = forces - actuator_limits .>-1e-6

feasible_configuration = num_robot .== 4 .* true_feasibility
feasible_configurations = sum(feasible_configuration[:,end])
println("Found $(feasible_configurations) feasible configurations in $(size(data_array,1)) trials")

current_feasibility = map(x->x.current_feasibility, data_array)
true_feasibility = map(x->x.true_feasibility, data_array)

#######################
# plot normalized error
#######################
println("normalized error")

figure()
errors = map(x->x.error, data_array)
mean_errors = mean(errors, 1)
@show size(mean_errors)

colors = ["b", "g", "r", "m"]

indices = collect(1:length(mean_errors))
plot(indices, mean_errors[:], color = "k", linewidth = 2.0)

for ii = 1:size(errors, 1)
  plot(indices, errors[ii,:][:], color = "k", alpha = 0.2,
  linewidth = 1.0)

  #robot_added_inds = find(diff(num_robot[jj, ii, :][:]))+1
  #scatter(robot_added_inds, errors[jj, ii, robot_added_inds], color =
  #colors[jj], alpha = 0.2, marker = "1")

  #for kk = 2:size(errors, 3)
    #if
      #scatter(kk, errors[jj, ii, kk], color = colors[jj], alpha = 0.2)
    #end
  #end
end

ylim(0.0, 0.5)
xlim(0.0, size(errors, 2))

xlabel("Iteration")
ylabel("Normalized error")

if save_plots
  matplotlib2tikz.save("$(folder)/convergence.tex", figureheight="\\figureheight",
    figurewidth="\\figurewidth")
end

#######################
# plot number of robots
#######################
println("number robots")

figure()

mean_robots = mean(num_robot, 1)

plot(indices, mean_robots[:], color = "k", linewidth = 2.0)

for ii = 1:size(num_robot, 1)
  plot(indices, num_robot[ii,:][:], color = "k", alpha = 0.2,
    linewidth = 1.0)
end

xlabel("Iteration")
ylabel("Num. robots")

if save_plots
  matplotlib2tikz.save("$(folder)/num_robots.tex", figureheight="\\figureheight",
    figurewidth="\\figurewidth")
end

##################
# plot feasibility
##################
println("feasibility")

figure()

mean_feasibility = mean(current_feasibility, 1)
plot(indices, mean_feasibility[:], color = "k", linewidth = 2.0)

for ii = 1:size(current_feasibility, 1)
  plot(indices, current_feasibility[ii,:][:], color = "k", alpha = 0.2,
    linewidth = 1.0)
end

xlabel("Iteration")
ylabel("Feasibility probability")

if save_plots
  matplotlib2tikz.save("$(folder)/feasibility.tex", figureheight="\\figureheight",
    figurewidth="\\figurewidth")
end

#############
# plot belief
#############
println("belief")

if save_plots && do_belief
  for ii = 1:size(errors,1)
    trial_folder = "$(folder)/$(ii)"
    mkpath(trial_folder)

    for jj = 1:size(errors,2) plt[:close]("all")
      println("belief: trial=$(ii), iter=$(jj)")
      data = data_array[ii,jj]
      robots = data.robots
      belief = data.belief
      robots_measurement = data.robots_measurement
      theta = thetas[:,ii]
      com_p = theta[1:2]
      mass = theta[3]

      figure()
      plot_attachment_points(attachment_ps)
      plot_occupied_points(attachment_ps[robots])
      cloud = to_cloud(belief, interior_q)
      scaled_ps = (15*cloud[4,:]'/maximum(cloud[4,:]))
      scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', "z", scaled_ps.^2, "purple", alpha = 0.5)

      scatter3D([com_p[1]], [com_p[2]], [mass], "z", 200, "red", marker="*", alpha=1)
      scatter3D([com_p[1]], [com_p[2]], [0], "z", 200, "red", marker=(5,2,0), alpha=1)
      plot3D(com_p[1]*ones(2), com_p[2]*ones(2), [0.0; mass],
        color = "k", alpha = 1, linestyle = "dashdot")

      plot_measurement_points(attachment_ps[robots_measurement])

      robots_polygon = sort(robots)
      push!(robots_polygon, robots_polygon[1])
      polygon = hcat(attachment_ps[robots_polygon]...)
      @show size(polygon)

      plot3D(polygon[1,:][:], polygon[2,:][:], polygon[3,:][:],
        color = "k", alpha = 1, linestyle = "solid")

      if jj > 1
        old_robots = data_array[ii,jj-1].robots
        if length(robots) > length(old_robots)
          plot_new_point(attachment_ps[setdiff(robots, old_robots)[1]])
        end
      end

      xlabel("X")
      ylabel("Y")
      zlabel("M")
      #fig[:axes][1][:get_yaxis]()[:set_visible](false)
      #fig[:axes][1][:get_xaxis]()[:set_visible](false)
      #axis("off")
      savefig("$(trial_folder)/belief_$(jj).png", pad_inches=0.01, bbox_inches="tight")
    end
  end
end

pygui(true)
