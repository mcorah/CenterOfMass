using HDF5, JLD
using PyCall
@pyimport matplotlib2tikz

max_immediate_feasibility = 0.476
mean_immediate_feasibility = 0.322

include("CenterOfMass.jl")
include("setup_parameters.jl")

save_plots = false
do_belief = false

plt[:close]("all")
pygui(!save_plots)

folder = "chance_constrained"

max_robots = 4

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

@load "$(folder)/data.fixed_probability.40" data_array csqmi_ratio thetas actuator_limit feasibility_constraint
n_trial = size(data_array, 1)

forces = map(x->x.applied_f, data_array)
num_measurement = map(x->length(x.robots_measurement), data_array)
num_robot = map(x->length(x.robots), data_array)
actuator_limits = actuator_limit * num_measurement
limits = forces - actuator_limits .>-1e-6

current_feasibility = map(x->x.current_feasibility, data_array)
true_feasibility = map(x->x.true_feasibility, data_array)


robots_feasible(robots, theta) =
  lifting_feasibility(attachment_ws[robots],
                      theta[3]*g*point_to_wrench(theta[1:2]),
                      actuator_limit)

current_lifting = Array{Bool}(size(data_array))
for ii = 1:size(data_array, 1)
  for jj = 1:size(data_array, 2)
    current_lifting[ii,jj] = robots_feasible(data_array[ii,jj].robots, thetas[:,ii])
  end
end

lifting_configurations = sum(current_lifting[:,end])

println("Found $(lifting_configurations) feasible lifting configurations in $(size(data_array,1)) trials")
println("$(sum(~true_feasibility[:,end])) trials had no feasible lifting configuration remaining")

##########################################################################
# compare to behavior of best immediate combination to iterative selection
##########################################################################

best_initial = map(x->robots_feasible([1,3,5,8], thetas[:,x]), 1:size(thetas,2))
println("The rate for the best initial configuration is $(mean(best_initial))")

#######################
# plot normalized error
#######################
println("normalized error")

figure()
errors = map(x->x.error, data_array)
mean_errors = mean(errors, 1)

colors = ["b", "g", "r", "m"]

indices = collect(1:length(mean_errors))
plot(indices, mean_errors[:], color = "k", linewidth = 2.0)

cutoff = 0.15
n = 2
stds = std(errors, 1)[:] / sqrt(size(errors,1))
fill([indices;reverse(indices)], [mean_errors[:]+stds;reverse(mean_errors[:]-stds)],
  color = "k", alpha=0.2, linewidth=0.0)

if false
for ii = 1:size(errors, 1)
  plot(indices, errors[ii,:][:], color = "k", alpha = 0.2,
  linewidth = 1.0)

  #robot_added_inds = find(diff(num_robot[jj, ii, :][:]))+1
  #scatter(robot_added_inds, errors[jj, ii, robot_added_inds], color =
  #colors[jj], alpha = 0.2, marker = "1")

  for kk = 2:size(errors, 3)
    scatter(kk, errors[jj, ii, kk], color = colors[jj], alpha = 0.2)
  end
end
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
#plot(indices, mean_feasibility[:], color = "k", linewidth = 2.0)
plot(indices, mean(true_feasibility,1)[:], color = "b")
plot(indices, mean(current_lifting,1)[:], color = "g")

plot([1;size(errors, 2)], max_immediate_feasibility*ones(2), "--r")
plot([1;size(errors, 2)], mean_immediate_feasibility*ones(2), "--c")

#stds = std(current_feasibility, 1)[:] / sqrt(size(current_feasibility,1))
#fill([indices;reverse(indices)], [mean_feasibility[:]+stds;reverse(mean_feasibility[:]-stds)],
  #color = "k", alpha=0.2, linewidth=0.0)

if false
for ii = 1:size(current_feasibility, 1)
  plot(indices, current_feasibility[ii,:][:], color = "k", alpha = 0.2,
    linewidth = 1.0)
end
end

legend(["feasibility existence", "current feasibility", "max configuration", "mean feasibility"], loc="upper left")


xlabel("Iteration")
#ylabel("Feasibility probability")

if save_plots
  matplotlib2tikz.save("$(folder)/feasibility.tex", figureheight="\\figureheight",
    figurewidth="\\figurewidth")
end

#################################
# feasibility by number of robots
#################################

feasibility_by_number = zeros(max_robots)
lifting_by_number = zeros(max_robots)
num_number = zeros(max_robots)
for ii = 1:max_robots
  for jj = 1:n_trial
    ind = findfirst(num_robot[1,:], ii)
    if ind > 0
      num_number[ii] += 1
      feasibility_by_number[ii] += true_feasibility[jj,ind]
      lifting_by_number[ii] += current_lifting[jj,ind]
    else
      feasibility_by_number[ii] += true_feasibility[jj,ind]
      lifting_by_number[ii] += current_lifting[jj,ind]
    end
  end
end

feasibility_by_number[:] ./= num_number
lifting_by_number[:] ./= num_number

figure()
plot(collect(1:max_robots), feasibility_by_number, color = "b")
plot(collect(1:max_robots), lifting_by_number, color = "g")
plot([1;max_robots], max_immediate_feasibility*ones(2), "--r")
plot([1;max_robots], mean_immediate_feasibility*ones(2), "--c")
legend(["feasibility existence", "current feasibility", "max configuration", "mean feasibility"], loc="upper left")

xlabel("Num. robots")
#ylabel("Feasibility probability")

if save_plots
  matplotlib2tikz.save("$(folder)/lifting.tex", figureheight="\\figureheight",
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

    for jj = 1:size(errors,2)
      plt[:close]("all")
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
      plot_occupied_points(attachment_ps[setdiff(robots, robots_measurement)])
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
      fill3d(hcat(circle_ps...), alpha=0.2)

      xlabel("X")
      ylabel("Y")
      zlabel("M")
      #fig[:axes][1][:get_yaxis]()[:set_visible](false)
      #fig[:axes][1][:get_xaxis]()[:set_visible](false)
      #axis("off")
      xlim(-1.0,1.0)
      ylim(-1.0,1.0)
      zlim(0.0,1.405)
      savefig("$(trial_folder)/belief_$(jj).png", pad_inches=0.01, bbox_inches="tight")
    end
  end
end

pygui(true)
