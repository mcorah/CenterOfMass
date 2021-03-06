using HDF5, JLD
using PyCall
@pyimport matplotlib2tikz

include("CenterOfMass.jl")
include("setup_parameters.jl")

save_plots = false
do_belief = false

plt[:close]("all")
pygui(!save_plots)

folder = "multi_robot"

type TrialData
  belief::Histogram
  estimate
  error
  applied_f
  f_hat
  csqmi
  robots_measurement
  robots
end

@load "$(folder)/data.25_iter.0_4" data_array csqmi_ratios thetas actuator_limit

forces = map(x->x.applied_f, data_array)
num_measurement = map(x->length(x.robots_measurement), data_array)
num_robot = map(x->length(x.robots), data_array)
actuator_limits = actuator_limit * num_measurement
limits = forces - actuator_limits .>-1e-6
partial_limits = forces - actuator_limit .>-1e-6

#######################
# plot normalized error
#######################
println("normalized error")

figure()
errors = map(x->x.error, data_array)
mean_errors = mean(errors, 2)

colors = ["b", "g", "r", "c"]

indices = collect(1:size(mean_errors,3))
for ii = 1:size(mean_errors, 1)
#for ii = 1:3
  plot(indices, mean_errors[ii, 1, :][:], color = colors[ii], linewidth = 2.0)
end
legend(map(string, csqmi_ratios))

cutoff = 0.15
for ii = 1:size(errors, 1)
  n = 2
  stds = std(errors[ii,:,:], 2)[:]/sqrt(size(errors,2))
  es = mean_errors[ii,1,:][:]
  fill([indices;reverse(indices)], [es+stds;reverse(es-stds)], color = colors[ii],
    alpha=0.2, linewidth=0.0)
end
#plot([0;25],[cutoff;cutoff],linestyle="--", color="k")
#plot([9;9],[0.0;cutoff],linestyle="--", color="k")
#plot([13;13],[0.0;cutoff],linestyle="--", color="k")
#plot([17;17],[0.0;cutoff],linestyle="--", color="k")

if false
for jj = 1:size(errors, 1)
  for ii = 1:size(errors, 2)
    plot(1:size(errors,3), errors[jj, ii,:][:], color = colors[jj], alpha = 0.2,
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
end
end

#ylim(0.0, 0.8)
xlim(0.0, size(errors, 3))


xlabel("Iteration")
ylabel("Normalized error")

matplotlib2tikz.save("$(folder)/convergence.tex", figureheight="\\figureheight",
  figurewidth="\\figurewidth")

#######################
# plot number of robots
#######################
println("number robots")

figure()

mean_robots = mean(num_robot, 2)
for ii = 1:size(mean_robots, 1)
  plot(indices, mean_robots[ii, 1, :][:], color = colors[ii], linewidth = 2.0)
end

legend(map(string, csqmi_ratios))

for ii = 1:size(num_robot, 2)
  for jj = 1:size(num_robot, 1)
    plot(1:size(errors,3), num_robot[jj, ii,:][:], color = colors[jj], alpha = 0.2,
    linewidth = 1.0)
  end
end

xlabel("Iteration")
ylabel("Num. robots")

matplotlib2tikz.save("$(folder)/num_robots.tex", figureheight="\\figureheight",
  figurewidth="\\figurewidth")

#############
# plot belief
#############
println("belief")

if save_plots && do_belief
  for (ii, csqmi_ratio) = enumerate(csqmi_ratios)
    for jj = 1:size(errors,2)
      trial_folder = "$(folder)/$(csqmi_ratio)_$(jj)"
      mkpath(trial_folder)

      for kk = 1:size(errors,3)
        plt[:close]("all")

        println("belief: ratio=$(csqmi_ratio), trial=$(jj), iter=$(kk)")
        data = data_array[ii,jj,kk]
        robots = data.robots
        belief = data.belief
        robots_measurement = data.robots_measurement
        theta = thetas[:,jj]
        com_p = theta[1:2]
        mass = theta[3]

        figure()
        plot_attachment_points(attachment_ps)
        plot_occupied_points(attachment_ps[setdiff(robots, robots_measurement)])
        cloud = to_cloud(belief, interior_q)
        scaled_ps = (15*cloud[4,:]'/maximum(cloud[4,:]))
        scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', "z", scaled_ps.^2, "purple", alpha = 0.5)

        scatter3D([com_p[1]], [com_p[2]], [mass], "z", 200, "red", marker="*", alpha=1)

        plot_measurement_points(attachment_ps[robots_measurement])

        if kk > 1
          old_robots = data_array[ii,jj,kk-1].robots
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
        savefig("$(trial_folder)/belief_$(kk).png", pad_inches=0.01, bbox_inches="tight")
      end
    end
  end
end

pygui(true)
