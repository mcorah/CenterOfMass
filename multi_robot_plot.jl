using HDF5, JLD

include("CenterOfMass.jl")
include("setup_parameters.jl")

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

@load "$(folder)/data" data_array csqmi_ratios thetas actuator_limit

forces = map(x->x.applied_f, data_array)
num_measurement = map(x->length(x.robots_measurement), data_array)
num_robot = map(x->length(x.robots), data_array)
actuator_limits = actuator_limit * num_measurement
limits = forces - actuator_limits .>-1e-6

#######################
# plot normalized error
#######################

figure()
errors = map(x->x.error, data_array)
mean_errors = mean(errors, 2)
@show size(mean_errors)

colors = ["b", "g", "r", "c", "m"]

indices = collect(1:size(mean_errors,3))
for ii = 1:size(mean_errors, 1)
#for ii = 1:3
  plot(indices, mean_errors[ii, 1, :][:], color = colors[ii], linewidth = 2.0)
end
legend(map(string, csqmi_ratios))

for ii = 1:size(errors, 2)
  for jj = 1:size(errors, 1)
  #for ii = 1:3
    plot(1:size(errors,3), errors[jj, ii,:][:], color = colors[jj], alpha = 0.2,
    linewidth = 1.0)

    for kk = 2:size(errors, 3)
      if num_robot[jj, ii, kk] > num_robot[jj, ii, kk-1]
        scatter(kk, errors[jj, ii, kk], color = colors[jj], alpha = 0.2)
      end
    end
  end
end

ylim(0.0, 0.8)
xlim(0.0, size(errors, 3))


xlabel("Iteration")
ylabel("Normalized error")

#######################
# plot number of robots
#######################

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

#############
# plot belief
#############

pygui(false)
pygui(true)
