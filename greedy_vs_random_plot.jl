using PyPlot
using Convex
using PyCall
using HDF5, JLD
@pyimport matplotlib2tikz

save_plots = true
do_belief = true

plt[:close]("all")
pygui(!save_plots)

include("CenterOfMass.jl")
include("setup_parameters.jl")

type TrialData
  belief::Histogram
  estimate
  error
  applied_f
  f_hat
  csqmis
  robot_measurement
end

folder = "greedy_vs_random"

@load "$(folder)/data" data_array thetas

forces = map(x->x.applied_f, data_array)
errors = map(x->x.error, data_array)
n_measurement = size(data_array, 3)

colors = ["b" "g" "m"]

mean_error = mean(errors, 1)
for ii = 1:size(mean_error, 2)
  plot(1:n_measurement, mean_error[:,ii,:][:], color = colors[ii], linewidth = 3.0)
end
legend(["greedy", "random", "cyclic"])

indices = collect(1:n_measurement)
for ii = 1:size(errors,2)
  stds = std(errors[:,ii,:],1)[:]/sqrt(size(errors,1))
  es = mean_error[:,ii,:][:]
  fill([indices;reverse(indices)], [es+stds;reverse(es-stds)], color = colors[ii],
    alpha=0.2, linewidth=0.0)
end

if false
for ii = 1:size(errors, 2)
  for jj = 1:size(errors, 1)
    plot(1:n_measurement, errors[jj, ii,:][:], color = colors[ii], alpha = 0.1,
      linewidth = 1.0)
  end
end
end

plot([0;20],[0.1;0.1],linestyle="--", color="k")
plot([7;7],[0.0;0.1],linestyle="--", color="b")
plot([15;15],[0.0;0.1],linestyle="--", color="g")
plot([11;11],[0.0;0.1],linestyle="--", color="m")

xlabel("Iteration")
ylabel("Normalized error")

if save_plots
  matplotlib2tikz.save("$(folder)/convergence.tex", figureheight="\\figureheight",
    figurewidth="\\figurewidth")
end

if save_plots && do_belief
  for ii = 1:min(10,size(errors,1)) # num robots

    for jj = 1:size(errors,2) # function
      trial_folder = "$(folder)/fun_$(jj)_$(ii)"
      mkpath(trial_folder)
      for kk = 1:size(errors,3) # trial
        plt[:close]("all")

        println("belief: trial=$(ii), fun=$(jj), iter=$(kk)")

        data = data_array[ii,jj,kk]
        belief = data.belief
        csqmis = data.csqmis
        robot_measurement = data.robot_measurement
        theta = thetas[:,ii]
        com_p = theta[1:2]
        mass = theta[3]
        applied_p = attachment_ps[data.robot_measurement]

        figure()
        cloud = to_cloud(belief, interior_q)

        plot_attachment_csqmis(circle_ps, attachment_ps, csqmis)
        fill3d(hcat(circle_ps...), alpha=0.2, color="k")

        plot_solution(applied_p)

        scaled_ps = (15*cloud[4,:]'/maximum(cloud[4,:]))

        # plot probabilities
        scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', "z", scaled_ps.^2, "purple", alpha = 0.5)

        # plot CoM point
        scatter3D([com_p[1]], [com_p[2]], [mass], "z", 200, "red", marker="*", alpha=1)

        xlabel("X")
        ylabel("Y")
        zlabel("M")
        savefig("$(trial_folder)/belief_$(kk).png", pad_inches=0.01, bbox_inches="tight")
      end
    end
  end
end

pygui(true)
