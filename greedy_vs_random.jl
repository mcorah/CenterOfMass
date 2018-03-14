using PyPlot
using Convex
using PyCall
using HDF5, JLD
@pyimport matplotlib2tikz

#using CenterOfMass
include("CenterOfMass.jl")
include("setup_parameters.jl")

n_trial = 100
n_measurement = 20
#n_trial = 5
#n_measurement = 10

plt[:close]("all")

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

if false
@time critical_forces_by_point = map(attachment_ws) do wrench
  get_critical_values(circle_ws, wrench, prior, interior_q)
end
end

if false
@time normals_matrices = map(critical_forces_by_point) do forces
  normal_matrix(forces[:], sigma^2)
end
end

rand_com() = r_attach * rand_in_circle()[1:2] # hack
rand_mass() = min_mass + rand() * (max_mass - min_mass)

rand_parameters() = [rand_com(); rand_mass()]

thetas = hcat(map(x->rand_parameters(), 1:n_trial)...)

select_random(prior) = rand(1:length(attachment_ps))

function select_csqmi(prior)
  n = length(critical_forces_by_point)
  csqmis = map(1:n) do ii
    compute_mutual_information(get_data(prior)[:], normals_matrices[ii])
  end

  indmax(csqmis)
end

selection_functions = [select_csqmi, select_random]
nf = length(selection_functions)

estimates = cell(n_trial, nf, n_measurement)
errors = zeros(n_trial, nf, n_measurement)

if false
for ii = 1:n_trial
  theta = thetas[:,ii]
  com_p = [theta[1:2];0]
  mass = theta[3]

  println("trial $(ii)")

  for jj = 1:nf
    println("function $(jj)")
    selection_function = selection_functions[jj]
    belief = deepcopy(prior)


    for kk = 1:n_measurement
      index = selection_function(belief)
      applied_p = attachment_ps[index]

      applied_f = critical_force_from_points(circle_ps, com_p, applied_p, mass)

      f_hat = applied_f + sigma * randn()

      critical_forces = critical_forces_by_point[index]

      update_prior!(belief, critical_forces, f_hat, sigma)

      estimate = weighted_average(belief)

      estimates[ii,jj,kk] = estimate
      errors[ii,jj,kk] = normalized_error(estimate - theta)
    end
  end
end
else
@load "greedy_vs_random/data.old.good" errors thetas estimates
for ii = 1:n_trial
  theta = thetas[:,ii]
  for jj = 1:nf
    for kk = 1:n_measurement
      estimate = estimates[ii,jj,kk]
      errors[ii,jj,kk] = normalized_error(estimate - theta)
    end
  end
end
end

colors = ["b" "g"]

mean_error = mean(errors, 1)
plot(1:n_measurement, mean_error[:,1,:][:], color = "b", linewidth = 3.0)
plot(1:n_measurement, mean_error[:,2,:][:], color = "g", linewidth = 3.0)
legend(["csqmi", "random"])

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
plot([13;13],[0.0;0.1],linestyle="--", color="g")

xlabel("Iteration")
ylabel("Normalized error")

#@save "greedy_vs_random/data" errors thetas estimates

matplotlib2tikz.save("greedy_vs_random/convergence.tex", figureheight="\\figureheight",
  figurewidth="\\figurewidth")
