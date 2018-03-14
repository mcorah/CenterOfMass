using Convex
using HDF5, JLD

folder = "greedy_vs_random"

#using CenterOfMass
include("CenterOfMass.jl")
include("setup_parameters.jl")

n_trial = 1000
n_measurement = 20

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

@time critical_forces_by_point = map(attachment_ws) do wrench
  get_critical_values(circle_ws, wrench, prior, interior_q)
end

@time normals_matrices = map(critical_forces_by_point) do forces
  normal_matrix(forces[:], sigma^2)
end

rand_com() = r_attach * rand_in_circle()[1:2] # hack
rand_mass() = min_mass + rand() * (max_mass - min_mass)

rand_parameters() = [rand_com(); rand_mass()]

thetas = hcat(map(x->rand_parameters(), 1:n_trial)...)

select_random(prior) = (rand(1:length(attachment_ps)), ones(length(attachment_ps)))

cycle_val = 1
select_cyclic(prior) = begin
  global cycle_val
  val = cycle_val
  cycle_val = (cycle_val-1+3)%length(attachment_ps) + 1
  (val, ones(length(attachment_ps)))
end

function select_csqmi(prior)
  n = length(critical_forces_by_point)
  csqmis = map(1:n) do ii
    compute_mutual_information(get_data(prior)[:], normals_matrices[ii])
  end

  indmax(csqmis), csqmis
end

selection_functions = [select_csqmi, select_random, select_cyclic]
nf = length(selection_functions)

type TrialData
  #belief::Histogram
  estimate
  error
  applied_f
  f_hat
  csqmis
  robot_measurement
end

data_array = Array{TrialData}(n_trial, nf, n_measurement)

belief = deepcopy(prior)

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
      index, csqmis = selection_function(belief)
      applied_p = attachment_ps[index]

      applied_f = critical_force_from_points(circle_ps, com_p, applied_p, mass)

      f_hat = applied_f + sigma * randn()

      critical_forces = critical_forces_by_point[index]

      update_prior!(belief, critical_forces, f_hat, sigma)

      estimate = weighted_average(belief)
      error = normalized_error(estimate - theta)

      data = TrialData(deepcopy(estimate), error, applied_f,
        f_hat, csqmis, index)
      data_array[ii,jj,kk] = data
    end
  end
end

if ~isdir(folder)
  mkdir(folder)
end
@save "$(folder)/data_new" data_array thetas
Void
