include("CenterOfMass.jl")
include("setup_parameters.jl")

actuator_limit = 4.0
max_robots = 4

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

data = get_data(prior)[:]
@time for ii = 1:max_robots
  for c = combinations(1:length(attachment_ps), ii)
    total_limit = actuator_limit * ii
    action_w = mean(attachment_ws[c])

    field = get_critical_values(circle_ws, action_w, prior, interior_q, total_limit)
    normals = normal_matrix(field[:], sigma^2)

    csqmi = compute_mutual_information(data, normals)
  end
end
