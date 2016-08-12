include("CenterOfMass.jl")
include("setup_parameters.jl")

actuator_limit = 5.0
max_robots = 4

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

all_robots = collect(1:10)
attached_robots = [1,4,7]

@time @show maximize_csqmi_combinations(attached_robots, 2, prior,
  circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

@time @show maximize_csqmi_available_robots(attached_robots, prior,
  circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

@time @show maximize_csqmi_additional_robot(attached_robots, prior,
  circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

println("null tests")
no_robots = Array{Int64,1}()

@time @show maximize_csqmi_combinations(no_robots, 2, prior,
  circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

@time @show maximize_csqmi_available_robots(no_robots, prior,
  circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

@time @show maximize_csqmi_additional_robot(no_robots, prior,
  circle_ws, attachment_ws, sigma, interior_q, actuator_limit)
