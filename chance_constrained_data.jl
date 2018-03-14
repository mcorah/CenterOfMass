using HDF5, JLD

include("CenterOfMass.jl")
include("setup_parameters.jl")

plt[:close]("all")

feasibility_constraint = 0.9

folder = "chance_constrained"

actuator_limit = 6.0
max_robots = 4
csqmi_ratio = 5.0
n_trial = 40
n_measurement = 20

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

belief = deepcopy(prior)

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

data_array = Array{TrialData}(n_trial, n_measurement)

rand_com() = r_attach * rand_in_circle()[1:2] # hack
rand_mass() = min_mass + rand() * (max_mass - min_mass)

rand_parameters() = [rand_com(); rand_mass()]

thetas = Array{Float64}(3,0)
n_bad = 0
while size(thetas, 2) < n_trial
  theta = rand_parameters()

  gravity_w = theta[3]*g*point_to_wrench(theta[1:2])
  feasible = check_feasible_configuration(Array{Array{Float64}}(0), attachment_ws,
    gravity_w, actuator_limit, max_robots)

  if feasible
    thetas = hcat(thetas, theta)
  else
    n_bad += 1
    feasible = check_feasible_configuration(Array{Array{Float64}}(0), attachment_ws,
      gravity_w, 1e6, 10)
    if feasible
      #println("infeasible because of actuator limit")
    else
      #println("infeasible despite actuator limit")
    end
  end
end
println("$(n_bad) infeasible thetas encountered during instantiation")


for ii = 1:n_trial
  theta = thetas[:,ii]
  gravity_w = theta[3]*g*point_to_wrench(theta[1:2])
  belief = deepcopy(prior)
  robots = Array{Int64,1}()

  no_attach_count = 1

  for jj = 1:n_measurement
    Convex.clearmemory()

    println("\n$ii: Measurement: $(jj), robots: $(robots)\n")

    remaining_robots = setdiff(collect(1:length(attachment_ps)), robots)
    current_feasibility = feasibility_probability(belief, attachment_ws[robots], attachment_ws[remaining_robots], actuator_limit, max_robots)
    println("Current feasibility: $(current_feasibility)")

    @time @show csqmi_current, robots_current = maximize_csqmi_available_robots(
      robots, belief, circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

    csqmi = csqmi_current
    robots_measurement = robots_current

    if length(robots) < max_robots
      inner_feasibility_constraint = feasibility_constraint^no_attach_count * current_feasibility

      if length(robots) == 0
        inner_feasibility_constraint = 0.0
      end

      @time @show csqmi_additional, robots_additional, additional_robot =
        maximize_csqmi_additional_robot_feasibility(robots, belief,
          circle_ws, attachment_ws, sigma, interior_q, actuator_limit,
          inner_feasibility_constraint)

      if length(robots) == 0 || csqmi_additional / csqmi_current > csqmi_ratio
        println("\nAdding robot $(additional_robot), csqmi: $(csqmi_additional) vs: $(csqmi_current)\n")

        csqmi = csqmi_additional
        robots_measurement = robots_additional
        push!(robots, additional_robot)

        no_attach_count = 1
      else
        no_attach_count += 1
      end
    end

    # check feasibility of current figuration against true theta
    all_robots = collect(1:length(attachment_ws))
    remaining_robots = setdiff(all_robots, robots)

    true_feasibility = check_feasible_configuration(attachment_ws[robots],
      attachment_ws[remaining_robots], gravity_w, actuator_limit, max_robots)
    feasibility_no_limit = check_feasible_configuration(attachment_ws[robots],
      attachment_ws[remaining_robots], gravity_w, 1e6, max_robots)
    if true_feasibility
      println("The current configuration is still feasible")
    else
      if feasibility_no_limit
        println("The current configuration NO LONGER feasible but is feasible if
          limits are removed")
      else
        println("The current configuration NO LONGER feasible")
      end
    end


    applied_w = mean(attachment_ws[robots_measurement])
    total_actuator_limit = length(robots_measurement) * actuator_limit

    applied_f = critical_force_iterative(applied_w, gravity_w,
      circle_ws, total_actuator_limit)

    f_hat = applied_f + sigma * randn()

    println("\nRobots measurement: $(robots_measurement)")
    println("Measurement: $(applied_f)/$(f_hat), total limit: $(total_actuator_limit)\n")

    critical_forces = get_critical_values(circle_ws, applied_w, belief,
      interior_q, total_actuator_limit)

    update_prior!(belief, critical_forces, f_hat, sigma)

    estimate = weighted_average(belief)

    error = normalized_error(estimate - theta)
    println("Estimate: $(estimate), actual: $(theta), normalized error: $(error)")

    data = TrialData(deepcopy(belief), deepcopy(estimate), error, applied_f, f_hat, csqmi,
      deepcopy(robots_measurement), deepcopy(robots), current_feasibility,
      true_feasibility, feasibility_no_limit)
    data_array[ii,jj] = data
  end
end

if ~isdir(folder)
  mkdir(folder)
end
@save "$(folder)/data" data_array csqmi_ratio thetas actuator_limit feasibility_constraint
Void
