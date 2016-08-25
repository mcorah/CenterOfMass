using HDF5, JLD

include("CenterOfMass.jl")
include("setup_parameters.jl")

plt[:close]("all")

actuator_limit = 4.0
max_robots = 4
csqmi_ratios = [1.0; 2.0; 5.0; 10.0]
#csqmi_ratios = [1.0; 10.0]
#n_trial = 10
n_trial = 20
#n_trial = 5
n_measurement = 25

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
end

data_array = Array{TrialData}(length(csqmi_ratios), n_trial, n_measurement)

rand_com() = r_attach * rand_in_circle()[1:2] # hack
rand_mass() = min_mass + rand() * (max_mass - min_mass)

rand_parameters() = [rand_com(); rand_mass()]

thetas = hcat(map(x->rand_parameters(), 1:n_trial)...)

for (ii, csqmi_ratio) = enumerate(csqmi_ratios)
  for jj = 1:n_trial
    theta = thetas[:,jj]
    gravity_w = theta[3]*g*point_to_wrench(theta[1:2])
    belief = deepcopy(prior)
    robots = Array{Int64,1}()
    for kk = 1:n_measurement
      println("\n$ii,$jj: Measurement: $(kk), robots: $(robots)\n")

      @time @show csqmi_current, robots_current = maximize_csqmi_available_robots(
        robots, belief, circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

      csqmi = csqmi_current
      robots_measurement = robots_current

      if length(robots) < max_robots
        @time @show csqmi_additional, robots_additional, additional_robot =
          maximize_csqmi_additional_robot(robots, belief,
          circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

        if length(robots) == 0 || csqmi_additional / csqmi_current > csqmi_ratio
          println("\nAdding robot $(additional_robot), csqmi: $(csqmi_additional) vs: $(csqmi_current)\n")

          csqmi = csqmi_additional
          robots_measurement = robots_additional
          push!(robots, additional_robot)
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
        deepcopy(robots_measurement), deepcopy(robots))
      data_array[ii,jj,kk] = data
    end
  end
end

@save "multi_robot/data" data_array csqmi_ratios thetas actuator_limit
Void
