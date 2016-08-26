include("CenterOfMass.jl")
include("setup_parameters.jl")

plt[:close]("all")

feasibility_constraint = 0.8

actuator_limit = 10.0
max_robots = 4
csqmi_ratio = 10.0

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

mass = min_mass + rand() * (max_mass - min_mass)
com_p = r_attach * rand_in_circle()
gravity_w = mass*g*point_to_wrench(com_p)
theta = [com_p[1:2]; mass]
println("\nCOM: $(com_p), mass: $(mass)")

robots = Array{Int64,1}()

n_measurement = 10

belief = deepcopy(prior)

for ii = 1:n_measurement
  figure()
  plot_attachment_points(attachment_ps)
  plot_occupied_points(attachment_ps[robots])

  cloud = to_cloud(belief, interior_q)
  scaled_ps = (15*cloud[4,:]'/maximum(cloud[4,:]))
  scatter3D(cloud[1,:]', cloud[2,:]', cloud[3,:]', "z", scaled_ps.^2, "purple", alpha = 0.5)

  scatter3D([com_p[1]], [com_p[2]], [mass], "z", 200, "red", marker="*", alpha=1)
  xlabel("X")
  ylabel("Y")
  zlabel("M")
  title("Iteration: $(ii)")


  println("\nMeasurement: $(ii), robots: $(robots)\n")

  current_feasibility = feasibility_probability(belief, Array[], attachment_ws, actuator_limit, max_robots)
  println("Current feasibility: $(current_feasibility)")

  @time @show csqmi_current, robots_current = maximize_csqmi_available_robots(
    robots, belief, circle_ws, attachment_ws, sigma, interior_q, actuator_limit)

  csqmi = csqmi_current
  robots_measurement = robots_current

  if length(robots) < max_robots
    inner_feasibility_constraint = feasibility_constraint * current_feasibility

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

      plot_new_point(attachment_ps[additional_robot])
    end
  end

  plot_measurement_points(attachment_ps[robots_measurement])

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
end
