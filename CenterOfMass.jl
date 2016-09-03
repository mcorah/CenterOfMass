#module CenterOfMass

using PyPlot
using Convex
using SCS
using GLPKMathProgInterface
include("Histogram.jl")
include("plotting_tools.jl")

# all wrenches computed around the origin
point_to_wrench(p::Array{Float64,1}) = [1.0;-p[2]; p[1]]

# normal(x, var) = exp(-0.5*x*x/var)/sqrt(2*pi*var)
const one_over_sqrt_2pi = 1.0 / sqrt(2*pi)
function normal(x, var::Float64)
  ov = 1.0/var
  sqrt2pivar = sqrt(ov) * one_over_sqrt_2pi
  normal(x, -0.5*ov, sqrt2pivar)
end

normal(x::Float64, nhalfov::Float64, ov2sqrt2pi::Float64) = exp(x*x*nhalfov) * ov2sqrt2pi

function critical_force(wrench_applied, wrench_offset, W_boundary)
  fa = Variable(1)
  Fr = Variable(size(W_boundary, 2))

  stability = wrench_applied * fa +
              W_boundary * Fr +
              wrench_offset == 0
  feasibility = [fa >= 0; Fr >= 0]

  problem = maximize(fa, [stability; feasibility])

  solver = SCSSolver(verbose = 0)

  solve!(problem, solver)

  problem.optval, evaluate(Fr)
end

function solve_minimal(a, w1, w2, b)
  A = [a w1 w2]

  #if abs(det(A)) <= 1e-3
    #return -1, [-1;-1]
  #end

  x = A \ (-b)

  for ii = 1:3
    if x[ii] < -1e-3
      return -1.0
    end
  end

  return x[1]
end

# assume points form the convex hull and are arranged clockwise or
# counter-clockwise
function critical_force_iterative(wrench_applied, wrench_offset, W_boundary,
  actuator_limit::Float64 = 1e6)

  max_val::Float64 = -1

  n = size(W_boundary, 2)

  # TODO: implement binary search
  for ii = 1:n
    i2 = mod(ii, n) + 1
    val = solve_minimal(wrench_applied, W_boundary[:,ii], W_boundary[:,i2], wrench_offset)

    if val > max_val
      if val > actuator_limit
        max_val = actuator_limit
      else
        max_val = val
      end

      break
    end
  end

  max_val
end

flatten(v::Array) = hcat(v...)
flatten(v) = v

function rand_in_circle()
  r = sqrt(rand())
  theta = 2*pi*rand()
  r * [cos(theta);sin(theta);0]
end

function to_wrench_matrix(points)
  n = length(points)
  ws = zeros(3, n)
  for ii = 1:n
    ws[:,ii] = point_to_wrench(points[ii])
  end

  ws
end

function critical_force_from_points(boundary_ps, com_p, applied_p, mass,
    actuator_limit = 1e6, solver = critical_force_iterative)

  g = -9.8

  boundary_ws = to_wrench_matrix(boundary_ps)
  gravity_w = mass*g*point_to_wrench(com_p)
  applied_w = point_to_wrench(applied_p)

  fa = solver(applied_w, gravity_w, boundary_ws, actuator_limit)

  #stability = applied_w * fa +
              #boundary_ws * f_boundary +
              #gravity_w

  fa
end

function get_reaction_points(boundary_ps, boundary_fs)
  inds = find(boundary_fs .>= 1e-3)
  force_points = boundary_ps[inds]

  force_points
end

function plot_solution(applied_p)
  scatter3D(applied_p[1,:]', applied_p[2,:]', [0], color="k", s=800, alpha=0.8)
end

function plot_attachment_points(attachment_ps)
  for p = attachment_ps
    #scatter3D(p[1,:]', p[2,:]', [0], s=400, facecolors="",edgecolors = "k",
    #alpha = 0.8)
    plot_circle(p, radius = 0.2)
  end
end

function plot_new_point(point)
  # nothing to see here
end

function plot_occupied_points(occupied_points)
  for p = occupied_points
    plot_quadrotor(p+[0;0;0.025], color="k", scale=0.15)
  end
end

function plot_measurement_points(measurement_points)
  for p = measurement_points
    plot_quadrotor(p+[0;0;0.025], color="r", scale=0.15)
  end
end

function initialize_prior(boundary_ps, resolution, interior_q, masses)
  boundary_vec = flatten(boundary_ps)

  min_p = minimum(boundary_vec, 2)[1:2]
  max_p = maximum(boundary_vec, 2)[1:2]
  position_ranges = map(x->x[1]:resolution:x[2], zip(min_p, max_p))
  ranges = (position_ranges..., masses)

  prior = Histogram(ranges)

  num_inside = 0
  for kk = 1:length(1:length(get_range(prior, 3)))
    for jj = 1:length(1:length(get_range(prior, 2)))
      for ii = 1:length(1:length(get_range(prior, 1)))
        p = from_indices(prior, [ii,jj,kk])
        if interior_q(p[1:2])::Bool
          num_inside += 1
        end
      end
    end
  end
  p_interior = (1 / (num_inside))

  for kk = 1:length(1:length(get_range(prior, 3)))
    for jj = 1:length(1:length(get_range(prior, 2)))
      for ii = 1:length(1:length(get_range(prior, 1)))
        p = from_indices(prior, [ii,jj,kk])
        value = 0
        if interior_q(p[1:2])::Bool
          value = p_interior
        end
        get_data(prior)[ii,jj,kk] = value
      end
    end
  end

  prior
end

function update_prior!(prior, interior_q, boundary_ws, applied_w, f_hat, sigma, actuator_limit = 1e6)
  critical_fs = get_critical_values(boundary_ws, applied_w, prior, interior_q, actuator_limit)
  update_prior!(prior, critical_fs, f_hat, sigma)
end

shift_dims(field) = reshape(field[1,:,:], (size(field, 2), size(field, 3)))

function update_prior!(prior, critical_fs, f_hat, sigma)
  p_prior = get_data(prior)

  p_f_given_com = map(y->normal(y, sigma^2), f_hat - critical_fs)

  p_f_and_com = p_f_given_com .* p_prior

  p_f = sum(p_f_and_com)

  p_com_given_f = p_f_and_com / p_f



  get_data(prior)[:] = p_com_given_f[:]

  assert(abs(sum(get_data(prior)) - 1.0) < 1e-6)

  prior
end

function get_critical_values(boundary_ws, applied_w, prior, interior_q, actuator_limit = 1e6)
  values = zeros(size(prior))

  for kk = 1:length(get_range(prior, 3))
    for jj = 1:length(get_range(prior, 2))
      for ii = 1:length(get_range(prior, 1))
        p = from_indices(prior, [ii, jj, kk])
        if interior_q(p[1:2])::Bool
          mass = p[3]
          gravity_w = mass*g*point_to_wrench(p)
          applied_f = critical_force_iterative(applied_w, gravity_w, boundary_ws, actuator_limit)
          values[ii,jj,kk] = applied_f
          if isnan(applied_f)
            values[ii,jj,kk] = 0.0
          end
        end
      end
    end
  end

  values
end

function plot_attachment_csqmis(boundary_ps, attachment_ps, csqmis)
  attachment_vec = hcat(attachment_ps...)

  csqmi_scaled = 800 * csqmis / maximum(csqmis)
  scatter3D(attachment_vec[1,:]', attachment_vec[2,:]',
    zeros(size(attachment_vec, 2)), s = csqmi_scaled,
    color="b", alpha = 0.5, edgecolor="k", linewidth=2)
end

function to_cloud(prior, interior_q)
  out = Any[]
  ind = 1
  for kk = 1:length(get_range(prior, 3))
    for jj = 1:length(get_range(prior, 2))
      for ii = 1:length(get_range(prior, 1))
        p = from_indices(prior, [ii, jj, kk])
        if(interior_q(p[1:2]))::Bool
          mass = p[3]
          probability = get_data(prior)[ii, jj, kk]
          push!(out, [p; probability])
        end
      end
    end
  end
  out = hcat(out...)

  out
end

function plot_field(field, title_string="")
  fig, ax = plt[:subplots](1)
  imshow(field' , cmap= "BuPu",
         interpolation="nearest", origin="lower")
  fig[:axes][1][:get_yaxis]()[:set_visible](false)
  fig[:axes][1][:get_xaxis]()[:set_visible](false)
  title(title_string)
  axis("off")
end

include("csqmi.jl")

typealias CombinationMap Dict{Array{Int64,1},Array{Float64,2}}
function generate_over_combinations(action_ws, num_choose, prior,
  boundary_ws, actuator_limit = 1e6, interior_q = x->true)

  total_limit = actuator_limit * num_choose

  d = CombinationMap()
  for c = combinations(1:length(action_ws), num_choose)
    action_w = mean(action_ws[c])
    field = get_critical_values(boundary_ws, action_w, prior, interior_q, total_limit)
    normals = normal_matrix(field[:], sigma^2)
    d[c] = normals
  end
  d
end

function generate_combinations_maps(action_ws, max_choose, prior,
  boundary_ws, actuator_limit = 1e6, interior_q = x->true)
  out = Array{CombinationMap,1}()
  for ii = 1:max_choose
    println("Generating normals for $(ii)")
    @time combination_map = generate_over_combinations(action_ws, ii, prior,
      boundary_ws, actuator_limit, interior_q)
    push!(out, combination_map)
  end
  out
end

function maximize_csqmi_combinations(robot_indices, num_select, prior,
  boundary_ws, action_ws, sigma, interior_q, actuator_limit)

  max_csqmi = 0.0
  best_combination = Array{Float64,1}()
  data = get_data(prior)[:]

  total_limit = actuator_limit * num_select

  for c = combinations(robot_indices, num_select)
    action_w = mean(action_ws[c])

    field = get_critical_values(boundary_ws, action_w, prior, interior_q, total_limit)
    normals = normal_matrix(field[:], sigma^2)

    csqmi = compute_mutual_information(data, normals)

    if csqmi > max_csqmi
      max_csqmi = csqmi
      best_combination = c
    end
  end

  max_csqmi, best_combination
end

function maximize_csqmi_available_robots(robot_indices, prior, boundary_ws,
  action_ws, sigma, interior_q, actuator_limit)

  max_csqmi = 0.0
  best_combination = Array{Float64,1}()

  for ii = 1:length(robot_indices)
    csqmi, c = maximize_csqmi_combinations(robot_indices, ii, prior,
      boundary_ws, action_ws, sigma, interior_q, actuator_limit)

    if csqmi > max_csqmi
      max_csqmi = csqmi
      best_combination = c
    end
  end

  max_csqmi, best_combination
end

function maximize_csqmi_additional_robot(robot_indices, prior, boundary_ws,
  action_ws, sigma, interior_q, actuator_limit)

  max_csqmi = 0.0
  best_combination = Array{Float64,1}()
  additional_robot = 0

  all_robots = collect(1:length(action_ws))

  remaining_robots = setdiff(all_robots, robot_indices)

  # iterate over all combinations of robots including the additional robot
  for remaining_robot = remaining_robots
    for ii = 0:length(robot_indices)
      for c = combinations(robot_indices, ii)
        robots = [remaining_robot; c]
        total_limit = actuator_limit * length(robots)

        action_w = mean(action_ws[robots])

        field = get_critical_values(circle_ws, action_w, prior, interior_q, total_limit)
        normals = normal_matrix(field[:], sigma^2)

        csqmi = compute_mutual_information(get_data(prior)[:], normals)

        if csqmi > max_csqmi
          max_csqmi = csqmi
          best_combination = robots
          additional_robot = remaining_robot
        end
      end
    end
  end

  max_csqmi, best_combination, additional_robot
end

function maximize_csqmi_additional_robot_feasibility(robot_indices, prior, boundary_ws,
  action_ws, sigma, interior_q, actuator_limit, feasibility_constraint)

  max_csqmi = 0.0
  best_combination = Array{Float64,1}()
  additional_robot = 0

  all_robots = collect(1:length(action_ws))

  remaining_robots = setdiff(all_robots, robot_indices)

  # iterate over all combinations of robots including the additional robot
  for remaining_robot = remaining_robots
    robot_indices_p = [robot_indices; remaining_robot]
    robots = action_ws[robot_indices_p]
    doubly_remaining = action_ws[setdiff(all_robots, robot_indices_p)]

    fp = feasibility_probability(belief, robots, doubly_remaining, actuator_limit, max_robots)

    if fp >= feasibility_constraint
      println("adding robot $(remaining_robot) is feasible: $(fp) >= $(feasibility_constraint)")
      for ii = 0:length(robot_indices)
        for c = combinations(robot_indices, ii)
          robots = [remaining_robot; c]
          total_limit = actuator_limit * length(robots)

          action_w = mean(action_ws[robots])

          field = get_critical_values(circle_ws, action_w, prior, interior_q, total_limit)
          normals = normal_matrix(field[:], sigma^2)

          csqmi = compute_mutual_information(get_data(prior)[:], normals)

          if csqmi > max_csqmi
            max_csqmi = csqmi
            best_combination = robots
            additional_robot = remaining_robot
          end
        end
      end
    else
      println("adding robot $(remaining_robot) isn't feasible: $(fp) < $(feasibility_constraint)")
    end
  end

  max_csqmi, best_combination, additional_robot
end

# feasibility
function lifting_feasibility(applied_wrenches, wrench_offset, actuator_limit)
  @show W = hcat(applied_wrenches...)

  @show fs = Variable(length(applied_wrenches))

  lifting_condition = W*fs + wrench_offset == 0

  feasibility = [fs >= 0.0; fs <= actuator_limit]

  problem = maximize(0, [lifting_condition; feasibility])

  solver = SCSSolver(verbose = 0)
  solve!(problem, solver)

  ret = problem.status != :Infeasible

  ret
end

function check_feasible_configuration(chosen_ws, remaining_ws, offset_w, actuator_limit,
  max_num_robots)

  num_remaining = max_num_robots - length(chosen_ws)

  W_chosen = hcat(chosen_ws...)
  W_remaining = hcat(remaining_ws...)

  fs_chosen = Variable(length(chosen_ws))
  fs_remaining = Variable(length(remaining_ws))

  fs_enabled = Variable(length(remaining_ws), :Bin)

  feasibility = Convex.Constraint[]
  if length(remaining_ws) > 0 && length(chosen_ws) > 0
    #println("remaining and chosen")
    feasibility = [
      W_chosen*fs_chosen + W_remaining*fs_remaining + offset_w == 0.0;
      fs_chosen >= 0.0;
      fs_remaining >= 0.0;
      fs_chosen <= actuator_limit;
      fs_remaining <= fs_enabled*actuator_limit;
      sum(fs_enabled) <= num_remaining
    ]
  elseif length(remaining_ws) > 0
    #println("remaining")
    feasibility = [
      W_remaining*fs_remaining + offset_w == 0.0;
      fs_remaining >= 0.0;
      fs_remaining <= fs_enabled*actuator_limit;
      sum(fs_enabled) <= num_remaining
    ]
  else
    #println("chosen")
    feasibility = [
      W_chosen*fs_chosen + offset_w == 0.0;
      fs_chosen >= 0.0;
      fs_chosen <= actuator_limit;
    ]
  end

  problem = minimize(0.0, feasibility)

  solver = GLPKSolverMIP(presolve=true, msg_lev=GLPK.MSG_OFF)

  #TT=STDERR
  #out_read, out_write = redirect_stderr()
  #close(out_write)

  try
    solve!(problem, solver)
  catch ex
    #data = readavailable(out_read)
    #close(out_read)

    #redirect_stderr(TT)

    if isa(ex, InterruptException)
      println("Caught Interrupt")
    else
      println("Caught other exception")
      @show ex
    end
    throw(ex)
  end

  ret = problem.status == :Optimal

  ret
end

function milp()
  x = Variable(4, :Bin)

  problem = minimize(sum(x), x>=0.5)

  solver = GLPKSolverMIP(presolve=true)
  solve!(problem, solver)

  ret = problem.optval

  if problem.status == :Infeasible
    ret = 1000
  end

  ret
end

# probability of feasible configuration
function feasibility_probability(belief, chosen_ws, remaining_ws,
  actuator_limit, max_num_robots)
  g = -9.8
  ranges = get_range(belief)

  data = get_data(belief)

  total = 0.0

  for ii = 1:length(ranges[1])
    for jj = 1:length(ranges[2])
      for kk = 1:length(ranges[3])
        p = from_indices(belief, [ii;jj;kk])
        gravity_w = g * p[3] * point_to_wrench(p[1:2])

        feasible = check_feasible_configuration(chosen_ws, remaining_ws,
          gravity_w, actuator_limit, max_num_robots)

        total += feasible * data[ii,jj,kk]
      end
    end
  end

  total
end

# optimality

# expected configuration value

#end
