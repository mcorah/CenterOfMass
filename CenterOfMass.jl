#module CenterOfMass

using PyPlot
using Convex
using SCS
include("Histogram.jl")

# all wrenches computed around the origin
point_to_wrench(p::Array{Float64,1}) = [1.0;-p[2]; p[1]]

# normal(x, var) = exp(-0.5*x*x/var)/sqrt(2*pi*var)
one_over_sqrt_2pi = 1 / sqrt(2*pi)
function normal(x, var)
  ov = 1/var
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

  fa, f_boundary = solver(applied_w, gravity_w, boundary_ws, actuator_limit)

  stability = applied_w * fa +
              boundary_ws * f_boundary +
              gravity_w

  fa, f_boundary
end

function get_reaction_points(boundary_ps, boundary_fs)
  inds = find(boundary_fs .>= 1e-3)
  force_points = boundary_ps[inds]

  force_points
end

#function solve_and_plot(boundary_ps, com_p, applied_p, attachment_ps)
  #applied_f, boundary_fs = critical_force_from_points(boundary_ps, com_p, applied_p)
#
  #reaction_ps = get_reaction_points(boundary_ps, boundary_fs)
  #plot_solution(boundary_ps, com_p, applied_p, reaction_ps, attachment_ps)
#
  #applied_f, boundary_fs
#end

function plot_solution(applied_p)
  scatter3D(applied_p[1,:]', applied_p[2,:]', [0], color="k", s=800, alpha=0.8)
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
  #plot_field(shift_dims(p_f_given_com), "p_f_given_com")

  p_f_and_com = p_f_given_com .* p_prior

  #plot_field(p_f_and_com[1], "p_f_and_com")

  p_f = sum(p_f_and_com)

  p_com_given_f = p_f_and_com / p_f

  #plot_field(p_com_given_f[1], "p_com_given_f")

  get_data(prior)[:] = p_com_given_f[:]

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

function to_cloud(prior, masses, interior_q)
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

#end
