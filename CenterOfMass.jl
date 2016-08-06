#module CenterOfMass

using PyPlot
using Mapping
using Convex
using SCS

# all wrenches computed around the origin
point_to_wrench(p::Array{Float64,1}) = [1.0;-p[1]; p[2]]

normal(x, var) = exp(-0.5*x*x/var)/sqrt(2*pi*var)

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
      return -1, [-1;-1]
    end
  end

  return x[1], x[2:3]
end

# assume points form the convex hull and are arranged clockwise or
# counter-clockwise
function critical_force_iterative(wrench_applied, wrench_offset, W_boundary)
  max_val = -1

  n = size(W_boundary, 2)

  max_forces = zeros(n)

  for ii = 1:n
    i2 = mod(ii, n) + 1
    val, forces = solve_minimal(wrench_applied, W_boundary[:,ii], W_boundary[:,i2], wrench_offset)

    if val > max_val
      max_val = val
      max_forces[[ii,i2]] = forces
    end
  end

  max_val, max_forces
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

function critical_force_from_points(boundary_ps, com_p, applied_p, mass, solver
  = critical_force_iterative)
  g = -9.8

  boundary_ws = to_wrench_matrix(boundary_ps)
  gravity_w = mass*g*point_to_wrench(com_p)
  applied_w = point_to_wrench(applied_p)

  fa, f_boundary = solver(applied_w, gravity_w, boundary_ws)

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

function initialize_prior(boundary_ps, resolution, interior_q, masses, mass_resolution)
  boundary_vec = flatten(boundary_ps)

  min_p = minimum(boundary_vec, 2)[1:2]
  max_p = maximum(boundary_vec, 2)[1:2]
  grid_size = convert(Array{Int64}, floor((max_p - min_p)/resolution))

  prior = map(m->OccupancyGrid(resolution, -min_p, grid_size), masses)

  cell_volume = resolution^2 * mass_resolution

  num_inside = 0
  for ii = 1:length(masses)
    for jj = 1:size(prior[ii].cells, 1)
      for kk = 1:size(prior[ii].cells, 2)
        ind = OccupancyGridIndex((jj, kk))
        p = to_world(prior[ii], ind)
        if interior_q(p)
          num_inside += 1
        end
      end
    end
  end
  p_occupied = (1 / (cell_volume * num_inside))

  for ii = 1:length(masses)
    for jj = 1:size(prior[ii].cells, 1)
      for kk = 1:size(prior[ii].cells, 2)
        ind = OccupancyGridIndex((jj, kk))
        p = to_world(prior[ii], ind)
        if interior_q(p)
          set!(prior[ii], ind, p_occupied)
        end
      end
    end
  end

  prior
end

function update_prior!(prior, interior_q, boundary_ps, applied_p, f_hat, sigma, masses)
  critical_fs = get_critical_values(boundary_ps, applied_p, prior, interior_q,
  masses)
  update_prior!(prior, critical_fs, f_hat, sigma, masses[2] - masses[1])
end

shift_dims(field) = reshape(field[1,:,:], (size(field, 2), size(field, 3)))

function update_prior!(prior, critical_fs, f_hat, sigma, mass_resolution)
  cell_volume = prior[1].resolution^2 * mass_resolution

  p_prior = map(x->x.cells, prior)
  @show typeof(p_prior)

  p_f_given_com = map(y->normal(y, sigma^2), f_hat - critical_fs)
  #@show typeof(p_f_given_com)
  #plot_field(shift_dims(p_f_given_com), "p_f_given_com")

  p_f_and_com = Array[]
  for ii = 1:length(p_prior)
    push!(p_f_and_com, shift_dims(p_f_given_com[ii,:,:]).*p_prior[ii])
  end
  #p_f_and_com = map(x->x[1] .* x[2], zip(p_f_given_com, p_prior))
  #@show typeof(p_f_and_com)
  #plot_field(p_f_and_com[1], "p_f_and_com")

  p_f = sum(flatten(p_f_and_com)) * cell_volume

  p_com_given_f = map(x->map(x->x / p_f, x), p_f_and_com)
  @show typeof(p_com_given_f), size(p_com_given_f[1]), size(prior[1].cells)
  #plot_field(p_com_given_f[1], "p_com_given_f")

  for ii = 1:length(prior)
    prior[ii].cells[:] = p_com_given_f[ii]
  end

  prior
end

function get_critical_values(boundary_ps, applied_p, grids, interior_q, masses)
  values = zeros(length(masses), size(grids[1].cells)...)
  for ii = 1:length(masses)
    for jj = 1:size(grids[ii].cells, 1)
      for kk = 1:size(grids[ii].cells, 2)
        ind = OccupancyGridIndex((jj, kk))
        p = [to_world(grids[ii], ind);0]
        if interior_q(p)
          applied_f, boundary_fs = critical_force_from_points(boundary_ps, p, applied_p, masses[ii])
          values[ii,jj,kk] = applied_f
          if isnan(applied_f)
            values[ii,jj,kk] = 0
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
  for ii = 1:length(masses)
    for jj = 1:size(prior[ii].cells, 1)
      for kk = 1:size(prior[ii].cells, 2)
        occ_ind = OccupancyGridIndex((jj, kk))
        point = to_world(prior[ii], occ_ind)
        if(interior_q(point))
          mass = masses[ii]
          probability = prior[ii].cells[jj,kk]

          push!(out, [point; mass; probability])
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
