#module CenterOfMass

using PyPlot
using Mapping
using Convex

# all wrenches computed around the origin
point_to_wrench(p) = [1;cross(p,z)[1:2]]

normal(x, var) = 1/sqrt(2*pi*var)*exp(-0.5*x^2/var)

function critical_force(wrench_applied, wrench_offset, W_boundary)
  fa = Variable(1)
  Fr = Variable(size(W_boundary, 2))

  stability = wrench_applied * fa +
              W_boundary * Fr +
              wrench_offset == 0
  feasibility = [fa >= 0; Fr >= 0]

  problem = maximize(fa, [stability; feasibility])

  TT=STDOUT
  redirect_stdout()
  solve!(problem)
  redirect_stdout(TT)

  #if problem.status != :Optimal
    #error("Solution not optimal: $(problem.status)")
  #end

  problem.optval, evaluate(Fr)
end

flatten(v) = hcat(v...)

function rand_in_circle()
  r = sqrt(rand())
  theta = 2*pi*rand()
  r * [cos(theta);sin(theta);0]
end

function critical_force_from_points(boundary_ps, com_p, applied_p)
  m = 1
  g = -9.8

  boundary_ws = hcat(map(point_to_wrench, boundary_ps)...)
  gravity_w = m*g*point_to_wrench(com_p)
  applied_w = point_to_wrench(applied_p)

  fa, f_boundary = critical_force(applied_w, gravity_w, boundary_ws)
  fa, f_boundary
end

function get_reaction_points(boundary_ps, boundary_fs)
  inds = find(boundary_fs .>= 1e-3)
  force_points = boundary_ps[inds]

  force_points
end

function solve_and_plot(boundary_ps, com_p, applied_p, attachment_ps)
  applied_f, boundary_fs = critical_force_from_points(boundary_ps, com_p, applied_p)

  reaction_ps = get_reaction_points(boundary_ps, boundary_fs)
  plot_solution(boundary_ps, com_p, applied_p, reaction_ps, attachment_ps)

  applied_f, boundary_fs
end

function plot_solution(boundary_ps, com_p, applied_p, reaction_ps, attachment_ps)
  marker_size = 200

  boundary_vec = hcat(boundary_ps..., boundary_ps[1])
  attachment_vec = hcat(attachment_ps...)

  plot(boundary_vec[1,:]', boundary_vec[2,:]',"-k")
  scatter(applied_p[1,:]', applied_p[2,:]', color="b", s=marker_size)
  scatter(flatten(reaction_ps)[1,:]', flatten(reaction_ps)[2,:]', color="r", s=marker_size)
  scatter(com_p[1,:]', com_p[2,:]', marker="*", color="k", s=marker_size)
  scatter(attachment_vec[1,:]', attachment_vec[2,:]', marker="x", color="k", s = marker_size)
  lgnd = legend(["\$\\mathcal{C}\$", "\$q_a\$", "\$q_r\$" ,"\$q_g\$",
                "\$\\mathcal{A}\$"], loc = 4, ncol = 5, scatterpoints = 1,
                handletextpad=0.1,
                columnspacing=0.1)
  map(x->lgnd[:legendHandles][x][:_sizes] = [100], 2:5)
  #title("Applied, gravity, and reaction forces")
end

function initialize_prior(boundary_ps, resolution, interior_q)
  boundary_vec = flatten(boundary_ps)

  min_p = minimum(boundary_vec, 2)[1:2]
  max_p = maximum(boundary_vec, 2)[1:2]
  grid_size = convert(Array{Int64}, floor((max_p - min_p)/resolution))

  prior = OccupancyGrid(resolution, -min_p, grid_size)


  num_inside = 0
  for ii = 1:size(prior.cells, 1)
    for jj = 1:size(prior.cells, 2)
      ind = OccupancyGridIndex((ii, jj))
      p = to_world(prior, ind)
      if interior_q(p)
        num_inside += 1
      end
    end
  end
  l_occupied = to_log_odds(1 / (resolution^2 * num_inside))

  for ii = 1:size(prior.cells, 1)
    for jj = 1:size(prior.cells, 2)
      ind = OccupancyGridIndex((ii, jj))
      p = to_world(prior, ind)
      if interior_q(p)
        set!(prior, ind, l_occupied)
      end
    end
  end

  prior
end

# currently implemented for a prior independent of the measurement
function update_prior!(prior, interior_q, boundary_ps, applied_p, f_hat, sigma)
  cell_volume = prior.resolution^2

  critical_fs = get_critical_values(boundary_ps, applied_p, prior, interior_q)

  p_prior = map(to_probability, prior.cells)

  p_f_given_com = map(x->normal(x, sigma^2), f_hat - critical_fs)

  p_f_and_com = p_f_given_com .* p_prior

  p_f = sum(p_f_and_com) / cell_volume

  p_com_given_f = p_f_and_com / p_f

  prior.cells[:] = map(to_log_odds, p_com_given_f)

  #figure()
  #critical_fs_normalized = critical_fs / maximum(critical_fs)

  #ps = get_grid_points(prior)

  #z_min = minimum(critical_fs_normalized)
  #z_max = maximum(critical_fs_normalized)
  #b_min = minimum(ps, 2)
  #b_max = maximum(ps, 2)
  #imshow(critical_fs_normalized', cmap="hot", vmin=z_min, vmax=z_max,
             #extent=[b_min[1], b_max[1], b_min[2], b_max[2]],
             #interpolation="nearest", origin="lower")
  #title("critical forces")
end

function get_critical_values(boundary_ps, applied_p, grid, interior_q)
  values = zeros(size(grid.cells))
  for ii = 1:size(grid.cells, 1)
    for jj = 1:size(grid.cells, 2)
      ind = OccupancyGridIndex((ii, jj))
      p = [to_world(grid, ind);0]
      if interior_q(p)
        applied_f, boundary_fs = critical_force_from_points(boundary_ps, p, applied_p)
        values[ii,jj] = applied_f
        if isnan(applied_f)
          values[ii,jj] = 0
        end
      end
    end
  end

  values
end

function plot_attachment_csqmis(boundary_ps, attachment_ps, csqmis)
  boundary_vec = hcat(boundary_ps..., boundary_ps[1])
  attachment_vec = hcat(attachment_ps...)
  plot(boundary_vec[1,:]', boundary_vec[2,:]',"-k")
  #plot(attachment_vec[1,:]', attachment_vec[2,:]', "og", s = csqmis)

  csqmi_scaled = csqmis * 800 / maximum(csqmis)
  scatter(attachment_vec[1,:]', attachment_vec[2,:]', s = csqmi_scaled, color="b")
  scatter(attachment_vec[1,:]', attachment_vec[2,:]', marker="x", color="k", s=200)
  #legend(["boundary", "attachments"], loc = 4)
  #title("csqmi values")
  lgnd = legend(["\$\\mathcal{C}\$", "\$I_{CS}\$", "\$\\mathcal{A}\$"], loc = 4,
                ncol = 3, scatterpoints = 1, handletextpad=0.1, columnspacing=0.1)
  lgnd[:legendHandles][2][:_sizes] = [100]
  lgnd[:legendHandles][3][:_sizes] = [100]
end

include("csqmi.jl")

#end
