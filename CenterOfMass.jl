#module CenterOfMass

function critical_force(wrench_applied, wrench_offset, W_boundary)
  fa = Variable(1)
  Fr = Variable(size(W_boundary, 2))

  stability = wrench_applied * fa +
              W_boundary * Fr +
              wrench_offset == 0
  #feasibility = [fa >= 0; Fr .>= 0]
  feasibility = [fa >= 0; Fr >= 0]
  println("$(typeof(stability)) $(typeof(feasibility))")

  problem = maximize(fa, [stability; feasibility])

  solve!(problem)

  #if problem.status != :Optimal
    #error("Solution not optimal: $(problem.status)")
  #end

  problem.optval, evaluate(Fr)
end

point_to_wrench(p) = [1;cross(p,z)[1:2]]

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

function solve_and_plot(boundary_ps, com_p, applied_p)
  applied_f, boundary_fs = critical_force_from_points(boundary_ps, com_p, applied_p)

  reaction_ps = get_reaction_points(boundary_ps, boundary_fs)
  plot_solution(boundary_ps, com_p, applied_p, reaction_ps)

  applied_f, boundary_fs
end

function plot_solution(boundary_ps, com_p, applied_p, reaction_ps)
  plot(flatten(boundary_ps)[1,:]', flatten(boundary_ps)[2,:]',"-b")
  plot(com_p[1,:]', com_p[2,:]', "og")
  plot(applied_p[1,:]', applied_p[2,:]', "oy")
  plot(flatten(reaction_ps)[1,:]', flatten(reaction_ps)[2,:]',"or")
  legend(["boundary", "com", "applied", "reaction"])
end

function initialize_prior(boundary_ps, resolution, interior_q)

end

#end
