# Maximum feasibility of 0.4769921436588103 for [1,3,5,8]
# Mean feasibility of 0.3221046443268665

include("CenterOfMass.jl")
include("setup_parameters.jl")

mass_resolution = 0.1
masses = collect(min_mass:mass_resolution:max_mass)
resolution = 0.1

max_robots = 4
actuator_limit = 6.0

#########################
# Myopic feasibility rate
#########################
boundary_vec = flatten(circle_ps)
min_p = minimum(boundary_vec, 2)[1:2]
max_p = maximum(boundary_vec, 2)[1:2]
position_ranges = map(x->x[1]:resolution:x[2], zip(min_p, max_p))
ranges = (position_ranges..., masses)

robots_feasible(robots, theta) =
  lifting_feasibility(attachment_ws[robots],
                      theta[3]*g*point_to_wrench(theta[1:2]),
                      actuator_limit)

maximum_feasibility_rate = 0
maximum_feasibility_combination = zeros(Int64,0)

good_inds = zeros(Bool,map(length,ranges))
for (ii,x) = enumerate(ranges[1])
  for (jj,y) = enumerate(ranges[2])
    for (kk,m) = enumerate(ranges[3])
      Convex.clearmemory()
      if interior_q([x;y])
        gravity_w = m*g*point_to_wrench([x;y])
        feasible = check_feasible_configuration(Array{Array{Float64}}(0), attachment_ws,
          gravity_w, actuator_limit, max_robots)

        if feasible
          good_inds[ii,jj,kk] = true
        end
      end
    end
  end
end
num_good = sum(good_inds)

mean_feasibility = 0.0
cs = combinations(1:length(attachment_ws), max_robots)
for (ii, c) = enumerate(cs)
  num_feasible = 0
  for (jj,x) = enumerate(ranges[1])
    for (kk,y) = enumerate(ranges[2])
      for (ll,m) = enumerate(ranges[3])
        Convex.clearmemory()
        theta = [x,y,m]
        if good_inds[jj,kk,ll]
          num_feasible += robots_feasible(c, theta)
        end
      end
    end
  end
  feasibility = num_feasible / num_good
  mean_feasibility += feasibility
  if feasibility > maximum_feasibility_rate
    maximum_feasibility_combination = c
    maximum_feasibility_rate = feasibility
    println("New best: $(feasibility)")
  end
  println("Finished $(ii) of $(length(cs)), best: $(maximum_feasibility_rate)")
end
mean_feasibility /= length(cs)

println("Maximum feasibility of $(maximum_feasibility_rate) for $(maximum_feasibility_combination)")
println("Mean feasibility of $(mean_feasibility)")
