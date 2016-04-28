using PyPlot
using Convex
using Mapping
#using CenterOfMass
include("CenterOfMass.jl")
plt[:close]("all")

m = 1
g = -9.8
z = [0;0;1]

# set up the occupancy prior
resolution = 0.02
grid = OccupancyGrid(resolution, -[-1,-1], convert(Array{Int64},[2; 2]/resolution))
cells_inside = 0
for ii = 1:size(grid.cells, 1)
  for jj = 1:size(grid.cells, 1)
    p = to_world(grid, OccupancyGridIndex((ii, jj)))
    if norm(p) < 1
      cells_inside += 1
    end
  end
end
l_occupied = to_log_odds(1 / cells_inside)
for ii = 1:size(grid.cells, 1)
  for jj = 1:size(grid.cells, 1)
    ind = OccupancyGridIndex((ii, jj))
    p = to_world(grid, ind)
    if norm(p) < 1
      set!(grid, ind, l_occupied)
    end
  end
end

fig, ax = plt[:subplots](1)
plot_grid(grid, :hot)
#ax[:set_ylim]([-1,1])
#ax[:set_xlim]([-1,1])

np = 100
thetas = 2*pi * (1/np:1/np:1)
circle_ps = map(theta->[sin(theta);cos(theta);0], thetas)

com_p = rand_in_circle()

# all wrenches computed around the origin

applied_p = [1;0;0]

figure()
solve_and_plot(circle_ps, com_p, applied_p)
