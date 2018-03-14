using PyPlot

function circle(p; radius=1, scale=1, dth=0.1)
  scale*hcat(map(x->p+[radius*cos(x);radius*sin(x)], 0:dth:(2*pi)+dth)...)
end

function plot_circle(p; scale=1, radius=1, color="k", linestyle="-", linewidth=1.0)
  c = circle(p[1:2], scale=scale, radius=radius)
  plot3D(c[1,:][:], c[2,:][:], p[3]*ones(size(c,2)), color=color, linestyle=linestyle, linewidth=linewidth)
end

so2(theta) = [cos(theta) -sin(theta); sin(theta) cos(theta)]

function fill3d(ps; color = "k", alpha=1.0)
  ax = gca()
  tuples = map(x->tuple(ps[:,x]...), 1:size(ps,2))
  poly = art3D[:Poly3DCollection](Array[tuples])
  poly[:set_color](color)
  poly[:set_alpha](alpha)

  ax[:add_collection3d](poly)
end

function plot_quadrotor(p; scale=1, color="k", linewidth=2.0, theta=0.0, alpha=1.0)
  r = 0.6
  rot = so2(theta)
  circle_ps = rot*scale*[[1;0] [0;1] [-1;0] [0;-1]]

  p3 = 0.0
  if length(p) == 3
    p3 = p[3]
  end

  p2 = p[1:2]

  l1 = [p2+scale*rot[:,1] p2-scale*rot[:,1]]
  l2 = [p2+scale*rot[:,2] p2-scale*rot[:,2]]
  plot3D(l1[1,:][:], l1[2,:][:], p3*ones(2), color=color, linewidth=linewidth,
          alpha=alpha)
  plot3D(l2[1,:][:], l2[2,:][:], p3*ones(2), color=color, linewidth=linewidth,
          alpha = alpha)

  for ii = 1:size(circle_ps,2)
    c = circle(p2+circle_ps[:,ii], radius=scale*r)
    plot3D(c[1,:][:], c[2,:][:], p3*ones(size(c,2)), color=color,
            linestyle="--", linewidth=linewidth, alpha=alpha)
    #fill(c[1,:][:], c[2,:][:], p3*ones(size(c,2)), color=color, alpha=0.5)
    fill3d([c;p3*ones(1,size(c,2))], color=color, alpha=alpha*0.5)
  end
end

# iterations in 2, trials in 1
function standard_error(data, color = "k")
  stds = std(data, 1)[:] / sqrt(size(data,1))
  means = mean(data,1)[:]

  indices = collect(1:size(data,2))
  fill([indices;reverse(indices)], [means+stds;reverse(means-stds)],
    color = color, alpha=0.2, linewidth=0.0)
end
