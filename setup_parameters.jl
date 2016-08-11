min_mass = 0.6
max_mass = 1.4
g = -9.8
#z = [0;0;1]
sigma = 1

mass_resolution = 0.2
masses = collect(min_mass:mass_resolution:max_mass)

np = 20
thetas = 2*pi * (1/np:1/np:1)
circle_ps = map(theta->[cos(theta);sin(theta);0], thetas)

# move to separate attachment points
na = 10
thetas = 2*pi * (1/na:1/na:1)
r_attach = 0.8
attachment_ps = map(theta->r_attach*[cos(theta);sin(theta);0], thetas)

# set up the occupancy prior
resolution = 0.1

interior_q(x) = norm(x) <= 1 - resolution * 3/2