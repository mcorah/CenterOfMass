include("CenterOfMass.jl")

np = 5
thetas = 2*pi * (1/np:1/np:1)
circle_ps = map(theta->[cos(theta);sin(theta);0], thetas)

na = 5
thetas = 2*pi * (1/na:1/na:1)
r_attach = 0.8
attachment_ps = map(theta->r_attach*[cos(theta);sin(theta);0], thetas)

get_com() = r_attach * rand_in_circle()

com = get_com()

lin_prog, lin_fs = critical_force_from_points(circle_ps, com, attachment_ps[1], 1)
iterative, it_fs = critical_force_from_points(circle_ps, com, attachment_ps[1], 1, critical_force_iterative)

println("Lin prog solution: $(lin_prog), iterative solution $(iterative)")
