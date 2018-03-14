using HDF5, JLD

include("CenterOfMass.jl")
include("setup_parameters.jl")

actuator_limit = 4.0
max_robots = 4

prior = initialize_prior(circle_ps, resolution, interior_q, masses)

data_file = "dict_file_$(max_robots)_$(actuator_limit)"

precomputed_normals = Array{CombinationMap}()

if isreadable(data_file)
  println("loading normals")
  @time @load data_file precomputed_normals
else
  println("precomputing normals")

  @time precomputed_normals = generate_combinations_maps(attachment_ws, max_robots, prior,
    circle_ws, actuator_limit, interior_q)

  println("saving normals")
  @time @save data_file precomputed_normals
end

println("succesfully got normals")

data = get_data(prior)[:]
@time for ii = 1:length(precomputed_normals)
  for key = keys(precomputed_normals[ii])
    normals = precomputed_normals[ii][key]
    csqmi = compute_mutual_information(data, normals)
  end
end
