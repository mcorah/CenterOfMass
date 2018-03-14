using PyPlot
using Convex
using ProfileView
include("CenterOfMass.jl")

sigma = 1.0
forces = rand(2205)
normals = normal_matrix(forces, sigma^2)

function test_csqmi()
  prior = rand(2205)
  prior = prior / sum(prior)
  #@elapsed @profile compute_mutual_information(forces, prior, sigma)
  @elapsed @profile compute_mutual_information(prior, normals)
end

Profile.clear()

times = Float64[]
for ii = 1:100
  time = test_csqmi()
  push!(times, time)
end

println("Mean time: $(mean(times))")

ProfileView.view()
