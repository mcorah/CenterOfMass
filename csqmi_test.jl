using PyPlot
using Convex
using ProfileView
include("CenterOfMass.jl")

function test_csqmi()
  sigma = 1
  forces = rand(2000)
  prior = rand(2000)
  prior = prior / sum(prior)
  @elapsed @profile compute_mutual_information(forces, prior, sigma)
end

Profile.clear()

times = Float64[]
for ii = 1:100
  time = test_csqmi()
  push!(times, time)
end

println("Mean time: $(mean(times))")

ProfileView.view()
