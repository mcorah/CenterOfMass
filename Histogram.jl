#module

#export Histogram, reset_distribution, ndim, get_data, get_range, generate_prior,
       #test_histogram, weighted_average, size

import Base.size

type Histogram
  range
  data::Array
  Histogram(range, data) = new(map(collect, range), data)
end

Histogram(range) = Histogram(range, generate_prior(range))

function reset_distribution!(x::Histogram)
  x.data = generate_prior(x)
  Void
end

ndim(x::Histogram) = length(x.range)

size(x::Histogram) = size(x.data)

get_data(x::Histogram) = x.data

get_range(x::Histogram) = x.range
get_range(x::Histogram, index) = x.range[index]

function generate_prior(range)
  lengths = map(length, range)
  num_cells = prod(lengths)

  ones(lengths) / num_cells
end
generate_prior(x::Histogram) = generate_prior(x.range)

weighted_average(x::Histogram, dim) = dot(sum_all_dims_but(x.data, dim), collect(x.range[dim]))
weighted_average(h::Histogram) = map(x->weighted_average(h, x), 1:length(h.range))

#from_indices(h::Histogram, inds) = convert(Array{Float64},
  #map(x->h.range[x[1]][x[2]], zip(1:ndim(h), inds)))
function from_indices(h::Histogram, inds::Array{Int64})
  n = length(h.range)
  out = zeros(n)
  for ii = 1:n
    out[ii] = h.range[ii][inds[ii]]
  end
  out
end

###########
# Test code
###########

test_total_probability(x::Histogram) = abs(sum(get_data(x)) - 1) < 1e-3

test_positive(x::Histogram) = all(get_data(x) .>= 0)

function test_histogram()
  h = Histogram((1:0.5:2, 1:0.5:2))

  assert(test_total_probability(h))
  assert(test_positive(h))
end

function sum_all_dims_but{N,T}(data::Array{N,T}, dim)
  all_but = filter(x->x!=dim, 1:T)
  sum(data, all_but)[:]
end

#end
