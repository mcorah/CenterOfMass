module Histograms

using LinearAlgebra

export Histogram, duplicate, reset_distribution, ndim, get_data, get_values,
  get_buffer, swap_buffer!, copy_filter!, get_range, generate_prior,
  test_histogram, weighted_average, size

import Base.size

mutable struct Histogram{RangeType <: Real}
  range::Tuple{Vector{RangeType},Vector{RangeType}}
  data::Array{Float64, 2}

  buffer::Array{Float64, 2}

  # The default constructor specializes for when we can infer the type of the
  # range from the signature. (Alternatively, see the outer constructor).
  function Histogram(range::Tuple{Vector{D}, Vector{D}}, data) where D <: Real
    new{D}(range, data, Array{Float64}(undef, size(data)))
  end
end

include("sparse_histograms.jl")

# Outer constructor that defers determination of the histogram type until after
# pulling the ranges
Histogram(range, data) = Histogram(map(collect, range), data)

Histogram(range) = Histogram(range, generate_prior(range))

function reset_distribution!(x::Histogram)
  x.data = generate_prior(x)
  Void
end

# Copy constructor. Note that this only duplicates the data.
# We assume that nobody is crazy enough to modify the range.
duplicate(x::Histogram) = Histogram(x)
function Histogram(x::Histogram)
  Histogram(get_range(x), Array(get_data(x)))
end

ndim(x::AnyHistogram) = length(x.range)

size(x::Histogram) = size(x.data)

# Returns the data matrix
get_data(x::Histogram) = x.data
# Returns the data values themselves (as in for sparse matrices)
get_values(x::AnyHistogram) = x.data
get_buffer(x::AnyHistogram) = x.buffer

function swap_buffer!(x::AnyHistogram)
  old_data = x.data

  x.data = x.buffer
  x.buffer = old_data
end

# Copies filter data (exclusively)
function copy_filter!(x::Histogram; out::Histogram)
  out.data .= x.data
end

get_range(x::AnyHistogram) = x.range
get_range(x::AnyHistogram, index) = x.range[index]

function generate_prior(range)
  lengths = map(length, range)
  num_cells = prod(lengths)

  ones(lengths) / num_cells
end
generate_prior(x::Histogram) = generate_prior(x.range)

weighted_average(x::Histogram, dim) = dot(sum_all_dims_but(x.data, dim), collect(x.range[dim]))
weighted_average(h::Histogram) = map(x->weighted_average(h, x), 1:length(h.range))

# Maps indices in the ranges to the space they represent
function from_indices(h::Histogram, inds)
  n = length(h.range)
  out = Array{Float64}(undef, n)
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

function sum_all_dims_but(data::Array{T,N}, dim) where {T, N}
  all_but = filter(x->x!=dim, 1:N)
  sum(data, all_but)[:]
end

end
