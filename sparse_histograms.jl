using SparseArrays

# I am using a separate type here because I want to preserve my ability to load
# Histogram types using JLD2. Oh boy...

export SparseHistogram, to_sparse, set_threshold!, drop_below_threshold!,
  sparsity

# Construct a sparse vector based on the input data
function to_sparse(x; threshold)
  ret = spzeros(length(x))
  for (ii, val) in enumerate(x)
    if val > threshold
      push!(ret.nzind, ii)
      push!(ret.nzval, val)
    end
  end

  ret
end

mutable struct SparseHistogram{RangeType <: Real}
  range::Tuple{Vector{RangeType},Vector{RangeType}}
  data::SparseVector{Float64, Int64}

  buffer::SparseVector{Float64, Int64}

  # Threshold defines the minimum value to keep in the filter
  threshold::Float64

  # The default constructor specializes for when we can infer the type of the
  # range from the signature. (Alternatively, see the outer constructor).
  function SparseHistogram(range::Tuple{Vector{D}, Vector{D}},
                           data::SparseVector;
                           threshold) where D <: Real
    new{D}(range, data, spzeros(length(data)), threshold)
  end
end

AnyHistogram = Union{SparseHistogram, Histogram}

# Constructor for when the input data is not sparse
function SparseHistogram(range, data; threshold)
  SparseHistogram(range, to_sparse(data, threshold=threshold);
                  threshold=threshold)
end

# Outer constructor that defers determination of the histogram type until after
# pulling the ranges
function SparseHistogram(range, data::SparseVector; threshold)
  SparseHistogram(map(collect, range), data; threshold=threshold)
end

# Copy constructor. Note that this only duplicates the data.
# We assume that nobody is crazy enough to modify the range.
duplicate(x::SparseHistogram) = SparseHistogram(x)
# Note, this will fail if no threshold is given when converting a Histogram
function SparseHistogram(x::AnyHistogram; threshold = x.threshold)
  SparseHistogram(get_range(x), to_sparse(get_values(x), threshold=threshold),
                  threshold=threshold)
end

set_threshold!(x::SparseHistogram; threshold) = (x.threshold = threshold)

# Returns the data matrix
size(x::SparseHistogram) = map(length, x.range)
get_data(x::SparseHistogram) = reshape(x.data, size(x))

#
# Sparse helper methods
#

function clear!(x::SparseVector)
  resize!(x.nzind, 0)
  resize!(x.nzval, 0)
end

function copy_sparse!(out, in, threshold)
  nzind = in.nzind
  nzval = in.nzval
  n = length(nzind)

  out_ind = out.nzind
  out_val = out.nzval

  # Presume that we will copy most of the input
  resize!(out_ind, n)
  resize!(out_val, n)

  # Copy values above the threshold
  end_ind = 1
  @inbounds for ii in 1:n
    if nzval[ii] > threshold
      out_ind[end_ind] = nzind[ii]
      out_val[end_ind] = nzval[ii]

      end_ind += 1
    end
  end

  resize!(out_ind, end_ind-1)
  resize!(out_val, end_ind-1)

  nothing
end

# Copies filter data, keeping nonzeros
function copy_filter!(x::SparseHistogram; out::SparseHistogram)
  copy_sparse!(out.data, x.data, x.threshold)
  clear!(out.buffer)

  out.threshold = x.threshold
end

# Remove values from the histogram below a given threshold
#
# Trim resizes the matrix. We will generally continue to operate in place so we
# will end up using the empty space
function drop_below_threshold!(x::SparseHistogram;
                               threshold = x.threshold,
                               trim = true
                              )
  droptol!(x.data, threshold, trim = trim)
  droptol!(x.buffer, threshold, trim = trim)
end
drop_below_threshold!(x::Histogram; kwargs...) = nothing

sparsity(x::SparseHistogram) =  1.0 - nnz(get_values(x)) / length(get_values(x))
