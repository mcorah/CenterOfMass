using SparseArrays

# I am using a separate type here because I want to preserve my ability to load
# Histogram types using JLD2. Oh boy...

export SparseHistogram, to_sparse, set_threshold!, drop_below_threshold!,
  sparsity

mutable struct SparseHistogram{RangeType <: Real}
  range::Tuple{Vector{RangeType},Vector{RangeType}}
  data::SparseMatrixCSC{Float64, Int64}

  buffer::SparseMatrixCSC{Float64, Int64}

  # Threshold defines the minimum value to keep in the filter
  threshold::Float64

  # The default constructor specializes for when we can infer the type of the
  # range from the signature. (Alternatively, see the outer constructor).
  function SparseHistogram(range::Tuple{Vector{D}, Vector{D}}, data;
                          threshold = 0.0) where D <: Real
    new{D}(range, data, spzeros(size(data)...), threshold)
  end
end

# Outer constructor that defers determination of the histogram type until after
# pulling the ranges
function SparseHistogram(range, data; kwargs...)
  SparseHistogram(map(collect, range), data; kwargs...)
end

function SparseHistogram(hist::Histogram; threshold=0.0)
  SparseHistogram(hist.range,
                  to_sparse(hist.data, threshold=threshold),
                  threshold=threshold)
end

# Copy constructor. Note that this only duplicates the data.
# We assume that nobody is crazy enough to modify the range.
duplicate(x::SparseHistogram) = SparseHistogram(x)
function SparseHistogram(x::SparseHistogram)
  SparseHistogram(get_range(x), sparse(get_data(x)), threshold = x.threshold)
end

function to_sparse(matrix::Matrix; threshold=0.0)
  indices = findall(x -> x>threshold, matrix.data)
  I = map(x->x.I[1], indices)
  J = map(x->x.I[2], indices)
  V = map(x->matrix.data[x], indices)
  sparse(I, J, V)
end

set_threshold!(x::SparseHistogram; threshold) = (x.threshold = threshold)

get_values(x::SparseHistogram) = nonzeros(x.data)

# Remove values from the histogram below a given threshold
#
# Trim resizes the matrix. We will generally continue to operate in place so we
# will end up using the empty space
function drop_below_threshold!(x::SparseHistogram;
                               threshold = x.threshold,
                               trim = false
                              )
  droptol!(x.data, theshold, trim = trim)
end

sparsity(x::SparseHistogram) =  1.0 - nnz(get_data(x)) / length(get_data(x))
