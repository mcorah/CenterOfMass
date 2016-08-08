function compute_mutual_information(boundary_ps, applied_p, prior, sigma, interior_q)
  critical_fs = get_critical_values(boundary_ps, applied_p, prior, interior_q)
  compute_mutual_information(critical_fs, prior, sigma)
end

function compute_mutual_information{T <: Map}(critical_fs, prior::Array{T}, sigma, cell_volume)
  nu = sigma^2

  belief = flatten(map(prior->prior.cells, prior))[:]

  field = flatten(critical_fs)[:]

  normals = normal_matrix(field, 2*nu)

  i1 = integral_cross(field, belief, cell_volume, normals)
  i2 = integral_joint(belief, cell_volume, normals)
  i3 = integral_marginals(field, belief, cell_volume, normals)

  out = -2*log(e, i1) + log(e, i2) + log(e, i3)

  out
end

function compute_mutual_information(critical_fs, prior, sigma, cell_volume)
  nu = sigma^2

  belief = flatten(prior)[:]

  field = flatten(critical_fs)[:]

  normals = normal_matrix(field, 2*nu)

  i1 = integral_cross(field, belief, cell_volume, normals)
  i2 = integral_joint(belief, cell_volume, normals)
  i3 = integral_marginals(field, belief, cell_volume, normals)

  out::Float64 = -2*log(e, i1) + log(e, i2) + log(e, i3)

  out
end

function normal_matrix(field, nu)
  n = length(field)
  normals = zeros(n, n)
  @inbounds for ii = 1:n
    @fastmath @simd for jj = ii+1:n
      normals[jj,ii] = normal(field[ii] - field[jj], nu)
      normals[ii,jj] = normals[jj,ii]
    end
  end

  @inbounds @fastmath @simd for ii = 1:n
    normals[ii,ii] = normal(0, nu)
  end

  normals
end

function integral_cross(field, belief, cell_volume::Float64, normals)
  out::Float64 = 0.0

  @inbounds for ii = 1:length(belief)
    @fastmath @simd for jj = 1:length(belief)
      out += belief[ii]^2 * belief[jj] * normals[ii,jj]
    end
  end
  out *= cell_volume^2

  out
end

function integral_joint(belief, cell_volume, normals)
  out::Float64 = 0.0

  @fastmath @inbounds @simd for ii = 1:length(belief)
    out += belief[ii]^2 * normals[1,1]
  end
  out *= cell_volume

  out
end

function integral_marginals(field, belief, cell_volume, normals)
  val1::Float64 = sum(belief.^2) * cell_volume

  val2::Float64 = 0.0
  @inbounds for ii = 1:length(belief)
    @fastmath @simd for jj = 1:length(belief)
      val2 += belief[ii] * belief[jj] * normals[ii,jj]
    end
  end
  val2 *= cell_volume^2

  out::Float64 = val1 * val2

  out
end
