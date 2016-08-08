function compute_mutual_information(boundary_ps, applied_p, prior, sigma, interior_q)
  critical_fs = get_critical_values(boundary_ps, applied_p, prior, interior_q)
  compute_mutual_information(critical_fs, prior, sigma)
end

function compute_mutual_information(critical_fs, prior::Histogram, sigma)
  nu = sigma^2

  belief = get_data(prior)[:]

  field = critical_fs[:]

  normals = normal_matrix(field, 2*nu)

  i1 = integral_cross(belief, normals)
  i2 = integral_joint(belief, normals)
  i3 = integral_marginals(belief, normals)

  out = -2*log(e, i1) + log(e, i2) + log(e, i3)

  out
end

function compute_mutual_information(critical_fs, prior, sigma)
  nu = sigma^2

  belief = flatten(prior)[:]

  field = flatten(critical_fs)[:]

  normals = normal_matrix(field, 2*nu)

  i1 = integral_cross(belief, normals)
  i2 = integral_joint(belief, normals)
  i3 = integral_marginals(belief, normals)

  out::Float64 = -2*log(e, i1) + log(e, i2) + log(e, i3)

  out
end

function normal_matrix(field, nu)
  n = length(field)
  normals = zeros(n, n)

  nhalf_over_nu = - 0.5 / nu
  over_sqrt2pinu = 1/sqrt(2*pi*nu)

  @inbounds for ii = 1:n
    @fastmath @simd for jj = ii+1:n
      normals[jj,ii] = normal(field[ii] - field[jj], nhalf_over_nu, over_sqrt2pinu)
      normals[ii,jj] = normals[jj,ii]
    end
  end

  @inbounds @fastmath @simd for ii = 1:n
    normals[ii,ii] = normal(0, nhalf_over_nu, over_sqrt2pinu)
  end

  normals
end

function integral_cross(belief, normals)
  out::Float64 = 0.0

  @inbounds for ii = 1:length(belief)
    @fastmath @simd for jj = 1:length(belief)
      out += belief[ii]^2 * belief[jj] * normals[jj,ii]
    end
  end
  out /= length(belief)^2

  out
end

function integral_joint(belief, normals)
  out::Float64 = 0.0

  @fastmath @inbounds @simd for ii = 1:length(belief)
    out += belief[ii]^2 * normals[ii,ii]
  end
  out /= length(belief)

  out
end

function integral_marginals(belief, normals)
  val1::Float64 = sum(belief.^2) / length(belief)

  val2::Float64 = 0.0
  @inbounds for ii = 1:length(belief)
    @fastmath @simd for jj = 1:length(belief)
      val2 += belief[ii] * belief[jj] * normals[jj,ii]
    end
  end
  val2 /= length(belief)^2

  out::Float64 = val1 * val2

  out
end
