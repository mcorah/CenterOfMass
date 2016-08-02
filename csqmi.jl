function compute_mutual_information(boundary_ps, applied_p, prior, sigma, interior_q)
  critical_fs = get_critical_values(boundary_ps, applied_p, prior, interior_q)
  compute_mutual_information(critical_fs, prior, sigma)
end

function compute_mutual_information(critical_fs, prior, sigma, cell_volume)
  nu = sigma^2

  belief = flatten(map(prior->prior.cells, prior))[:]

  field = flatten(critical_fs)[:]

  i1 = integral1(field, belief, cell_volume, nu)
  i2 = integral2(belief, cell_volume, nu)
  i3 = integral3(field, belief, cell_volume, nu)

  out = -2*log(e, i1) + log(e, i2) + log(e, i3)

  out
end

function integral1(field, belief, cell_volume, nu)
  out = 0

  @inbounds @simd for ii = 1:length(belief)
    for jj = 1:length(belief)
      out += belief[ii]^2 * belief[jj] * normal(field[ii] - field[jj], 2*nu)
    end
  end
  out *= cell_volume^2

  out
end

function integral2(belief, cell_volume, nu)
  out = 0

  @inbounds @simd for ii = 1:length(belief)
    out += belief[ii]^2 * normal(0, 2*nu)
  end
  out *= cell_volume

  out
end

function integral3(field, belief, cell_volume, nu)
  val1 = sum(belief.^2) * cell_volume

  val2 = 0
  @inbounds @simd for ii = 1:length(belief)
    for jj = 1:length(belief)
      val2 += belief[ii] * belief[jj] * normal(field[ii] - field[jj], 2*nu)
    end
  end
  val2 *= cell_volume^2

  out = val1 * val2

  out
end
