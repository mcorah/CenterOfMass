function compute_mutual_information(boundary_ps, applied_p, prior, sigma, interior_q)
  critical_fs = get_critical_values(boundary_ps, applied_p, prior, interior_q)
  compute_mutual_information(critical_fs, prior, sigma)
end

function compute_mutual_information(critical_fs, prior, sigma, cell_volume)
  belief = flatten(map(prior->prior.cells, prior))[:]

  field = flatten(critical_fs)[:]

  i1 = integral1(field, belief, cell_volume, sigma)
  i2 = integral2(belief, cell_volume, sigma)
  i3 = integral3(field, belief, cell_volume, sigma)

  out = -2*log(e, i1) + log(e, i2) + log(e, i3)

  out
end

function integral1(field, belief, cell_volume, sigma)
  out = 0

  for ii = 1:length(belief)
    for jj = 1:length(belief)
      out += belief[ii]^2 * belief[jj] * normal(field[ii] - field[jj], 2*sigma^2)
    end
  end
  out *= cell_volume^2

  out
end

function integral2(belief, cell_volume, sigma)
  out = 0

  for ii = 1:length(belief)
    out += belief[ii]^2 * normal(0, 2*sigma^2)
  end
  out *= cell_volume

  out
end

function integral3(field, belief, cell_volume, sigma)
  val1 = sum(belief.^2) / cell_volume

  val2 = 0
  for ii = 1:length(belief)
    for jj = 1:length(belief)
      val2 += belief[ii] * belief[jj] * normal(field[ii] - field[jj], 2*sigma^2)
    end
  end
  val2 *= cell_volume^2

  out = val1 * val2

  out
end
