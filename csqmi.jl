function compute_mutual_information(boundary_ps, applied_p, prior, sigma, interior_q)
  critical_fs = get_critical_values(boundary_ps, applied_p, prior, interior_q)
  p_com = map(to_probability, prior.cells)
  cell_volume = prior.resolution^2

  i1 = integral1(critical_fs, p_com, cell_volume, sigma)
  i2 = integral2(p_com, cell_volume, sigma)
  i3 = integral3(critical_fs, p_com, cell_volume, sigma)

  out = -2*log(e, i1) + log(e, i2) + log(e, i3)

  out
end

function integral1(critical_fs, p_com, cell_volume, sigma)
  out = 0

  for ii = 1:length(p_com)
    for jj = 1:length(p_com)
      out += p_com[ii]^2 * p_com[jj] * normal(critical_fs[ii] - critical_fs[jj], 2*sigma^2)
    end
  end
  out /= cell_volume^2

  out
end

function integral2(p_com, cell_volume, sigma)
  out = 0

  for ii = 1:length(p_com)
    for jj = 1:length(p_com)
      out += p_com[ii]^2 * normal(0, 2*sigma^2)
    end
  end
  out /= cell_volume^2

  out
end

function integral3(critical_fs, p_com, cell_volume, sigma)
  val1 = sum(p_com.^2) / cell_volume^2

  val2 = 0
  for ii = 1:length(p_com)
    for jj = 1:length(p_com)
      val2 += p_com[ii] * p_com[jj] * normal(critical_fs[ii] - critical_fs[jj], 2*sigma^2)
    end
  end
  val2 /= cell_volume^2

  out = val1 * val2

  out
end
