# Center of Mass

This package provides a simulation of an active sensing method for teams of
aerial robots to localize the center of mass of objects while they are still on
the ground.

Actions are generally selected to maximize a specialized measure of information
gain:
Cauchy-Schwarz Quadratic Mutual Information (CSQMI).

Estimation is based on a histogram filter, assuming Guassian noise and allowing
for limits on applied force.

## References

If you use this package in published work, pleace consider citing:
```
@inproceedings{corah2017icra,
  title={Active estimation of mass properties for safe cooperative lifting},
  author={Corah, Micah and Michael, Nathan},
  booktitle={Proc. of the {IEEE} Intl. Conf. on Robot. and Autom.},
  year={2017},
  month=may,
  address={Singapore},
}
```
