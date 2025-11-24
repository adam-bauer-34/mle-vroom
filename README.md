# MLE Vroom

`mle-vroom` is a barebones implementation of maximum likelihood estimation (MLE) in Python. The goal is to provide users flexible fitting functions to carry out MLE, rather than relying on heavy and rigid `scipy` distribution objects. Another useful feature is the ability to incorporate covariates -- i.e., parameters that vary in time, space, or both.

Users further have the option to accelerate their computations using `numba`, but the package is usable with or without it. We provide benchmarks when utilizing `numba` will likely accelerate computation versus when it will slow it down relative to `numpy`-based calculations.

try!
