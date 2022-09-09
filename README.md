# Optimization with forward gradients on test functions

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code used for benchmarking [forward gradients](https://arxiv.org/abs/2202.08587) against canonical test functions. This was used in my research work at [Lokad](https://www.lokad.com/home) as part of my end-of-study internship for the [MVA master](https://www.master-mva.com/).

## Usage

We recommend you use a virtual environment. To install, run

```
$ pip install [path-to-forward-repository]
```

Then the data for the experiment can be generated with
```
$ python3 benchmark/make.py
```

And the figures with
```
$ python3 benchmark/figures.py
```

## Acknowledgments

This implementation relies on the following works, which I would like to credit and thank
- the [autograd](https://github.com/HIPS/autograd) library for all things autodifferentiation
- the exhaustive collection of test functions implemented by [Axel Thevenot](https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective)
