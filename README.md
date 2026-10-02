# Sublevel Flood bifiltration: scalable 2-parameter persistent homology

This repository contains the code for the manuscript *Sublevel Flood bifiltration: scalable 2-parameter persistent homology*.
The aim is to provide a way to compute 2-parameter persistent homology of large scale point clouds (up to $10^6$).
To do so, it adapts the [*Flood filtration*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/aba03e6f25decb32bda9c5bf81c58305-Abstract-Conference.html) to the 2-parameter context.

The Python package constructs a ```SimplexTreeMulti``` from the [```multipers```](https://github.com/DavidLapous/multipers) library.

## Installation
`slflood` can be installed via PyPI.

```pip install slflood```

## Usage

The following example computes the sublevel Flood bifiltration of a random point set in $\mathbb{R}^3$ with random values and 100 landmarks, using CUDA if available, then computes and plots its multiparameter module approximation with `multipers`. 

```python
import slflood

import matplotlib.pyplot as plt
import multipers
import torch

pts = torch.rand((10000,3))
fun = torch.rand(10000)
slf = slflood.slflood_bifiltration(pts.cuda(), 100, fun.cuda())
mma = multipers.module_approximation(slf)
mma.plot(box=[[0, 0], [1, 1]])
plt.show()
```

## Citation

