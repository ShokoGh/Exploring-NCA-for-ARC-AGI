# Notice

This repository contains code developed for a master's thesis project on Neural Cellular Automata for ARC-AGI tasks.

Parts of the implementation are adapted from and inspired by Google Research's `self-organising-systems` repository, especially the Growing Neural Cellular Automata notebook:

https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb

Related article:

**Growing Neural Cellular Automata: Differentiable Model of Morphogenesis**
By:
Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin.
Distill, 2020.
https://distill.pub/2020/growing-ca/

Original Google Research code:

```text
Copyright 2020 Google LLC
Licensed under the Apache License, Version 2.0.
```

The code in this repository has been modified and extended for ARC-AGI experiments. The main modifications include:

* ARC grid and palette representation.
* ARC input-output task setup instead of image growth from a seed cell.
* Padding support for variable-sized ARC grids.
* Spatial masks and target masks.
* ARC-specific loss, accuracy, evaluation, trace saving, and logging.
* Learnable perception filters.
* Extended fixed perception filters.
* Color-permutation augmentation.
* Rotation-aware training.
* Dihedral augmentation using rotations and flips.
* Mixed-rule NCA variants with selector mechanisms.

This repository is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.
