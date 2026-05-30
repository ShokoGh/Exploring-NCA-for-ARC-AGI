# Exploring-NCA-for-ARC-AGI
# Neural Cellular Automata Experiments for ARC-AGI_1 and ARC-AGI_2

This repository contains code and selected result visualizations from a master's thesis project investigating Neural Cellular Automata (NCA) models for ARC-AGI tasks.

The project investigates Neural Cellular Automata (NCA) models for solving ARC-AGI tasks in a task-wise few-shot setting. The experiments include baseline NCA models, learnable perception filters, extended fixed filters, color-permutation training, rotation-aware variants, dihedral augmentation, and mixed-rule selector models.

## Repository Structure

### Padding-based package variants

The following folders contain padding-based NCA variants. These variants are structured as Python packages and are intended for experiments where ARC tasks may have different input and output sizes. These versions can also be used for full-dataset evaluation.

* `NCA_Baseline_PAD/`
  Baseline padded NCA implementation.

* `NCA_Extended_Fixed_Filter_PAD/`
  NCA with extended fixed perception filters.

* `NCA_Learnable_Filter_PAD/`
  NCA with learnable depthwise perception filters.

* `NCA_Color_Permutation_PAD/`
  NCA with ARC-specific color-permutation augmentation.

* `NCA_Rotated_Learnable_Filter_PAD/`
  NCA with rotated learnable perception filters and rotation-conditioning.

* `NCA_Combined_PAD/`
  Combined color-permutation and rotation-aware variant.

Example command for running a package variant:

```bash
python -m NCA_Color_Permutation_PAD.main
python -m NCA_Baseline_PAD.main
python -m NCA_Learnable_Filter_PAD.main
python -m NCA_Rotated_Learnable_Filter_PAD.main
```

### Non-padding sampled-task variants

The following scripts are non-padding variants. They were used mainly for sampled ARC-AGI-1 experiments where the input and output grids have the same size.

* `MNCA_selector_cell.py`
  Mixed-rule NCA with a cell-level selector.

* `MNCA_selector_task.py`
  Mixed-rule NCA with a task-level selector.

* `NCA_dihedral_1loss.py`
  Dihedral augmentation variant using a one-loss setup.

* `NCA_dihedral_2loss.py`
  Dihedral augmentation variant using a two-loss setup.

These scripts can be run directly, for example:

```bash
python MNCA_selector_cell.py
python MNCA_selector_task.py
python NCA_dihedral_1loss.py
python NCA_dihedral_2loss.py
```

### Visualization tool

* `GUI.py`
  A Tkinter-based GUI tool for visualizing generated IO files and trace files.

The GUI expects generated JSON files from the experiment scripts. The following two lines in `GUI.py` can be changed depending on which model output folders should be visualized:

```python
ROOT_TAG_IO = "json_g_io_MNCA_selector"
ROOT_TAG_TR = "json_g_traces_MNCA_selector"
```

For example, if another experiment produces different output folders, these two values should be updated to match those folders before running the GUI.

Run the GUI with:

```bash
python GUI.py
```

## Results

The `Results/` folder contains selected visual result files from the experiments. These include examples of:

* solved tasks,
* almost solved tasks,
* reasoning pitfalls,
* input/target/prediction comparisons.

These files are included to make it easier to inspect model behavior qualitatively, not only through numerical scores.

## Datasets

The ARC-AGI datasets are not included in this repository.

They must be downloaded separately from the official repositories:

* ARC-AGI-1: https://github.com/fchollet/ARC-AGI/tree/master
* ARC-AGI-2: https://github.com/arcprize/ARC-AGI-2/tree/main

After downloading the datasets, place the task JSON files in the local data folders expected by the scripts. Some scripts currently use paths such as:

 task_root = "data/test"

```
task_root = "ARC-AGI-master/data/training"
```

These paths can be changed inside the scripts depending on the local dataset location.

## Requirements

The code uses Python with libraries such as:

* JAX
* Flax
* Optax
* NumPy
* Matplotlib
* tqdm
* Pillow
* Tkinter

JAX installation depends on the local CPU/GPU environment, so users should install the JAX version appropriate for their own system.

## Attribution

Parts of this project are adapted from and inspired by Google Research's Growing Neural Cellular Automata implementation and the related Distill article:

**Growing Neural Cellular Automata: Differentiable Model of Morphogenesis**
Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin.
Distill, 2020.
https://distill.pub/2020/growing-ca/

Reference implementation:

https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb

See `NOTICE.md` for more details.

## License

This repository is licensed under the Apache License, Version 2.0.
