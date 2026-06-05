# Solved Task Videos

This folder contains MP4 visualizations of solved ARC-AGI tasks from the full-dataset evaluation. The videos were generated for the six evaluated NCA model variants and are organized by model and dataset split.

The main model folders are:

* `Result_basline`
* `Result_fixed-filter`
* `Result_learnable-filter`
* `Result_color-permutation`
* `Result_rotating-filter`
* `Result_combined`

Each model folder may contain videos for the following dataset splits:

* `train1`: ARC-AGI-1 training set
* `eval1`: ARC-AGI-1 evaluation set
* `train2`: ARC-AGI-2 training set
* `eval2`: ARC-AGI-2 evaluation set

## Video content

Each video shows the development of the model prediction for one solved test task. The top row shows the available training input-target examples for the task. The bottom row shows the test input, the test target, the final model prediction, and the NCA trace over update steps.

The videos are intended as qualitative visualizations of how the NCA prediction develops during the iterative update process.

## Notes

Only tasks marked as solved by exact match are included. A task is considered solved when the final prediction matches the target output exactly according to the stored full-dataset evaluation results.

These videos are provided to make the model behavior easier to inspect visually, especially for comparing how different NCA variants solve tasks across ARC-AGI-1 and ARC-AGI-2.
