import glob
import os
import jax
from tqdm import tqdm

from .trainer import train_one_task
from .config import NUM_TRAIN_ITERS

def main():
    """Run training over all ARC tasks for the configured number of runs."""
    task_root = "ARC-AGI-master/data/training"
    task_paths = sorted(glob.glob(os.path.join(task_root, "*.json")))

    rng_key = jax.random.PRNGKey(0)
    num_runs = 1
    print("Available JAX devices:", jax.devices())

    for run_index in range(1, num_runs + 1):
        print(f"\n===== Run {run_index}/{num_runs} =====")

        results_dir = "run_results_4LF_pad_train1"
        os.makedirs(results_dir, exist_ok=True)

        results_csv = os.path.join(results_dir, f"run_{run_index:02d}.csv")
        with open(results_csv, "w") as file:
            file.write(
                "task,final_acc,final_exact_match,"
                "final_foreground_acc,first_solve_step,task_runtime_seconds\n"
            )

        for task_path in tqdm(
            task_paths,
            desc=f"Run {run_index} | Training tasks",
            ncols=80,
        ):
            rng_key, task_key = jax.random.split(rng_key)
            train_one_task(
                task_path=task_path,
                rng_key=task_key,
                num_iters=NUM_TRAIN_ITERS,
                run_id=run_index,
            )


if __name__ == "__main__":
    main()
