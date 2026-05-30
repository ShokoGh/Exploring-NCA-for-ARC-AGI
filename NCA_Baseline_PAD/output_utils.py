import json
import os
import csv

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .palette_utils import snap_to_palette_rgb
from .data_utils import extract_rgb_image,extract_rgb_images,extract_trace_rgb

def trace_to_frame_list(trace_rgb: np.ndarray) -> list:
    """
    Convert a trace array of shape (T, H, W, 3) into a list of RGB frames.
    """
    trace_rgb = np.asarray(trace_rgb)
    trace_rgb = np.clip(trace_rgb, 0.0, 1.0)
    return [trace_rgb[t] for t in range(trace_rgb.shape[0])]

def save_trace_json(rgb_frames: list, output_path: str) -> None:
    """
    Save a sequence of RGB frames to a JSON file.

    Each frame is stored as an (H, W, 3) array, and the full sequence is written
    as a JSON list.

    Args:
        rgb_frames: List of RGB arrays with shape (H, W, 3).
        output_path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        file.write("[\n")
        for index, frame in enumerate(rgb_frames):
            json.dump(frame.tolist(), file)
            file.write(",\n" if index < len(rgb_frames) - 1 else "\n")
        file.write("]\n")


def save_io_bundle_json(
    input_images: list,
    target_images: list,
    predicted_images: list,
    output_path: str,
) -> None:
    """
    Save input, target, and predicted RGB images to a JSON file.

    Args:
        input_images: List of input RGB images.
        target_images: List of target RGB images.
        predicted_images: List of predicted RGB images.
        output_path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bundle = {
        "input": [image.tolist() for image in input_images],
        "target": [image.tolist() for image in target_images],
        "pred": [image.tolist() for image in predicted_images],
    }

    with open(output_path, "w") as file:
        json.dump(bundle, file)


def save_training_outputs(
    iteration_index: int,
    pair_index: int,
    num_unroll_steps: int,
    input_state,
    target_state,
    final_state,
    state_trace,
    trace_output_dir: str,
    io_output_dir: str,
) -> None:
    """
    Save training traces and input/target/prediction bundles for one training pair.
    """

    snapped_trace = snap_to_palette_rgb(state_trace)
    trace_rgb = extract_trace_rgb(snapped_trace)
    _, pool_size, _, _, _ = trace_rgb.shape

    for pool_index in range(pool_size):
        rgb_frames = trace_to_frame_list(trace_rgb[:, pool_index])
        filename = f"iter_{iteration_index:04d}_pair{pair_index}_P{pool_index}_{num_unroll_steps}x.json"
        save_trace_json(rgb_frames, os.path.join(trace_output_dir, filename))

    input_rgb = extract_rgb_image(input_state)
    target_rgb = extract_rgb_image(target_state)
    predicted_rgb = extract_rgb_images(snap_to_palette_rgb(final_state))

    input_images = [input_rgb] * pool_size
    target_images = [target_rgb] * pool_size

    io_filename = f"iter_{iteration_index:04d}_pair{pair_index}_{num_unroll_steps}x_io.json"
    save_io_bundle_json(
        input_images,
        target_images,
        predicted_rgb,
        os.path.join(io_output_dir, io_filename),
    )

def save_test_outputs(
    input_state_nhwc: jnp.ndarray,
    target_state_nhwc: jnp.ndarray,
    final_state: jnp.ndarray,
    state_trace: jnp.ndarray,
    trace_test_dir: str,
    io_test_dir: str,
) -> None:
    """
    Save test trace frames and final input/target/prediction bundle.
    """


    snapped_trace = snap_to_palette_rgb(state_trace)
    trace_rgb = extract_trace_rgb(snapped_trace)[:, 0]
    rgb_frames = trace_to_frame_list(trace_rgb)
    save_trace_json(rgb_frames, os.path.join(trace_test_dir, "iter_final_test.json"))

    input_rgb = extract_rgb_image(input_state_nhwc)
    target_rgb = extract_rgb_image(target_state_nhwc)

    predicted_rgb = extract_rgb_image(snap_to_palette_rgb(final_state))

    save_io_bundle_json(
        [input_rgb],
        [target_rgb],
        [predicted_rgb],
        os.path.join(io_test_dir, "iter_final_test_io.json"),
    )

def save_train_metrics_csv(
    lr_log_csv: str,
    iteration_index: int,
    current_lr: float,
    loss: jnp.ndarray,
    metrics: dict,
) -> None:
    """
    Append one training metrics row to the training CSV log.
    """
    with open(lr_log_csv, "a") as file:
        file.write(
            f"{iteration_index},{current_lr:.8e},{float(loss):.8f},"
            f"{metrics['log_loss']:.8f},{metrics['mean_accuracy']:.4f}\n"
        )


def save_test_metrics_csv(
    lr_log_dir: str,
    metrics: dict,
) -> None:
    """
    Save per-step and final test metrics to CSV.
    """
    test_metrics_csv = os.path.join(lr_log_dir, "test_metrics.csv")

    step_loss = metrics["step_loss"]
    step_log_loss = metrics["step_log_loss"]
    step_accuracy = metrics["step_accuracy"]
    step_exact_match = metrics["step_exact_match"]
    step_foreground_accuracy = metrics["step_foreground_accuracy"]

    with open(test_metrics_csv, "w") as file:
        file.write("step,loss,log_loss,acc,exact_match,foreground_accuracy\n")
        for step_index in range(step_loss.shape[0]):
            file.write(
                f"{step_index},{float(step_loss[step_index]):.10f},"
                f"{float(step_log_loss[step_index]):.6f},"
                f"{float(step_accuracy[step_index]):.6f},"
                f"{float(step_exact_match[step_index]):.1f},"
                f"{float(step_foreground_accuracy[step_index]):.6f}\n"
            )

        file.write(
            f"final,{metrics['final_loss']:.10f},{metrics['final_log_loss']:.6f},"
            f"{metrics['final_accuracy']:.6f},{metrics['final_exact_match']:.1f},"
            f"{metrics['final_foreground_accuracy']:.6f}\n"
        )


def save_padding_debug_log(log_lines: list[str], output_path: str) -> None:
    """
    Save padding debug lines to a text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as file:
        for line in log_lines:
            file.write(line + "\n")


def plot_lr_and_loss(log_csv_path: str, output_dir: str) -> None:
    """
    Plot learning rate, loss, log-loss, and accuracy curves from a CSV log file.

    The CSV file is expected to contain the columns:
    iter, lr, loss, log_loss, accuracy

    Missing optional values are handled gracefully when possible.

    Args:
        log_csv_path: Path to the training log CSV file.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    steps, learning_rates, losses = [], [], []
    log_losses, accuracies = [], []

    with open(log_csv_path, newline="") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue

            try:
                step = int(row[0])
                learning_rate = float(row[1])
                loss = float(row[2])
            except (ValueError, IndexError):
                continue

            steps.append(step)
            learning_rates.append(learning_rate)
            losses.append(loss)

            if len(row) >= 4 and row[3].strip():
                try:
                    log_losses.append(float(row[3]))
                except ValueError:
                    pass
            elif log_losses:
                log_losses.append(log_losses[-1])

            if len(row) >= 5 and row[4].strip():
                try:
                    accuracies.append(float(row[4]))
                except ValueError:
                    pass
            elif accuracies:
                accuracies.append(accuracies[-1])

    if steps:
        plt.figure()
        plt.plot(steps, learning_rates)
        plt.xlabel("Iteration")
        plt.ylabel("Learning rate")
        plt.title("Learning rate schedule")
        plt.savefig(os.path.join(output_dir, "lr.png"))
        plt.close()

        plt.figure()
        plt.plot(steps, losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()

    if log_losses and len(log_losses) == len(steps):
        plt.figure()
        plt.plot(steps, log_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Log loss")
        plt.title("Training log-loss")
        plt.savefig(os.path.join(output_dir, "log_loss.png"))
        plt.close()

    if accuracies and len(accuracies) == len(steps):
        plt.figure()
        plt.plot(steps, accuracies)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Training accuracy")
        plt.savefig(os.path.join(output_dir, "accuracy.png"))
        plt.close()
