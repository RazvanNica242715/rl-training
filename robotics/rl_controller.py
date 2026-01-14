import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import ot2_env  # needed to trigger registration
import wandb
import os
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================
# W&B run IDs to evaluate
RUN_IDS = ["fhseidkq", "pavt1wmv", "n4lpr7th", "cl89vtyw"]

WANDB_PROJECT = "230395-breda-university-of-applied-sciences/ot2-rl-control"
ARTIFACTS_DIR = "artifacts"

# Evaluation settings
MAX_STEPS = 500
DT = 1.0 / 240.0
NUM_TARGETS = 8
CONVERGENCE_THRESHOLD = 0.001  # meters

# Workspace limits
WORKSPACE_LIMITS = {
    "x": (-0.1875, 0.2532),
    "y": (-0.17010, 0.2198),
    "z": (0.1195, 0.2903),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def download_model(run_id, api):
    """Download model from W&B run and return path."""
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    run_name = run.name
    model_filename = f"model_{run_name}.zip"
    model_path = os.path.join(ARTIFACTS_DIR, model_filename)

    # Download to temp location then move
    run.file("model.zip").download(root=ARTIFACTS_DIR, replace=True)
    temp_path = os.path.join(ARTIFACTS_DIR, "model.zip")

    if os.path.exists(temp_path):
        shutil.move(temp_path, model_path)
        print(f"Downloaded model for run '{run_name}' -> {model_path}")
        return model_path, run_name
    else:
        print(f"Failed to download model for run {run_id}")
        return None, run_name


def generate_targets(num_targets, seed=42):
    """Generate random targets within workspace."""
    np.random.seed(seed)
    targets = []
    for _ in range(num_targets):
        x = np.random.uniform(*WORKSPACE_LIMITS["x"])
        y = np.random.uniform(*WORKSPACE_LIMITS["y"])
        z = np.random.uniform(*WORKSPACE_LIMITS["z"])
        targets.append([x, y, z])
    return targets


def evaluate_model(model_path, targets, verbose=False):
    """Evaluate a model on all targets and return metrics."""
    model = PPO.load(model_path)

    all_distances = []
    final_distances = []
    convergence_times = []

    for t_idx, target in enumerate(targets):
        if verbose:
            print(f"  Target {t_idx + 1}/{len(targets)}: {target}")

        env = gym.make(
            "OT2ENV-v0", target=target, max_steps=MAX_STEPS, render_mode="none"
        )
        obs, info = env.reset(seed=42 + t_idx)

        # Set pipette to starting position
        start_position = (0.10775, 0.062, 0.1215)
        env.unwrapped.sim.set_start_position(*start_position)
        env.unwrapped.target = np.array(target)

        states = env.unwrapped.sim.get_states()
        obs = env.unwrapped._get_obs(states)
        info = env.unwrapped._get_info(states)

        env.unwrapped.dwell_steps = MAX_STEPS + 1
        if hasattr(env.unwrapped, "terminate_on_target"):
            env.unwrapped.terminate_on_target = False

        robot_key = list(info["pipette_positions"].keys())[0]
        distances = []
        threshold_reached_step = None

        for step in range(MAX_STEPS):
            pipette_pos = np.array(info["pipette_positions"][robot_key])
            error = np.array(target) - pipette_pos
            distance = np.linalg.norm(error)
            distances.append(distance)

            if threshold_reached_step is None and distance < CONVERGENCE_THRESHOLD:
                threshold_reached_step = step

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if truncated:
                break

        env.close()

        all_distances.append(distances)
        final_distances.append(distances[-1])
        convergence_times.append(
            threshold_reached_step * DT if threshold_reached_step else None
        )

    # Compute summary metrics
    valid_conv = [t for t in convergence_times if t is not None]
    metrics = {
        "final_distance_mean": np.mean(final_distances),
        "final_distance_std": np.std(final_distances),
        "final_distance_max": np.max(final_distances),
        "convergence_rate": len(valid_conv) / len(targets),
        "convergence_time_mean": np.mean(valid_conv) if valid_conv else float("inf"),
        "all_distances": all_distances,
        "final_distances": final_distances,
        "convergence_times": convergence_times,
    }
    return metrics


def compute_score(metrics):
    """Compute overall score (lower is better)."""
    # Weighted combination: prioritize final distance, then convergence rate
    score = (
        metrics["final_distance_mean"] * 1000  # Convert to mm
        + (1 - metrics["convergence_rate"]) * 10  # Penalty for not converging
        + metrics["convergence_time_mean"] * 0.1  # Small penalty for slow convergence
    )
    return score


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("=" * 60)
    print("MULTI-MODEL EVALUATION")
    print("=" * 60)

    # Initialize W&B API
    api = wandb.Api()

    # Generate consistent targets for fair comparison
    targets = generate_targets(NUM_TARGETS)
    print(f"\nGenerated {NUM_TARGETS} random targets within workspace")

    # Download all models
    print("\n" + "-" * 60)
    print("DOWNLOADING MODELS")
    print("-" * 60)

    models = {}
    for run_id in RUN_IDS:
        model_path, run_name = download_model(run_id, api)
        if model_path:
            models[run_id] = {"path": model_path, "name": run_name}

    if not models:
        print("No models downloaded successfully. Exiting.")
        exit(1)

    # Evaluate each model
    print("\n" + "-" * 60)
    print("EVALUATING MODELS")
    print("-" * 60)

    results = {}
    for run_id, model_info in models.items():
        print(f"\nEvaluating: {model_info['name']} (run_id: {run_id})")
        metrics = evaluate_model(model_info["path"], targets, verbose=True)
        metrics["score"] = compute_score(metrics)
        results[run_id] = metrics

        print(
            f"  Final distance: {metrics['final_distance_mean']*1000:.3f} ± {metrics['final_distance_std']*1000:.3f} mm"
        )
        print(f"  Convergence rate: {metrics['convergence_rate']*100:.1f}%")
        print(f"  Score: {metrics['score']:.4f}")

    # Find best model
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    best_run_id = min(results, key=lambda x: results[x]["score"])
    best_model = models[best_run_id]

    print("\nAll models ranked (lower score = better):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["score"])
    for rank, (run_id, metrics) in enumerate(sorted_results, 1):
        marker = " ★ BEST" if run_id == best_run_id else ""
        print(
            f"  {rank}. {models[run_id]['name']}: score={metrics['score']:.4f}, "
            f"dist={metrics['final_distance_mean']*1000:.3f}mm, "
            f"conv={metrics['convergence_rate']*100:.0f}%{marker}"
        )

    # Keep only best model, remove others
    print("\n" + "-" * 60)
    print("CLEANUP")
    print("-" * 60)

    for run_id, model_info in models.items():
        if run_id != best_run_id:
            os.remove(model_info["path"])
            print(f"Removed: {model_info['path']}")

    # Rename best model to model.zip for easy access
    best_final_path = os.path.join(ARTIFACTS_DIR, "model.zip")
    shutil.copy(best_model["path"], best_final_path)
    print(f"Best model saved as: {best_final_path}")
    print(f"Original also kept: {best_model['path']}")

    # Visualization of best model
    print("\n" + "=" * 60)
    print(f"BEST MODEL PERFORMANCE: {best_model['name']}")
    print("=" * 60)

    best_metrics = results[best_run_id]
    all_distances = best_metrics["all_distances"]

    # Pad and compute mean trajectories
    max_len = max(len(d) for d in all_distances)
    timestamps = [i * DT for i in range(max_len)]

    def pad_and_stack(arrays, max_len):
        padded = []
        for arr in arrays:
            if len(arr) < max_len:
                arr = arr + [arr[-1]] * (max_len - len(arr))
            padded.append(arr)
        return np.array(padded)

    dist_array = pad_and_stack(all_distances, max_len)
    mean_distances = np.mean(dist_array, axis=0)
    std_distances = np.std(dist_array, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        timestamps,
        mean_distances * 1000,
        linewidth=2,
        color="blue",
        label="Mean Distance",
    )
    ax.fill_between(
        timestamps,
        (mean_distances - std_distances) * 1000,
        (mean_distances + std_distances) * 1000,
        alpha=0.3,
        color="blue",
        label="±1 Std Dev",
    )
    ax.axhline(
        y=CONVERGENCE_THRESHOLD * 1000,
        color="r",
        linestyle="--",
        label="Target threshold",
        alpha=0.5,
    )
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Distance to Target (mm)", fontsize=12)
    ax.set_title(f'Best Model Performance: {best_model["name"]}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "best_model_performance.png"), dpi=150)
    plt.show()

    print(f"\nFinal Statistics:")
    print(f"  Mean final distance: {best_metrics['final_distance_mean']*1000:.3f} mm")
    print(f"  Std final distance:  {best_metrics['final_distance_std']*1000:.3f} mm")
    print(f"  Max final distance:  {best_metrics['final_distance_max']*1000:.3f} mm")
    print(f"  Convergence rate:    {best_metrics['convergence_rate']*100:.1f}%")
    if best_metrics["convergence_time_mean"] != float("inf"):
        print(f"  Mean convergence time: {best_metrics['convergence_time_mean']:.3f} s")
    print("=" * 60)
