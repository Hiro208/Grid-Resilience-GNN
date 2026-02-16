import argparse
import csv
import os
import shutil
from datetime import datetime

import matplotlib
import networkx as nx
import numpy as np
from tqdm import tqdm

from simulation import CascadingSimulator


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_once(sim: CascadingSimulator, initial_failures: int) -> tuple[float, nx.Graph]:
    sim.trigger_failure(num_initial_failures=initial_failures)
    total_failed, graph = sim.run_cascade()
    failure_rate = total_failed / sim.num_nodes
    return failure_rate, graph


def save_graph_snapshot(graph: nx.Graph, output_dir: str, filename: str) -> str:
    pos = nx.get_node_attributes(graph, "pos")
    colors = ["red" if graph.nodes[n]["status"] == "failed" else "steelblue" for n in graph.nodes]

    fig = plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, node_color=colors, node_size=40, edge_color="gray", alpha=0.45)
    plt.title("Cascade Snapshot (red=failed, blue=active)")
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_distribution_plot(failure_rates: list[float], output_dir: str, filename: str) -> str:
    fig = plt.figure(figsize=(10, 6))
    plt.hist(failure_rates, bins=20, color="royalblue", alpha=0.85, edgecolor="white")
    plt.xlabel("Failure Rate")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cascade Failure Rate")
    plt.grid(alpha=0.2)
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_curve_plot(
    x: list[int], means: list[float], stds: list[float], output_dir: str, filename: str
) -> str:
    x_np = np.array(x)
    means_np = np.array(means)
    stds_np = np.array(stds)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_np, means_np, marker="o", color="darkorange", linewidth=2, label="Mean Failure Rate")
    plt.fill_between(
        x_np,
        np.clip(means_np - stds_np, 0, 1),
        np.clip(means_np + stds_np, 0, 1),
        alpha=0.2,
        color="darkorange",
        label="±1 std",
    )
    plt.xlabel("Number of Initial Failures")
    plt.ylabel("Failure Rate")
    plt.ylim(0, 1.0)
    plt.title("Attack Intensity vs Failure Rate")
    plt.grid(alpha=0.2)
    plt.legend()
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_summary_csv(
    x: list[int], means: list[float], stds: list[float], output_dir: str, filename: str
) -> str:
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["initial_failures", "mean_failure_rate", "std_failure_rate"])
        for xi, mean, std in zip(x, means, stds):
            writer.writerow([xi, f"{mean:.6f}", f"{std:.6f}"])
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulation visualizations for README.")
    parser.add_argument("--num-nodes", type=int, default=200)
    parser.add_argument("--radius", type=float, default=0.12)
    parser.add_argument("--batch-runs", type=int, default=80)
    parser.add_argument("--curve-runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/visualizations")
    args = parser.parse_args()

    np.random.seed(args.seed)
    ensure_dir(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim = CascadingSimulator(num_nodes=args.num_nodes, radius=args.radius)

    # 1) Single snapshot
    snapshot_rate, snapshot_graph = run_once(sim, initial_failures=3)
    snapshot_file = f"snapshot_{timestamp}.png"
    snapshot_path = save_graph_snapshot(snapshot_graph, args.output_dir, snapshot_file)

    # 2) Distribution under fixed attack size
    distribution_rates = []
    for _ in tqdm(range(args.batch_runs), desc="Batch simulation"):
        rate, _ = run_once(sim, initial_failures=3)
        distribution_rates.append(rate)
    distribution_file = f"distribution_{timestamp}.png"
    distribution_path = save_distribution_plot(distribution_rates, args.output_dir, distribution_file)

    # 3) Attack intensity curve
    x_values = list(range(1, 9))
    means = []
    stds = []
    for x in tqdm(x_values, desc="Attack intensity sweep"):
        rates = []
        for _ in range(args.curve_runs):
            rate, _ = run_once(sim, initial_failures=x)
            rates.append(rate)
        means.append(float(np.mean(rates)))
        stds.append(float(np.std(rates)))

    curve_file = f"intensity_curve_{timestamp}.png"
    curve_path = save_curve_plot(x_values, means, stds, args.output_dir, curve_file)
    csv_file = f"summary_{timestamp}.csv"
    csv_path = save_summary_csv(x_values, means, stds, args.output_dir, csv_file)

    # Create stable filenames for README embedding.
    shutil.copyfile(snapshot_path, os.path.join(args.output_dir, "snapshot_latest.png"))
    shutil.copyfile(distribution_path, os.path.join(args.output_dir, "distribution_latest.png"))
    shutil.copyfile(curve_path, os.path.join(args.output_dir, "intensity_curve_latest.png"))
    shutil.copyfile(csv_path, os.path.join(args.output_dir, "summary_latest.csv"))

    print("Visualization artifacts generated:")
    print(f"- Snapshot        : {snapshot_path} (failure rate={snapshot_rate:.2%})")
    print(f"- Distribution    : {distribution_path}")
    print(f"- Intensity curve : {curve_path}")
    print(f"- Summary CSV     : {csv_path}")


if __name__ == "__main__":
    main()
