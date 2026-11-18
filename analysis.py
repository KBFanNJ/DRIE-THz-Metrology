"""Regenerate plots and check the released MS2 feature and grid analyses."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from models import MODELS, load_weights, predict
from run import ROOT, SCALE, inputs, write_csv


SOURCE = ROOT / "source_data"
COLORS = {"MLP": "#377eb8", "CNN": "#e08214", "Transformer": "#328a5d"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def noise_plot(out):
    rows = read_csv(SOURCE / "figure4b_noise_curves.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, color in COLORS.items():
        subset = sorted((r for r in rows if r["model"] == kind), key=lambda r: float(r["sigma"]))
        for dim, line in (("S", "--"), ("H", "-")):
            ax.plot([float(r["sigma"]) for r in subset], [float(r[f"{dim}_MAE_um"]) for r in subset],
                    linestyle=line, marker="o", color=color, label=f"{kind} {dim}")
    ax.set(xlabel="Added noise standard deviation", ylabel="MAE (µm)")
    ax.legend(frameon=False, ncols=2)
    fig.savefig(out / "figure4b.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_archived_noise(out):
    truth = inputs("MS2", "measured_references", 2)
    with np.load(ROOT / "data/MS2/archived_noise_inputs.npz", allow_pickle=False) as bundle:
        sigma, spectra = bundle["sigma"], bundle["spectra"]
    levels, samples, width = spectra.shape
    rows = []
    for kind in MODELS:
        net = load_weights(kind, width, ROOT / "weights/MS2" / f"{kind}.npz")
        predicted = predict(net, spectra.reshape(-1, width))[:, 2:].reshape(levels, samples, 2)
        errors = abs(predicted - truth) * SCALE
        rows.extend((kind, level, *mean) for level, mean in zip(sigma, errors.mean(axis=1)))
    write_csv(out / "figure4b_checkpoint_recalculation.csv", ["model", "sigma", "S_MAE_um", "H_MAE_um"], rows)


def feature_analysis(out, recompute_projections):
    labels = inputs("MS2", "simulation_labels", 4)[:, 2:]
    truth = inputs("MS2", "measured_references", 2)
    spectra = np.vstack([inputs("MS2", name, 215) for name in ("simulation_inputs", "measured_inputs")])
    metrics, neighbour_rows = [], []
    coordinates = read_csv(SOURCE / "figure5_last_hidden_coordinates.csv") if not recompute_projections else []
    with np.load(SOURCE / "ms2_last_hidden_features.npz", allow_pickle=False) as bundle:
        for kind in MODELS:
            net = load_weights(kind, 215, ROOT / "weights/MS2" / f"{kind}.npz")
            output_layer = {"MLP": lambda: net.linears[-1], "CNN": lambda: net.head[-1],
                            "Transformer": lambda: net.fc}[kind]()
            activations = []
            with output_layer.register_forward_pre_hook(lambda _, args: activations.append(args[0].cpu().numpy())):
                predict(net, spectra)
            features = np.concatenate(activations)
            np.testing.assert_allclose(features, bundle[kind], rtol=3e-4, atol=5e-5)
            sim, exp = features[:len(labels)], features[len(labels):]
            distances = cdist(exp, sim)
            within_sim = pdist(sim).mean() * 0.1
            within_exp = pdist(exp).mean() * 0.1
            between = distances.mean() * 0.1
            metrics.append((kind, features.shape[1], within_sim, within_exp, between,
                            between / (within_sim + within_exp)))
            nearest = np.argsort(distances, axis=1, kind="stable")[:, :5]
            errors = np.abs(labels[nearest] - truth[:, None]) * SCALE
            neighbour_rows.append((kind, len(truth), len(labels), 5, *errors.mean(axis=(0, 1))))
            if recompute_projections:
                with threadpool_limits(limits=1):
                    coords = {
                        "PCA": PCA(n_components=2, svd_solver="full").fit_transform(features),
                        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000,
                                      init="pca", learning_rate="auto", n_jobs=1).fit_transform(features),
                    }
                for method, values in coords.items():
                    for i, (x, y) in enumerate(values):
                        coordinates.append(dict(model=kind, method=method, x=x, y=y,
                                                domain="simulation" if i < len(sim) else "experiment"))
    write_csv(out / "figure4c.csv", ["model", "feature_dimensions", "S_sim_scaled", "S_exp_scaled",
                                      "S_inter_scaled", "A_score"], metrics)
    write_csv(out / "table_s6_neighbours.csv", ["model", "n_measured", "n_simulated", "k",
                                                "lateral_difference_um", "depth_difference_um"], neighbour_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(4)
    for i, row in enumerate(metrics):
        ax.bar(x + (i - 1) * 0.24, row[2:], width=0.24, label=row[0], color=COLORS[row[0]])
    ax.set_xticks(x, ["Simulation", "Experiment", "Cross-domain", "Ratio"])
    ax.legend(frameon=False)
    fig.savefig(out / "figure4c.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for i, method in enumerate(("PCA", "t-SNE")):
        for j, kind in enumerate(MODELS):
            ax = axes[i, j]
            for domain, label, color, size in (("simulation", "Simulation", "#6c9cc4", 8),
                                               ("experiment", "Experiment", "#d54531", 16)):
                points = [(float(r["x"]), float(r["y"])) for r in coordinates
                          if r["model"] == kind and r["method"].replace("-", "").lower() == method.replace("-", "").lower()
                          and r["domain"] == domain]
                if not points:
                    raise ValueError(f"No {method} coordinates for {kind} {domain}")
                ax.scatter(*zip(*points), label=label, color=color, s=size, linewidths=0)
            ax.set(title=f"{kind} {method}", xlabel="Dimension 1", ylabel="Dimension 2")
            ax.legend(frameon=False, fontsize=7)
    fig.savefig(out / "figure5.png", dpi=220)
    plt.close(fig)


def grid_analysis(out):
    coarse_x = inputs("MS2", "simulation_inputs", 215)
    coarse = inputs("MS2", "simulation_labels", 4)
    fine_x = inputs("MS2", "fine_grid_inputs", 215)
    fine = inputs("MS2", "fine_grid_labels", 4)
    match = np.all(np.isclose(coarse[:, :2], fine[0, :2]), axis=1)
    a, b = (np.unique(coarse[match, j]) for j in (2, 3))
    fine_off = (~np.isin(fine[:, 2].round(8), a.round(8)) & ~np.isin(fine[:, 3].round(8), b.round(8)))
    labels = fine[fine_off, 2:]
    spectra = fine_x[fine_off]
    historical = coarse[:int(0.8 * len(coarse))]
    fixed = np.all(np.isclose(historical[:, :2], fine[0, :2]), axis=1)
    known = {tuple(row.round(8)) for row in historical[fixed, 2:]}
    i, j = (np.searchsorted(a, labels[:, 0]), np.searchsorted(b, labels[:, 1]))
    inside = [all(tuple(np.round((s, h), 8)) in known for s in a[m-1:m+1] for h in b[n-1:n+1])
              for m, n in zip(i, j)]
    labels, spectra = labels[inside], spectra[inside]
    model = load_weights("Transformer", 215, ROOT / "weights/MS2/Transformer.npz")
    pred = predict(model, spectra)[:, 2:]
    xy = coarse[match, 2:]
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    interp = RegularGridInterpolator((a, b), coarse_x[match][order, 1:].reshape(len(a), len(b), -1))
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 201), indexing="ij"), -1).reshape(-1, 2)
    grid_spectra = interp(grid)
    start = grid[cdist(spectra[:, 1:], grid_spectra, metric="sqeuclidean").argmin(1)]
    fit = np.array([least_squares(lambda u: interp(u)[0] - target, xy, bounds=(0, 1),
                                  ftol=1e-9, xtol=1e-9, gtol=1e-10).x
                    for target, xy in zip(spectra[:, 1:], start)])
    rows = []
    for kind, p in (("Bilinear spectral fitting", fit), ("Transformer", pred)):
        err = abs(p - labels) * SCALE
        rows.append((kind, len(err), *err.mean(axis=0), np.sum(np.all(err < [0.2, 0.5], axis=1))))
    write_csv(out / "table_s5_off_grid.csv", ["method", "cases", "lateral_MAE_um", "depth_MAE_um",
                                              "cases_below_both_half_grid_thresholds"], rows)


def summarize_training_runs(out):
    rows = []
    truth = inputs("MS2", "measured_references", 2)
    with np.load(SOURCE / "ms2_100_training_predictions.npz", allow_pickle=False) as bundle:
        for kind in MODELS:
            errors = abs(bundle[kind][:, 0, :, 2:] - truth) * SCALE
            mae_per_run = errors.mean(axis=1)
            rows.append((kind, len(mae_per_run), *mae_per_run.mean(0), *mae_per_run.std(0, ddof=1)))
    write_csv(out / "ms2_100_runs_clean.csv", ["model", "runs", "S_mean_MAE_um", "H_mean_MAE_um",
                                                "S_SD_of_MAE_um", "H_SD_of_MAE_um"], rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    cli.add_argument("--recompute-projections", action="store_true", help="Fit PCA and t-SNE to the supplied features")
    cli.add_argument("--off-grid", action="store_true", help="Also run the off-grid spectral fitting comparison")
    args = cli.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    noise_plot(args.output)
    evaluate_archived_noise(args.output)
    feature_analysis(args.output, args.recompute_projections)
    summarize_training_runs(args.output)
    if args.off_grid:
        grid_analysis(args.output)


if __name__ == "__main__":
    main()
