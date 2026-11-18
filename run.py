"""Train, predict, or compare retrieval methods on the released spectral inputs."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models import MODELS, load_weights, predict, save_weights


ROOT = Path(__file__).resolve().parent
DESIGNS = ("MS1", "MS2", "MS3", "DMS2")
SCALE = np.array([4.0, 20.0])
OFFSET = np.array([-2.0, -10.0])


def metadata():
    return json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))


def matrix(path, width):
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if not len(values) or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"{path.name} must contain one or more finite rows of {width} values")
    return values


def inputs(design, name, width):
    return matrix(ROOT / "data" / design / f"{name}.csv", width)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def fit_library(query, library, labels, frequency_step):
    """Search the supplied library using the manuscript spectral-match objective."""
    values = library[:, 1:]
    resonance = np.argmin(values, axis=1) * frequency_step
    predictions = []
    for row in query[:, 1:]:
        spectrum_mse = np.mean(np.square(values - row), axis=1)
        resonance_error = np.square(resonance - np.argmin(row) * frequency_step)
        predictions.append(labels[np.argmin(spectrum_mse + resonance_error)])
    return np.array(predictions)


def run_predictions(args, design_info):
    n = design_info["input_columns"]
    if args.input is not None:
        spectra = matrix(args.input, n)
    else:
        spectra = inputs(args.design, "measured_inputs", n)
    kinds = [*MODELS, "fitting"] if args.model == "all" else [args.model]
    references = None if args.input is not None else inputs(args.design, "measured_references", 2)
    rows = []
    for kind in kinds:
        if kind == "fitting":
            outputs = fit_library(spectra, inputs(args.design, "simulation_inputs", n),
                                  inputs(args.design, "simulation_labels", 4), design_info["frequency_step_THz"])
        else:
            weights = args.weights or ROOT / "weights" / args.design / f"{kind}.npz"
            outputs = predict(load_weights(kind, n, weights, args.device), spectra, args.device)
        dimensions = outputs[:, 2:] * SCALE + OFFSET
        refs_um = references * SCALE + OFFSET if references is not None else None
        if refs_um is not None:
            error = dimensions - refs_um
            print(f"{args.design} {kind}: sample MAE in um, lateral {abs(error[:, 0]).mean():.3f}, "
                  f"depth {abs(error[:, 1]).mean():.3f}")
        for i, (raw, dim) in enumerate(zip(outputs, dimensions)):
            reference = refs_um[i] if refs_um is not None else ("", "")
            rows.append([kind, i + 1, *raw, *dim, *reference])
    write_csv(args.output, ["model", "row", "angle_label", "carrier_label", "normalized_S", "normalized_H",
                            "lateral_deviation_um", "depth_deviation_um",
                            "reference_lateral_deviation_um", "reference_depth_deviation_um"], rows)


def train_model(args, design_info, defaults):
    profile = defaults[args.model]
    epochs = args.epochs or profile["epochs"]
    torch.manual_seed(args.seed)
    n = design_info["input_columns"]
    x = torch.tensor(inputs(args.design, "simulation_inputs", n), dtype=torch.float32)
    y = torch.tensor(inputs(args.design, "simulation_labels", 4), dtype=torch.float32)
    with np.load(ROOT / "data" / args.design / "split_indices.npz", allow_pickle=False) as split:
        train, validation = split["train"], split["validation"]
    tx, ty, vx, vy = x[train], y[train], x[validation], y[validation]
    net = MODELS[args.model](n).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=profile["lr"], weight_decay=profile["weight_decay"])
    scheduler = None
    if args.model != "Transformer":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=10,
                                                                cooldown=500, threshold=1e-5)
    history, state = [], None
    best = float("inf")
    for epoch in range(1, epochs + 1):
        net.train()
        loss_sum = 0.0
        for batch in torch.randperm(len(tx)).split(profile["batch"]):
            opt.zero_grad()
            for part in batch.split(profile["microbatch"]):
                loss = F.mse_loss(net(tx[part].to(args.device)), ty[part].to(args.device)) * profile["loss_scale"]
                (loss * len(part) / len(batch)).backward()
                loss_sum += loss.item() * len(part)
            opt.step()
        training_loss = loss_sum / len(tx)
        validation_loss = None
        if args.model == "Transformer" or (epoch - 1) % 20 != 0 or epochs == 1:
            net.eval()
            loss_sum = 0.0
            with torch.inference_mode():
                for bx, by in zip(vx.split(profile["microbatch"]), vy.split(profile["microbatch"])):
                    loss_sum += F.mse_loss(net(bx.to(args.device)), by.to(args.device)).item() * len(bx)
            validation_loss = loss_sum * profile["loss_scale"] / len(vx)
            if validation_loss < best and scheduler is not None:
                best, chosen = validation_loss, epoch
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if not np.isfinite(training_loss) or (validation_loss is not None and not np.isfinite(validation_loss)):
            raise RuntimeError(f"A non-finite loss occurred at epoch {epoch}")
        history.append((epoch, training_loss, validation_loss, opt.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step(training_loss)
    if scheduler is None:
        chosen, state = epochs, net.state_dict()
    args.output.mkdir(parents=True, exist_ok=True)
    save_weights(args.output / "weights.npz", state)
    write_csv(args.output / "history.csv", ["epoch", "training_loss", "validation_loss", "learning_rate"], history)
    print(f"{args.design} {args.model}: saved seed {args.seed}, epoch {chosen}/{epochs} to {args.output}")


def verify():
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        candidate = ROOT / name
        if not candidate.is_file() or sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or modified release file: {name}")
    table = ROOT / "source_data" / "checkpoint_expected_predictions.csv"
    with table.open(newline="", encoding="utf-8") as handle:
        benchmark = list(csv.DictReader(handle))
    for design, config in metadata()["designs"].items():
        n = config["input_columns"]
        x = inputs(design, "simulation_inputs", n)
        y = inputs(design, "simulation_labels", 4)
        exp = inputs(design, "measured_inputs", n)
        ref = inputs(design, "measured_references", 2)
        with np.load(ROOT / "data" / design / "split_indices.npz", allow_pickle=False) as split:
            train, val = split["train"], split["validation"]
            np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(len(x)))
        assert len(x) == len(y) == config["simulation_rows"] and len(exp) == len(ref) == config["measured_rows"]
        for model in MODELS:
            path = ROOT / "weights" / design / f"{model}.npz"
            pred = predict(load_weights(model, n, path), exp)
            rows = [row for row in benchmark if row["design"] == design and row["model"] == model]
            expected = [[float(x[key]) for key in ("angle_label", "carrier_label", "normalized_S", "normalized_H")]
                        for x in rows]
            np.testing.assert_allclose(pred, expected, atol=5e-5, rtol=3e-4)
        print(f"{design}: {n} inputs, {len(x)} simulations, {len(exp)} paired spectra, three checkpoints verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU thread count")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("predict")
    p.add_argument("--design", required=True, choices=DESIGNS)
    p.add_argument("--model", choices=[*MODELS, "fitting", "all"], default="all")
    p.add_argument("--input", type=Path, help="Optional headerless CSV of processed input spectra")
    p.add_argument("--weights", type=Path, help="An alternative checkpoint for one neural-network model")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    t = commands.add_parser("train")
    t.add_argument("--design", required=True, choices=DESIGNS)
    t.add_argument("--model", required=True, choices=MODELS)
    t.add_argument("--seed", type=int, default=43000)
    t.add_argument("--epochs", type=int, help="Override the training budget for a new run")
    t.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    t.add_argument("--output", type=Path, required=True)
    commands.add_parser("verify")
    args = parser.parse_args()
    epochs = getattr(args, "epochs", None)
    if args.threads < 1 or (epochs is not None and epochs < 1):
        parser.error("Thread and epoch counts must be positive")
    if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the current PyTorch environment")
    torch.set_num_threads(args.threads)
    if args.command == "verify":
        verify()
        return
    configuration = metadata()
    design = configuration["designs"][args.design]
    if args.command == "train":
        if (args.output / "weights.npz").exists():
            parser.error("Choose a new training output directory")
        train_model(args, design, configuration["training_profiles"])
    else:
        if args.weights is not None and args.model not in MODELS:
            parser.error("--weights requires one neural-network model")
        run_predictions(args, design)


if __name__ == "__main__":
    main()
