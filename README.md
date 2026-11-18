# Transformer-Enabled Terahertz Metasurface Metrology for High-Aspect-Ratio Deep Reactive-Ion Etching

This repository provides processed terahertz reflectance spectra, paired microscopy reference dimensions, PyTorch implementations, and source data for the DRIE metasurface study. The retrieval methods are numerical fitting, MLP, CNN, and Transformer.

## Quick start

The package was checked with Python 3.11 and PyTorch 2.3.0. From the repository root, activate a clean Python environment and run:

```sh
python -m pip install -r requirements.txt
python run.py verify
python run.py predict --design MS2 --model all --output outputs/MS2_predictions.csv
python analysis.py --output outputs/figures --off-grid
```

The verification command checks the download, the dataset splits, and the shipped model predictions. The prediction command uses the packaged measurements by default and reports their errors against the paired microscopy references. For a new sample, use `--input path/to/processed_spectra.csv` with one headerless row per spectrum in the format described below. The analysis command regenerates plots and tables, evaluates the supplied checkpoints on the archived noisy spectra, and recomputes the last-hidden-layer features and associated metrics. Add `--recompute-projections` to fit PCA and t-SNE again on those features.

To train a new model on the simulation data:

```sh
python run.py train --design MS2 --model CNN --seed 43000 --output outputs/MS2_CNN_43000
python run.py predict --design MS2 --model CNN --weights outputs/MS2_CNN_43000/weights.npz --output outputs/MS2_CNN_predictions.csv
```

The same command accepts `MLP` and `Transformer`. Use `--epochs 2` for a quick training check or `--device cuda` with a CUDA-enabled PyTorch installation. The default profiles and fixed simulation train/validation indices are recorded in `metadata.json` and `data/<design>/split_indices.npz`. These profiles describe the controlled retraining protocol. The procedure for obtaining the archived manuscript checkpoints is described separately in the study.

## Datasets

| Paper label | Archive label | Simulation rows | Paired measured spectra | Input columns |
|---|---|---:|---:|---:|
| MS1 | MS4 | 861 | 8 | 215 |
| MS2 | MS6 | 693 | 9 | 215 |
| MS3 | MS7 | 693 | 9 | 215 |
| DMS2 | NMS4 | 231 | 12 | 198 |

- `data/<design>/simulation_inputs.csv` and `simulation_labels.csv` have one row per simulation in matching order. The four labels are the archived sidewall-angle and carrier labels, followed by normalized lateral dimension and etch depth. The label values for angle and carrier concentration follow the archived simulation encodings.
- `measured_inputs.csv` and `measured_references.csv` are paired row by row for the corresponding design. The two reference columns are normalized lateral dimension and etch depth. The stored test inputs are processed model inputs. The first column is the archived resonance-related feature and the remaining columns are sampled reflectance values, ordered as in the archive. Raw time-domain waveforms and the electromagnetic solver projects are outside the scope of this deposit.
- `article_measured_*` files, where present, preserve the alternate archived measurement selection. The command-line demo uses `measured_*`. For MS3, the demo preserves the nine spectra and their raw reference dimensions from the paired legacy files. Table S3 records the manuscript's historical results. Treat the subgroups and reference rows as paired within each file pair.
- To obtain signed size errors in micrometres, subtract the reference from the prediction in normalized coordinates and multiply the lateral and depth components by 4 and 20, respectively. The prediction CSV also reports deviations in the coordinate system with offsets of −2 and −10 µm. These are deviation coordinates. Absolute geometric dimensions require applying the appropriate design reference.
- `data/MS2/archived_noise_inputs.npz` contains the original processed reflectance inputs for the noise analysis, ordered by `sigma`, including the original spectrum at zero. `retraining_noise_inputs.npz` contains the corresponding inputs for the separate 100-run comparison. Its independent perturbations are indexed by noise level, repeat, measured spectrum and input column.
- `data/MS2/fine_grid_*` contains the archived finer-grid spectra and labels used in the off-grid test. The script reconstructs the original script's ordered training subset and selects geometries whose four surrounding coarse-grid points are in that subset.

## Models and provenance

The released CNN is the six-layer dilated 1D convolutional network with group normalization and adaptive average pooling. All public model labels use `CNN`. `weights/` contains one example of each neural network per design in NumPy tensor archives, with provenance in `metadata.json`. The original Transformer checkpoints are used where identifiable. The MS2 CNN weight is the archived checkpoint used for the noise and feature analyses. The MS1, MS3 and DMS2 CNN files are documented retraining examples. One checkpoint per model and design provides a runnable example, while the 100-run and manuscript statistics below apply to their documented cohorts.

Numerical fitting searches the supplied simulated library using the sum of the reflectance mean squared error and the squared difference in resonance frequency. The resonance minima are evaluated on the common spectral grid of each packaged design. The numerical fitting demo uses the processed release files. The archival comparisons are available as source data below.

## Figure and table source data

| Result | Files in `source_data/` |
|---|---|
| Fig. 3 | `figure3_marker_counts.csv` contains the published-plot counting analysis. `checkpoint_expected_predictions.csv` contains the predictions for the shipped checkpoints on the packaged examples. |
| Fig. 4b | `figure4b_noise_curves.csv` records the archived MLP curve and the archived-input evaluations of the CNN and Transformer for MS2. The script also exports a separate checkpoint-based recalculation that includes the zero-noise point. Small differences for the MLP arise when evaluating the archived checkpoint on the nine spectra in this package. |
| Fig. 4c | `ms2_last_hidden_features.npz` and `figure4c_reference_metrics.csv` contain the activations and distance statistics for MS2. The plotted ratio is the between-domain distance divided by the sum of the two within-domain distances. |
| Fig. 5 | `figure5_last_hidden_coordinates.csv` gives the matched last-hidden-feature visualization used by the analysis script. `figure5_earlier_coordinates.csv` preserves the earlier archived-layout comparison, with separate provenance in the `source` column. |
| Fig. S6 | `figure_s6_matched_signed_errors.npz` holds matched archived signed errors in µm, with axes training run, spectrum and dimension (S, H). Deterministic fitting has one row in the run axis. `figure_s6_per_spectrum_SD.csv` reports the per-spectrum run-to-run sample standard deviations. |
| Tables S1, S3 and S4 | `table_s1_nominal_dimensions.csv`, `table_s3_revised_MAE.csv` and `table_s4_revised_mean_MAE.csv` give the design dimensions, per-design errors and cross-design averages, respectively. The revised MS2 CNN row uses the 100-run comparison below. |
| Table S5 | `table_s5_off_grid_reference.csv` gives the off-grid comparison for the matched set. |
| Table S6 | `table_s6_neighbour_reference.csv` and `table_s6_neighbour_details.csv` give five neighbours per measured MS2 spectrum and model. They compare the simulation labels of the feature nearest neighbours with the microscopy references. |
| Table S7 | `table_s7_fabrication_variation.csv` and `table_s7_microscopy_repeatability.csv` report the fabrication sampling and microscopy repeatability summaries. |
| MS2 independent retraining comparison | `ms2_100_training_predictions.npz` contains arrays of shape `(100, 211, 9, 4)` for each model and a separate `seeds` vector. Input order is one clean group, ten archived noisy groups, then 200 groups indexed first by noise level and then by repeat. This comparison uses its own checkpoints and is separate from the historical Fig. S6 analysis. |

`SHA256SUMS` records the files in this release. The manuscript title above identifies the associated study. Cite the manuscript and the version of the repository or release you use. A code and data reuse licence has yet to be selected for this deposit.
