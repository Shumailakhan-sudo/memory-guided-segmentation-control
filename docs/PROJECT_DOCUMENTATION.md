# Memory-Guided Video Object Segmentation

Technical documentation for a video object segmentation and motion-analysis toolkit built around pretrained SAM-2, DAVIS datasets, RAFT optical flow, confidence gating, and a separate two-dimensional Cartesian simulation.

This document covers the software workflow, experiment configuration, metric definitions, data integrity, runtime measurement, and known limitations. **Working analysis code and synthetic tests do not establish the accuracy of historical experimental results.** Source records, settings, and hardware logs are required to independently verify those results.

## 1. System overview

The video segmentation pipeline initializes an object from a first-frame spatial prompt, propagates masks using SAM-2 temporal memory, and evaluates predictions against annotated masks. Optional confidence gating decides which predicted masks are accepted. Separate analysis modules calculate object visibility, centroid trajectories, optical flow, uncertainty calibration, and latency.

| Component | Purpose |
|---|---|
| SAM-2 | Pretrained video object segmentation and temporal mask propagation |
| DAVIS 2016/2017 | Annotated video sequences for segmentation and motion analysis |
| Confidence gating | Accept non-empty masks only when confidence reaches a configured threshold |
| RAFT | Forward/backward optical-flow estimation |
| Visibility analysis | Detect low-area intervals and their restoration in annotations |
| Centroid analysis | Track object-centre motion and compute Allan variance |
| Calibration | Evaluate confidence reliability and acceptance/error trade-offs |
| Runtime profiling | Record processing-stage latencies and deadline compliance |
| Cartesian simulation | Test confidence-dependent goal updates in a simplified point-motion environment |

The segmentation benchmark and Cartesian simulation are independent components. Cartesian endpoint success is **not** a physical grasping result.

## 2. Evaluation configuration

### 2.1 Video segmentation

| Setting | Configuration or scope |
|---|---|
| Dataset family | DAVIS 2016/2017 |
| Dataset resolution | 480p |
| Reference frame rate | 24 FPS |
| Reference cohort | 50 sequences in the available evaluation description; exact membership requires a saved manifest |
| Segmentation model | Pretrained SAM-2; no model fine-tuning described |
| First-frame prompt types | `bbox`, `center_point`, `none` |
| Memory-update interval | `u ∈ {1, 2, 3, 5}` |
| Confidence score | Per-prediction `q ∈ [0, 1]` |
| Gating thresholds | `q* ∈ {0, 0.40, 0.55, 0.65}` |
| Optical flow | RAFT, with `raft-things.pth` for the Princeton backend |
| Flow consistency threshold | 1 pixel |
| Low-visibility area threshold | `α = 0.05` |
| Minimum completed visibility-gap length | `β = 3` frames |
| Uncertainty intervals | Sequence-cluster bootstrap, 2,000 resamples |

The exact SAM-2 checkpoint, confidence extraction method, memory-update mechanism, and behavior of the `none` prompt are not established by the available experimental records. Each must be configured explicitly for a new run. Four existing configuration combinations jointly change prompt type, gating threshold, and memory interval; differences between combinations cannot isolate the causal effect of any one setting.

### 2.2 Cartesian point-motion simulation

| Setting | Configuration or scope |
|---|---|
| Geometry | 2-D Cartesian point motion |
| Workspace | 1.5 × 1.5 m |
| Control interval | 0.05 s |
| Timeout | 10 s |
| Endpoint success tolerance | 0.03 m |
| Goal-update confidence threshold | 0.55 |
| Episode reference set | 20 episodes; separate 10-episode reward records |

A valid target estimate with confidence at least 0.55 replaces the active goal. Otherwise, the prior goal is retained; motion pauses if no valid goal is available. The reconstruction requires explicit motion-speed, closure-radius, and per-step observation inputs. These cannot be inferred reliably from summary statistics.

This simulation does not include a robot arm, joint kinematics, a camera, contact dynamics, object lifting, or grasp verification. A Franka Panda/PyBullet configuration exists as a design specification only; it is not evidence of an executed robotics experiment. The 10-episode reward records carry an internal label that does not imply physical robot data; the reward formula has not been established.

## 3. Dataset preparation and integrity

Download DAVIS from [davischallenge.org](https://davischallenge.org/) and extract the dataset so this directory is available:

```text
data/raw/DAVIS/JPEGImages/480p/
```

The dataset is not redistributed in this repository. Configure its location through `configs/davis.toml` or the applicable command-line override.

Create a sequence manifest:

```bash
python scripts/prepare_davis.py --config configs/davis.toml
```

The manifest records sequence identifiers, frame and annotation counts, image dimensions, object counts, known DAVIS split membership, frame-content hashes, and reference-frame-rate alignment. Duplicate content hashes are reported for inspection. Frame-rate alignment and any resizing must be included in the run's provenance.

### 3.1 Dataset membership and partitions

The reference evaluation lists **10,365 frame–strategy evaluations**, or **3,455 evaluations per prompt type**. A total of 3,455 annotated frames is consistent with DAVIS 2016's 50-sequence collection, but does **not** by itself prove which exact masks or sequence cohort were used. Single-object versus merged multi-object evaluation is likewise unresolved without the frame-level records.

The published reference figures are pooled across 50 sequences. They should not be described as held-out test performance.

The following partition schemes must remain distinct:

- **Official DAVIS 2016 partition:** 30 training and 20 validation sequences.
- **Configurable sequence-disjoint partition:** 30 development, 10 validation, and 10 test sequences (60:20:20), using a complexity-based stratification implementation with weights `0.4 / 0.4 / 0.2`, terciles, and seed 42.

The second scheme can be generated deterministically by the toolkit, but a newly generated assignment is **not proof of an earlier assignment**. A seed alone cannot reconstruct an unspecified shuffling process. No held-out performance estimate is available from the pooled reference summaries.

### 3.2 Leakage checks

Use sequence-level membership when partitioning videos. Do not place different frames, objects, or prompt variants of the same sequence in different partitions. The library provides `dataset_split.check_partition_integrity` to detect sequence overlap when records include partition identifiers.

Other safeguards are:

- Save the exact sequence assignment and input hashes alongside results.
- Inspect duplicate frame-content hashes before using partitions.
- Choose confidence thresholds and memory intervals using development/validation data rather than inspecting test outcomes.
- Report official-split and pooled statistics separately, when corresponding data exist.
- Check whether a pretrained model's training material overlaps evaluated videos; overlap has not been established or ruled out here.
- Treat repeated observations from one sequence as dependent. For example, 150 sequence-by-strategy groups derived from 50 source videos are not 150 independent videos.

These checks provide controls for *new* evaluations; they do not retroactively verify dataset partitions used in unavailable experiments.

## 4. Metric definitions

Unless specified otherwise, masks are binary and all numerical evaluations should use a documented aggregation unit. In particular, per-pixel, per-frame, per-object, and per-sequence averages are not interchangeable.

### 4.1 Segmentation accuracy

**Region Jaccard index:**

\[
J(\widehat M,G)=\frac{|\widehat M\cap G|}{|\widehat M\cup G|}
\]

An empty prediction against a non-empty reference mask receives 0. If both masks are empty, the implementation assigns 1, following the chosen DAVIS-style convention.

**Prediction correctness:**

\[
y=\mathbf{1}[J\geq 0.50]
\]

**Boundary F-score:** One-pixel mask boundaries are extracted and matched using a tolerance of `ceil(0.008 × image diagonal)` pixels. With boundary precision `P` and recall `R`:

\[
F=\frac{2PR}{P+R}
\]

Both-empty boundary cases are assigned 1. The boundary tolerance is an explicit implementation choice and may differ from settings used to produce earlier results.

### 4.2 Area-based visibility

For object area `A_t` at frame `t`, let `A_ref` be the first positive annotated area:

\[
a_t=\frac{A_t}{A_{ref}},\qquad b_t=\mathbf{1}[a_t<0.05]
\]

Frames before the first positive reference area are undefined (`NaN`) and are not treated as low-visibility frames. Consecutive low-visibility frames form runs:

| Run category | Meaning |
|---|---|
| `short` | Completed run shorter than 3 frames |
| `confirmed` | Completed run of at least 3 frames |
| `censored` | Run of at least 3 frames still active at the last frame |
| `short_censored` | Open run shorter than 3 frames at the last frame |

For a completed run, gap length is `t_end − t_start + 1` frames and the first restored-area frame is `t_end + 1`. At 24 FPS, the duration in milliseconds is `1,000 × frames / 24`.

A separately defined, Jaccard-based recovery delay may count frames after visible-area restoration until a prediction again reaches a specified Jaccard threshold. That threshold has no established default and this quantity must not be conflated with area restoration or identity-verified reacquisition.

Kaplan–Meier survival analysis can treat completed runs as events and open runs as censored. Cantelli's inequality provides an upper probability bound for deviations above a mean:

\[
P(X\geq\mu+k)\leq\frac{\sigma^2}{\sigma^2+k^2}
\]

**Interpretation limit:** A reduction in annotated area may reflect occlusion, truncation, scale change, or movement outside the image. Restored area alone does not verify the object's identity.

### 4.3 Centroid trajectories and Allan variance

The centroid is the mean `(x, y)` position of foreground pixels; empty masks produce `NaN` coordinates. Path length sums distances between consecutive valid centroids, omitting steps adjacent to undefined positions.

Allan variance uses non-overlapping windows of length `τ`, with `K = floor(T / τ)` available windows. At least two windows are required. Differences involving entirely empty windows are excluded. Each `(object track, τ)` pair is one Allan-variance observation. Results depend on this choice of windowing and missing-data handling.

### 4.4 Optical flow

RAFT estimates forward and backward pixel displacement. Pixel motion magnitude is the Euclidean norm of the estimated displacement. A forward/backward consistency residual compares the forward flow with the backward flow sampled at its displaced target position.

Backward flow is sampled bilinearly. Locations mapped beyond the image are assigned `NaN` and excluded from consistency-violation statistics. The violation rate is the fraction of valid tested pixels with residual greater than **1 pixel**.

A robust Huber-location estimate uses `δ = 1` and iteratively reweighted least squares initialized from the median. The implemented record unit is one `(sequence, object, frame pair)` observation, restricted to object pixels; other aggregation schemes may produce different counts.

### 4.5 Confidence calibration and gating

Confidence values use 10 bins, with bin index `min(floor(10q), 9)`, so confidence 1.0 belongs in the final bin. Expected calibration error (ECE) is the weighted absolute difference between bin accuracy and mean bin confidence:

\[
\operatorname{ECE}=\sum_{b\in\mathcal B}\frac{n_b}{N}\left|\operatorname{acc}(b)-\operatorname{conf}(b)\right|
\]

The acceptance rule requires both sufficient confidence and a non-empty predicted mask:

\[
g=\mathbf{1}[q\geq q^*]\,\mathbf{1}[|\widehat M|>0]
\]

Counts are separated into accepted-correct (`AC`), accepted-wrong (`AW`), deferred-correct (`DC`), and deferred-wrong (`DW`). Using `N = AC + AW + DC + DW`:

| Metric | Definition |
|---|---|
| Coverage | `(AC + AW) / N` |
| Error rejection | `DW / (AW + DW)` |
| False acceptance | `AW / (AW + DW)` |
| Error among accepted outputs | `AW / (AC + AW)` |

Ratios with zero denominators are undefined rather than zero. Confidence intervals can be estimated with **2,000 sequence-cluster bootstrap resamples**, using whole-sequence resampling and percentile 95% intervals. The toolkit uses seed 42 as an explicit reproducibility choice; invalid or undefined bootstrap replicates are counted and excluded.

### 4.6 Descriptive statistics

Sample standard deviation uses `ddof=1`. Quartiles use NumPy linear interpolation. Statistical results should identify the sampled unit and whether repeated frames from the same video were clustered.

Some quantities—including a composite temporal-stability score, drift, an unspecified interruption-delay definition, reward formula, and certain simulation state summaries—lack sufficiently defined computation rules. They should not be assigned invented formulas or treated as reproduced values.

## 5. Runtime measurement

A processing record separates four stages:

| Stage | Measured work in the segmentation runner |
|---|---|
| Preprocess | Video state initialization and prompt registration, amortized across frames |
| Inference | A video-propagation step |
| Postprocess | Convert model outputs to a binary mask and obtain confidence |
| Publish | Compute J/F metrics and append the per-frame record |

Total per-frame latency is the sum of stage latencies. Nominal throughput is `1,000 / mean latency (ms)`. Deadline compliance is the fraction of frames whose total latency falls within **33 ms**. Nominal FPS is an average and does not imply every frame satisfies a deadline.

### 5.1 Reference latency values

The following figures are retained only as **unverified reference summaries**. They have not been reproduced from frame-level timing data.

| Quantity | Reference value |
|---|---:|
| Mean total latency | 10.1521 ms |
| Sample standard deviation | 4.68 ms |
| Maximum latency | 111.11 ms |
| Nominal FPS from mean latency | 98.50 FPS |
| Mean preprocess time | 2.3023 ms |
| Mean inference time | 0.3270 ms |
| Mean postprocess time | 4.8003 ms |
| Mean publish time | 2.7225 ms |

The four stage means sum to the reported total mean. This arithmetic agreement **does not validate the timing measurements**. In particular, the 0.3270-ms inference figure is difficult to interpret without synchronization and stage-boundary evidence. Asynchronous GPU execution is one possible explanation, but the cause is unknown. The maximum also exceeds the 33-ms reference deadline. The reference data do not establish p95, p99, or the fraction of missed deadlines.

Available descriptions mention more than one software environment (including PyTorch 2.1/CUDA 11 and Python 3.10/PyTorch 2.5.1/CUDA 12.4), but do not identify a single verified hardware/software configuration for the timing rows.

### 5.2 Reproducible profiling

Use:

```bash
python scripts/run_segmentation.py --config configs/sam2.toml --profile
```

When CUDA is available, profiling synchronizes execution at the measured stage boundaries. The runtime utilities can record environment information, warm-up exclusions, stage statistics, nominal FPS, p90/p95/p99, and deadline misses. Preserve the GPU model, driver, CUDA version, PyTorch version, SAM-2 checkpoint, numeric precision, batch/sequence conditions, synchronization choice, and whether loading/warm-up was included.

The reference documents do **not** establish that this GPU profiling path has been executed. Newly measured timings must remain separate from the unverified reference summaries.

## 6. Installation

Python 3.11 or newer is required by the package configuration.

Create and activate a virtual environment:

**Linux/macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows Command Prompt**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

For GPU-dependent work, also install the GPU requirements and the official SAM-2 package:

```bash
pip install -r requirements-gpu.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Obtain the required checkpoint from the [SAM-2 project](https://github.com/facebookresearch/sam2). For the Princeton RAFT backend, install the [RAFT implementation](https://github.com/princeton-vl/RAFT), make its `core` package importable, and provide `raft-things.pth`. Compatible PyTorch, CUDA, and torchvision versions depend on the machine and selected backend.

## 7. Execution guide

### 7.1 Tests and synthetic checks

```bash
python -m pytest
python scripts/smoke_test.py
```

These validate code behavior on fixtures or generated examples. They do not validate historical benchmark scores or actual GPU performance.

### 7.2 Dataset preparation and CPU analysis

```bash
python scripts/prepare_davis.py --config configs/davis.toml
python scripts/run_visibility_analysis.py --config configs/davis.toml
```

These commands need an accessible DAVIS dataset. Outputs include dataset manifests, visibility frames/runs, centroid trajectories, Allan variance, and metadata.

### 7.3 Record-based evaluation

Place experimental CSVs in a known input directory and use the required schemas from `src/memory_guided_vos/records.py`.

```bash
python scripts/run_calibration.py --records data/original/calibration_pairs.csv
python scripts/run_configuration_comparison.py --records data/original/configuration_frames.csv
python scripts/run_cartesian_simulation.py --summarise
```

The Cartesian summary reads paths configured in `configs/cartesian_simulation.toml`. Record-based commands require the corresponding files and must not silently substitute synthetic inputs for missing measurements.

For a separately configured simulation run:

```bash
python scripts/run_cartesian_simulation.py --reconstruct
```

The reconstruction requires all necessary motion parameters and step-level target observations. It produces new synthetic/reconstructed data, not recovered historical episode logs.

### 7.4 GPU evaluation

Set the SAM-2 model configuration, checkpoint, confidence source, memory mechanism, and device in `configs/sam2.toml`. Configure the RAFT backend and checkpoint in `configs/raft.toml`.

```bash
python scripts/run_segmentation.py --config configs/sam2.toml --profile
python scripts/run_optical_flow.py --config configs/raft.toml
```

Segmentation outputs include per-frame masks' evaluation metrics, confidence values, stage timing, and run metadata. Optical-flow outputs include object-level frame-pair measurements, forward/backward consistency, robust summaries, and metadata. Actual GPU execution must be verified on the target machine.

### 7.5 Data lineage and outputs

Keep experimental inputs and output reports separate:

```text
data/raw/DAVIS/              External benchmark files
data/original/               Input records supplied for comparison
data/manifests/              Dataset inventory and split information
results/                    Generated measurements and metadata
figures/generated/          Generated visualizations
```

Record input paths, configuration settings, software/hardware environment, sequence membership, and content hashes. The tool's provenance metadata distinguishes synthetic checks from new runs and external records. A missing input is an error, not evidence that the corresponding result was reproduced.

## 8. Verification status and constraints

The available source documents describe CPU analysis and synthetic testing, but **do not contain the underlying records or source implementation for the historical experimental results**. Consequently:

- Numerical reference summaries cannot be independently recomputed from their originating observations.
- The exact 50-sequence membership, partition assignment, and some group definitions are not established.
- Certain runtime stage measurements cannot be attributed to a single verified device and measurement procedure.
- The SAM-2 and RAFT GPU execution paths are described but were not run in the documented CPU-only environment.
- Several metric definitions and simulation parameters were absent and therefore require explicit new-run choices.
- Area-based restoration does not demonstrate identity-verified recovery.
- Joint configuration comparisons do not isolate individual hyperparameter effects.
- No cross-dataset validation, physical robotics experiment, or guaranteed real-time operation is established.

A previous CPU-only software environment used to exercise the toolkit was reported as Windows 11, Python 3.15.0, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, Pillow 13.0.0, and pytest 9.1.1. This is environment context, **not** a verified GPU measurement setup or a locked dependency environment.

The correct interpretation is that the toolkit supports repeatable evaluation *when the necessary data and settings are supplied*. Existing synthetic checks, internally consistent arithmetic, and plausible reference figures must remain distinct from independently verified experimental outcomes.

## 9. Scope and responsible interpretation

This project evaluates video segmentation and related tracking signals in a benchmark setting. Its results do not, by themselves, establish generalization to other datasets, recovery of an object's identity after occlusion, full-system deadline guarantees, or robotic grasp performance.

Any new dataset split, threshold selection, simulator reconstruction, or hardware performance run should be identified by its own configuration and provenance. Unknown quantities should stay undefined until their source records or operational definitions are available.
