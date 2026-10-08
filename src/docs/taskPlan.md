# Task Plan: `java-cnn-char-recog` Implementation Roadmap

This document serves as the implementation tracking sheet for executing tasks via **Antigravity CLI**.

---

## Progress Overview
- [x] **Phase 0: Core Foundation & Bug Fixes** (Completed)
- [x] **Phase 1: CLI Configuration & Interactive Control Suite** (Completed)
- [ ] **Phase 2: Model Weights Persistence**
- [ ] **Phase 3: Preprocessing & Drawing Canvas**
- [ ] **Phase 4: Core Java GUI Shell & Real-time Prediction**
- [ ] **Phase 5: Intermediate Activation & Feature Map Visualization**
- [ ] **Phase 6: Live Training Dashboard & Analytics**
- [ ] **Phase 7: Packaging, Polishing & Verification**

---

## Detailed Task Breakdown

### Phase 0: Core Foundation & Bug Fixes (Completed)
- [x] **TASK-001: DataReader Header Resilience & Multi-Path Resolution**
  - **Goal**: Make `DataReader.java` handle CSV header lines (`"label,..."`) and auto-resolve dataset paths (`data/`, `../../data/`).
  - **Status**: Completed and verified against full 60,000 train / 10,000 test dataset.

- [x] **TASK-002: Fix MaxPoolLayer Dimension Bug**
  - **Goal**: Correct `MaxPoolLayer.getOutputElements()` calculation from `_inRows * ...` to `getOutputLength() * ...`.
  - **Status**: Completed; eliminated `ArrayIndexOutOfBoundsException` on forward pass.

- [x] **TASK-003: Core End-to-End Sanity Test Suite**
  - **Goal**: Verify forward pass, loss calculation, backprop, and accuracy evaluation on a controlled slice.
  - **Status**: Completed; tested 3 epochs with `--quick` flag showing accuracy improvement (9% -> 17%).

---

### Phase 1: CLI Configuration & Interactive Control Suite (Immediate Focus)

- [x] **TASK-101: `AppConfig` Model & Immutable Configuration**
  - **Goal**: Create `cli.AppConfig` to encapsulate all hyperparameter, training, and runtime settings.
  - **Details**:
    - Default configuration matching current stable settings (Seed: 123, ScaleFactor: 256*100, Conv: 8 filters of 5x5, stride 1, lr 0.1; Pool: window 3, stride 2; FC: 10 classes, lr 0.1; Epochs: 3).
    - Immutable getters and builder pattern for safe partial overrides.
  - **Status**: Completed; implemented `cli.AppConfig` with immutable state, fluent `Builder`, derived CNN output dimension calculations, and comprehensive tests in `test.Phase1Test`.

- [x] **TASK-102: `ConfigValidator` (Math & Constraint Validation Engine)**
  - **Goal**: Create `cli.ConfigValidator` with mathematical validity and type/range verification.
  - **Details**:
    - Validate dimensions: $\text{ConvOutRows} = (28 - \text{FilterSize})/\text{ConvStride} + 1 > 0$.
    - Validate pooling: $\text{PoolOutRows} = (\text{ConvOutRows} - \text{PoolWindow})/\text{PoolStride} + 1 > 0$.
    - Check ranges: Learning rates $(0.0, 1.0]$, epochs $\ge 1$, scale factor $> 0$, filters $\ge 1$.
    - Throw descriptive `ConfigValidationException` explaining the mathematical reason for any failure.
  - **Status**: Completed; implemented `cli.ConfigValidator` and `cli.ConfigValidationException` enforcing CNN mathematical constraints and parameter bounds, verified with invalid input tests in `test.Phase1Test`.

- [x] **TASK-103: `CliArgsParser` (Flag & Option Parser)**
  - **Goal**: Build pure Core Java flag parser supporting standard POSIX/GNU-style arguments.
  - **Details**:
    - Flags: `--help`/`-h`, `--custom-configuration`/`-c`, `--verbose`/`-v`, `--quick`/`-q`.
    - Hyperparameter flags: `--epochs`/`-e`, `--filters`, `--filter-size`, `--conv-stride`, `--conv-lr`, `--pool-window`, `--pool-stride`, `--fc-lr`, `--scale-factor`, `--seed`, `--train-limit`, `--test-limit`.
    - Partial override support: Passing only `--epochs 5` overrides epochs while keeping all other defaults.
    - Helpful `--help` screen with syntax, default values, and usage examples.
  - **Status**: Completed; implemented `cli.CliArgsParser` and `cli.CliParseException` supporting GNU/POSIX flags, synonyms, inline syntax (`--flag=val`), partial overrides, and comprehensive help manual, verified in `test.Phase1Test`.

- [x] **TASK-104: `InteractiveConfigWizard` (Step-by-Step Terminal Prompts)**
  - **Goal**: Implement guided interactive CLI wizard for `--custom-configuration`.
  - **Details**:
    - Sequentially prompt the user for each customizable parameter.
    - Display default value in prompt (e.g. `Convolution Filters [default: 8]: `); pressing Enter accepts default.
    - Real-time input parsing and re-prompting on invalid syntax or out-of-bounds values.
  - **Status**: Completed; implemented `cli.InteractiveConfigWizard` with sequential prompts, default value fallbacks, immediate mathematical/range validation, and error recovery, verified in `test.Phase1Test`.

- [x] **TASK-105: Educational Verbose Mode (`--verbose` / `-v`)**
  - **Goal**: Integrate parameter explanations into both flag execution and the interactive wizard.
  - **Details**:
    - For each parameter, provide:
      1. **Meaning**: Conceptual explanation of the parameter in CNN architecture.
      2. **Effect**: What changes when increased or decreased (e.g. computation time, feature depth, overfitting risk).
      3. **Recommended Range**: Practical bounds for MNIST handwritten digits.
  - **Status**: Completed; implemented `cli.ParameterExplainer` with comprehensive architectural meaning, impact analysis, and recommended MNIST ranges, integrated into `InteractiveConfigWizard` and CLI verbose mode, verified in `test.Phase1Test`.

- [x] **TASK-106: `ConfigRenderer` (ASCII Summary Formatter)**
  - **Goal**: Format and display an aesthetic ASCII summary table before model execution.
  - **Details**:
    - Display table with:
      - Dataset paths and sample limits.
      - Conv layer specs, kernel size, output feature shape ($8 \times 24 \times 24$).
      - MaxPool specs, window, output pooled shape ($8 \times 11 \times 11$, total 968 elements).
      - Fully connected specs, input nodes, output classes, and learning rate.
      - Training epochs, scale factor, and seed.
  - **Status**: Completed; implemented `cli.ConfigRenderer` displaying cleanly aligned ASCII hyperparameter cards and layer tensor shapes, verified in `test.Phase1Test`.

- [x] **TASK-107: `Main.java` Integration & Execution Pipeline**
  - **Goal**: Wire up CLI subsystem with core CNN training pipeline.
  - **Details**:
    - Parse `args` -> Validate -> Print ASCII summary -> Build network via `NetworkBuilder` -> Train & evaluate.
    - If run with no args (`java Main`), immediately runs default mode with the ASCII summary printed first.
  - **Status**: Completed; integrated `CliArgsParser`, `InteractiveConfigWizard`, `ConfigValidator`, and `ConfigRenderer` into `Main.java`, verified default execution, flag-override execution (`--epochs 1 --filters 4 --quick`), interactive wizard (`--custom-configuration`), help system (`--help`), and educational mode (`--verbose`).

---

### Phase 2: Model Weights Persistence
- [ ] **TASK-201: `ModelSerializer` Implementation**
  - **Goal**: Export trained model weights to disk and reload them without retraining.
  - **Details**:
    - Serialize convolution filter weights and fully connected weight matrix.
    - Add `--save <path>` and `--load <path>` flags to CLI.
  - **Verification**: Train for 1 epoch, save weights, reload into fresh network, verify identical test outputs.

---

### Phase 3: Preprocessing & Drawing Canvas (Future GUI Phase)
- [ ] **TASK-301: ImagePreprocessor (Canvas to MNIST Format)**
  - **Goal**: Convert freehand drawings to clean $28 \times 28$ inputs with center-of-mass alignment.
  - **Verification**: Unit test comparing drawn digit downsampling with MNIST sample distributions.

- [ ] **TASK-302: Interactive Drawing Canvas Component**
  - **Goal**: Custom Swing `JComponent` with anti-aliasing, stroke dilation, and clear actions.
  - **Verification**: Smooth 60 FPS drawing in UI.

---

### Phase 4: Core Java GUI Shell & Real-time Prediction
- [ ] **TASK-401: Main Application Window (`MainWindow`)**
  - **Goal**: Establish responsive Swing layout (Canvas, Feature Maps, Prediction Bar Chart).
  - **Verification**: Clean rendering across high-DPI displays.

- [ ] **TASK-402: Prediction Probability Bar Chart (`PredictionBarChart`)**
  - **Goal**: Java2D horizontal bar chart showing live 0–9 digit confidences.
  - **Verification**: Real-time updates as user strokes canvas.

---

### Phase 5: Intermediate Activation & Feature Map Visualization
- [ ] **TASK-501: Layer Activation Capture (`NetworkState`)**
  - **Goal**: Expose intermediate activations (Conv 8 maps, MaxPool 8 maps, FC logits).
  - **Verification**: Non-null matrices verified with matching dimensions.

- [ ] **TASK-502: Feature Map Visualizer Component (`FeatureMapVisualizer`)**
  - **Goal**: Render intermediate activations as dynamic heatmaps.
  - **Verification**: Heatmaps update in sync with canvas drawing.

---

### Phase 6: Live Training Dashboard & Analytics
- [ ] **TASK-601: Background Training Worker (`TrainingWorker`)**
  - **Goal**: Non-blocking training execution using `SwingWorker`.
  - **Verification**: UI remains responsive during active training.

- [ ] **TASK-602: Training Dashboard Panel (`TrainingDashboardPanel`)**
  - **Goal**: Java2D line plots for loss and accuracy over epochs.
  - **Verification**: Real-time curve updates on epoch completion.

---

### Phase 7: Packaging, Polishing & Verification
- [ ] **TASK-701: Standalone Application Launcher & README Update**
  - **Goal**: Complete CLI and GUI documentation in `README.md`.
  - **Verification**: Clean build and execution from terminal using standard JDK.
