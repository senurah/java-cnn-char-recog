# Agent Rules & Guidelines: `java-cnn-char-recog`

This document defines the operational constraints, design guidelines, and code conventions for autonomous AI agents working on this repository via Antigravity CLI.

---

## 1. Core Engineering Principles

### 1.1 Zero External ML Bloat (Pure Core Java)
- **Constraint**: Do not introduce heavy third-party deep learning libraries (e.g., PyTorch, Deeplearning4j, TensorFlow Java, Weka).
- **Rationale**: The educational and core objective of this project is a transparent, handwritten implementation of convolutional neural networks in pure Java.
- **Allowed Libraries**:
  - Java Standard Library (`java.desktop` / Swing / AWT, `java.util.concurrent`, `java.nio`, `java.io`).
  - Lightweight utility libraries (e.g., modern FlatLaf for Swing styling or lightweight JSON/serialization if strictly necessary and approved).

### 1.2 Mathematical & Algorithmic Integrity
- Never alter core mathematical formulas (convolution filters, pooling logic, ReLU activations, learning rate updates, and backpropagation gradients) unless fixing verified dimension or calculation bugs.
- Always document the input/output tensor dimensions (channels, rows, columns) when modifying or creating layer classes.

### 1.3 Non-Blocking GUI Threading
- **Strict Rule**: Never execute neural network training, batch testing, or disk I/O on the Swing **Event Dispatch Thread (EDT)**.
- Use `javax.swing.SwingWorker` or `java.util.concurrent.ExecutorService` with `SwingUtilities.invokeLater(...)` to push UI updates, progress metrics, and feature maps asynchronously.

---

## 2. Code Quality & Architecture Standards

### 2.1 Package Organization & Boundaries
Maintain strict separation of concerns across packages:
- `data`: Dataset loading, sample representations, normalization, preprocessing, and matrix utilities.
- `layers`: Individual CNN layers (`ConvolutionLayer`, `MaxPoolLayer`, `FullyConnectedLayer`) inheriting from abstract `Layer`.
- `network`: Network orchestration, forward/backward execution, builder pattern, and model serialization.
- `gui`: Swing presentation layer, drawing canvases, layer activation renderers, and chart panels.
- `service`: Background training workers, model persistence manager, and event listeners.

### 2.2 Error Handling & Logging
- **Anti-Pattern**: Do NOT swallow exceptions or mask specific runtime errors with generic error messages (e.g., catching `Exception` and throwing `IllegalArgumentException("File not found")`).
- Always preserve the root cause using chained exceptions (`new IOException("Failed to parse MNIST CSV at line " + lineNum, e)`).
- Log actionable diagnostic messages with row numbers, expected dimensions, and actual dimensions.

### 2.3 Verification & Testing
- Before marking any task complete:
  1. Compile with the project JDK (`javac`).
  2. Run a verification test with a small data slice (e.g., 50–100 samples) to ensure forward pass, backward pass, and GUI event handling work without crashes.
  3. Validate that UI repaints cleanly without memory leaks or dropped frames.

---

## 3. Workflow & Task Execution Checklist

When taking up a task from [`taskPlan.md`](file:///home/senura/projects/personal/java-cnn-char-recog/src/docs/taskPlan.md):
1. **Scope Check**: Read the relevant section in [`architecture.md`](file:///home/senura/projects/personal/java-cnn-char-recog/src/docs/architecture.md) and confirm acceptance criteria.
2. **Implementation**: Maintain backward compatibility with existing constructors where feasible.
3. **Self-Review**: Verify thread safety, dimension correctness, and absence of hardcoded absolute paths.
4. **Update Status**: Mark the task checkbox `[x]` in [`taskPlan.md`](file:///home/senura/projects/personal/java-cnn-char-recog/src/docs/taskPlan.md) with brief notes on changes made.
