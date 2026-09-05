# Perceptron

A lightweight single-layer Perceptron (Logistic Regression) binary classifier implemented in Python, backed by a high-performance C++20 matrix operations library (`matrixlib`) via Pybind11 bindings.

[![Continuous Integration](https://github.com/Kawenka/perceptron/actions/workflows/ci.yml/badge.svg)](https://github.com/Kawenka/perceptron/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Input Data Format](#input-data-format)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option 1: Standard Installation via pip (Recommended)](#option-1-standard-installation-via-pip-recommended)
  - [Option 2: Compiling `matrixlib` from Source via Pybind11 / CMake](#option-2-compiling-matrixlib-from-source-via-pybind11--cmake)
  - [Option 3: Using a Git Submodule](#option-3-using-a-git-submodule)
- [Usage & Execution](#usage--execution)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Data Security & Privacy](#data-security--privacy)
- [License](#license)

---

## Overview

This project implements a single-layer neural network (Perceptron) for binary classification tasks. The computational heavy lifting (matrix multiplications, additions, element-wise transformations, reductions) is performed by `matrixlib`, an external C++20 matrix library exposed to Python through Pybind11.

Key features:
- **Binary Cross-Entropy Loss** with numerical stabilization (log-loss clipping).
- **Gradient Descent Optimization** computing exact analytical gradients for weights ($dW$) and bias ($db$).
- **Dynamic Feature Handling**: automatically adapts to any number of features defined in the CSV header.
- **Dynamic Min-Max Normalization**: scales input feature columns to the range $[0, 1]$ based on observed maximums during training.
- **Configurable CLI Interface**: fully customizable training dataset, hyperparameters (epochs, learning rate), and single-entry inference using `argparse`.

---

## Project Architecture

```
perceptron/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (multi-version Python testing)
├── data/
│   └── example.csv            # Synthetic demonstration dataset for binary classification
├── model/
│   ├── __init__.py            # Model package marker
│   └── neuron.py              # Neuron class (forward, backward, loss, parameter update)
├── .gitignore                 # Excludes cache, venv, build artifacts, and private data
├── LICENSE                    # MIT Open Source License
├── main.py                    # Main CLI entry point (data loading, training loop, evaluation)
├── README.md                  # Comprehensive project documentation
└── requirements.txt           # Python dependencies (NumPy and matrixlib from GitHub)
```

### Component Details

- **`main.py`**:
  The orchestrator of the pipeline. It parses CLI arguments, loads and validates CSV input data, scales features dynamically, initializes the `Neuron`, executes the gradient descent training loop, reports progress and loss metrics, outputs the final learned weights and bias, and performs inference on custom samples.

- **`model/neuron.py`**:
  Contains the core `Neuron` class implementing:
  - **Weight Initialization**: Initialized with uniform random values in $[-1.0, 1.0]$ and bias set to $0.0$.
  - **Forward Pass**: Computes affine transformation $Z = X \cdot W + b$, followed by Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  - **Loss Function**: Binary Cross-Entropy loss $L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i)]$, clamped with $\epsilon = 10^{-15}$ to prevent numerical overflow/underflow.
  - **Backward Pass**: Vectorized gradient computation:
    $$dW = \frac{1}{N} X^T (\hat{Y} - Y), \quad db = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
  - **Update Parameters**: Gradient descent parameter step:
    $$W \leftarrow W - \alpha \cdot dW, \quad b \leftarrow b - \alpha \cdot db$$

- **`matrixlib`** (External C++20 Dependency):
  Underlying C++ matrix library providing bounds-checked element access, memory management, and linear algebra operations exposed via Pybind11.

---

## Input Data Format

The pipeline enforces a **strict format** for input datasets to ensure valid training and evaluation:

1. **File Format**: Standard comma-separated values (`.csv`).
2. **Header Row**: The first line must contain comma-separated column names (e.g., `feature_1,feature_2,feature_3,target`).
3. **Values**: All data cells must contain strictly **numerical values** (`int` or `float`). Non-numerical strings or missing/empty values will cause the loader to abort with an error.
4. **Target Column (Last Column)**:
   - The binary target variable **must strictly reside in the last column**.
   - Target values **must strictly be binary**: either `0.0` or `1.0`.
   - Any value other than `0.0` or `1.0` in the target column will trigger an immediate validation error.
5. **Feature Columns**:
   - All columns preceding the final column are treated as input features.
   - The model dynamically adjusts its input dimension to match the number of feature columns.

### Example Valid CSV (`data/example.csv`):

```csv
feature_1,feature_2,feature_3,target
12.5,4.0,3.0,1.0
10.0,3.5,2.0,1.0
14.0,4.0,3.0,1.0
8.5,2.0,1.0,0.0
5.0,1.5,0.0,0.0
```

---

## Requirements

- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **Python**: 3.10, 3.11, or 3.12
- **C++ Compiler**: C++20 compatible (GCC 11+, Clang 13+, or MSVC 2019+)
- **CMake**: 3.20 or newer
- **Git**: For repository management and cloning dependencies

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kawenka/perceptron.git
cd perceptron
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(On Windows PowerShell: `.venv\Scripts\Activate.ps1`)*

---

### Option 1: Standard Installation via pip (Recommended)

The repository's `requirements.txt` points directly to the `matrixlib` GitHub repository. Pip downloads, compiles the C++20 code via Pybind11, and installs the module in one step:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Option 2: Compiling `matrixlib` from Source via Pybind11 / CMake

If you wish to compile `matrixlib` manually or develop against its C++ source code:

1. **Clone the `matrixlib` repository:**
   ```bash
   git clone https://github.com/Kawenka/matrixlib.git /tmp/matrixlib
   cd /tmp/matrixlib
   ```

2. **Configure with CMake (enabling Python bindings):**
   ```bash
   cmake -B build \
     -DMATRIXLIB_BUILD_PYTHON_BINDINGS=ON \
     -DPython_EXECUTABLE=$(which python3)
   ```
   *(Note: Pybind11 is fetched automatically during the CMake configuration via `FetchContent`)*

3. **Build the extension module:**
   ```bash
   cmake --build build --target matrixlib_py
   ```

4. **Install into your environment or add to `PYTHONPATH`:**
   ```bash
   # Either install with pip from the matrixlib root:
   pip install .

   # Or export the build directory to your PYTHONPATH:
   export PYTHONPATH=/tmp/matrixlib/build:$PYTHONPATH
   ```

---

### Option 3: Using a Git Submodule

If you prefer managing `matrixlib` as a submodule inside the project:

```bash
git submodule add https://github.com/Kawenka/matrixlib.git lib/matrixlib
pip install ./lib/matrixlib
```

To clone the repository and initialize submodules in a single step in the future:

```bash
git clone --recurse-submodules https://github.com/Kawenka/perceptron.git
pip install ./lib/matrixlib
```

---

## Usage & Execution

### Command-Line Arguments

The `main.py` entry point accepts the following arguments:

| Argument | Short Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | `-d` | `str` | `data/example.csv` | Path to the CSV dataset file. |
| `--epochs` | `-e` | `int` | `3000` | Total number of training iterations. |
| `--lr`, `--learning-rate` | `-l` | `float` | `0.5` | Learning rate for gradient descent. |
| `--predict` | `-p` | `float ...` | `None` | Optional space-separated feature values for single-sample inference. |
| `--help` | `-h` | - | - | Displays usage instructions and option summary. |

---

### Examples

#### 1. Run with Default Demonstration Dataset

Trains the model on `data/example.csv` with default parameters (3000 epochs, lr=0.5):

```bash
python main.py
```

#### 2. Run with a Custom Dataset

Specify your own CSV dataset file:

```bash
python main.py --dataset path/to/my_data.csv
```

#### 3. Customize Hyperparameters (Epochs & Learning Rate)

```bash
python main.py --dataset data/example.csv --epochs 5000 --lr 0.1
```

#### 4. Perform Single-Entry Prediction on New Values

Pass feature values via `--predict` to evaluate a custom input sample after training:

```bash
python main.py --dataset data/example.csv --predict 14.0 4.0 3.0
```

*Expected output excerpt:*
```
--- Evaluating Single Entry ---
feature_1: 14.0
feature_2: 4.0
feature_3: 3.0
Result -> Probability: 0.9999 | Predicted Class: 1.0
```

---

## Data Security & Privacy

This repository complies with data security best practices:
- **No Private Data in History**: Real candidate evaluation records or sensitive client data must never be committed.
- **Git Ignore**: The `.gitignore` configuration explicitly excludes sensitive dataset files (`data/data.csv`), Python virtual environments (`.venv/`), bytecode caches (`__pycache__/`, `*.pyc`), and compiled C++ extension objects (`*.so`, `build/`).
- **Demonstration Data**: Only synthetic/anonymized demonstration datasets (`data/example.csv`) are tracked in source control to verify algorithmic correctness.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
