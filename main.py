from matrixlib import Matrix
from model.neuron import Neuron
import argparse
import csv
import os
import sys

def load_data(filepath: str) -> tuple:
    """
    Load and preprocess data from a CSV file.
    Assumes the last column is the binary target (0.0 or 1.0)
    and all preceding columns are numerical features.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        tuple: (X_matrix, Y_matrix, max_values_list, feature_names_list)
    """
    if not os.path.exists(filepath):
        print(f"Error: Dataset file not found at '{filepath}'.", file=sys.stderr)
        sys.exit(1)

    raw_data = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            print(f"Error: Dataset file '{filepath}' is empty.", file=sys.stderr)
            sys.exit(1)

        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(field.strip() for field in row):
                continue
            try:
                float_row = [float(val.strip()) for val in row]
            except ValueError as e:
                print(
                    f"Error: Line {row_idx} in '{filepath}' contains non-numerical values: {row}",
                    file=sys.stderr
                )
                sys.exit(1)
            raw_data.append(float_row)

    if not raw_data:
        print(f"Error: No data rows found in '{filepath}'.", file=sys.stderr)
        sys.exit(1)

    num_samples = len(raw_data)
    num_features = len(header) - 1

    if num_features < 1:
        print(
            f"Error: Dataset must have at least one feature column and one target column.",
            file=sys.stderr
        )
        sys.exit(1)

    # Validate binary target column (last column must be 0.0 or 1.0)
    for row_idx, row in enumerate(raw_data, start=2):
        target = row[-1]
        if target not in (0.0, 1.0):
            print(
                f"Error: Invalid target value '{target}' at row {row_idx}. "
                f"The target in the last column must be strictly 0.0 or 1.0.",
                file=sys.stderr
            )
            sys.exit(1)

    X = Matrix(num_samples, num_features)
    Y = Matrix(num_samples, 1)

    max_values = []
    for j in range(num_features):
        column_values = [abs(row[j]) for row in raw_data]
        max_val = max(column_values)
        if max_val == 0.0:
            max_val = 1.0
        max_values.append(max_val)

    for i in range(num_samples):
        for j in range(num_features):
            X[i, j] = raw_data[i][j] / max_values[j]
        Y[i, 0] = raw_data[i][-1]

    feature_names = [name.strip() for name in header[:-1]]
    return X, Y, max_values, feature_names

def predict_single_entry(model: Neuron, new_data: list, max_values: list, feature_names: list) -> None:
    """
    Predict the outcome for a single new entry using dynamic normalization.
    
    Args:
        model (Neuron): The trained neural network model.
        new_data (list): The raw feature values to test.
        max_values (list): The maximum values used during training for normalization.
        feature_names (list): The names of the features for display.
    """
    num_features = len(new_data)
    X_new = Matrix(1, num_features)
    
    # Apply the exact same normalization as the training phase
    for j in range(num_features):
        X_new[0, j] = new_data[j] / max_values[j]
        
    prediction_matrix = model.forward(X_new)
    probability = prediction_matrix[0, 0]
    status = 1.0 if probability >= 0.5 else 0.0
    
    print("\n--- Evaluating Single Entry ---")
    for j in range(num_features):
        print(f"{feature_names[j]}: {new_data[j]}")
    print(f"Result -> Probability: {probability:.4f} | Predicted Class: {status}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-layer Perceptron (Logistic Regression) training and prediction pipeline."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="data/example.csv",
        help="Path to the input CSV dataset (default: data/example.csv)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3000,
        help="Number of training epochs (default: 3000)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", "-l",
        type=float,
        default=0.5,
        dest="learning_rate",
        help="Learning rate for gradient descent (default: 0.5)"
    )
    parser.add_argument(
        "--predict", "-p",
        nargs="+",
        type=float,
        default=None,
        help="Space-separated feature values for single-entry prediction (optional)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("--- Loading Data ---")
    print(f"Dataset path: {args.dataset}")
    X, Y, max_values, feature_names = load_data(args.dataset)

    num_features = len(feature_names)
    print(f"Loaded {X.rows} training samples with {num_features} features: {feature_names}")

    model = Neuron(num_features=num_features)

    print(f"\n--- Starting Training Process ({args.epochs} epochs, lr={args.learning_rate}) ---")
    for epoch in range(args.epochs):
        predictions = model.forward(X)
        loss = model.compute_loss(Y, predictions)
        dW, db = model.backward(X, Y, predictions)
        model.update_parameters(dW, db, args.learning_rate)
        
        if epoch % 500 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    print("\n--- Final Predictions on Training Samples ---")
    final_predictions = model.forward(X)
    display_limit = min(10, final_predictions.rows)
    for i in range(display_limit):
        pred_val = final_predictions[i, 0]
        expected_val = Y[i, 0]
        print(f"Sample {i + 1:2d} -> Prediction: {pred_val:.4f} | Expected: {expected_val}")

    print("\n--- Learned Feature Weights ---")
    for j in range(num_features):
        print(f"{feature_names[j]:<20} Weight: {model.W[j, 0]:.4f}")
    print(f"{'Global Bias':<20} : {model.bias:.4f}")

    print("\n--- Single-Entry Prediction Execution ---")
    if args.predict is not None:
        if len(args.predict) != num_features:
            print(
                f"Error: Expected {num_features} feature values for prediction, but received {len(args.predict)}.",
                file=sys.stderr
            )
            sys.exit(1)
        predict_single_entry(model, args.predict, max_values, feature_names)
    else:
        # If no specific input was provided via CLI, run sample predictions to demonstrate functionality
        if num_features == 3:
            sample_high = [14.0, 4.0, 3.0]
            sample_low = [5.0, 1.0, 0.0]
            predict_single_entry(model, sample_high, max_values, feature_names)
            predict_single_entry(model, sample_low, max_values, feature_names)
        else:
            sample_first = [raw_val * max_values[idx] for idx, raw_val in enumerate([X[0, j] for j in range(num_features)])]
            predict_single_entry(model, sample_first, max_values, feature_names)

if __name__ == "__main__":
    main()
