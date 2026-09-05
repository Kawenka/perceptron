from matrixlib import Matrix
from model.neuron import Neuron
import csv

def load_data(filepath: str) -> tuple:
    """
    Load data from a CSV file.
    Assumes the last column is the binary target (0.0 or 1.0)
    and all preceding columns are features.

    Args:
        fileparth (str): The path to the CSV file.

    Returns:
        tuple: (X_matrix, Y_matrix, max_values_list, feature_name_list)
    """
    raw_data = []

    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            # Converts all string values in the row in floats
            float_row = [float(val) for val in row]
            raw_data.append(float_row)

    num_samples = len(raw_data)
    num_features = len(header) - 1

    X = Matrix(num_samples, num_features)
    Y = Matrix(num_samples, 1)

    max_values = []
    for j in range(num_features):
        column_values = [row[j] for row in raw_data]
        max_val = max(column_values)

        if max_val == 0.0:
            max_val = 1.0

        max_values.append(max_val)

    for i in range(num_samples):
        for j in range(num_features):
            X[i, j] = raw_data[i][j] / max_values[j]

        Y[i, 0] = raw_data[i][-1]

    features_names = header[:-1]
    return X, Y, max_values, features_names

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
    
    print("\n--- Evaluating New Entry ---")
    for j in range(num_features):
        print(f"{feature_names[j]}: {new_data[j]}")
    print(f"Result -> Probability: {probability:.4f} | Predicted Class: {status}")

def main():
    print(" --- Loading data ---")
    X, Y, max_values, feature_names = load_data("data/data.csv")

    num_features = len(feature_names)
    print(f"Loaded {X.rows} training samples with {num_features} features.")

    model = Neuron(num_features=num_features)

    epochs = 3000
    learning_rate = 0.5

    print("--- Starting Training Process ---")
    
    for epoch in range(epochs):
        predictions = model.forward(X)
        loss = model.compute_loss(Y, predictions)
        dW, db = model.backward(X, Y, predictions)
        model.update_parameters(dW, db, learning_rate)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    print("\n--- Final Predictions after Training ---")
    final_predictions = model.forward(X)

    display_limit = min(10, final_predictions.rows)
    for i in range(display_limit):
        pred_val = final_predictions[i, 0]
        expected_val = Y[i, 0]
        print(f"Sample {i + 1} -> Prediction: {pred_val:.4f} | Expected: {expected_val}")

    print("\n--- Learned Feature Weights ---")
    print(f"Modules Weight: {model.W[0, 0]:.4f}")
    print(f"Exams Weight:   {model.W[1, 0]:.4f}")
    print(f"Rushes Weight:  {model.W[2, 0]:.4f}")
    print(f"Global Bias:    {model.bias:.4f}")

    print("\n --- Individual prediction execution ---")
    a = [14.0, 4.0, 3.0]
    predict_single_entry(model, a, max_values, feature_names)

    b = [5.0, 1.0, 0.0]
    predict_single_entry(model, b, max_values, feature_names)
    
if __name__ == "__main__":
    main()
