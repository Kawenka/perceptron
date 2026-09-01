from matrixlib import Matrix
from model.neuron import Neuron
import csv

def load_data(filepath: str) -> tuple:
    """
    Load candidate data form a CSV file, normalize features, and format matrices.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        tuple, A tuple containing the normalized input Matrix (X) and target Matrix (Y).
    """
    raw_modules = []
    raw_exams = []
    raw_rushes = []
    target_status = []

    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            raw_modules.append(float(row[0]))
            raw_exams.append(float(row[1]))
            raw_rushes.append(float(row[2]))
            target_status.append(float(row[3]))

    num_samples = len(raw_modules)
    num_features = 3

    X = Matrix(num_samples, num_features)
    Y = Matrix(num_samples, 1)

    max_modules = 15.0
    max_exams = 4.0
    max_rushes = 3.0

    for i in range(num_samples):
        X[i, 0] = raw_modules[i] / max_modules
        X[i, 1] = raw_exams[i] /max_exams
        X[i, 2] = raw_rushes[i] / max_rushes
        Y[i, 0] = target_status[i]

    return X, Y


def main():
    print(" --- Loading data ---")
    X, Y = load_data("data.csv")
    print(f"Loaded {X.rows} training samples.")

    num_features = 3
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
    
if __name__ == "__main__":
    main()
