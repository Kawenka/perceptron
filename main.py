from matrixlib import Matrix
from model.neuron import Neuron

def main():
    num_samples = 4
    num_features = 3

    X = Matrix(num_samples, num_features)
    X.randomize(-2.0, 2.0)

    Y = Matrix(num_samples, 1)
    Y[0, 0] = 1.0
    Y[1, 0] = 0.0
    Y[2, 0] = 1.0
    Y[3, 0] = 0.0

    model = Neuron(num_features=num_features)

    prediction = model.forward(X)

    for i in range(prediction.rows):
        pred_val = prediction[i, 0]
        expected_val = Y[i, 0]
        print(f"Sample {i + 1} - Prediction: {pred_val:.4f} | Expected: {expected_val}")

if __name__ == "__main__":
    main()
