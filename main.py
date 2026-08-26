from matrixlib import Matrix
from model.neuron import Neuron

def main():
    num_samples = 4
    num_features = 3

    X = Matrix(num_samples, num_features)
    X.randomize(-2.0, 2.0)

    model = Neuron(num_features=num_features)

    prediction = model.forward(X)

    for i in range(prediction.rows):
        value = prediction[i, 0]
        print(f"Sample {i + 1}: {value:.4f}")

if __name__ == "__main__":
    main()
