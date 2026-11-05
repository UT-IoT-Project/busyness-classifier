"""Small dependency-light recurrent regressor for windowed occupancy features."""
import numpy as np


class RecurrentRegressor:
    """A tanh RNN that predicts the occupancy level from recent feature windows."""

    def __init__(self, hidden_size=16, sequence_length=5, epochs=120,
                 learning_rate=0.005, random_state=42):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _sequences(self, values):
        sequences = np.empty(
            (len(values), self.sequence_length, values.shape[1]), dtype=float)
        for index in range(len(values)):
            start = max(0, index - self.sequence_length + 1)
            sequence = values[start:index + 1]
            if len(sequence) < self.sequence_length:
                sequence = np.vstack([
                    np.repeat(sequence[[0]], self.sequence_length - len(sequence), axis=0),
                    sequence,
                ])
            sequences[index] = sequence
        return sequences

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.asarray(y, dtype=float)
        self.input_mean_ = values.mean(axis=0)
        self.input_scale_ = values.std(axis=0)
        self.input_scale_[self.input_scale_ == 0] = 1
        scaled = (values - self.input_mean_) / self.input_scale_
        sequences = self._sequences(scaled)

        rng = np.random.default_rng(self.random_state)
        input_size = values.shape[1]
        self.weights_input_ = rng.normal(0, 0.1, (input_size, self.hidden_size))
        self.weights_hidden_ = rng.normal(0, 0.1, (self.hidden_size, self.hidden_size))
        self.weights_output_ = rng.normal(0, 0.1, self.hidden_size)
        self.bias_hidden_ = np.zeros(self.hidden_size)
        self.bias_output_ = 0.0

        for _ in range(self.epochs):
            for sequence, target in zip(sequences, targets):
                hidden = np.zeros((self.sequence_length + 1, self.hidden_size))
                for step, row in enumerate(sequence, start=1):
                    hidden[step] = np.tanh(
                        row @ self.weights_input_
                        + hidden[step - 1] @ self.weights_hidden_
                        + self.bias_hidden_)
                error = hidden[-1] @ self.weights_output_ + self.bias_output_ - target
                gradient_output = error
                gradient_input = np.zeros_like(self.weights_input_)
                gradient_hidden = np.zeros_like(self.weights_hidden_)
                gradient_bias_hidden = np.zeros_like(self.bias_hidden_)
                gradient_weights_output = hidden[-1] * gradient_output
                gradient_bias_output = gradient_output
                next_gradient = np.zeros(self.hidden_size)
                for step in range(self.sequence_length, 0, -1):
                    gradient = (
                        gradient_output * self.weights_output_
                        + next_gradient) * (1 - hidden[step] ** 2)
                    gradient_input += np.outer(sequence[step - 1], gradient)
                    gradient_hidden += np.outer(hidden[step - 1], gradient)
                    gradient_bias_hidden += gradient
                    next_gradient = gradient @ self.weights_hidden_.T
                gradient_input = np.clip(gradient_input, -1, 1)
                gradient_hidden = np.clip(gradient_hidden, -1, 1)
                gradient_bias_hidden = np.clip(gradient_bias_hidden, -1, 1)
                gradient_weights_output = np.clip(gradient_weights_output, -1, 1)
                gradient_bias_output = np.clip(gradient_bias_output, -1, 1)
                learning_rate = self.learning_rate
                self.weights_output_ -= learning_rate * gradient_weights_output
                self.bias_output_ -= learning_rate * gradient_bias_output
                self.weights_input_ -= learning_rate * gradient_input
                self.weights_hidden_ -= learning_rate * gradient_hidden
                self.bias_hidden_ -= learning_rate * gradient_bias_hidden
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = (values - self.input_mean_) / self.input_scale_
        sequences = self._sequences(scaled)
        predictions = []
        for sequence in sequences:
            hidden = np.zeros(self.hidden_size)
            for row in sequence:
                hidden = np.tanh(
                    row @ self.weights_input_
                    + hidden @ self.weights_hidden_
                    + self.bias_hidden_)
            predictions.append(hidden @ self.weights_output_ + self.bias_output_)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=3.0, neginf=0.0)
        return np.clip(np.rint(predictions), 0, 3).astype(int)
