import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow import float32
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the labels to one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(124, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# Wrap the model in a tff.learning.Model
def model_fn():
    keras_model = create_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, 10), dtype=tf.float32)),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Create a federated dataset
def preprocess_fn(client_data):
    return client_data['x'], client_data['y']

@tff.tf_computation
def make_federated_data(client_data):
    x = client_data['x']
    y = client_data['y']
    
    federated_data = [(x[i], y[i]) for i in range(len(x))]
    return federated_data



train_data = {'x': x_train, 'y': y_train}
test_data = {'x': x_test, 'y': y_test}

train_data = make_federated_data(train_data)
test_data = make_federated_data(test_data)

# Create a federated learning process
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.002),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# Run the federated learning process
train_state = iterative_process.initialize()
losses = []
accuracies = []
for _ in range(10):
    train_state, metrics = iterative_process.next(train_state, train_data)
    losses.append(metrics.loss)
    accuracies.append(metrics.accuracy)

# Plot the learning graph
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Round')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies, marker='o')
plt.title('Training Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()