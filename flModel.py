import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# Function for preprocessing the human detection dataset for each client
def preprocess_data_for_client(client_id, data_dir, img_height=64, img_width=64):
    images = []
    labels = []

    # Construct the path for the client's data directory
    client_data_dir = os.path.join(data_dir, str(client_id))

    # Iterate through each image in the client's data directory
    for image_name in os.listdir(client_data_dir):
        image_path = os.path.join(client_data_dir, image_name)
        # Load and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (img_height, img_width))
        image = image.astype(np.float32) / 255.0  # Normalize pixel values
        images.append(image)
        labels.append(client_id)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Function to create a federated dataset from preprocessed data
def create_federated_dataset(data_dir):
    federated_train_data = []

    # Iterate through client IDs
    for client_id in [0, 1]:
        images, labels = preprocess_data_for_client(client_id, data_dir)
        client_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(images)).batch(32)
        federated_train_data.append(client_dataset)

    return federated_train_data

# Load human detection dataset and preprocess it
data_dir = "/content/drive/MyDrive/human detection dataset"
federated_train_data = create_federated_dataset(data_dir)

# Wrap a Keras model for use with TFF.
def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(64, 64)),
        tf.keras.layers.Dense(10, tf.nn.softmax)
    ])
    return tff.learning.models.from_keras_model(
        model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
state = trainer.initialize()

losses = []  # Store loss values for plotting

for _ in range(5):
    state, metrics = trainer.next(state, federated_train_data)
    print(metrics['client_work']['train']['loss'])
    losses.append(metrics['client_work']['train']['loss'])


# Plotting the learning curve
plt.plot(losses, marker='o')
plt.title('Learning Curve')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
