import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, BatchNormalization
import matplotlib.pyplot as plt

#loading the data and split it into train set and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#normalize the inputs from 0-255 to between 0 and 1 dividing by 255
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the labels to one-hot encoded vectors
y_train = to_categitgorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
model = tf.keras.models.Sequential()

# Convolutional Layer 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Convolutional Layer 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Flatten the output of the convolutional layers
model.add(layers.Flatten())
model.add(Dropout(0.2))

# Dense Layer 1
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Dense Layer 2
model.add(layers.Dense(10, activation='softmax'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Plotting the learning curve
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('learning_curve.png')  # Save the plot in the current directory
plt.show()
