#Import the necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#Load the dataset
#For this task, we use the IMDB dataset provided by keras.datasets.
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#Data Preprocessing: Vectorize the sequences
#We transform the word sequences into vectors using one-hot encoding so that they can be input into the neural network.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
#Convert the labels to categorical format
#We convert the labels into binary format (positive or negative).
y_train = np.array(train_labels).astype("float32")
y_test = np.array(test_labels).astype("float32")

#Build the neural network model
#We create a simple model with two fully connected (Dense) layers with 16 units each and an output layer with 1 unit (since it's binary classification)
model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(10000,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

#Compile the model
#We use the RMSprop optimizer, binary cross-entropy as the loss function (since it's binary classification), and accuracy as the metric.
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

#Prepare the validation set
#We'll split the training data into a training set and a validation set to monitor the model's performance on unseen data during training.
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#Train the model
#We train the model using the fit() method. We'll train for 20 epochs, with a batch size of 512 samples.
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#Evaluate the model
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

#Visualize the training and validation loss and accuracy
#We plot the results to see how well the model performed during training and validation.

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# Plot the loss
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

