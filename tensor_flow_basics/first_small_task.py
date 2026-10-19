
import os
import tensorflow as tf
import keras

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Sequential in tf.keras:", hasattr(tf.keras, "Sequential"))
print("Sequential in keras:", hasattr(keras, "Sequential"))

# First machine learning example
# There are many house for sale in the city. The price of the house depends on the size of the house.
# We have a dataset with the size of the house and the price of the house.
# We will use this dataset to train a model to predict the price of the house based on the size of the house.
# house_sizes = [30, 50, 70, 90]
# house_prices = [60, 100, 140, 180]
# formula: y = mx + b
# where y is the price of the house, x is the size of the house,
# m - weight, controlling how strongly the input affects the result
# b - bias, controlling the output when the input is zero

# Input examples, house sizes:
x = tf.constant([30, 50, 70, 90], dtype=tf.float32)

# Output examples, house prices:
y = tf.constant([60, 100, 140, 180], dtype=tf.float32)

# Model with one input and one output
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Input layer with one input
    tf.keras.layers.Dense(1)  # Output layer with one output
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

model.compile(optimizer=optimizer, loss='mean_squared_error')

# Learn from the examples
model.fit(x, y, epochs=200, verbose=0)

print(model.predict(tf.constant([60.0])))
print(model.get_weights())
