import numpy as np
import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Feature values and labels
X = np.array([[0.1], [0.2], [0.3],
              [0.7], [0.8], [0.9]])

y = np.array([0, 0, 0, 1, 1, 1])

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X, y, epochs=100, verbose=0)

# Predict a new example
probability = model.predict(
    np.array([[0.85]]),
    verbose=0
)[0][0]

print("Probability:", probability)
print("Predicted class:", int(probability >= 0.5))
