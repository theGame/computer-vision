import numpy as np
import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Generate training data
x = np.linspace(-2, 2, 101).reshape(-1, 1)
y = x ** 2

# 2. Build a neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

# 3. Configure training
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# 4. Train
model.fit(x, y, epochs=500, verbose=0)

# 5. Test predictions
test_values = np.array([[-1.5], [0.5], [1.5]])
predictions = model.predict(test_values, verbose=0)

for x_value, prediction in zip(test_values, predictions):
    print(f"x={x_value[0]:.1f}, predicted y={prediction[0]:.3f}")

