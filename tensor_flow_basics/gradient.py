
import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# A trainable parameter
w = tf.Variable(1.0)

x = tf.constant(2.0)
target = tf.constant(6.0)

# Track the calculations
with tf.GradientTape() as tape:
    prediction = w * x
    loss = (prediction - target) ** 2

# Calculate d(loss) / d(w)
gradient = tape.gradient(loss, w)

print("Prediction:", prediction.numpy())
print("Loss:", loss.numpy())
print("Gradient:", gradient.numpy())

# Perform an actual parameter update
learning_rate = 0.1

w.assign_sub(learning_rate * gradient)

print("Updated weight:", w.numpy())
