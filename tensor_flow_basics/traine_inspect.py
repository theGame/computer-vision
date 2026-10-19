import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# traine the data
x = tf.constant([[30.0], [50.0], [70.0], [90.0]], dtype=tf.float32)
y = tf.constant([[60.0], [100.0], [140.0], [180.0]], dtype=tf.float32)

# normalize the data
x_scaled = x / 100.0
y_scaled = y / 200.0

# create a model 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Input layer with one input
    tf.keras.layers.Dense(1)  # Output layer with one output
])

# configure training
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')

# train the model
history = model.fit(x_scaled, y_scaled, epochs=300, verbose=0)

# inspect the learning params 
weights, biases = model.layers[0].get_weights()

print("Weights:", weights[0, 0])
print("Biases:", biases[0])

# predict the price of a 60 square meter house
new_house = tf.constant([[60.0 / 100.0]], dtype=tf.float32)
prediction_scaled = model.predict(new_house, verbose=0)
prediction = prediction_scaled[0, 0] * 200.0

print("Predicted price for a 60 square meter house:", prediction)

# inspect the final loss
print("Final loss:", history.history['loss'][-1])