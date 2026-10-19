import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

x = tf.constant([1, 2, 3])

print('----------------------------------------------------------------')
print(x)
print("Shape:", x.shape)
print("Dimensions:", x.ndim)

y = x * 2

print(y)

# single number
a = tf.constant(5)
# one-dimensional tensor
b = tf.constant([1, 2, 3, 4, 5, 6])
# two-dimensional tensor
c = tf.constant([
    [1, 2, 3], 
    [4, 5, 6]
])

print(a.shape)  # ()
print(b.shape)  # (6,)
print(c.shape)  # (2, 3)

# changing the shape with tf.reshape
d = tf.reshape(b, (3, 2))
print(d.shape)  # (3, 2)
print(d) # [[1 2] [3 4] [5 6]]