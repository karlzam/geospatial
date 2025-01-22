import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
# lr = 0.1
optimizer = tf.keras.optimizers.Adam(0.1)

def train_step():
    # gradient tape computes the gradient for us
    # only have to implement forward prop
    with tf.GradientTape() as tape:
        # example cost function
        cost = w**2 - 10*w + 25
    trainable_variables = [w]
    grads = tape.gradient(cost, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

print(w)

train_step()
print(w)

for i in range(1000):
    train_step()
print(w)

# the minimum of the cost function is 5, so tensorflow automatically takes derivatives wrt the cost function

# what if the cost function is also a parameter of the training set?

w = tf.Variable(0, dtype=tf.float32)
x = np.array([1.0, -10.0, 25.0], dtype=np.float32)
optimizer = tf.keras.optimizers.Adam(0.1)

def training(x, w, optimizer):
    def cost_fn():
        # data controls coefficients
        return x[0]*w**2 + x[1]*w + x[2]
    for i in range(1000):
        optimizer.minimize(cost_fn, [w])

    return w

w = training(x, w, optimizer)
print(w)

#print(w)
# does the same thing as the multiple lines above
#optimizer.minimize(cost_fn, [w])
#print(w)






