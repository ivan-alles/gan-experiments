# An TF2 implementation of the GAN from "GANs in Action", Chapter 3.

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100


def build_gan(img_shape, z_dim):

    z = keras.layers.Input(shape=z_dim, name='z')
    x_real = keras.layers.Input(shape=img_shape, name='x_real')

    g = keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=0.01), name='G-Dense-1')(z)
    g = keras.layers.Dense(28 * 28 * 1, activation='tanh', name='G-Dense-2')(g)
    g = keras.layers.Reshape(img_shape, name='G-Reshape')(g)

    x = keras.layers.Concatenate(axis=0)([g, x_real])  # Concatenate fake and real images along the batch axis_

    d = keras.layers.Flatten(input_shape=img_shape)(x)
    d = keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=0.01), name='D-Dense-1')(d)
    d = keras.layers.Dense(1, activation='sigmoid', name='D-Dense-2')(d)

    gan = keras.Model(inputs=(x_real, z), outputs=(g, d))
    keras.utils.plot_model(gan, to_file='gan.svg', dpi=50, show_shapes=True)
    g_trainable = gan.trainable_variables[:4]
    d_trainable = gan.trainable_variables[4:]
    return gan, g_trainable, d_trainable


gan, g_trainable, d_trainable = build_gan(img_shape, z_dim)

g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()

losses = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):

    # Load the MNIST dataset
    (X_train, _), (_, _) = keras.datasets.mnist.load_data()

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)

    y_true_real = np.ones((batch_size // 2, 1), dtype=np.float32)

    y_true_fake = np.zeros((batch_size // 2, 1), dtype=np.float32)

    for iteration in range(iterations):
        # Real images.
        idx = np.random.randint(0, X_train.shape[0], batch_size // 2)
        x_real = X_train[idx]

        # Random inputs for the generator
        z = np.random.normal(0, 1, (batch_size // 2, z_dim))

        with tf.GradientTape(persistent=True) as tape:
            x_fake, y_pred = gan((x_real, z), training=True)
            y_pred_fake = y_pred[:batch_size // 2]
            y_pred_real = y_pred[batch_size // 2:]

            loss_d_fake = tf.math.reduce_mean(keras.losses.binary_crossentropy(y_true_fake, y_pred_fake))
            loss_d_real = tf.math.reduce_mean(keras.losses.binary_crossentropy(y_true_real, y_pred_real))
            loss_g = tf.math.reduce_mean(keras.losses.binary_crossentropy(y_true_real, y_pred_fake))

            loss_d = loss_d_fake + loss_d_real

        grads_d = tape.gradient(loss_d, d_trainable)
        d_optimizer.apply_gradients(zip(grads_d, d_trainable))

        grads_g = tape.gradient(loss_g, g_trainable)
        g_optimizer.apply_gradients(zip(grads_g, g_trainable))

        if (iteration + 1) % sample_interval == 0:
            # Save results so they can be plotted after training
            losses.append((loss_d_fake + loss_d_real, loss_g))
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print(f"{iteration + 1} D loss fake: {loss_d_fake:.4f} D loss real: {loss_d_real:.4f} "
                  f"G loss: {loss_g:.4f} Total loss: {loss_d_fake + loss_d_real + loss_g:.4f}")

            sample_images(x_fake)


def sample_images(x_fake, image_grid_rows=4, image_grid_columns=4):
    # Generate images from random noise
    gen_imgs = x_fake[:image_grid_rows * image_grid_columns]

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    plt.draw()
    plt.pause(0.001)


# Set hyperparameters
iterations = 20000
batch_size = 128
sample_interval = 1000

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)

losses = np.array(losses)

# Plot training losses for Discriminator and Generator
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

plt.show()

