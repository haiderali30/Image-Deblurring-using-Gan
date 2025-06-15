import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, mixed_precision
from generator_model import build_generator
from discriminator_model import build_discriminator
from utils import load_random_image_pairs

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Load data
blur_dir = "your_path/blurr"
non_blur_dir = "your_path/non blurr"
image_pairs = load_random_image_pairs(blur_dir, non_blur_dir, 500)

# Build models
generator = build_generator()
discriminator = build_discriminator()

generator.compile(loss='mse', optimizer=optimizers.Adam(0.0002, 0.5))
discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5))

# Training loop
def train(generator, discriminator, image_pairs, epochs=110, batch_size=8):
    for epoch in range(epochs):
        np.random.shuffle(image_pairs)
        for i in range(0, len(image_pairs), batch_size):
            batch = image_pairs[i:i+batch_size]
            blur_batch = np.array([x[0] for x in batch])
            sharp_batch = np.array([x[1] for x in batch])

            fake_images = generator(blur_batch)
            real_labels = np.ones((len(batch), 1))
            fake_labels = np.zeros((len(batch), 1))

            d_loss_real = discriminator.train_on_batch(sharp_batch, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

            discriminator.trainable = False
            g_loss = generator.train_on_batch(blur_batch, sharp_batch)
            discriminator.trainable = True

        print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {(d_loss_real + d_loss_fake)/2:.4f}")

train(generator, discriminator, image_pairs)
