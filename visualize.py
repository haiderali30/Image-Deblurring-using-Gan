import matplotlib.pyplot as plt
import numpy as np

def visualize_generated_images(generator, image_pairs, num_images=5):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 20))
    for i in range(num_images):
        blur_img, _ = image_pairs[i]
        generated_img = generator(np.expand_dims(blur_img, axis=0), training=False)[0]
        generated_img = (generated_img + 1) / 2
        generated_img = np.clip(generated_img, 0, 1)

        axes[i, 0].imshow(blur_img)
        axes[i, 0].set_title("Blurred Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated_img)
        axes[i, 1].set_title("Generated Image")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()
