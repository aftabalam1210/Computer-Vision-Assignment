import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load an image
image_path = 'Lenna.png'
original_image = cv2.imread(image_path)

# Step 2: Perform image rotation, scaling, and translation
rows, cols, _ = original_image.shape

# Rotation
rotation_angle = 45
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
rotated_image = cv2.warpAffine(original_image, rotation_matrix, (cols, rows))

# Scaling
scaling_factor = 1.5
scaled_image = cv2.resize(original_image, None, fx=scaling_factor, fy=scaling_factor)

# Translation
translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])  # Translate by (50, 30)
translated_image = cv2.warpAffine(original_image, translation_matrix, (cols, rows))

# Step 3: Visualize the transformed images
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title('Scaled Image')

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
plt.title('Translated Image')

plt.show()
