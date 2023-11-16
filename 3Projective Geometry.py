import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load an image
image_path = 'Lenna.png'
original_image = cv2.imread(image_path)

# Step 2: Define four source points for the projective transformation
height, width, _ = original_image.shape
src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

# Step 3: Define four destination points for the projective transformation
# You can adjust these points to control the transformation effect
dst_points = np.float32([[50, 50], [width - 100, 100], [100, height - 50], [width - 50, height - 50]])

# Step 4: Compute the projective transformation matrix
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Step 5: Apply the projective transformation
projective_image = cv2.warpPerspective(original_image, projective_matrix, (width, height))

# Step 6: Visualize the original and transformed images
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(projective_image, cv2.COLOR_BGR2RGB))
plt.title('Projective Transformed Image')

plt.show()
