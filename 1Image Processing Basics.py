import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load an image
image_path = 'Lenna.png'
original_image = cv2.imread(image_path)

# Step 2: Apply common image processing operations
# Blur
blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

# Sharpening
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
sharpened_image = cv2.filter2D(original_image, -1, kernel)

# Edge detection
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 50, 150)

# Step 3: Visualize and compare the results
plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title('Sharpened Image')

plt.subplot(2, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')

plt.show()
