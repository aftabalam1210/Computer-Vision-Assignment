import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load depth image (replace "your_depth_image.png" with your actual depth image)
depth_image = cv2.imread("Lenna.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

# Normalize the depth map to [0, 1]
normalized_depth = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))

# Perform depth smoothing
smoothed_depth = cv2.medianBlur(normalized_depth, 5)

# Perform morphological operations (e.g., dilation)
kernel = np.ones((5, 5), np.uint8)
dilated_depth = cv2.dilate(normalized_depth, kernel, iterations=1)

# Visualize the results
plt.subplot(1, 3, 1), plt.imshow(normalized_depth, cmap='jet'), plt.title('Original Depth')
plt.subplot(1, 3, 2), plt.imshow(smoothed_depth, cmap='jet'), plt.title('Smoothed Depth')
plt.subplot(1, 3, 3), plt.imshow(dilated_depth, cmap='jet'), plt.title('Dilated Depth')

# Show colorbar for reference
plt.colorbar()

plt.show()
