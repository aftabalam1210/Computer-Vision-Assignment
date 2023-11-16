import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("Lenna.png")

# Convert image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Apply color filter (e.g., isolate blue color)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
blue_result = cv2.bitwise_and(image, image, mask=blue_mask)

# Enhance color (e.g., histogram equalization)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(gray_image)

# Display original and processed images
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB)), plt.title('HSV')
plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(blue_result, cv2.COLOR_BGR2RGB)), plt.title('Blue Filter')
plt.subplot(2, 2, 4), plt.imshow(equalized_image, cmap='gray'), plt.title('Histogram Equalization')
plt.show()
