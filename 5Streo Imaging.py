import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load stereo images
left_image_path = 'teddy/im2.png'
right_image_path = 'teddy/im6.png'

left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Implement StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity_map = stereo.compute(left_image, right_image)

# Step 3: Visualize and interpret the depth map
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(left_image, cmap='gray')
plt.title('Left Image')

plt.subplot(1, 2, 2)
plt.imshow(disparity_map, cmap='viridis')
plt.title('Disparity Map')

plt.show()
