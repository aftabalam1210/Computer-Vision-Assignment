import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Create a virtual 3D cube
cube_size = 100
cube_points = np.float32([[0, 0, 0], [cube_size, 0, 0], [cube_size, cube_size, 0], [0, cube_size, 0],
                          [0, 0, cube_size], [cube_size, 0, cube_size], [cube_size, cube_size, cube_size], [0, cube_size, cube_size]])

# Step 2: Define camera parameters
focal_length = 500  # Focal length in pixels
camera_matrix = np.array([[focal_length, 0, cube_size / 2],
                          [0, focal_length, cube_size / 2],
                          [0, 0, 1]])

# Step 3: Generate rotation and translation matrices
rotation_matrix = np.eye(3)  # No rotation for simplicity
translation_vector = np.array([cube_size / 2, cube_size / 2, -cube_size * 2])

# Step 4: Perform the perspective transformation
transformed_points = cv2.projectPoints(cube_points, rotation_matrix, translation_vector, camera_matrix, None)[0].squeeze()

# Step 5: Visualize the projected points
plt.figure(figsize=(8, 8))
plt.scatter(transformed_points[:, 0], transformed_points[:, 1])
plt.title('Simulated Camera Projection')
plt.xlabel('X-axis (pixels)')
plt.ylabel('Y-axis (pixels)')
plt.show()
