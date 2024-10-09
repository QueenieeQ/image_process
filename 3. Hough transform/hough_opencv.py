import cv2
import numpy as np
import math

# Step 1: Load the grayscale image and apply edge detection (using Canny for simplicity here)
grayscale_image = cv2.imread('../Road_image.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(grayscale_image, 50, 150)

# Step 2: Initialize parameters
height, width = edges.shape
rho_max = int(np.sqrt(height**2 + width**2))  # Diagonal of the image for max rho
theta_max = 180  # We will search theta from 0 to 180 degrees
accumulator = np.zeros((2 * rho_max, theta_max), dtype=np.int32)  # Hough accumulator

# Step 3: Populate the accumulator by voting for each edge pixel
for y in range(height):
    for x in range(width):
        if edges[y, x] > 0:  # Only process edge points
            for theta in range(0, theta_max):
                theta_rad = np.deg2rad(theta)  # Convert theta to radians
                rho = int(x * np.cos(theta_rad) + y * np.sin(theta_rad))
                rho_index = rho + rho_max  # Shift rho to positive index
                accumulator[rho_index, theta] += 1  # Vote in the accumulator

# Step 4: Filter out lines with votes greater than a threshold (e.g., 200)
threshold = 200
lines = np.argwhere(accumulator > threshold)  # Get (rho, theta) for high votes

# Step 5: Draw the detected lines on the image
for rho_idx, theta_idx in lines:
    rho = rho_idx - rho_max  # Convert rho index back to actual rho
    theta = np.deg2rad(theta_idx)  # Convert theta index to radians

    # Convert polar coordinates (rho, theta) to Cartesian coordinates for drawing
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    # Calculate two points on the line to draw
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Draw the line on the original grayscale image
    cv2.line(grayscale_image, (x1, y1), (x2, y2), (0, 0, 255), 2 )
    cv2.line

# Step 6: Display the result
cv2.imshow('Hough Transform Lines', grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
