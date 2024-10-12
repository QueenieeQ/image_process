import cv2
import numpy as np

# Read the grayscale image
grayscale_image = cv2.imread('../images/Road_image.jpg', cv2.IMREAD_GRAYSCALE)

if grayscale_image is None:
    print("Error: Unable to read the image.")
    exit()

#  Apply edge detection using Canny
edges = cv2.Canny(grayscale_image, 50, 150, apertureSize=3) # Canny has an optional third parameter

#  Hough Transform to detect lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=200, minLineLength=100, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw the detected line on the image
        cv2.line(grayscale_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#  Display the result with lines drawn
cv2.imshow('Hough Transform Lines', grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
