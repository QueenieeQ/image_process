import cv2
import numpy

grayscaled = cv2.imread('grayscaled.jpg')
opencv_grayscaled = cv2.imread('grayscaled_opencv.jpg')

diff = cv2.absdiff(grayscaled, opencv_grayscaled)
total_diff = numpy.sum(diff)

if total_diff == 0:
    print("Perfectly identical")
else:
    print("Total difference:", total_diff)
cv2.imshow('title',diff)
cv2.waitKey(0)
cv2.destroyAllWindows