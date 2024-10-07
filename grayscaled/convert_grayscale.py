import cv2
import numpy

pic = cv2.imread('Road_image.jpg')
cv2.imshow('before', pic)

height, width, channels = pic.shape


grayscaled = numpy.zeros((height,width), dtype = numpy.uint8)

for row in range(height):
    for collum in range(width):
        r,g,b = pic[row,collum]
        grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)

        grayscaled[row, collum] = grayscale_value

cv2.imshow('after', grayscaled)
cv2.imwrite('grayscaled.jpg', grayscaled)
cv2.waitKey(0)
cv2.destroyAllWindows