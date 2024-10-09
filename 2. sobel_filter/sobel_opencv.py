import cv2
import numpy

grayscale_image = cv2.imread('../Road_image.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('title',grayscale_image)

# apply sobel filter in horizontal (x) and vertical (y) direction

sobel_x= cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize = 3)
sobel_y= cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize = 3)


# calculate gradient magnitude

sobel_magnitude = numpy.sqrt((sobel_x ** 2) + (sobel_y ** 2))

# normalize the magnitude in range from 0 darkness to 255 whiteness

sobel_magnitude_normalized = numpy.uint8((255*sobel_magnitude) / numpy.max(sobel_magnitude))

# apply threshold to create binary edge

# threshold = 50
threshold=(int(input("Type in threshold values:")))
_,binary_ouput = cv2.threshold(sobel_magnitude_normalized, threshold, 255, cv2.THRESH_BINARY)

cv2.imshow('This is result of sobel edge detection', binary_ouput)
cv2.waitKey(0)
cv2.destroyAllWindows