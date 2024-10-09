import cv2
import numpy

# grayscale_image = cv2.imread('../Road_image.jpg', cv2.IMREAD_GRAYSCALE)
# open image using opencv 
# step 1
pic = cv2.imread('../Road_image.jpg')
cv2.imshow('before', pic)
# get image dimensions and color channel
height, width, channels = pic.shape
#declear a array variable for grayscale image
grayscale_image = numpy.zeros((height,width), dtype = numpy.uint8)

# interate over every pixel from top to bottom, left to right and apply luminance formula
for row in range(height):
    for collum in range(width):
        # take r,g,b value
        r,g,b = pic[row,collum]
        # calculate grayscale luminance
        grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)
        # save the grayscale value to new image
        grayscale_image[row, collum] = grayscale_value
cv2.imshow('grayscaled', grayscale_image)
# step 2
# get grayscaled image size
height, width = grayscale_image.shape
#sobel_kernel horizontal an vertical formula
sobel_x_kernel = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y_kernel = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# empty array to sore data
sobel_x = numpy.zeros_like(grayscale_image, dtype=numpy.float32)
sobel_y = numpy.zeros_like(grayscale_image, dtype=numpy.float32)
# convolution image 
# start from 1 because inoring image borders
for row in range(1, height -1):
    for column in range(1, width - 1):
        # get sourounding 3x3 region from pixel
        region = grayscale_image[row - 1: row +2, column -1: column+2]
        #calculate horizontal and vertical kernel
        result_x = numpy.sum(region*sobel_x_kernel)
        result_y = numpy.sum(region*sobel_y_kernel)
        # save result
        sobel_x[row, column] = result_x
        sobel_y[row, column] = result_y
# calculate gradient magnitude
sobel_magnitude = numpy.sqrt((sobel_x**2)+(sobel_y**2))
#normalized the magnitude number (0-255)
sobel_magnitude_normalized = numpy.uint8(255 * sobel_magnitude / numpy.max(sobel_magnitude))
# get threshold value
sobel_threshold = int(input("Enter threshold value(50-100):"))
# apply a threshold value to create binary edge iamge
_,binary_ouput = cv2.threshold(sobel_magnitude_normalized, sobel_threshold, 255, cv2.THRESH_BINARY)
# show result
cv2.imwrite('../sobel_edge_detection.jpg', binary_ouput)
cv2.imshow('title', binary_ouput)
cv2.waitKey(0)
cv2.destroyAllWindows

