import cv2
import numpy
import math

# open image using opencv 
# step 1
pic = cv2.imread('../images/Road_image.jpg')
cv2.imshow('0. input image', pic)
# get image dimensions and color channel
height, width, channels = pic.shape
# declare an empty array variable for grayscale image
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
cv2.imshow('1. grayscale image', grayscale_image)
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
_,binary_output = cv2.threshold(sobel_magnitude_normalized, sobel_threshold, 255, cv2.THRESH_BINARY)

cv2.imshow('2. sobel_filter',binary_output)
# Step 3 
# Hough Transform implementation
rho_max=int(numpy.sqrt(height ** 2 + width ** 2)) # maximum posible slope of a line that can be detected in the image
theta_max= 180 # 180 degree 
votes = numpy.zeros((2* rho_max, theta_max),dtype=numpy.int32)
# votes for each possible line orientation
# calculate vote in hough transform accumulator
for h in range(height):
    for w in range(width):
        if binary_output[h,w ] > 0:
            for theta in range(0, theta_max):
                theta_radian = numpy.deg2rad(theta)
                rho = int(w * numpy.cos(theta_radian)+ h * numpy.sin(theta_radian))
                rho_index = rho + rho_max
                votes[rho_index, theta] += 1
print(votes)
# apply threshold to accumulator and detect line
hough_threshold = 200 
lines = numpy.argwhere(votes > hough_threshold)
#draw detected line on images
for rho_index, theta_index in lines:
    rho = rho_index - rho_max 
    # convert rho index bach to to rho
    theta = numpy.deg2rad(theta_index)
    #convert polar coordinate (rho,theta) to cartesian coordinates to drawing
    x0 = (numpy.cos(theta)) * rho
    y0 = (numpy.sin(theta)) * rho
# calculate 2 point on the line to draw
    x1 = int(x0 + 1000 * (-1*( numpy.sin(theta))))
    y1 = int( y0 +1000 * (1 *( numpy.cos(theta))))
    x2 = int( x0 - 1000 * (-1*( numpy.sin(theta))))
    y2 = int( y0 -1000 * (1 *( numpy.cos(theta))))
    # now draw the line on image using opencv line
    # Or you can draw the line by yourself use bresenham line algorithm
    cv2.line(pic, (x1, y1), (x2, y2), (0,0,255),2 )

# show result
cv2.imshow('3. Hough Transform lines', pic)
cv2.waitKey(0)
cv2.destroyAllWindows
           


    






