import cv2

picture = cv2.imread('Road_image.jpg')
# step 1 show the picture
cv2.imshow('before',picture)
# step 2 convert image using opencv 
grayscale_pic = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
cv2.imshow('after', grayscale_pic)
cv2.imwrite('grayscaled_opencv.jpg', grayscale_pic)
cv2.waitKey(0)
cv2.destroyAllWindows