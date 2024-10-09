# Import the necessary packages
# %matplotlib inline

import typing as tp
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path
best_tl = 634
best_th = 854
print(f'tl={best_tl}, th={best_th}')

# Function to plot images inline in the notebook
def show_img(img, title=None):
    #Check if the image is in BGR format and convert it to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #Plot the image
    plt.figure(figsize=(10,5)) 
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

# Function to perform the conversion between polar and cartesian coordinates
def polar2cartesian(radius: np.ndarray, angle: np.ndarray, cv2_setup: bool = True) -> np.ndarray:
    if cv2_setup:
        return radius * np.array([np.cos(angle), np.sin(angle)])
    else:
        return radius * np.array([np.sin(angle), np.cos(angle)])
#Funtion to add lines to an image
def draw_lines(img: np.ndarray, lines: np.ndarray, color: tp.List[int] = [0, 0, 255], thickness: int = 1, cv2_setup: bool = True) -> tp.Tuple[np.ndarray]:
    new_image = np.copy(img)
    empty_image = np.zeros(img.shape[:2])

    if len(lines.shape) == 1:
        lines = lines[None, ...]

    # Draw found lines
    for rho, theta in lines:
        x0 = polar2cartesian(rho, theta, cv2_setup)
        direction = np.array([x0[1], -x0[0]])
        pt1 = np.round(x0 + 1000*direction).astype(int)
        pt2 = np.round(x0 - 1000*direction).astype(int)
        empty_image = cv.line(img=empty_image,pt1=pt1, pt2=pt2, color=255, thickness=thickness)

    # Keep lower part of each line until intersection
    mask_lines = empty_image != 0
    min_diff = np.inf
    max_line = 0
    for i in range(mask_lines.shape[0]):
        line = mask_lines[i]
        indices = np.argwhere(line)
        if indices[-1] - indices[0] < min_diff:
            min_diff = indices[-1] - indices[0]
            max_line = i

    mask_boundaries = np.zeros_like(empty_image)
    mask_boundaries[max_line:] = 1
    mask = (mask_lines * mask_boundaries).astype(bool)

    new_image[mask] = np.array(color)
    
    return new_image, mask

# Function that given the image and the mask of the lines, fill the area between the lines
def fill_lines(img: np.ndarray, mask: np.ndarray, color: tp.List[int] = [0, 0, 255]) -> np.ndarray:
    border = np.where(mask)

    possible_vertex = np.where(border[0] == np.min(border[0]))
    vertex = np.array([border[0][int(len(possible_vertex[0]) / 2)], border[1][int(len(possible_vertex[0]) / 2)]])[::-1]

    possible_bottom = np.where(border[0] == np.max(border[0]))
    bottom_pos = [np.min(possible_bottom[0]), np.max(possible_bottom[0])]
    bottom_left = np.array([border[0][bottom_pos[0]], border[1][bottom_pos[0]]])[::-1]
    bottom_right = np.array([border[0][bottom_pos[1]], border[1][bottom_pos[1]]])[::-1]
    points = np.array([vertex, bottom_left, bottom_right])

    return cv.fillConvexPoly(np.copy(img), points=points, color=color)

check_params = lambda x: [i if i > 0 else 1 for i in x]

# Load image (img path './data/road2.png')
image_folder = Path('./data')
image_name = '../Road_image.jpg'
image_path = image_folder / image_name
image_path = image_name
# Open the image
image = cv.imread(str(image_path))

grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY);
# Show the image
# Suggestion: to show the image inline use the function show_img()
show_img(image)

#Create a cv window with two trackbars to change the thresholds values of the Canny algorithm
# Using the trackbars choose the best thresholds values to extract the edges of the road
# Print also the best parameters
win_name = 'CannyParameters'

cv.namedWindow(win_name)
max_th = 1000

class CannyFilter:
    def __init__(self, tl, th):
        self.th = th
        self.tl = tl
        
    def set_th(self, val):
        #if self.th >= self.tl:
        self.th = val
        
    def set_tl(self, val):
        #if self.tl <= self.th:
        self.tl = val
    
    def get_th(self):
        return self.th
    
    def get_tl(self):
        return self.tl
    
    def __call__(self, image, tl: float, th: float):
        self.set_th(th)
        self.set_tl(tl)
        return cv.Canny(image, self.get_tl(), self.get_th(), 4)
        
filter = CannyFilter(600, 900)
canny_edges = filter(grey_image, filter.get_tl(), filter.get_th())

cv.createTrackbar('threshold_1', win_name, filter.get_tl(), max_th, lambda x: cv.imshow(win_name,filter(grey_image, x, filter.get_th())))
cv.createTrackbar('threshold_2', win_name, filter.get_th(), max_th, lambda x: cv.imshow(win_name, filter(grey_image, filter.get_tl(), x)))



cv.imshow(win_name, canny_edges)
cv.waitKey(0)
cv.destroyAllWindows()
best_canny_res = cv.Canny(grey_image, best_tl, best_th, 7)
show_img(best_canny_res, 'Result of Canny edge detector')

# Visualize the result of the Hough transform with the best parameters
rho = 9
theta = 0.261
threshold = 101
min_theta = 0
max_theta = np.pi

lines = cv.HoughLines(best_canny_res, rho, theta, threshold, None, 0, 0)
print(lines)
lines = np.squeeze(lines, axis=1)

result, mask = draw_lines(image, lines)

show_img(result, 'Hough Result Lines')

# Color all the pixel inside the lines in red (you can use the provided fill_lines function)
filled_image = fill_lines(result, mask)

show_img(filled_image, 'Hough Result Filled')

#Visualize the result of the Canny algorithm with the best thresholds values
# Suggestions: use the function show_img() and check that the edges are suitable for the next step (see the example on the slides)



# Create a window with a set of trackbars to change the parameters of the Hough transform
# Using the trackbars choose the best parameters to find the lines of the road
win_name = 'Hough_transform'

theta = np.pi/180
rho = 1
threshold = 100
min_theta = 0
max_theta = np.pi

hough_params = [rho, theta, threshold, min_theta, max_theta]

cv.namedWindow(win_name)

def updateParam(val, index):
    if index != 2 and index != 0:
        hough_params[index] = val / 1000.
    else:
        hough_params[index] = val
       
    lines = cv.HoughLines(best_canny_res, hough_params[0], hough_params[1], hough_params[2], None, 0, 0, hough_params[3], hough_params[4])

    if lines is not None:
        lines = np.squeeze(lines, axis=1)
        result, _ = draw_lines(image, lines)
        cv.imshow(win_name, result)
        

cv.createTrackbar('rho', win_name, int(hough_params[0] * 10), 20, lambda x: updateParam(x, 0))
cv.createTrackbar('theta * 100', win_name, int(hough_params[1] * 1000), int(np.pi * 1000), lambda x: updateParam(x, 1))
cv.createTrackbar('threshold', win_name, hough_params[2], 200, lambda x: updateParam(x, 2))
cv.createTrackbar('min_theta * 100', win_name, int(hough_params[3] * 1000), int(np.pi * 1000), lambda x: updateParam(x, 3))
cv.createTrackbar('max_theta * 100', win_name, int(hough_params[4] * 1000), int(np.pi * 1000), lambda x: updateParam(x, 4))

cv.imshow(win_name, image)
cv.waitKey(0)
cv.destroyAllWindows()

print(hough_params)

# Visualize the result of the Hough transform with the best parameters
rho = 9
theta = 0.261
threshold = 101
min_theta = 0
max_theta = np.pi

lines = cv.HoughLines(best_canny_res, rho, theta, threshold, None, 0, 0)
print(lines)
lines = np.squeeze(lines, axis=1)

result, mask = draw_lines(image, lines)

show_img(result, 'Hough Result Lines')

# Color all the pixel inside the lines in red (you can use the provided fill_lines function)
filled_image = fill_lines(result, mask)

show_img(filled_image, 'Hough Result Filled')