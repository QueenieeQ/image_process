#####################################################################

# Example :  hough line detection based on canny edge detection
# for a a video file specified on the command line (e.g. python FILE.py
# video_file) or from an attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2021 Dept. Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys
import numpy as np

#####################################################################

keep_processing = True
use_probablistic_hough = False

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Perform ' +
    sys.argv[0] +
    ' example operation on incoming camera/video image')
parser.add_argument(
    "-c",
    "--camera_to_use",
    type=int,
    help="specify camera to use",
    default=0)
parser.add_argument(
    "-r",
    "--rescale",
    type=float,
    help="rescale image by this factor",
    default=1.0)
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
args = parser.parse_args()

#####################################################################

# this function is called as a call-back everytime the trackbar is moved
# (here we just do nothing)


def nothing(x):
    pass


#####################################################################

# define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    if not (args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera Input"  # window name
window_name2 = "Hough Lines"  # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings

    lower_threshold = 25
    cv2.createTrackbar("lower", window_name2, lower_threshold, 255, nothing)
    upper_threshold = 120
    cv2.createTrackbar("upper", window_name2, upper_threshold, 255, nothing)
    smoothing_neighbourhood = 3
    cv2.createTrackbar(
        "smoothing",
        window_name2,
        smoothing_neighbourhood,
        15,
        nothing)
    sobel_size = 3  # greater than 7 seems to crash
    cv2.createTrackbar("sobel size", window_name2, sobel_size, 7, nothing)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()  # rescale if specified

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # get parameters from track bars

        lower_threshold = cv2.getTrackbarPos("lower", window_name2)
        upper_threshold = cv2.getTrackbarPos("upper", window_name2)
        smoothing_neighbourhood = cv2.getTrackbarPos("smoothing", window_name2)
        sobel_size = cv2.getTrackbarPos("sobel size", window_name2)

        # check neighbourhood is greater than 3 and odd

        smoothing_neighbourhood = max(3, smoothing_neighbourhood)
        if not (smoothing_neighbourhood % 2):
            smoothing_neighbourhood = smoothing_neighbourhood + 1

        sobel_size = max(3, sobel_size)
        if not (sobel_size % 2):
            sobel_size = sobel_size + 1

        # convert to grayscale

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # performing smoothing on the image using a 5x5 smoothing mark (see
        # manual entry for GaussianBlur())

        smoothed = cv2.GaussianBlur(
            gray_frame, (smoothing_neighbourhood, smoothing_neighbourhood), 0)

        # perform canny edge detection

        canny = cv2.Canny(
            smoothed,
            lower_threshold,
            upper_threshold,
            apertureSize=sobel_size)

        # perform hough line detection
        # based on tutorial at:
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

        if not (use_probablistic_hough):
            lines = cv2.HoughLines(canny, 1, np.pi/180, 40)
            if lines is not None:
                for rho, theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        else:

            # use use probablistic hough transform

            min_line_length = 100   # requires tuning
            max_line_gap = 10       # requires tuning

            lines = cv2.HoughLinesP(canny, 1, np.pi/180, 10,
                                    min_line_length, max_line_gap)
            if lines is not None:
                for x1, y1, x2, y2 in lines[0]:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # display image

        cv2.imshow(window_name, frame)
        cv2.imshow(window_name2, canny)

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in
        # milliseconds). It waits for specified milliseconds for any keyboard
        # event. If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of
        # multi-byte response)

        # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(40) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen
        # display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(
                window_name2,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)
        elif (key == ord('p')):
            use_probablistic_hough = not (use_probablistic_hough)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################