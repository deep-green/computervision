import os
import cv2
import numpy as np
from pathlib import Path

import chess_math


#from matplotlib import pyplot as plt

# Config
resized_norm = 400
clahe_cliplimit = 30
clahe_tilegridsize = 4
threshold_value = 160
threshold_type = 0
canny_min = 400
canny_max = 500
canny_size = 3
houghlinesp_rho = 8
houghlinesp_theta = 180
houghlinesp_threshold = 50
houghlinesp_minlength = 25
houghlinesp_maxgap = 100

debug=False



def nothing(x):
    pass


def readimage(filename):
    destination = cv2.imread(filename, cv2.IMREAD_COLOR)
    if debug:
        cv2.imshow("Debug Window", destination)
        cv2.waitKey(2000)
    return destination


def resize(source):
    source_height, source_width = source.shape[:2]
    if debug:
        print("source height:" + str(source_height))
        print("source width:" + str(source_width))
    if (source_height != 0 and source_width != 0):
        if (source_height >= source_width):
            destination_width=int(resized_norm)
            destination_height=int(source_height*(resized_norm/source_width))
        if (source_height < source_width):
            destination_height=int(resized_norm)
            destination_width=int(source_width*(resized_norm/source_height))
        if debug:
            print("new height:" + str(destination_height))
            print("new width:" + str(destination_width))
        destination = cv2.resize(source, (destination_width, destination_height), interpolation=cv2.INTER_CUBIC)
        if debug:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(2000)
        return destination


def gray(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        if debug:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(2000)
        return destination


def clahe(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        clahe_def = cv2.createCLAHE(clipLimit=clahe_cliplimit/10, tileGridSize=(clahe_tilegridsize, clahe_tilegridsize))
        destination = clahe_def.apply(source)
        if debug:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(2000)
        return destination


def threshold(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        _, destination = cv2.threshold(source, threshold_value, 255, threshold_type)
        if debug:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(2000)
        return destination


def canny(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = cv2.Canny(source, canny_min, canny_max, canny_size)
        if debug:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(2000)
        return destination


def houghlinesp(source,result):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        linesp = cv2.HoughLinesP(image=source, rho=houghlinesp_rho/10, theta=(np.pi / houghlinesp_theta), threshold=houghlinesp_threshold, minLineLength=houghlinesp_minlength, maxLineGap = houghlinesp_maxgap)
        if debug:
            destination = result.copy()
            if linesp is not None:
               for i in linesp:
                    x1, y1, x2, y2 = i.ravel()
                    cv2.line(destination, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(2000)
        return linesp


def toolchain(filename):
    if debug:
        cv2.namedWindow("Debug Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug Window', 600, 600)
    image_original = readimage(filename)
    image_resize = resize(image_original)
    image_gray = gray(image_resize);
    image_clahe = clahe(image_gray);
    image_threshold = threshold(image_clahe);
    image_canny = canny(image_clahe);
    line_set = houghlinesp(image_canny,image_resize);
    print(chess_math.get_main_angles(line_set))
    #chma.print_line_length(line_set);
    line_list=chess_math.get_list_from_linearray(line_set)
    print(line_list)

	#std::cout << "line_set_size: " << line_set.size() << "\n";
	#//line_set = reduce_lines(line_set);
	#std::cout << "line_set_size: " << line_set.size() << "\n";



	#// draw the result
	#image_resize.copyTo(image_result);
	#image_result=draw_lines(image_result,line_set);
	#image_result=draw_main_angles(image_result);
	#cv::imshow(windowname[10], image_result);






# bild einlesen
dir_path = os.path.dirname(os.path.realpath(__file__))
current_pic_path = "/work_data/0.jpg"
filename=dir_path + current_pic_path
toolchain(filename)

