# INSTALLATIONS:
# first try:
# sudo apt-get install python3-numpy 
# pip install opencv-python

# alternatives:
# sudo apt-get install python-opencv
# pip install opencv-contrib-python


import sys
import os
import cv2
import numpy as np
import base64
import math
import chess_math

# debug options
debug = 0
# use debug = 0 for no debug information
# use debug = 1 for only text information
# use debug = 2 for full debug information and to show all working images
# use debug = 3 for additional saving working images to file
debug_delay_time = 20
debug_delay_time_long = 2000


# globals
resized_norm = 400
blurtype = 0
blursize = 15  # 15
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

filename =""






def resize(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        if (source_height >= source_width):
            destination_width=int(resized_norm)
            destination_height=int(source_height*(resized_norm/source_width))
        if (source_height < source_width):
            destination_height=int(resized_norm)
            destination_width=int(source_width*(resized_norm/source_height))
        destination = cv2.resize(source, (destination_width, destination_height), interpolation=cv2.INTER_CUBIC)
        if debug == 3:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(debug_delay_time)
        return destination


def gray(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        if debug == 3:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(debug_delay_time)
        return destination


def blur(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        if blurtype == 0:
            destination = source
        if blurtype == 1:
            destination = cv2.GaussianBlur(source, (blursize, blursize), 0)
        if blurtype == 2:
            destination = cv2.bilateralFilter(source, blursize, blursize * 2, blursize / 2)
        if debug == 3:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(debug_delay_time)
        return destination


def clahe(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        clahe_def = cv2.createCLAHE(clipLimit=clahe_cliplimit/10, tileGridSize=(clahe_tilegridsize, clahe_tilegridsize))
        destination = clahe_def.apply(source)
        if debug == 3:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(debug_delay_time)
        return destination


def threshold(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        #_, destination = cv2.threshold(source, threshold_value, 255, threshold_type)
        _, destination = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #destination = cv2.adaptiveThreshold(source, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        if debug == 3:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(debug_delay_time)
        return destination


def canny(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = cv2.Canny(source, canny_min, canny_max, canny_size)
        if debug == 3:
            cv2.imshow("Debug Window", destination)
            cv2.waitKey(debug_delay_time)
        return destination


def houghlinesp(source):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        linesp = cv2.HoughLinesP(image=source, rho=houghlinesp_rho/10, theta=(np.pi / houghlinesp_theta), threshold=houghlinesp_threshold, minLineLength=houghlinesp_minlength, maxLineGap = houghlinesp_maxgap)
        return linesp



def draw_lines(source,lines):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = source.copy()
        if lines is not None:
            for i in lines:
                x1, y1, x2, y2 = i.ravel()
                cv2.line(destination, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return destination


def draw_line(source, line, color, thickness):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = source.copy()
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        cv2.line(destination, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness)
    return destination


def draw_line_from_points(source, point_1, point_2, color, thickness):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = source.copy()
        x1 = int(point_1[0])
        y1 = int(point_1[1])
        x2 = int(point_2[0])
        y2 = int(point_2[1])
        cv2.line(destination, (x1, y1), (x2, y2), color, thickness)
        return destination



def get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    point_1_3D = np.array([(point_3D[0], point_3D[1], point_3D[2])])
    (point_1_2D, jacobian) = cv2.projectPoints(point_1_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    point_1_2D = (((point_1_2D[0])[0])[0], ((point_1_2D[0])[0])[1])
    return point_1_2D



def draw_line_jacobian(source, line_3D, color, thickness, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    point_1_3D = np.array([(line_3D[0], line_3D[1], line_3D[2])])
    point_2_3D = np.array([(line_3D[3], line_3D[4], line_3D[5])])
    (point_1_2D, jacobian) = cv2.projectPoints(point_1_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_2_2D, jacobian) = cv2.projectPoints(point_2_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    point_1_2D = (int(((point_1_2D[0])[0])[0]), int(((point_1_2D[0])[0])[1]))
    point_2_2D = (int(((point_2_2D[0])[0])[0]), int(((point_2_2D[0])[0])[1]))
    destination = source.copy()
    cv2.line(destination, point_1_2D, point_2_2D, color, thickness)
    return destination


def draw_quader_jacobian(source, line_3D, color, thickness, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    point_1_3D = np.array([(line_3D[0], line_3D[1], line_3D[2])])
    point_2_3D = np.array([(line_3D[3], line_3D[1], line_3D[2])])
    point_3_3D = np.array([(line_3D[0], line_3D[4], line_3D[2])])
    point_4_3D = np.array([(line_3D[3], line_3D[4], line_3D[2])])
    point_5_3D = np.array([(line_3D[0], line_3D[1], line_3D[5])])
    point_6_3D = np.array([(line_3D[3], line_3D[1], line_3D[5])])
    point_7_3D = np.array([(line_3D[0], line_3D[4], line_3D[5])])
    point_8_3D = np.array([(line_3D[3], line_3D[4], line_3D[5])])
    (point_1_2D, jacobian) = cv2.projectPoints(point_1_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_2_2D, jacobian) = cv2.projectPoints(point_2_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_3_2D, jacobian) = cv2.projectPoints(point_3_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_4_2D, jacobian) = cv2.projectPoints(point_4_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_5_2D, jacobian) = cv2.projectPoints(point_5_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_6_2D, jacobian) = cv2.projectPoints(point_6_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_7_2D, jacobian) = cv2.projectPoints(point_7_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    (point_8_2D, jacobian) = cv2.projectPoints(point_8_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    point_1_2D = (int(((point_1_2D[0])[0])[0]), int(((point_1_2D[0])[0])[1]))
    point_2_2D = (int(((point_2_2D[0])[0])[0]), int(((point_2_2D[0])[0])[1]))
    point_3_2D = (int(((point_3_2D[0])[0])[0]), int(((point_3_2D[0])[0])[1]))
    point_4_2D = (int(((point_4_2D[0])[0])[0]), int(((point_4_2D[0])[0])[1]))
    point_5_2D = (int(((point_5_2D[0])[0])[0]), int(((point_5_2D[0])[0])[1]))
    point_6_2D = (int(((point_6_2D[0])[0])[0]), int(((point_6_2D[0])[0])[1]))
    point_7_2D = (int(((point_7_2D[0])[0])[0]), int(((point_7_2D[0])[0])[1]))
    point_8_2D = (int(((point_8_2D[0])[0])[0]), int(((point_8_2D[0])[0])[1]))
    destination = source.copy()
    cv2.line(destination, point_1_2D, point_2_2D, color, thickness)
    cv2.line(destination, point_2_2D, point_4_2D, color, thickness)
    cv2.line(destination, point_4_2D, point_3_2D, color, thickness)
    cv2.line(destination, point_3_2D, point_1_2D, color, thickness)
    cv2.line(destination, point_5_2D, point_6_2D, color, thickness)
    cv2.line(destination, point_6_2D, point_8_2D, color, thickness)
    cv2.line(destination, point_8_2D, point_7_2D, color, thickness)
    cv2.line(destination, point_7_2D, point_5_2D, color, thickness)
    cv2.line(destination, point_1_2D, point_5_2D, color, thickness)
    cv2.line(destination, point_2_2D, point_6_2D, color, thickness)
    cv2.line(destination, point_4_2D, point_8_2D, color, thickness)
    cv2.line(destination, point_3_2D, point_7_2D, color, thickness)
    return destination


def draw_line_list(source, line_list, color, thickness):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = source.copy()
        if line_list is not None:
            for line in line_list:
                x1 = line[0]
                y1 = line[1]
                x2 = line[2]
                y2 = line[3]
                cv2.line(destination, (x1, y1), (x2, y2), color, thickness)
        return destination




def draw_points(source, points ,color, thickness):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = source.copy()
        if points is not None:
            for point in points:
                x = int(point[0])
                y = int(point[1])
                cv2.circle(destination, (x, y), 5, color, thickness)
        return destination



def draw_main_angles(source,angles):
    source_height, source_width = source.shape[:2]
    if (source_height != 0 and source_width != 0):
        destination = source.copy()
        if angles is not None:
            for angle in angles:
                x1=int(source_width/2)
                y1=int(source_height/2)
                x2=int(source_width / 2 + math.cos(angle) * 100)
                y2=int(source_height / 2 + math.sin(angle) * 100)
                cv2.line(destination, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return destination


# toolchain to get best horizontals
def toolchain_01(given_image):
    image_original = given_image
    image_resize = resize(image_original)
    image_blur = blur(image_resize)
    image_gray = gray(image_blur)
    image_clahe = clahe(image_gray)
    image_canny = canny(image_clahe)
    line_set = houghlinesp(image_canny)
    line_list = chess_math.get_list_from_linearray(line_set)
    # divide the houghlines in two parts
    line_list_horizontal = chess_math.divide_line_list_by_direction(line_list,0)
    line_list_horizontal = chess_math.expand_lines_to_image_size(line_list_horizontal, image_resize.shape[1], image_resize.shape[0])
    line_list_horizontal = chess_math.homogen_lines(line_list_horizontal)
    line_list_horizontal = chess_math.delete_identical_lines_from_list(line_list_horizontal)
    if debug == 2:
        image_result = draw_line_list(image_resize, line_list_horizontal,(0, 255, 0),1)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time)
    return line_list_horizontal


# toolchain to get best verticals
def toolchain_02(given_image, already_used_angle):
    image_original = given_image
    image_resize = resize(image_original)
    image_blur = blur(image_resize)
    image_gray = gray(image_blur)
    image_clahe = clahe(image_gray)
    image_canny = canny(image_clahe)
    line_set = houghlinesp(image_canny)
    line_list = chess_math.get_list_from_linearray(line_set)
    # divide the houghlines in two parts
    line_list_vertical = chess_math.divide_line_list_by_direction(line_list, 1)
    average_angle = chess_math.get_average_angle(line_list_vertical)
    if abs(average_angle-already_used_angle) < 45/180 * math.pi:
        line_list_vertical = chess_math.divide_line_list_by_direction(line_list, 0)
    line_list_vertical = chess_math.expand_lines_to_image_size(line_list_vertical, image_resize.shape[1], image_resize.shape[0])
    line_list_vertical = chess_math.homogen_lines(line_list_vertical)
    line_list_vertical = chess_math.delete_identical_lines_from_list(line_list_vertical)
    if debug > 1:
        image_result = draw_line_list(image_resize, line_list_vertical, (255, 0, 0), 1)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time)
    return line_list_vertical


def toolchain_03(given_image):
    image_original = given_image
    image_resize = resize(image_original)
    image_blur = blur(image_resize)
    image_gray = gray(image_blur)
    image_clahe = clahe(image_gray)
    image_canny = canny(image_clahe)
    line_set = houghlinesp(image_canny)
    line_list = chess_math.get_list_from_linearray(line_set)
    line_list = chess_math.expand_lines_to_image_size(line_list, image_resize.shape[1], image_resize.shape[0])
    line_list = chess_math.homogen_lines(line_list)
    if debug > 1:
        image_result = draw_line_list(image_resize, line_list, (0, 255, 0), 1)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time)
    line_list = chess_math.delete_identical_lines_from_list(line_list)
    return line_list




def toolchain_00(original_image):
    global resized_norm
    global blurtype
    global blursize
    global clahe_cliplimit
    global clahe_tilegridsize
    global threshold_value
    global threshold_type
    global canny_min
    global canny_max
    global canny_size
    global houghlinesp_rho
    global houghlinesp_theta
    global houghlinesp_threshold
    global houghlinesp_minlength
    global houghlinesp_maxgap

    global filename
    if debug > 1:
        cv2.namedWindow("Debug Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug Window', 1200, 900)

    resized_norm = 400
    blurtype = 0
    blursize = 5
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

    number_of_horizontal_lines = 1000
    while (number_of_horizontal_lines > 10):
        houghlinesp_threshold = houghlinesp_threshold + 2
        horizontals=toolchain_01(original_image)
        number_of_horizontal_lines = len(horizontals)
    houghlinesp_threshold = 50
    number_of_vertical_lines = 1000
    average_horizontal_angle = chess_math.get_average_angle(horizontals)
    while (number_of_vertical_lines > 10):
        houghlinesp_threshold = houghlinesp_threshold + 2
        verticals = toolchain_02(original_image, average_horizontal_angle)
        number_of_vertical_lines = len(verticals)
    horizontal_pitch_angle = chess_math.pitch_angle_parallel(horizontals)
    vertical_pitch_angle = chess_math.pitch_angle_parallel(verticals)

    if debug > 0:
        print("horizontal_pitch_angle: " + str(horizontal_pitch_angle))
        print("vertical_pitch_angle: " + str(vertical_pitch_angle))
    if debug > 1:
        image_resize = resize(original_image)
        image_result = draw_line_list(image_resize, horizontals, (0, 255, 0), 1)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time)
        image_result = draw_line_list(image_resize, verticals, (255, 0, 0), 1)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time)

    # get the horizontal lines
    if horizontal_pitch_angle == 10000: # horizontal lines are not parallel
        if debug >1:
            image_resize = resize(original_image)
            image_result = draw_line_list(image_resize, horizontals, (0, 255, 0), 1)
            cv2.imshow("Debug Window", image_result)
            cv2.waitKey(debug_delay_time)
        section_points_list = chess_math.section_point_list(horizontals)
        section_points_list = chess_math.reduce_points_to_number_by_distance_to_average(section_points_list, 20)
        horizontal_average = chess_math.average_point_of_point_list(section_points_list)
        horizontals = chess_math.reduce_line_list_by_distance_to_point(horizontals, horizontal_average, 100)
        if debug == 2:
            image_resize = resize(original_image)
            image_result = draw_line_list(image_resize, horizontals, (0, 255, 0), 1)
            cv2.imshow("Debug Window", image_result)
            cv2.waitKey(debug_delay_time)
        section_points_list = chess_math.section_point_list(horizontals)
        section_points_list = chess_math.reduce_points_to_number_by_distance_to_average(section_points_list, 20)
        horizontal_average = chess_math.average_point_of_point_list(section_points_list)
        max_to_line_distance = 0
        for i in range(len(horizontals)):
            distance = chess_math.min_distance_point_line_2D(horizontal_average, horizontals[i])
            if distance > max_to_line_distance:
                max_to_line_distance = distance
        # set search parameters for non-parallel horizontal lines
        resized_norm = 400
        blurtype = 0
        blursize = 5
        clahe_cliplimit = 30
        clahe_tilegridsize = 4
        threshold_value = 160
        threshold_type = 0
        canny_min = 400
        canny_max = 500
        canny_size = 3
        houghlinesp_rho = 8
        houghlinesp_theta = 180
        houghlinesp_threshold = 40
        houghlinesp_minlength = 25
        houghlinesp_maxgap = 100
        horizontals = toolchain_03(original_image)

        horizontals = chess_math.reduce_line_list_by_distance_to_point(horizontals, horizontal_average, 3.0 * max_to_line_distance)
        horizontals = chess_math.sort_line_list_by_height(horizontals)

        if debug > 1:
            image_resize = resize(original_image)
            image_result = draw_line_list(image_resize, horizontals, (0, 255, 0), 1)
            cv2.imshow("Debug Window", image_result)
            cv2.waitKey(debug_delay_time)

    else: # horizontal lines are parallel

        if debug > 1:
            image_resize = resize(original_image)
            image_result = draw_line_list(image_resize, horizontals, (0, 255, 0), 1)
            cv2.imshow("Debug Window", image_result)
            cv2.waitKey(debug_delay_time)

        horizontal_average = [9999, 9999]
        # set search parameters for horizontal parallel lines
        blurtype = 0
        blursize = 5
        clahe_cliplimit = 30
        clahe_tilegridsize = 4
        threshold_value = 160
        threshold_type = 0
        canny_min = 400
        canny_max = 500
        canny_size = 3
        houghlinesp_rho = 8
        houghlinesp_theta = 180
        houghlinesp_threshold = 30
        houghlinesp_minlength = 25
        houghlinesp_maxgap = 100
        if abs(horizontal_pitch_angle)< 0.05:
            houghlinesp_threshold = 70
        horizontals = toolchain_03(original_image)
        horizontals = chess_math.reduce_line_list_by_angle(horizontals, horizontal_pitch_angle, 0.2)

    # get the vertical lines
    if vertical_pitch_angle == 10000: # vertical lines are not parallel
        if debug > 1:
            image_resize = resize(original_image)
            image_result = draw_line_list(image_resize, verticals, (255, 0, 0), 1)
            cv2.imshow("Debug Window", image_result)
            cv2.waitKey(debug_delay_time)
        section_points_list = chess_math.section_point_list(verticals)
        section_points_list = chess_math.reduce_points_to_number_by_distance_to_average(section_points_list, 20)
        vertical_average = chess_math.average_point_of_point_list(section_points_list)
        verticals = chess_math.reduce_line_list_by_distance_to_point(verticals, vertical_average, 100)
        section_points_list = chess_math.section_point_list(verticals)
        section_points_list = chess_math.reduce_points_to_number_by_distance_to_average(section_points_list, 20)
        vertical_average = chess_math.average_point_of_point_list(section_points_list)
        if debug == 2:
            image_resize = resize(original_image)
            image_result = draw_line_list(image_resize, verticals, (255, 0, 0), 1)
            cv2.imshow("Debug Window", image_result)
            cv2.waitKey(debug_delay_time)
        max_to_line_distance = 0
        for i in range(len(verticals)):
            distance = chess_math.min_distance_point_line_2D(vertical_average, verticals[i])
            if distance > max_to_line_distance:
                max_to_line_distance = distance

        # set search parameters for vertical lines
        blurtype = 0
        blursize = 5
        clahe_cliplimit = 30
        clahe_tilegridsize = 4
        threshold_value = 160
        threshold_type = 0
        canny_min = 300
        canny_max = 500
        canny_size = 3
        houghlinesp_rho = 8
        houghlinesp_theta = 180
        houghlinesp_threshold = 50
        houghlinesp_minlength = 25
        houghlinesp_maxgap = 100

        # get new line list and reduce them by calculated section point
        verticals = toolchain_03(original_image)
        verticals = chess_math.reduce_line_list_by_distance_to_point(verticals, vertical_average, 3.0 * max_to_line_distance)

    # define the sizes
    image_resize = resize(original_image)
    image_height, image_width = image_resize.shape[:2]

    # sort the lines
    verticals = chess_math.sort_line_list_by_height(verticals)
    horizontals = chess_math.sort_line_list_by_height(horizontals)

    # eliminate short lines
    verticals = chess_math.reduce_line_list_by_vertical_length_threshold(verticals, image_height / 3)
    horizontals = chess_math.reduce_line_list_by_horizontal_length_threshold(horizontals, image_width / 2)

    # get middle lines and first try pattern
    middle_horizontal_1 = chess_math.get_middle_line_of_line_list(horizontals)
    middle_vertical_1 = chess_math.get_middle_line_of_line_list(verticals)
    vertical_pattern_points = chess_math.get_pattern_distance(horizontals, middle_vertical_1, vertical_average)
    horizontal_pattern_points = chess_math.get_pattern_distance(verticals, middle_horizontal_1, horizontal_average)
    middle_horizontal_2 = chess_math.get_second_middle_line_of_line_list(horizontals, middle_horizontal_1, middle_vertical_1, chess_math.distance_points_2d(vertical_pattern_points[0], vertical_pattern_points[1]))
    middle_vertical_2 = chess_math.get_second_middle_line_of_line_list(verticals, middle_vertical_1, middle_horizontal_1, chess_math.distance_points_2d(horizontal_pattern_points[0], horizontal_pattern_points[1]))
    middles_horizontal = [middle_horizontal_1, middle_horizontal_2]
    middles_vertical = [middle_vertical_1, middle_vertical_2]
    middles_horizontal = chess_math.sort_line_list_by_height(middles_horizontal)
    middles_vertical = chess_math.sort_line_list_by_height(middles_vertical)

    # get point for camera definition
    camera_point_1 = chess_math.section_point_2d(middles_horizontal[0], middles_vertical[0])
    camera_point_2 = chess_math.section_point_2d(middles_horizontal[0], middles_vertical[1])
    camera_point_3 = chess_math.section_point_2d(middles_horizontal[1], middles_vertical[0])
    camera_point_4 = chess_math.section_point_2d(middles_horizontal[1], middles_vertical[1])
    camera_points = [camera_point_1, camera_point_2, camera_point_3, camera_point_4]

    # draw result
    if debug > 1:
        #cv2.namedWindow("Debug Window", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Debug Window', 800, 600)
        image_resize = resize(original_image)
        image_result = draw_line(image_resize, middle_vertical_1, (255, 0, 0), 2)
        image_result = draw_line(image_result, middle_horizontal_1, (0, 255, 0), 2)
        image_result = draw_line(image_result, middle_vertical_2, (255, 0, 0), 2)
        image_result = draw_line(image_result, middle_horizontal_2, (0, 255, 0), 2)
        image_result = draw_points(image_result, camera_points, (0,0,255), 2)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time_long)


    # define world coordinates of the chess field
    chess_field_size = 50.0
    world_point_1 = [0.0, 0.0, 0.0]
    world_point_2 = [chess_field_size, 0.0, 0.0]
    world_point_3 = [0.0, chess_field_size, 0.0]
    world_point_4 = [chess_field_size, chess_field_size, 0.0]
    world_points = [world_point_1, world_point_2, world_point_3, world_point_4]

    # get initial camera matrix
    initial_camera_matrix = chess_math.get_camera_matrix(camera_points, chess_field_size, image_width, image_height)

    # calculate rotation vector and translation vector via solvePnP
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    model_points = np.array(world_points,np.float32)
    image_points = np.array(camera_points, np.float32)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # calculate rotation matrix out of the rotation vector via Rodrigues
    rmat = cv2.Rodrigues(rotation_vector)[0]

    # calculate the Euler angles from the rotation matrix
    beta = -math.acos(rmat[2, 2])
    alpha = math.acos(rmat[2, 1] / math.sin(beta))
    if (abs(math.sin(alpha) * math.sin(beta) - rmat[2, 0]) > 0.0001):
        alpha = alpha
    gamma = math.asin(rmat[0, 2] / math.sin(beta))
    if (abs(-1 * math.sin(beta) * math.cos(gamma) - rmat[1, 2]) > 0.0001):
        gamma = math.pi - gamma
    test_matrix = chess_math.rotation_matrix_from_euler_angles(alpha, beta, gamma)


    # searching good points on the board
    section_points_list = chess_math.section_point_list_from_two_line_lists(verticals, horizontals)
    search_distance = 5
    intrinsinc_guess = False
    flag = cv2.SOLVEPNP_DLS

    for i in range(1, 5):      # number of tries

        # y+
        checkpoint = [0.0 , i * chess_field_size, 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # x+
        checkpoint = [i * chess_field_size , 0.0 , 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # y-
        checkpoint = [0.0 , -i * chess_field_size, 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # x-
        checkpoint = [-i * chess_field_size , 0.0 , 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # x+y+
        checkpoint = [i * chess_field_size , i * chess_field_size, 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # x-y-
        checkpoint = [-i * chess_field_size , -i * chess_field_size, 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # x+y-
        checkpoint = [i * chess_field_size , -i * chess_field_size, 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # x-y+
        checkpoint = [-i * chess_field_size , +i * chess_field_size, 0.0]
        point_3D = checkpoint
        point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        distance = chess_math.get_min_distance_from_point_to_one_point_of_a_list(point_2D, section_points_list)
        if distance < search_distance:
            point_2D = chess_math.get_nearest_point_out_of_a_list(point_2D, section_points_list)
            if world_points.count(point_3D) == 0:
                world_points.append(point_3D)
                camera_points.append(point_2D)
            model_points = np.array(world_points, np.float32)
            image_points = np.array(camera_points, np.float32)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # get best focal length
        min_distance_sum = 1e12
        for i in range(400,2000,5):
            center = (image_width / 2, image_height / 2)
            initial_camera_matrix = np.array([[i, 0, center[0]], [0, i, center[1]], [0, 0, 1]], dtype="double")
            distance_sum = 0
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)
            for j in range(len(model_points)):
                point_3D = world_points[j]
                point_2D = get_2D_point_jacobian(point_3D, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
                distance_sum += chess_math.distance_points_2d(point_2D, camera_points[j])
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                focal_length = i
        initial_camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, initial_camera_matrix, dist_coeffs, intrinsinc_guess, flag)

        # calculate rotation matrix out of the rotation vector via Rodrigues
        rmat = cv2.Rodrigues(rotation_vector)[0]

        # calculate the Euler angles from the rotation matrix
        beta = -math.acos(rmat[2, 2])
        alpha = math.acos(rmat[2, 1] / math.sin(beta))
        if (abs(math.sin(alpha) * math.sin(beta) - rmat[2, 0]) > 0.0001):
            alpha = alpha
        gamma = math.asin(rmat[0, 2] / math.sin(beta))
        if (abs(-1 * math.sin(beta) * math.cos(gamma) - rmat[1, 2]) > 0.000000001):
            gamma = math.pi - gamma
        test_matrix = chess_math.rotation_matrix_from_euler_angles(alpha, beta, gamma)

    if debug > 0:
        print()
        print("image data:")
        print("-----------")
        print("rotation vector:" + str(rotation_vector))
        print("translation vector:" +str(translation_vector))
        print("rotation matrix: " +str(rmat))
        print("euler angles:")
        print("   alpha: " + str(alpha * 180 / math.pi))
        print("   beta : " + str(beta * 180 / math.pi))
        print("   gamma: " + str(gamma * 180 / math.pi))
        print("test matrix:" + str(test_matrix))
        print()
        print("camera and positional information:")
        print("----------------------------------")
        print("   focal length  : " + str(initial_camera_matrix[0, 0]))
        print("   swing angle   : " + str(-alpha * 180 / math.pi))
        print("   tilt angle    : " + str(-beta * 180 / math.pi))
        print("   tend angle    : " + str(-gamma * 180 / math.pi))
        print()


    # on debug: draw and show result
    if debug > 1:
        #cv2.namedWindow("Debug Window", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Debug Window', 800, 600)
        image_resize = resize(original_image)
        image_result = draw_line(image_resize, middle_vertical_1, (255, 0, 0), 2)
        image_result = draw_line(image_result, middle_horizontal_1, (0, 255, 0), 2)
        image_result = draw_line(image_result, middle_vertical_2, (255, 0, 0), 2)
        image_result = draw_line(image_result, middle_horizontal_2, (0, 255, 0), 2)
        #image_result = draw_points(image_result, first_camera_points, (0,0,255), 2)
        #image_result = draw_line_list(image_result, verticals, (255, 0, 0), 1)
        #image_result = draw_line_list(image_result, horizontals, (0, 255, 0), 1)
        #image_result = draw_points(image_result, vertical_pattern_points,(0,255,255), 1)
        #image_result = draw_points(image_result, horizontal_pattern_points,(0,255,255), 1)
        image_result = draw_points(image_result, camera_points, (0,0,255), 2)
        # draw quader over field
        number_of_quaders = 20
        for i in range(number_of_quaders):
            for j in range(number_of_quaders):
                #image_result = draw_quader_jacobian(image_result, [int(i-4)*chess_field_size, (j-4)*chess_field_size, 0.0, (i-3)*chess_field_size, (j-3)*chess_field_size, -3.0], (255, 255, 0), 1, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
                image_result = draw_quader_jacobian(image_result, [int(i - number_of_quaders/2) * chess_field_size, int(j - number_of_quaders/2) * chess_field_size, 0.0, (1 + int(i - number_of_quaders/2)) * chess_field_size, (1+ int(j - number_of_quaders/2)) * chess_field_size, -3.0], (255, 255, 0), 1, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time_long)
    #
    #
    #
    #
    #
    # --------------------------------------------------------------------------
    # get board colors and direction
    # --------------------------------------------------------------------------
    start_field = 8
    field_counter = 16
    border = 0
    field_color = [['u' for x in range(field_counter)] for y in range(field_counter)]
    for i in range(field_counter):
        for j in range(field_counter):
            world_point_1 = [int(0 + i - start_field) * chess_field_size, int(0 + j - start_field) * chess_field_size, 0.0]
            world_point_2 = [int(1 + i - start_field) * chess_field_size, int(0 + j - start_field) * chess_field_size, 0.0]
            world_point_3 = [int(0 + i - start_field) * chess_field_size, int(1 + j - start_field) * chess_field_size, 0.0]
            world_point_4 = [int(1 + i - start_field) * chess_field_size, int(1 + j - start_field) * chess_field_size, 0.0]
            image_point_1 = get_2D_point_jacobian(world_point_1, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_2 = get_2D_point_jacobian(world_point_2, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_3 = get_2D_point_jacobian(world_point_3, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_4 = get_2D_point_jacobian(world_point_4, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            min_x = min(image_point_1[0], image_point_2[0], image_point_3[0], image_point_4[0])
            max_x = max(image_point_1[0], image_point_2[0], image_point_3[0], image_point_4[0])
            min_y = min(image_point_1[1], image_point_2[1], image_point_3[1], image_point_4[1])
            max_y = max(image_point_1[1], image_point_2[1], image_point_3[1], image_point_4[1])
            if max_x - border <= image_width and max_y - border <= image_height and min_x + border >= 0 and min_y + border >= 0:
                field_contour = [np.array([image_point_1, image_point_2, image_point_4, image_point_3], dtype=np.int32)]
                mask = np.zeros((image_resize.shape[0], image_resize.shape[1], 3), np.uint8)
                mask = cv2.drawContours(mask, field_contour, -1, (255, 255, 255), -1)
                blurtype = 1
                blursize = 15
                clahe_tilegridsize = 10
                # check if it is a white field
                _, thresh_image = cv2.threshold(clahe(blur(gray(image_resize))), 180, 255, 0)
                field_image = thresh_image * gray(mask)
                field_image = field_image * gray(mask)
                ret_image, contours, hierarchy = cv2.findContours(field_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                area = 0
                for contour in contours:
                    area += cv2.contourArea(contour)
                if area > 0.6 * cv2.contourArea(field_contour[0]):
                    field_color[j][i] = 'w'
                # check if it is a black field
                _, thresh_image = cv2.threshold(clahe(blur(gray(image_resize))), 135, 255, 0)
                field_image = thresh_image * gray(mask)
                field_image = field_image * gray(mask)
                ret_image, contours, hierarchy = cv2.findContours(field_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                area = 0
                for contour in contours:
                    area += cv2.contourArea(contour)
                if area < 0.2 * cv2.contourArea(field_contour[0]):
                    field_color[j][i] = 'b'

                # on debug: show the result
                if debug > 1:
                    #field_image = field_image[int(min_y - border): int(max_y + border), int(min_x - border): int(max_x + border)]
                    #cv2.imshow("Debug Window", field_image)
                    cv2.waitKey(5)

                # on debug: save the result
                if debug > 2:
                    field_image = field_image[int(min_y - border): int(max_y + border), int(min_x - border): int(max_x + border)]
                    new_filename = filename + " field " + str(i) + "_" + str(j) + ".jpg"
                    cv2.imwrite(new_filename, field_image)
            else:
                # the field to check is outside the image
                field_color[j][i] = ' '

    # reduce possible fields
    columns = []
    for columncounter in range(field_counter):
        column = []
        for line in field_color:
            char = line[columncounter]
            column.append(char)
        columns.append(column)
    for columncounter in range(field_counter):
        if field_counter - columns[columncounter].count(' ') < 7:
            for line in range(len(field_color)):
                (field_color[line])[columncounter] = ' '
        if field_counter - columns[columncounter].count('w') < 7:
            for line in range(len(field_color)):
                (field_color[line])[columncounter] = ' '
        if field_counter - columns[columncounter].count('b') < 7:
            for line in range(len(field_color)):
                (field_color[line])[columncounter] = ' '
    for line in field_color:
        if field_counter - line.count(' ') < 7:
            for i in range(len(line)):
                line[i] = ' '
        if field_counter - line.count('w') < 7:
            for i in range(len(line)):
                line[i] = ' '
        if field_counter - line.count('b') < 7:
            for i in range(len(line)):
                line[i] = ' '

    # print the result
    if debug > 0:
        print()
        print("field colors detected:")
        print("----------------------")
        print()
        print("   ['0', [1], [2], [3], [4], [5], [6], [7], [8], [9], [10, [11, [12, [13, [14, [15")
        counter = 0
        for line in field_color:
            print(str(100 + counter) + str(line))
            counter += 1

    # find the best match for field colors and board direction
    long_list = []
    for line in field_color:
        for char in line:
            long_list.append(char)
    long_texts = []
    for start_line in range(len(field_color)-7):
        for start_column in range(len(field_color[0])-7):
            long_text = []
            long_text.append(str(start_line))
            long_text.append(str(start_column))
            for i in range(8):
                for j in range(8):
                    long_text.append(field_color[start_line+i][start_column+j])
            long_texts.append(long_text)
    max_ranking = 0
    max_ranking_right = 0
    max_ranking_wrong = 0
    target_line_right_direction = ['w', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'w']
    target_line_wrong_direction = ['b', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'b', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'w', 'b', 'w', 'b', 'w', 'b', 'w', 'b']
    target_line = 0
    target_start_line = 0
    target_start_column = 0
    for long_text in long_texts:
        start_line = long_text[0]
        start_column = long_text[1]
        for i in range(2, len(long_text)-63):
            test_line = []
            for j in range(64):
                test_line.append(long_text[i+j])
        ranking_right_direction = 0
        ranking_wrong_direction = 0
        for i in range(64):
            if test_line[i] == target_line_right_direction[i]:
                ranking_right_direction += 1
            if test_line[i] == 'b' and target_line_right_direction[i] == 'w':
                ranking_right_direction -= 1
            if test_line[i] == 'w' and target_line_right_direction[i] == 'b':
                ranking_right_direction -= 1
            if test_line[i] == ' ':
                ranking_right_direction -= 1
            if test_line[i] == target_line_wrong_direction[i]:
                ranking_wrong_direction += 1
            if test_line[i] == 'b' and target_line_wrong_direction[i] == 'w':
                ranking_wrong_direction -= 1
            if test_line[i] == 'w' and target_line_wrong_direction[i] == 'b':
                ranking_wrong_direction -= 1
            if test_line[i] == ' ':
                ranking_wrong_direction -= 1
        if ranking_right_direction > max_ranking_right:
            max_ranking_right = ranking_right_direction
        if ranking_wrong_direction > max_ranking_wrong:
            max_ranking_wrong = ranking_wrong_direction
        if ranking_wrong_direction > max_ranking:
            max_ranking = ranking_wrong_direction
            target_start_line = start_line
            target_start_column = start_column
            best_line = test_line
            board_direction = "wrong"
            target_line = target_line_wrong_direction
        if ranking_right_direction >= max_ranking:
            max_ranking = ranking_right_direction
            target_start_line = start_line
            target_start_column = start_column
            best_line = test_line
            board_direction = "right"
            target_line = target_line_right_direction

    # print the result
    if debug > 0:
        print()
        print("best matching color scheme and board direction:")
        print("-----------------------------------------------")
        print("best_start_column : " + str(target_start_column))
        print("best_start_line   : " + str(target_start_line))
        print("target_line       : " + str(target_line))
        print("best_line         : " + str(best_line))
        print("board direction   : " + board_direction)
        print("max_ranking       : " + str(max_ranking))
        print("max_ranking_wrong : " + str(max_ranking_wrong))
        print("max_ranking_right : " + str(max_ranking_right))
        print()

    # calculate the board corners and draw them into result image
    board_corners = []
    board_corner = [int(0 - start_field + int(target_start_column)) * chess_field_size, int(0 -start_field + int(target_start_line)) * chess_field_size, 0.0]
    board_corner = get_2D_point_jacobian(board_corner, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
    board_corners.append(board_corner)
    board_corner = [int(8 - start_field + int(target_start_column)) * chess_field_size, int(0 - start_field + int(target_start_line)) * chess_field_size, 0.0]
    board_corner = get_2D_point_jacobian(board_corner, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
    board_corners.append(board_corner)
    board_corner = [int(8 - start_field +int(target_start_column)) * chess_field_size, int(8 - start_field + int(target_start_line)) * chess_field_size, 0.0]
    board_corner = get_2D_point_jacobian(board_corner, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
    board_corners.append(board_corner)
    board_corner = [int(0 - start_field +int(target_start_column)) * chess_field_size, int(8 - start_field + int(target_start_line)) * chess_field_size, 0.0]
    board_corner = get_2D_point_jacobian(board_corner, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
    board_corners.append(board_corner)

    # on debug: show the result
    if debug > 1:
        image_result = draw_points(image_result, board_corners, (0, 255, 255), 5)
        cv2.imshow("Debug Window", image_result)
        cv2.waitKey(debug_delay_time_long)

    # on debug: save the result
    if debug > 2:
        new_filename = filename + "result.jpg"
        cv2.imwrite(new_filename , image_result)


    #
    #
    #
    #
    #
    # --------------------------------------------------------------------------
    # get figures
    # --------------------------------------------------------------------------
    start_x = start_field - int(target_start_column)
    start_y = start_field - int(target_start_line)
    field_counter = 8
    border = 5
    chess_field_height = chess_field_size * 2.0
    for i in range(field_counter):
        for j in range(field_counter):
            #print("i , j: " + str(i) + " , "+str(j))
            world_point_1 = [int(0 + i - start_x) * chess_field_size, int(0 + j - start_y) * chess_field_size, 0.0]
            world_point_2 = [int(1 + i - start_x) * chess_field_size, int(0 + j - start_y) * chess_field_size, 0.0]
            world_point_3 = [int(0 + i - start_x) * chess_field_size, int(1 + j - start_y) * chess_field_size, 0.0]
            world_point_4 = [int(1 + i - start_x) * chess_field_size, int(1 + j - start_y) * chess_field_size, 0.0]
            world_point_5 = [int(0 + i - start_x) * chess_field_size, int(0 + j - start_y) * chess_field_size, -chess_field_height]
            world_point_6 = [int(1 + i - start_x) * chess_field_size, int(0 + j - start_y) * chess_field_size, -chess_field_height]
            world_point_7 = [int(0 + i - start_x) * chess_field_size, int(1 + j - start_y) * chess_field_size, -chess_field_height]
            world_point_8 = [int(1 + i - start_x) * chess_field_size, int(1 + j - start_y) * chess_field_size, -chess_field_height]
            image_point_1 = get_2D_point_jacobian(world_point_1, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_2 = get_2D_point_jacobian(world_point_2, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_3 = get_2D_point_jacobian(world_point_3, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_4 = get_2D_point_jacobian(world_point_4, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_5 = get_2D_point_jacobian(world_point_5, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_6 = get_2D_point_jacobian(world_point_6, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_7 = get_2D_point_jacobian(world_point_7, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            image_point_8 = get_2D_point_jacobian(world_point_8, rotation_vector, translation_vector, initial_camera_matrix, dist_coeffs)
            min_x = min(image_point_1[0], image_point_2[0], image_point_3[0], image_point_4[0])
            max_x = max(image_point_1[0], image_point_2[0], image_point_3[0], image_point_4[0])
            min_y = min(image_point_1[1], image_point_2[1], image_point_3[1], image_point_4[1])
            max_y = max(image_point_1[1], image_point_2[1], image_point_3[1], image_point_4[1])
            scale_factor = original_image.shape[0] / image_height
            if max_x - border <= image_width and max_y - border <= image_height and min_x + border >= 0 and min_y + border >= 0:
                min_x = min(image_point_1[0], image_point_2[0], image_point_3[0], image_point_4[0], image_point_5[0], image_point_6[0], image_point_7[0], image_point_8[0])
                max_x = max(image_point_1[0], image_point_2[0], image_point_3[0], image_point_4[0], image_point_5[0], image_point_6[0], image_point_7[0], image_point_8[0])
                min_y = min(image_point_1[1], image_point_2[1], image_point_3[1], image_point_4[1], image_point_5[1], image_point_6[1], image_point_7[1], image_point_8[1])
                max_y = max(image_point_1[1], image_point_2[1], image_point_3[1], image_point_4[1], image_point_5[1], image_point_6[1], image_point_7[1], image_point_8[1])
                max_x = min(original_image.shape[1], scale_factor*(max_x + border))
                min_x = max(0, scale_factor*(min_x - border))
                max_y = min(original_image.shape[0], scale_factor*(max_y + border))
                min_y = max(0, scale_factor*(min_y - border))
                figure_image = original_image.copy()
                line_1 = [scale_factor * image_point_1[0], scale_factor * image_point_1[1], scale_factor * image_point_5[0], scale_factor * image_point_5[1]]
                line_2 = [scale_factor * image_point_2[0], scale_factor * image_point_2[1], scale_factor * image_point_6[0], scale_factor * image_point_6[1]]
                line_3 = [scale_factor * image_point_3[0], scale_factor * image_point_3[1], scale_factor * image_point_7[0], scale_factor * image_point_7[1]]
                line_4 = [scale_factor * image_point_4[0], scale_factor * image_point_4[1], scale_factor * image_point_8[0], scale_factor * image_point_8[1]]
                figure_image = figure_image[int(min_y): int(max_y), int(min_x): int(max_x)]
                # tilt the figure picture'
                average_tilt_angle = chess_math.get_average_angle([line_1, line_2, line_3, line_4])
                rows = figure_image.shape[0]
                cols = figure_image.shape[1]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (average_tilt_angle - math.pi/2) * 180 / math.pi, 1)
                figure_image = cv2.warpAffine(figure_image, M, (cols, rows))
                destination_height = 500
                destination_width = int(figure_image.shape[1] / figure_image.shape[0]  * destination_height)
                figure_image = cv2.resize(figure_image, (destination_width, destination_height), interpolation=cv2.INTER_CUBIC)
                clahe_cliplimit = 20
                clahe_tilegridsize = 4
                canny_min = 150
                canny_max = 200
                figure_image_canny = canny(clahe(gray(figure_image)))

                # on debug: show the figure picture
                if debug > 1:
                    cv2.imshow("Debug Window", figure_image)
                    cv2.waitKey(debug_delay_time)

                # on debug: save the figure pictures
                if debug > 2:
                    new_filename = filename + "x_field " + str(i) + "_" + str(j) + "arotated" ".jpg"
                    cv2.imwrite(new_filename, figure_image)
                    new_filename = filename + "x_field " + str(i) + "_" + str(j) + "canny" ".jpg"
                    cv2.imwrite(new_filename, figure_image_canny)

                """
                # template-matching doesn't work yet, so this part ist outcommented 
                # load template
                filenames = os.listdir("templates/")
                max_match_value = 0
                for filename in filenames:
                    if filename.endswith("png"):
                        #print("template: "+"/templates/" + filename)
                        dir_path = os.path.dirname(os.path.realpath(__file__))
                        pathname = dir_path + "/templates/"+ filename;
                        template_image = cv2.imread(pathname, cv2.IMREAD_COLOR)

                        destination_height = 500
                        destination_width = int(template_image.shape[1] / template_image.shape[0] * destination_height)
                        template_image = cv2.resize(template_image, (destination_width, destination_height), interpolation=cv2.INTER_CUBIC)
                        clahe_cliplimit = 7
                        clahe_tilegridsize = 4
                        canny_min = 150
                        canny_max = 200
                        template_image = canny(clahe(gray(template_image)))

                        destination_width = int(template_image.shape[1])
                        destination_height = int(template_image.shape[0])
                        template_image = cv2.resize(template_image, (destination_width, destination_height), interpolation=cv2.INTER_CUBIC)

                        for scale_template in range(10):
                            template_image = cv2.resize(template_image, (int(template_image.shape[1]*0.9), int(template_image.shape[0]*0.9)), interpolation=cv2.INTER_CUBIC)
                            # try to find match
                            if figure_image_canny.shape[0] > template_image.shape[0] and figure_image_canny.shape[1] > template_image.shape[1]:
                                match_result = cv2.matchTemplate(figure_image_canny, template_image, cv2.TM_CCOEFF_NORMED)
                                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(match_result)
                                (tH, tW) = template_image.shape[:2]
                                if maxVal > max_match_value:
                                    max_match_value = maxVal
                                    max_val_result = figure_image_canny.copy()
                                    max_val_result = cv2.cvtColor(max_val_result, cv2.COLOR_GRAY2BGR )
                                    max_val_result = cv2.rectangle(max_val_result, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                                    max_val_filename = filename
                                    max_val_template = template_image.copy()

                if debug > 1:
                    print("maxVal: "+str((maxVal)))
                    cv2.namedWindow("Match Result", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Match Result", (max_val_result.shape[1],max_val_result.shape[0]))
                    cv2.imshow("Match Result", max_val_result)
                    cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Template", (max_val_template.shape[1], max_val_template.shape[0]))
                    cv2.imshow("Template", max_val_template)
                    cv2.waitKey(0)
                    cv2.destroyWindow("Template")
                    cv2.destroyWindow("Match Result")

                """


# the function tests 24 default images
def test_ride():
    global filename
    original_image = read_image_from_file("/work_data/00.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/01.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/02.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/03.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/04.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/05.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/06.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/07.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/08.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/09.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/10.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/11.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/12.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/13.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/14.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/15.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/16.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/17.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/18.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/19.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/20.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/21.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/22.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/23.jpg")
    toolchain_00(original_image)
    original_image = read_image_from_file("/work_data/24.jpg")
    toolchain_00(original_image)


# the function reads the image via pipe
def read_image_via_pipe():
    # get the base64 image through the pipe
    data = sys.stdin.readline()
    # decode image
    image = base64.b64decode(data)
    nparr = np.fromstring(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # return the received image
    return img


# the function reads the image from file
def read_image_from_file(file):
    global filename
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + file
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if debug > 0:
        print()
        print()
        print()
        print("image loaded: " + filename)
        print("----------------------------------------------------------------------------")
    return img




# get the image wether via pipe or from file
#original_image = read_image_via_pipe()
if debug > 0:
    #original_image = read_image_from_file("/work_data/06.jpg")
    #toolchain_00(original_image)
    test_ride()

# generate FEN string
FEN='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
# send FEN string back
print(FEN)
sys.stdout.flush()













