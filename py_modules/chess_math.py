import math
import numpy as np
import cv2
import glob
import math




def reduce_line_list_by_horizontal_length_threshold(line_list, threshold):
    res_list = []
    for i in range(len(line_list)):
        if abs(line_list[i][2] - line_list[i][0]) > threshold:
            res_list.append(line_list[i])
    return res_list


def reduce_line_list_by_vertical_length_threshold(line_list, threshold):
    res_list = []
    for i in range(len(line_list)):
        if abs(line_list[i][3] - line_list[i][1]) > threshold:
            res_list.append(line_list[i])
    return res_list


def get_line_from_points(point1, point2):
    line = []
    line.append(point1[0])
    line.append(point1[1])
    line.append(point2[0])
    line.append(point2[1])
    return line


def get_min_distance_from_point_to_one_point_of_a_list(point, point_list):
    min_distance = 1e12
    for i in range(len(point_list)):
        test_point = point_list[i]
        distance = distance_points_2d(point, test_point)
        if distance < min_distance:
            min_distance = distance
    return min_distance


def get_nearest_point_out_of_a_list(point, point_list):
    min_distance = 1e12
    for i in range(len(point_list)):
        test_point = point_list[i]
        distance = distance_points_2d(point, test_point)
        if distance < min_distance:
            min_distance = distance
            res_point = test_point
    return res_point


"""
the function returns a rotation matrix calculated from three given euler angles
: param z_angle  : z euler angle    
: param x1_angle : x1 euler angle
: param z2_angle : z2 euler angle
: return   : the rotation matrix 
"""
def rotation_matrix_from_euler_angles(z_angle, x1_angle, z2_angle):
    alpha = z_angle
    beta = x1_angle
    gamma = z2_angle
    rotation_matrix = []
    r00 = math.cos(alpha) * math.cos(gamma) - math.sin(alpha) * math.cos(beta) * math.sin(gamma)
    r01 = -math.sin(alpha) * math.cos(gamma) - math.cos(alpha) * math.cos(beta) * math.sin(gamma)
    r02 = math.sin(beta) * math.sin(gamma)
    r10 = math.cos(alpha) * math.sin(gamma) + math.sin(alpha) * math.cos(beta) * math.cos(gamma)
    r11 = -math.sin(alpha) * math.sin(gamma) + math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    r12 = -math.sin(beta) * math.cos(gamma)
    r20 = math.sin(alpha) * math.sin(beta)
    r21 = math.cos(alpha) * math.sin(beta)
    r22 = math.cos(beta)
    line = []
    line.append(r00)
    line.append(r01)
    line.append(r02)
    rotation_matrix.append(line)
    line = []
    line.append(r10)
    line.append(r11)
    line.append(r12)
    rotation_matrix.append(line)
    line = []
    line.append(r20)
    line.append(r21)
    line.append(r22)
    rotation_matrix.append(line)
    rotation_matrix = np.array(rotation_matrix, np.float32)
    return rotation_matrix





"""
the function calculates a initial camera matrix from given points on the image
: param image_points     : list of four image points as corners of one chessfield in pixel coordinates    
: param chess_field_size : the (valued) size of one chessfield im mm
: param image_width      : the width of the image in pixels
: param image_height     : the height of the image in pixels
: return                 : 
"""
def get_camera_matrix(image_points, chess_field_size, image_width, image_height):
    img_points = nparray_2D_from_point_list(image_points)
    #mod_points = np.array([[0.0, 0.0, 0.0], [chess_field_size, 0.0, 0.0], [0.0, -chess_field_size, 0.0], [chess_field_size, -chess_field_size, 0.0]], np.float32)
    mod_points = np.array([[0.0, 0.0, 0.0], [chess_field_size, 0.0, 0.0], [0.0, chess_field_size, 0.0], [chess_field_size, chess_field_size, 0.0]], np.float32)
    objpoints = []
    imgpoints = []
    objpoints.append(mod_points)
    imgpoints.append(img_points)
    camera_matrix = cv2.initCameraMatrix2D(objpoints,imgpoints, (image_width,image_height), 1)
    return camera_matrix



def get_next_camera_matrix(image_points, model_points, image_width, image_height):
    img_points = nparray_2D_from_point_list(image_points)
    mod_points = nparray_3D_from_point_list(model_points)
    objpoints = []
    imgpoints = []
    objpoints.append(mod_points)
    imgpoints.append(img_points)
    camera_matrix = cv2.initCameraMatrix2D(objpoints,imgpoints, (image_width,image_height), 1)
    return camera_matrix


def homogen_lines(line_list):
    res_list = []
    new_line = []
    for i in range(len(line_list)):
        line = line_list[i]
        new_line = line
        if line[0] > line[2]:
            new_line[0] = line[2]
            new_line[1] = line[3]
            new_line[2] = line[0]
            new_line[3] = line[1]
        res_list.append(new_line)
    return res_list


def delete_identical_lines_from_list(line_list):
    res_list = []
    for i in range(len(line_list)):
        identical = False
        source = line_list[i]
        for j in range(len(res_list)):
            dest = res_list[j]
            if source == dest:
                identical = True
        if identical == False:
            res_list.append(source)
    return res_list


def get_second_middle_line_of_line_list(line_list, first_middle_line, section_line, target_distance):
    first_section_point = section_point_2d(first_middle_line, section_line)
    min_failure = 1e12
    for i in range(len(line_list)):
        second_section_point = section_point_2d(line_list[i], section_line)
        failure = abs(target_distance-distance_points_2d(first_section_point, second_section_point))
        if failure < min_failure:
            min_failure = failure
            second_line = line_list[i]
    return second_line


def get_sum_of_2D_vectors(vector1, vector2):
    new_vector = [vector1[0] + vector2[0], vector1[1] + vector2[1]]
    return new_vector


def get_2D_vector_length(vector):
    vector_x = vector[0]
    vector_y = vector[1]
    length = math.sqrt(vector_x * vector_x + vector_y * vector_y)
    return length


def change_2D_vector_to_length(vector, length):
    actual_length = get_2D_vector_length(vector)
    factor = length / actual_length
    new_vector = [vector[0] * factor , vector[1] * factor]
    return new_vector


def get_pattern_distance(line_list, section_line, section_center):
    increase_pattern_distance = False
    if section_center[0] != 9999:
        increase_pattern_distance = True
    section_points = []
    for i in range(len(line_list)):
        section_points.append(section_point_2d(line_list[i], section_line))
    min_pattern_distance = int(max_distance_of_point_list(section_points) / 12)
    max_pattern_distance = int(max_distance_of_point_list(section_points) / 8)
    min_distance_sum = 1e12
    if section_line[1] > section_line[3]:
        section_line_vector = [ (section_line[0] - section_line[2]) , section_line[1] - section_line[3] ]
    else:
        section_line_vector = [ (section_line[2] - section_line[0]) , section_line[3] - section_line[1] ]
    for test_distance in range(min_pattern_distance, max_pattern_distance):  #loop the distance
        for i in range(len(section_points)):                                 #loop the startpoint
            start_point = section_points[i]
            if increase_pattern_distance == True:
                increase_factor = 1.0 + 30 / distance_points_2d(section_center, start_point)
            else:
                increase_factor = 1.0
            test_points = []
            distance_sum = 0
            for j in range(0, 9):
                test_vector = change_2D_vector_to_length(section_line_vector, j*test_distance * math.pow(increase_factor, j))
                test_point = get_sum_of_2D_vectors(start_point, test_vector)
                test_points.append(test_point)
                min_distance = 1e12
                for k in range(len(section_points)):
                    distance = distance_points_2d(test_point, section_points[k])
                    if distance < min_distance:
                        min_distance = distance
                distance_sum += min_distance
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                pattern_distance = test_distance
                pattern_points = test_points
    return pattern_points


def get_pattern_distance_old(line_list, section_line, section_center):
    section_points = []
    for i in range(len(line_list)):
        section_points.append(section_point_2d(line_list[i], section_line))
    min_pattern_distance = int(max_distance_of_point_list(section_points) / 12)
    max_pattern_distance = int(max_distance_of_point_list(section_points) / 8)
    min_distance_sum = 1e12
    section_line_vector = [(section_line[2] - section_line[0]), section_line[3] - section_line[1]]
    for test_distance in range(min_pattern_distance, max_pattern_distance):  # loop the distance
        for i in range(len(section_points)):  # loop the startpoint
            start_point = section_points[i]
            test_points = []
            distance_sum = 0
            for j in range(0, 9):
                test_vector = change_2D_vector_to_length(section_line_vector,- j * test_distance)
                test_point = get_sum_of_2D_vectors(start_point, test_vector)
                test_points.append(test_point)
                min_distance = 1e12
                for k in range(len(section_points)):
                    distance = distance_points_2d(test_point, section_points[k])
                    if distance < min_distance:
                        min_distance = distance
                distance_sum += min_distance
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                pattern_distance = test_distance
                pattern_points = test_points
    return pattern_points


def sort_line_list_by_height(line_list):
    angle_sum = 0
    pair = []
    pair_list = []
    res_list = []
    for i in range(len(line_list)):
        angle_sum += get_angle_from_line(line_list[i])
    average_angle = angle_sum / len(line_list)
    if abs(average_angle) < abs(math.pi/2-average_angle):
        # horizontal style lines
        for i in range(len(line_list)):
            value = get_section_with_y_axis(line_list[i])
            pair = []
            pair.append(line_list[i])
            pair.append(value)
            pair_list.append(pair)
        pair_list = sorted(pair_list, key=lambda item: item[1])
        for i in range(len(pair_list)):
            res_list.append(pair_list[i][0])
    else:
        # vertical style lines
        for i in range(len(line_list)):
            value = get_section_with_x_axis(line_list[i])
            pair = []
            pair.append(line_list[i])
            pair.append(value)
            pair_list.append(pair)
        pair_list = sorted(pair_list, key=lambda item: item[1])
        for i in range(len(pair_list)):
            res_list.append(pair_list[i][0])
    return res_list


def get_pitch_of_line(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    if (x2 - x1) != 0:
        pitch = (y2-y1)/(x2-x1)
    else:
        pitch = 1e12
    return pitch


def get_section_with_y_axis(line):
    pitch = get_pitch_of_line(line)
    x1 = line[0]
    y1 = line[1]
    y0 = y1 - pitch * x1
    return y0


def get_section_with_x_axis(line):
    pitch = get_pitch_of_line(line)
    x1 = line[0]
    y1 = line[1]
    y0 = y1 - pitch * x1
    if pitch != 0:
        x0 = -y0 / pitch
    else:
        x0 = 1e12
    return x0


def get_middle_line_of_line_list(line_list):
    angle_sum = 0
    for i in range(len(line_list)):
        angle_sum += get_angle_from_line(line_list[i])
    average_angle = angle_sum / len(line_list)
    max_value = -1e12
    min_value = 1e12
    if abs(average_angle) < abs(math.pi/2-average_angle) or abs(math.pi - average_angle) < abs(math.pi/2-average_angle):
        # horizontal style lines
        for i in range(len(line_list)):
            value = get_section_with_y_axis(line_list[i])
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value
        middle_value = min_value + 4/5*(max_value - min_value) / 2
        min_distance_to_middle_value = 1e12
        for i in range(len(line_list)):
            value = get_section_with_y_axis(line_list[i])
            if abs(middle_value - value) < min_distance_to_middle_value:
                chosen_middle_value = value
                middle_line = line_list[i];
                min_distance_to_middle_value = abs(middle_value - value)
    else:
        # vertical style lines
        for i in range(len(line_list)):
            value = get_section_with_x_axis(line_list[i])
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value
        middle_value = min_value + (max_value - min_value) / 2
        min_distance_to_middle_value = 1e12
        for i in range(len(line_list)):
            value = get_section_with_x_axis(line_list[i])
            if abs(middle_value - value) < min_distance_to_middle_value:
                middle_line = line_list[i];
                min_distance_to_middle_value = abs(middle_value - value)
    return middle_line



def get_orthogonal_2D(line):
    vector1 = [line[2]-line[0], line[3]-line[1], 0]
    vector2 = [0, 0, 1]
    res_vector = cross_product(vector1, vector2)
    return res_vector[:2]


def cross_product(vector1, vector2):
    vector_x = vector1[1] * vector2[2] - vector1[2] * vector2[1]
    vector_y = vector1[2] * vector2[0] - vector1[0] * vector2[2]
    vector_z = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    vector = [vector_x, vector_y, vector_z]
    return vector


def min_distance_point_line_2D(point, line):
    orthogonal_vector = get_orthogonal_2D(line)
    v1x = line[2] - line[0]
    v1y = line[3] - line[1]
    v2x = orthogonal_vector[0]
    v2y = orthogonal_vector[1]
    x1 = line[0]
    y1 = line[1]
    x = point[0]
    y = point[1]
    if (v2y * v1x - v2x * v1y) != 0:
        lambda1 = (y * v1x - y1 * v1x - x * v1y + x1 * v1y) / (v2y * v1x - v2x * v1y)
    else:
        lambda1 = 1e12
    point1 = [0, 0]
    point2 = [lambda1 * v2x, lambda1 * v2y]
    distance = distance_points_2d(point1, point2)
    return distance


def max_distance_of_point_list(point_list):
    max_distance = 0
    for i in range(len(point_list)):
        for j in range(len(point_list)):
            distance = distance_points_2d(point_list[i], point_list[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def reduce_line_list_by_angle(line_list, target_angle, max_deviation):
    res_list = []
    for i in range(len(line_list)):
        angle = get_angle_from_line(line_list[i])
        if abs(angle - target_angle) < max_deviation:
            res_list.append(line_list[i])
        elif abs(angle - target_angle - math.pi) < max_deviation:
            res_list.append(line_list[i])
        elif abs(angle - target_angle + math.pi) < max_deviation:
            res_list.append(line_list[i])
    return res_list



def print_distance_to_point(line_list, target_point):
    for i in range(len(line_list)):
        distance = min_distance_point_line_2D(target_point, line_list[i])
        print("distance_to_point: " +str(distance))




def reduce_line_list_by_distance_to_point(line_list, target_point, max_distance):
    res_list = []
    for i in range(len(line_list)):
        distance = min_distance_point_line_2D(target_point, line_list[i])
        if distance <= max_distance:
            res_list.append(line_list[i])
    return res_list



def reduce_points_to_number_by_distance_to_average(point_list, number):
    res_list = point_list
    max_distance = 10000
    while (len(res_list) > number and max_distance) > 0.01:
        max_distance = max_distance * 0.9
        average = average_point_of_point_list(res_list)
        res_list = reduce_point_list_by_distance_to_point(res_list, average, max_distance)
    return res_list


def reduce_point_list_by_distance_to_point(point_list, target_point, max_distance):
    res_list = []
    for i in range(len(point_list)):
        distance = distance_points_2d(point_list[i], target_point)
        if distance <= max_distance:
            res_list.append(point_list[i])
    return res_list



def pitch_angle_parallel(line_list):
    max_parallels = 0
    for i in range(len(line_list)):
        angle1 = get_angle_from_line(line_list[i])
        parallels = 0
        for j in range(i, len(line_list)):
            angle2 = get_angle_from_line(line_list[j])
            if abs(angle1-angle2)<0.02:
                parallels = parallels + 1
        if parallels > max_parallels:
            res_angle = angle1
            max_parallels = parallels
    if max_parallels < 0.5*len(line_list):
        res_angle = 10000
    return res_angle



def section_point_list(line_list):
    section_points_list = []
    for i in range(len(line_list)):
        for j in range(len(line_list)):
            if i != j:
                section_point = section_point_2d(line_list[i], line_list[j])
                if section_point[0] != -1 or section_point[0] != -1:
                    section_points_list.append(section_point)
    return section_points_list


def section_point_list_from_two_line_lists(line_list_1, line_list_2):
    section_points_list = []
    for i in range(len(line_list_1)):
        for j in range(len(line_list_2)):
            section_point = section_point_2d(line_list_1[i], line_list_2[j])
            if section_point[0] != -1 or section_point[0] != -1:
                section_points_list.append(section_point)
    return section_points_list


def expand_lines_to_image_size(line_list, image_width, image_height):
    res_list = []
    for i in range(len(line_list)):
        x1 = line_list[i][0]
        y1 = line_list[i][1]
        x2 = line_list[i][2]
        y2 = line_list[i][3]
        if (x2-x1) != 0:
            m = (y2-y1)/(x2-x1)
        else:
            m = 1e12
        if m == 0:
            m = 0.000001
        n = y1 - m * x1
        y_x_0 = m * 0 +n
        y_x_width = m * image_width + n
        x_y_0 = -n / m
        x_y_height = (image_height - n) / m
        res_x1 = 0
        res_y1 = y_x_0
        res_x2 = image_width
        res_y2 = y_x_width
        if res_y1 > image_height:
            res_y1 = image_height
            res_x1 = x_y_height
        if res_y2 > image_height:
            res_y2 = image_height
            res_x2 = x_y_height
        if res_y1 < 0:
            res_y1 = 0
            res_x1 = x_y_0
        if res_y2 < 0:
            res_y2 = 0
            res_x2 = x_y_0
        res_line = [int(res_x1), int(res_y1), int(res_x2), int(res_y2)]
        res_list.append(res_line)
    return res_list



def expand_lines_to_image_size_old(line_list, image_width, image_height):
    res_list = []
    for i in range(len(line_list)):
        x1 = line_list[i][0]
        y1 = line_list[i][1]
        x2 = line_list[i][2]
        y2 = line_list[i][3]
        if (x2-x1) != 0:
            m = (y2-y1)/(x2-x1)
        else:
            m = 1e12
        y_x_0 = m * (0-x1)+y1
        y_x_image_width = m * (image_width-x1)+y1
        if m != 0:
            x_y_0 = 1/m * (0-y1)+x1
        else:
            x_y_0 = 1e12 * (0-y1)+x1
        if m != 0:
            x_y_image_height = 1/m * (image_height-y1)+x1
        else:
            x_y_image_height = 1e12 * (image_height - y1) + x1
        res_x1 = 10000
        res_y1 = 10000
        res_x2 = 10000
        res_y2 = 10000
        if y_x_0 >= 0 and y_x_0 <= image_height:
            res_x1=0
            res_y1=y_x_0
        if y_x_image_width >= 0 and y_x_image_width <= image_height:
            if res_x1 == 10000:
                res_x1 = image_width
                res_y1 = y_x_image_width
            else:
                res_x2 = image_width
                res_y2 = y_x_image_width
        if x_y_0 >= 0 and x_y_0 <= image_width:
            if res_x1 == 10000:
                res_x1 = x_y_0
                res_y1 = 0
            else:
                res_x2 = x_y_0
                res_y2 = 0
        if x_y_image_height >= 0 and x_y_image_height <= image_width:
            if res_x1 == 10000:
                res_x1=x_y_image_height
                res_y1=image_height
            else:
                res_x2=x_y_image_height
                res_y2=image_height
        res_line = [int(res_x1), int(res_y1), int(res_x2), int(res_y2)]
        res_list.append(res_line)
    return res_list




def average_point_of_point_list(point_list):
    sum_x = 0
    sum_y = 0
    sum_z = 0
    for i in range(len(point_list)):
        point = point_list[i]
        point_length = len(point)
        if point_length > 0:
            sum_x = sum_x + point_list[i][0]
        if point_length > 1:
            sum_y = sum_y + point_list[i][1]
        if point_length > 2:
            sum_z = sum_z + point_list[i][2]
    mid_x = sum_x / len(point_list)
    mid_y = sum_y / len(point_list)
    mid_z = sum_z / len(point_list)
    res_point = []
    if point_length > 0:
        res_point.append(mid_x)
    if point_length > 1:
        res_point.append(mid_y)
    if point_length > 2:
        res_point.append(mid_z)
    return res_point


def sort_section_points_by_distance_to_middle(section_points, middle_x, middle_y):
    middle_point = [middle_x, middle_y]
    tuple_list = []
    res_list = []
    for i in range(len(section_points)):
        distance_to_middle = distance_points_2d(middle_point, section_points[i])
        tuple = (section_points[i], distance_to_middle)
        tuple_list.append(tuple)
    sorted_tuple_list = sorted(tuple_list, key=lambda item: item[1])
    for i in range(len(sorted_tuple_list)):
        tuple = sorted_tuple_list[i]
        res_list.append(tuple[0])
    return res_list


def sort_points_by_angle_to_middle(point_list):
    sum_x = 0
    sum_y = 0
    tuple_list = []
    res_list = []
    for i in range(len(point_list)):
        sum_x = sum_x + point_list[i][0]
        sum_y = sum_y + point_list[i][1]
    mid_x = sum_x / len(point_list)
    mid_y = sum_y / len(point_list)
    for i in range(len(point_list)):
        x = point_list[i][0] - mid_x
        y = point_list[i][1] - mid_y
        angle = math.atan2(y, x)
        tuple = (point_list[i], angle)
        tuple_list.append(tuple)
    sorted_tuple_list = sorted(tuple_list, key=lambda item: item[1])
    for i in range(len(sorted_tuple_list)):
        tuple = sorted_tuple_list[i]
        res_list.append(tuple[0])
    return res_list



def nparray_3D_from_point_list(point_list):
    array = []
    for i in range(len(point_list)):
        res_point = []
        point = point_list[i]
        for j in range(3):
            if j < len(point):
                res_point.append(point[j])
            else:
                res_point.append(0)
        array.append(res_point)
    nparray = np.array(array, np.float32)
    return nparray


def nparray_2D_from_point_list(point_list):
    array = []
    for i in range(len(point_list)):
        res_point = []
        point = point_list[i]
        for j in range(2):
            if j < len(point):
                res_point.append(point[j])
            else:
                res_point.append(0)
        array.append(res_point)
    nparray = np.array(array, np.float32)
    return nparray








def section_nully_2d(line1):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    a1 = 0
    b1 = 0
    a2 = 0
    b2 = 10
    f = x2 - x1
    g = y2 - y1
    h = a2 - a1
    e = b2 - b1
    nue=0
    mue=0
    if ((g*h-e*f)!=0):
        nue = (b1*f-y1*f-a1*g+x1*g)/(g*h-e*f)
        if (f!=0):
            mue = (a1+h*nue-x1)/f
    section=[]
    if (mue!=0 and nue!=0):
        section_x = x1 + (x2 - x1) * mue
        section_y = y1 + (y2 - y1) * mue
    else:
        section_x = -1
        section_y = -1
    section.append(section_x)
    section.append(section_y)
    return section


def section_nullx_2d(line1):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    a1 = 0
    b1 = 0
    a2 = 10
    b2 = 0
    f = x2 - x1
    g = y2 - y1
    h = a2 - a1
    e = b2 - b1
    nue=0
    mue=0
    if ((g*h-e*f)!=0):
        nue = (b1*f-y1*f-a1*g+x1*g)/(g*h-e*f)
        if (f!=0):
            mue = (a1+h*nue-x1)/f
    section=[]
    if (mue!=0 and nue!=0):
        section_x = x1 + (x2 - x1) * mue
        section_y = y1 + (y2 - y1) * mue
    else:
        section_x = -1
        section_y = -1
    section.append(section_x)
    section.append(section_y)
    return section


def section_point_2d(line1,line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    a1 = line2[0]
    b1 = line2[1]
    a2 = line2[2]
    b2 = line2[3]
    f = x2 - x1
    g = y2 - y1
    h = a2 - a1
    e = b2 - b1
    nue=0
    mue=0
    if ((g*h-e*f)!=0):
        nue = (b1*f-y1*f-a1*g+x1*g)/(g*h-e*f)
        if (f!=0):
            mue = (a1+h*nue-x1)/f
    section=[]
    if (mue!=0 and nue!=0):
        section_x = x1 + (x2 - x1) * mue
        section_y = y1 + (y2 - y1) * mue
    else:
        section_x = -1
        section_y = -1
    if (abs(section_x) > 100000 or abs(section_y) > 100000):
        section_x = -1
        section_y = -1
    section.append(section_x)
    section.append(section_y)
    return section


def distance_points_2d(point1,point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    ax = x2 - x1
    ay = y2 - y1
    distance = math.sqrt(ax * ax + ay * ay)
    return distance



def reduce_points(points,distance):
    reduced_points=[]
    if points is not None:
        for point1 in points:
            takeover=True
            if reduced_points is not None:
                for point2 in reduced_points:
                    if (distance_points_2d(point1,point2))<distance:
                        takeover=False
            if takeover==True:
                reduced_points.append(point1)
    return reduced_points



def line_length_2d(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    ax = x2 - x1
    ay = y2 - y1
    length = math.sqrt(ax * ax + ay * ay)
    return length




def get_list_from_line(line):
    line_list = []
    x1, y1, x2, y2 = line.ravel()
    line_list.append(x1)
    line_list.append(y1)
    line_list.append(x2)
    line_list.append(y2)
    line_list.append(line_length_2d(line_list))
    line_list.append(get_angle_from_line(line_list))
    return line_list


def get_list_from_linearray(linearray):
    unsorted_list = []
    sorted_list = []
    if linearray is not None:
        for line in linearray:
            line_list = get_list_from_line(line)
            unsorted_list.append(line_list)
        sorted_list=sorted(unsorted_list, key=lambda line: line[4], reverse=True)
    return sorted_list


def divide_line_list_by_average_angle(line_list,direction):
    divided_list = []
    counter=0
    angle_sum=0
    for line in line_list:
        angle_sum += line[5]
    average_angle=angle_sum/len(line_list)
    for line in line_list:
        if (line[5]<=average_angle):
            if (direction == 0):
                divided_list.append(line)
        else:
            if (direction == 1):
                divided_list.append(line)
    return divided_list


def divide_line_list_by_direction(line_list,direction):
    line_list_1 = []
    line_list_2 = []
    angle_sum=0
    angle_sum_1=0
    angle_sum_2=0
    for line in line_list:
        if line[5]>math.pi*0.90:
            line[5]=line[5]-math.pi
    for line in line_list:
        angle_sum += line[5]
    average_angle=angle_sum/len(line_list)
    for line in line_list:
        if (line[5]<=average_angle):
            angle_sum_1 += line[5]
            line_list_1.append(line)
        else:
            angle_sum_2 += line[5]
            line_list_2.append(line)
    average_angle_1=angle_sum_1/len(line_list_1)
    average_angle_2=angle_sum_2/len(line_list_2)
    line_list_1 = []
    line_list_2 = []
    angle_sum_1=0
    angle_sum_2=0
    for line in line_list:
        if abs(line[5]-average_angle_1) < abs(line[5]-average_angle_2):
            line_list_1.append(line)
            angle_sum_1 += line[5]
        else:
            line_list_2.append(line)
            angle_sum_2 += line[5]
    average_angle_1=angle_sum_1/len(line_list_1)
    average_angle_2=angle_sum_2/len(line_list_2)
    if abs(math.pi/2-average_angle_1) > abs(math.pi/2-average_angle_2):
        horizontals = line_list_1
        verticals = line_list_2
    else:
        horizontals = line_list_2
        verticals = line_list_1
    if direction == 0:
        divided_list = horizontals
    else:
        divided_list = verticals
    return divided_list



def divide_line_list_by_direction_new(line_list,direction):
    horizontals = []
    verticals = []
    for line in line_list:
        angle = get_angle_from_line(line)
        if angle >= math.pi * 0/4 and angle < math.pi * 1/4:
            horizontals.append(line)
        if angle >= math.pi * 1/4 and angle < math.pi * 2/4:
            verticals.append(line)
        if angle >= math.pi * 2/4 and angle < math.pi * 3/4:
            verticals.append(line)
        if angle >= math.pi * 3/4 and angle <= math.pi * 4/4:
            horizontals.append(line)
    if direction == 0:
        divided_list = horizontals
    else:
        divided_list = verticals
    return divided_list




def eliminate_sectioners(line_list, source_image):
    source_height, source_width = source_image.shape[:2]
    new_list = []
    angle_sum=0
    for line1 in line_list:
        counter = 0
        for line2 in line_list:
            section_point=section_point_2d(line1,line2)
            if (section_point[0]!=-1 and section_point[1]!=-1):
                if (section_point[0]>=0 and section_point[0]<=source_width):
                    if (section_point[1] >= 0 and section_point[1] <= source_height):
                        counter += 1
        if (counter<8):
            new_list.append(line1)
    return new_list


def eliminate_identicals(line_list):
    sorted_list=sorted(line_list, key=lambda line: line[4])
    new_list = []
    counter_01=0
    for line1 in sorted_list:
        counter_01 += 1
        takeover = True
        section_line1_x = section_nullx_2d(line1)
        section_line1_y = section_nully_2d(line1)
        counter_02 = 0
        for line2 in sorted_list:
            counter_02 += 1
            section_line2_x = section_nullx_2d(line2)
            section_line2_y = section_nully_2d(line2)
            if counter_02>counter_01:
                if (abs(get_angle_from_line(line1)-get_angle_from_line(line2))<0.05):
                    if (section_line2_x[0]!=-1):
                        if (abs(section_line2_x[0] - section_line1_x[0]) < 5):
                            takeover=False
                    if (section_line2_y[1] != -1):
                        if (abs(section_line2_y[1] - section_line1_y[1]) < 5):
                            takeover = False
        if (takeover == True):
            new_list.append(line1)
    return new_list


def get_angle_from_line(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    y = y2 - y1
    x = x2 - x1
    if (x != 0):
        angle = math.atan(abs(y) / abs(x))
        if (y >= 0 and x < 0):
            angle = math.pi - angle
        if (y < 0 and x >= 0):
            angle = math.pi - angle
    else:
        angle = math.pi/2
    if abs(angle - math.pi) < 5/180 * math.pi:
        angle = angle - math.pi
    return angle


def get_average_angle(line_list):
    angle_sum = 0
    counter=0
    for line in line_list:
        counter = counter + 1
        line_angle = get_angle_from_line(line)
        #print("line_angle:"+str(line_angle*180/math.pi))
        angle_sum += line_angle
    average_angle = angle_sum / counter
    return average_angle




