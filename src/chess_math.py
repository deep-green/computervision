import math


def line_length_2d(line):
    x1, y1, x2, y2 = line.ravel()
    ax = x2 - x1
    ay = y2 - y1
    length = math.sqrt(ax * ax + ay * ay)
    return length


def print_line_length(lines):
    if lines is not None:
        counter=0
        for i in lines:
            counter=counter+1
            print("number:" + str(counter) + " length: "+str(line_length_2d(i)))


def sort_lines_by_length(lines):
    sorted = []
    while len(lines) > 0:
        max_length = 0
        for line in lines:
            length=line_length_2d(line)
            if length>max_length:
                max_length=length
                changer=line
        sorted.append(changer)
        lines.remove(changer)


def get_list_from_linearray(linearray):
    unsorted_list = []
    for line in linearray:
        line_list=[]
        x1, y1, x2, y2 = line.ravel()
        line_list.append(x1)
        line_list.append(y1)
        line_list.append(x2)
        line_list.append(y2)
        line_list.append(line_length_2d(line))
        unsorted_list.append(line_list)
    sorted_list=sorted(unsorted_list, key=lambda line: line[4])
    return sorted_list


def get_angle_from_line(line):
    x1, y1, x2, y2 = line.ravel()
    y = y2 - y1
    x = x2 - x1
    if (x != 0):
        angle = math.atan(abs(y) / abs(x))
        if (y >= 0 and x < 0):
            angle = math.pi - angle
        if (y < 0 and x >= 0):
            angle = math.pi - angle
    else:
        angle = math.pi
    return angle


def get_main_angles(linearray):
    main_angles=[]
    angle_sum_01 = 0
    angle_sum_02 = 0
    counter_01=0
    for line in linearray:
        counter_01 = counter_01 + 1
        line_angle = get_angle_from_line(line)
        angle_sum_01 += line_angle
    angle_average = angle_sum_01 / counter_01
    angle_sum_01 = 0
    angle_sum_02 = 0
    counter_01 = 0
    counter_02 = 0
    for line in linearray:
        line_angle = get_angle_from_line(line)
        if (line_angle < angle_average):
            counter_01 += 1
            angle_sum_01 += line_angle
        else:
            counter_02 += 1
            angle_sum_02 += line_angle
    main_angle_01 = angle_sum_01 / counter_01
    main_angle_02 = angle_sum_02 / counter_02
    print("main_angle_01:"+str(main_angle_01*180/math.pi))
    print("main_angle_02:"+str(main_angle_02*180/math.pi))
    main_angles.append(main_angle_01)
    main_angles.append(main_angle_02)
    return main_angles
