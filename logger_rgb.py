#!/usr/bin/env python

import rospy, os, tf
import numpy as np
from __builtin__ import round
from nav_msgs.msg import OccupancyGrid
import Queue, time, copy


def num_to_byte(number):
    if not isinstance(number, int):
        number = round(number, 4)
    byt = []
    for letter in str(number):
        if letter == '.':
            byt.append(46)
        elif letter == '-':
            byt.append(45)
        else:
            byt.append(int(letter)+48)
    return bytearray(byt)


# =========================================================
#   Representation of a pgm map
#   
#   P5
#   # r c1 c2 p1 p2 p3 p4 p5 p6 p7 t1 t2
#   height width
#   max_val
#   0x????
#
#   r: cell size
#   (c1, c2, c3): center of the map
#   (p1, p2, p3): location of pose
#   (p4, p5, p6, p7): angle of pose
#   t1.t2: time-stamp
# =========================================================

def print_map(message):
    global counter
    counter += 1
    global last_time
    #print "last_time %d" % last_time
    #current_time = time.clock()
    #print "current_time %d" % current_time
    #if current_time - last_time < 0.5:
    #    print current_time
    #    return
    #else:
    #   print current_time
    #   last_time = current_time
    maxint = 254
    undefined = 255
    global path
    global listener
    global target_frame
    global base_frame
    global map_index
    map_index += 1

    height = message.info.height
    width = message.info.width
    array = message.data
    r = message.info.resolution

    data = []
    # =========================================================
    #   Put in _data_ chars for .pgm format &
    #   Delete all rows that consist of only "-1"
    # =========================================================
    blank_rows = 0
    for y in range(height):
        ones_in_row = 0
        l = []
        for x in range(0,width,4):
            i = x + (height - y - 1) * width
            if array[i] == 100:
                ones_in_row += 1
                s = "%c%c%c" % (undefined,undefined,undefined)
            else:
                a = maxint - int(float(int(array[i]))*maxint/100)
                b = maxint - int(float(int(array[i+1]))*maxint/100)
                c = maxint - int(float(int(array[i+2]))*maxint/100)
                s = "%c%c%c" % (a,b,c)
            l.append(s)
        if ones_in_row == width/4:
            pass
        else:
            data.append(l)
    # =========================================================
    #   Delete all columns that consist of only "-1"
    # =========================================================
    x = 0
    while x < len(data[0]):
        ones_in_column = 0
        for y in range(len(data)):
            if ord(data[y][x][0]) == undefined:
                ones_in_column += 1
        if ones_in_column == len(data):
            for line in data:
                del line[x]
        else:
            x += 1
    # Header for .pgm black and white file
    header = bytearray([80, 54, 10])                   # P6\n
    header.extend(num_to_byte(len(data[0])))                   # width
    header.extend([32])                                        # <space>
    header.extend(num_to_byte(len(data)))                      # height
    header.extend([10])                                        # \n
    header.extend(num_to_byte(255))                            # 255
    header.extend([10])                                        # \n

    fout = open(path+'map%04d' % int(map_index) + '.ppm', 'wb')
    for i in range(len(data)):
        for j in range(len(data[0])):
            header.extend(data[i][j])
    fout.write(header)
    fout.close()
 

if __name__ == '__main__':
    counter = 0
    old_counter = 0
    map_index = 0

    rospy.init_node('ghost_map_creator', anonymous=True, log_level=rospy.INFO)
    path = rospy.get_param('~map_path', '')
    target_frame = rospy.get_param('~target_frame', 'map')
    base_frame = rospy.get_param('~base_frame', 'base_link')

    listener = tf.TransformListener()
    if path != '' and path[-1] != '/':
        path += '/'
    try:
        os.mkdir(path)
    except Exception:
        pass
    last_time = time.clock()
    rospy.Subscriber("tbm_map", OccupancyGrid, print_map)
    sleep_index = 0
    while sleep_index < 60 and not rospy.is_shutdown():
        try:
            if counter == old_counter:
                sleep_index += 1
                time.sleep(1)
            else:
                old_counter = counter
                sleep_index = 0
        except Exception:
            pass
