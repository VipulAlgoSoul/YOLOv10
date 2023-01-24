import os
import ast

import cv2
import numpy as np

def draw_from_imtxt(img_file, txt_file):
    '''This function draws bounding box from txt file'''
    print(img_file)
    img = cv2.imread(img_file)
    rows, cols, channel = img.shape

    with open(txt_file) as f:
        lines = f.readlines()

        line_fl = [[float(r) for r in i.split(" ")] for i in lines.copy()]
        cntre_crd = [get_cords_from_yolo(rows, cols,i) for i in line_fl]

    draw_img = img.copy()
    for crd in cntre_crd:
        draw_img = cv2.rectangle(draw_img, crd[1], crd[2], (0,255,200),3)

    # display_image(img)
    display_image(draw_img)

def draw_from_imtxt(img_file, txt_file, title):
    '''This function draws bounding box from txt file'''

    img = cv2.imread(img_file)
    rows, cols, channel = img.shape

    with open(txt_file) as f:
        lines = f.readlines()

        line_fl = [[float(r) for r in i.split(" ")] for i in lines.copy()]
        cntre_crd = [get_cords_from_yolo(rows, cols,i) for i in line_fl]

    draw_img = img.copy()
    for crd in cntre_crd:
        draw_img = cv2.rectangle(draw_img, crd[1], crd[2], (0,255,200),3)

    # display_image(img)
    display_image(draw_img, title = title)

def get_cords_from_yolo(img_r, img_c, single_line):
    '''this function gives coordinates from yolo'''
    class_nn = single_line[0]
    xc = int(single_line[1]*img_c)
    yc = int(single_line[2]*img_r)

    dx = int(0.5*single_line[3]*img_c)
    dy = int(0.5*single_line[4]*img_r)

    return [class_nn,(xc-dx, yc-dy), (xc+dx, yc+dy)]

def get_cords_from_yolo(img_r, img_c, single_line):
    '''this function gives coordinates from yolo'''
    class_nn = single_line[0]
    xc = int(single_line[1]*img_c)
    yc = int(single_line[2]*img_r)

    dx = int(0.5*single_line[3]*img_c)
    dy = int(0.5*single_line[4]*img_r)

    return [class_nn,(xc-dx, yc-dy), (xc+dx, yc+dy)]


def display_image(imag, shape = (600,800), title = "window", wait =0):

    image = cv2.resize(imag.copy(), shape)

    cv2.imshow(title,image)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()


# draw_from_imtxt('D:\Detection_Classification\data/train\c6.jpg', "D:\Detection_Classification\data/train\c6.txt")

