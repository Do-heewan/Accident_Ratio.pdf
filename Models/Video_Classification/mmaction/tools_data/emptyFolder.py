from math import pi
from time import process_time
import cv2
import os
import numpy


data_path = '/media/mingfan/DataHDD/211214_CarAccident_v1'
pathTXT = data_path + '/val-regen-tiny.txt'
labelTXT = open(pathTXT, 'r').readlines()

for i in range(len(labelTXT)):
    piece = labelTXT[i].strip().split()
    folder_name = os.path.join(data_path, piece[0])
    length_image = len(os.listdir(folder_name))
    print(length_image)
    if(length_image < 300):
        print("The number of image is {} in {}".format(length_image, folder_name))

