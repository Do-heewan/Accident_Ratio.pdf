import os
import cv2
import json

path_folder = '/media/mingfan/DataHDD/211214_CarAccident_v1'
path_txt = os.path.join(path_folder, 'train-regen.txt')
path_txt_new = os.path.join(path_folder, 'train-regen-new.txt')

trainTXT = open(path_txt, 'r').readlines()
trainTXT_New = open(path_txt_new, 'a')

for i in range(len(trainTXT)):
    piece = trainTXT[i].strip().split()
    folder_name = piece[0]
    folder_path = os.path.join(path_folder, folder_name)
    num_frame = int(len(os.listdir(folder_path)) / 2)

    label = piece[1]
    lines_train = "{} {} {}\n".format(folder_name, num_frame, label)
    trainTXT_New.writelines(lines_train)

trainTXT_New.close()