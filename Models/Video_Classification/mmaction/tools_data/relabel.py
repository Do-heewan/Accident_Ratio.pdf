from math import pi
from time import process_time
import cv2
import os
import numpy

pathTXT = '/media/mingfan/DataHDD/211214_CarAccident_v1/total.txt'
labelTXT = open(pathTXT, 'r').readlines()


labelSet = []
folderSet = []
for i in range(len(labelTXT)):
    piece = labelTXT[i].strip().split()
    labelSet.append(int(piece[1]))
    folderSet.append(piece[0])

sort_index = numpy.argsort(labelSet)

label_reordered = []
forder_reordered = []

for index in range(len(sort_index)):
    label_reordered.append(labelSet[sort_index[index]])
    forder_reordered.append(folderSet[sort_index[index]])

label_num = []
for label in range(14):
    num = label_reordered.count(label)
    label_num.append(num)

label_cum = [label_num[0]]
for item in label_num[1:]:
  label_cum.append(label_cum[-1] + item)


#According to the num of label, generate train/val label txt
label_txt_candidate = []
label_num_candidate = []

label_txt_wocandidate = []
label_num_wocandidate = []

for i in range(len(label_num)):
    if(label_num[i]<10):
        label_txt_wocandidate.append(forder_reordered[label_cum[i-1]:label_cum[i]])
        label_num_wocandidate.append(label_reordered[label_cum[i-1]:label_cum[i]])
    else:
        if(i==0):
            label_txt_candidate.append(forder_reordered[0:label_cum[i]])
            label_num_candidate.append(label_reordered[0:label_cum[i]])
        else:
            label_txt_candidate.append(forder_reordered[label_cum[i-1]:label_cum[i]])
            label_num_candidate.append(label_reordered[label_cum[i-1]:label_cum[i]])

# print(label_num)
# print(label_cum)
# #print(label_txt_candidate)
# print(label_num_candidate)

# print(label_txt_wocandidate)
# print(label_num_wocandidate)

trainTXT = open('/media/mingfan/DataHDD/211214_CarAccident_v1/train-regen.txt', 'a')
ValTXT = open('/media/mingfan/DataHDD/211214_CarAccident_v1/val-regen.txt', 'a')
val_ratio = 0.75
for i_list in range(len(label_num_candidate)):
    i_num_candidate = label_num_candidate[i_list]
    i_txt_candidate = label_txt_candidate[i_list]
    train_valid = int(numpy.floor(len(label_num_candidate[i_list]) * val_ratio))
    val_valid =len(i_num_candidate) - train_valid
    #For train
    for i_train in range(train_valid):
        lines_train = "{} {}\n".format(i_txt_candidate[i_train], i_num_candidate[i_train])
        trainTXT.write(lines_train)

    #For val
    for i_val in range(val_valid):
        lines_val = "{} {}\n".format(i_txt_candidate[train_valid+i_val], i_num_candidate[train_valid+i_val])
        ValTXT.write(lines_val)
ValTXT.close()

#For train add
for i_list in range(len(label_num_wocandidate)):
     i_num_candidate = label_num_wocandidate[i_list]
     i_txt_candidate = label_txt_wocandidate[i_list]
     length = len(i_num_candidate)
     if(length > 0):
        for i_add in range(len(i_num_candidate)):
            lines_train = "{} {}\n".format(i_txt_candidate[i_add], i_num_candidate[i_add])
            trainTXT.write(lines_train)
trainTXT.close()
