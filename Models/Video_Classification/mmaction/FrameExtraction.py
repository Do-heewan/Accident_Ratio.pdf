from time import process_time
import cv2
import os


path = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/val/'
data_folder_path = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/'
label_file = open('/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/kinetics_tiny_val_video.txt', 'r')
Lines = label_file.readlines()

#Label file for frames


for line in Lines:
    label = line.split(' ')[-1]
    name = line.split(' ')[-2]
    pure_name_video = name.split('.')[-2]
    frame_data_folder_name = path + pure_name_video
    os.makedirs(frame_data_folder_name, exist_ok=True)
    # print("Line{}: {}".format(count, line.strip()))
    #Get MP4 name
    print("Mp4:{}: Label:{}".format(name, label))
    #Read mp4 file
    name_mp4 = path + name
    vidcap = cv2.VideoCapture(name_mp4)
    success,image = vidcap.read()
    count = 0
    while success:
        #Image name
        name_img_out = frame_data_folder_name + '/' + 'img_' + str(count).zfill(5) + '.jpg'
        cv2.imwrite(name_img_out, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    
    label_file_for_frame = open('/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/kinetics_tiny_val_frame.txt', 'a+')
    Lable_line = pure_name_video + ' ' + str(count) + ' ' + str(label)
    label_file_for_frame.writelines(Lable_line)
label_file_for_frame.close()

# for f in files:
#     video_name = path + f
#     print(video_name)

# vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
# success,image = vidcap.read()
# count = 0
# while success:
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1



  # Tutorial 3: Adding New Dataset
# https://github.com/open-mmlab/mmaction2/blob/master/docs/tutorials/3_new_dataset.md#an-example-of-a-custom-dataset