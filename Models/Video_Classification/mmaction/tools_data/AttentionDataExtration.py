import os
import cv2
import json

path_in = '/media/mingfan/DataHDD/211214_CarAccident_v1/train'
path_out = '/media/mingfan/DataHDD/211214_CarAccident_v1/train_64_v2'


for dir in os.listdir(path_in):
    path_2 = os.path.join(path_in, dir)
    if(path_2.split('/')[-1].split('_')[0] == 'mask'):
        continue
    if os.path.isdir(path_2): #directory check
        if os.listdir(path_2): #empty or not
            img_name_lists = []
            img_name_lists_total = []
            #Operation in one folder
            for file_name in os.listdir(path_2):
                if(file_name.split('.')[-1] == 'jpg'):
                    mask_name = file_name.replace('.jpg', '.json')
                    img_path = os.path.join(path_2, file_name)
                    json_path = os.path.join(path_2, mask_name)
                    if(os.path.exists(json_path)):
                        json_file = open(json_path)
                        json_data = json.load(json_file)
                        objects = json_data['objects']
                        img_name_lists_total.append(img_path)
                        cunt_Object = 0
                        for obj in objects:
                            if(obj['isObjectA'] == True or obj['isObjectB'] == True):
                                cunt_Object = cunt_Object + 1
                        if(cunt_Object >= 1):
                            img_name_lists.append(img_path)

                            
            print(path_2)
            print(len(img_name_lists))
            acc_index = 20
            if not img_name_lists or len(img_name_lists) <=acc_index:
                index_lowerBound = 64
            else:
                print(img_name_lists[acc_index])
                #The accidient is occured in the range [index_lowerBound, index_upperBound]
                index_lowerBound = img_name_lists_total.index(img_name_lists[3])
                index_upperBound = img_name_lists_total.index(img_name_lists[-1])
            
            #Select range
            
            num_frames = 64
            index_start = 0
            index_end = 0
            if(index_lowerBound - num_frames <=0):
                index_start = index_lowerBound
                index_end = index_start+num_frames
                # print("Get frame from {} to {}".format(0, index_end))
            if(index_lowerBound - num_frames >0):
                index_start = index_lowerBound
                index_end = index_start+num_frames
                if(index_end > len(img_name_lists_total)):
                    index_start = len(img_name_lists_total)-num_frames
                    index_end = len(img_name_lists_total)
                # print("Get frame from {} to {}".format(index_start, index_end))
            
            

            print('\n')
            candidate_frames = img_name_lists_total[index_start:index_end]
            # candidate_frames = img_name_lists_total[0:120]

            out_img_path = path_2.replace('train', 'train_64_v2')
            out_mask_path = out_img_path.replace('video', 'mask')
            os.makedirs(out_img_path, exist_ok=True)
            os.makedirs(out_mask_path, exist_ok=True)

            for frame_name in candidate_frames:
                mask_name = frame_name.replace('video', 'mask')
                if(os.path.exists(mask_name) and os.path.exists(frame_name)):
                    img = cv2.imread(frame_name)
                    img = img[0:864, 0:1920, :]
                    mask = cv2.imread(mask_name)
                    mask = mask[0:864, 0:1920, :]
                    
                    out_img_name = frame_name.replace('train', 'train_64_v2')
                    out_mask_name = mask_name.replace('train', 'train_64_v2')
                    cv2.imwrite(out_img_name, img)
                    cv2.imwrite(out_mask_name, mask)

                
                
                

                


            # for i_lists in img_name_lists:
            #     print(i_lists)
                # if(length_image < 150):
                #     print("The number of image is {} in {}".format(length_image, path_2))