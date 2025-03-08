#%%
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmengine import Config
from mmdet.apis import inference_detector, init_detector

# from dataset import TrafficDataSet  # 데이터셋 클래스 import

import cv2
import time
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import os
import sys
from datetime import datetime, timedelta

def convert_kst(timestamp):
    dt_tm_utc = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    tm_kst = dt_tm_utc + timedelta(hours=9)
    str_datetime = tm_kst.strftime('%Y-%m-%d %H:%M:%S')
    return str_datetime

#%%
#config_path = "/root/detectors_cascadercnn/work_dir/cascade_rcnn_cfg.py"
#checkpoint_path = "/root/detectors_cascadercnn/work_dir/best_coco_bbox_mAP_epoch_10.pth"
#test_json = "/root/datasplit/test.json"
#save_dir = "/root/datasplit/"

config_path = "C:/Users/Noh/github/Accident_Prediction_Prevent/Models/Object_Detection/detectors_cascadercnn/config/cascade_rcnn_cfg.py"
checkpoint_path = "C:/Users/Noh/github/Accident_Prediction_Prevent/Models/Object_Detection/detectors_cascadercnn/work_dir/best_coco_bbox_mAP_epoch_10.pth"

test_data = "C:/Users/Noh/github/Accident_Prediction_Prevent/Models/Object_Detection/sample_data/" # 테스트 셋
test_json = test_data + "coco_annotations - bb_1_170402_pedestrian_112_329.json" # 테스트 데이터셋 모음 json파일

save_dir = "C:/Users/Noh/github/Accident_Prediction_Prevent/Models/Object_Detection/datasplit/" # 결과 저장 경로

test_set_coco = COCO(test_json)

real_categories_ids = [
    dict(id = 'vehicle', name='차량'),
    dict(id = 'pedestrian', name='보행자'),
    dict(id = 'two-wheeled-vehicle', name='이륜차'),
    dict(id = 'bike', name='자전거'),
    dict(id = 'traffic-sign', name='표지판'),
    dict(id = 'traffic-light-red', name='신호등(적색)'),
    dict(id = 'traffic-light-green', name='신호등(녹색)'),
    dict(id = 'traffic-light-etc', name='신호등(기타)'),
    dict(id = 'crosswalk', name='횡단보도'),]

categories_ids = test_set_coco.loadCats([0,1,2,3,4,5,6,7,8])

test_image_ids = test_set_coco.getImgIds()
cfg = Config.fromfile(config_path)
cfg.device = 'cpu'
model = init_detector(cfg, checkpoint_path, device = 'cpu')
eval_ids = []
eval_dict = dict()

if os.path.exists(os.path.join(save_dir, "pred.json")):
    os.remove(os.path.join(save_dir, "pred.json"))

if os.path.exists(os.path.join(save_dir, "eval_log.txt")):
    os.remove(os.path.join(save_dir, "eval_log.txt"))

pred_list_accum = []

init_time = time.time()
command = "실행 명령어:", "python " + " ".join(sys.argv)
print("%s %s" % (command[0], command[1]))

with open(os.path.join(save_dir, "eval_log.txt"), "a") as f:
    command = "실행 명령어:", "python " + " ".join(sys.argv)
    f.write("%s %s\n\nStart Time Stamp: %s\n" % (command[0], command[1], convert_kst(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(init_time)))))
    
for i in tqdm(range(len(test_image_ids))):
    
    if i == 0:
        timestamp = init_time
    else:
        timestamp = time.time()

    image_info = test_set_coco.loadImgs(test_image_ids[i])[0]
    image_path = os.path.join(test_data, image_info['file_name'])  # 전체 경로 생성

    '''
    if not os.path.exists(image_path):
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        continue
    '''

    img = cv2.imread(image_path)
    
    '''
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        continue
    '''

    result = inference_detector(model, image_path)
    
    labels = list(result.pred_instances.labels.detach().cpu().numpy())
    bboxes = list(result.pred_instances.bboxes.detach().cpu().numpy())
    scores = (result.pred_instances.scores.detach().cpu().numpy())

    pred_list = []
    for bbox, label, score in zip(bboxes, labels, scores):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
    
        w = xmax - xmin
        h = ymax - ymin
    
        pred_infos = {
            "image_id": int(test_image_ids[i]), 
            "category_id": int(label), 
            "bbox": [xmin,ymin,w, h], 
            "score": float(score),}

        pred_list.append(pred_infos)    
        pred_list_accum.append(pred_infos)

    if (i != 0 and i%1000 == 0) or i == len(test_image_ids) - 1:
        with open(os.path.join(save_dir, "pred_%s.json" % (i+1)), 'w') as file:
            json.dump(pred_list_accum, file, indent=4)
        pred_list_accum = []
    
    log = "Step: %s/%s \nTime Stamp : %s" % (i + 1, len(test_image_ids), convert_kst(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)))) + "\n"
    GT_i = test_set_coco.loadAnns(test_set_coco.getAnnIds(test_image_ids[i]))
    with open(os.path.join(save_dir, "eval_log.txt"), "a") as f:
        log += "Image Id : " + str(test_image_ids[i]) + "\n"
        log += "FilePath : " + image_path + "\n"
        log += "Ground Truth : \n"
        
        for i in range(0, len(GT_i)):
            class_name = categories_ids[GT_i[i]['category_id']]['name']
            for j in real_categories_ids:
                if j['id'] == class_name:
                    categories_name = j['name']
            log += '    category : ' + categories_name + ', bbox : ' + str(GT_i[i]['bbox']) + "\n"      
        log += "Prediction Result : \n"
        
        for i in range(0, len(pred_list)):
            class_name = categories_ids[pred_list[i]['category_id']]['name']
            for j in real_categories_ids:
                if j['id'] == class_name:
                    categories_name = j['name']
            log += '    category_id : ' + categories_name + ', bbox : ' + str(pred_list[i]['bbox']) + ', score : ' + str(pred_list[i]['score']) + "\n"      
        log += "\n"
        f.write(log)

pred_json_paths = glob(os.path.join(save_dir,"pred_*.json"))
anns = []
for i in range(len(pred_json_paths)):
    with open(pred_json_paths[i]) as f:
        a = json.load(f)
        anns += a

with open(os.path.join(save_dir,"pred.json"), 'w') as file:
    json.dump(anns, file, indent=4)

coco_pred = test_set_coco.loadRes(os.path.join(save_dir,"pred.json"))
coco_eval = COCOeval(test_set_coco, coco_pred, 'bbox')
coco_eval.params.imgIds = test_image_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
print()

end_time = time.time()
test_time = end_time - init_time

with open(os.path.join(save_dir,"eval_log.txt"), "a") as f:
    f.write("Evaluate annotation type *bbox*\n")
    f.write("Accumulating evaluation results...\n")
    f.write(" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[0],3))))
    f.write(" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[1],3))))
    f.write(" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[2],3))))
    f.write(" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[3],3))))
    f.write(" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[4],3))))
    f.write(" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[5],3))))
    f.write(" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = %s\n" % (str(np.round(coco_eval.stats[6],3))))
    f.write(" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = %s\n" % (str(np.round(coco_eval.stats[7],3))))
    f.write(" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[8],3))))
    f.write(" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[9],3))))
    f.write(" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[10],3))))
    f.write(" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %s\n" % (str(np.round(coco_eval.stats[11],3))))
    f.write("Test Complete Time Stamp : %s\n" % (convert_kst(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))))
    f.write("Test Execution Time : %s\n" % (time.strftime('%H:%M:%S', time.localtime(test_time))))
# %%